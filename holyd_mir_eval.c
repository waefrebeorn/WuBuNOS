/*
 * holyd_mir_eval.c  --  HolyD AST → MIR → ISA driver evaluation.
 *
 * Parses HolyD source, emits MIR (the hourglass neck), then compiles
 * and runs via any ISA driver. This is the cross-target path: one
 * frontend, N backends.
 *
 * C11, self-contained.
 */
#include "holyd_mir_eval.h"
#include "wubu_mir.h"
#include "wubu_mir_opt.h"
#include "wubu_isa_driver.h"
#include "holyd_ast.h"
#include "holyd_types.h"
#include "holyd_lexer.h"
#include "holyd_parser.h"
#include "wubu_preproc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MIR generation state */
#define MIRGEN_MAX_VARS 256
typedef struct {
    char name[HD_MAX_IDENT_LEN];
    wubu_vr_t vr;          /* home vr for scalars (also = memory base for load/store) */
    wubu_vr_t addr;        /* memory base address (0 if not yet allocated) */
    int is_unsigned;
    int is_float;
    int is_array;
    int array_size;        /* outermost dimension */
    int array_stride;      /* inner dimension for 2D+ arrays (1 for 1D) */
    int is_struct;         /* 1 if this variable is a struct instance */
    int is_ptr_struct;     /* 1 if this variable is a pointer to a struct */
    char struct_name[HD_MAX_IDENT_LEN]; /* struct type name */
    int fn_ptr_func_id;    /* func_id this pointer var points to, or -1 */
} mir_var_t;

/* struct member offset table */
#define MAX_STRUCTS 64
#define MAX_MEMBERS 32
typedef struct {
    char name[HD_MAX_IDENT_LEN];
    char member_names[MAX_MEMBERS][HD_MAX_IDENT_LEN];
    int member_offsets[MAX_MEMBERS];
    int member_is_unsigned[MAX_MEMBERS];
    char member_type_names[MAX_MEMBERS][HD_MAX_IDENT_LEN]; /* struct type name if member is struct */
    int n_members;
    int total_size;
} mir_struct_t;

typedef struct {
    wubu_mir_prog_t *prog;
    wubu_vr_t next_vr;      /* next virtual register index */
    int has_error;
    mir_var_t vars[MIRGEN_MAX_VARS];
    int n_vars;
    /* scope stack: track vars added at each scope level for shadowing */
    int scope_var_start[MIRGEN_MAX_VARS];
    int n_scopes;
    int in_function_body; /* set when generating func body — don't pop scope */
    mir_struct_t structs[MAX_STRUCTS];
    int n_structs;
    /* loop stack for break/continue resolution */
    uint32_t loop_top[MIRGEN_MAX_VARS];   /* continue target */
    uint32_t loop_done[MIRGEN_MAX_VARS];  /* break target */
    int n_loops;
    /* function-body early-return support: when generating a function body,
     * RETURN emits `result_vr = expr; jmp ret_label` so an early return skips
     * trailing statements (MIR is single-exit). 0 when not in a function body. */
    wubu_vr_t fn_ret_vr;
    uint32_t fn_ret_label;
    /* label tracking for goto */
    char label_names[64][HD_MAX_IDENT_LEN];
    uint32_t label_ids[64];
    int n_labels;
    /* function table for MIR_CALL (collected during TU generation) */
    const HDASTNode *func_ast[MIR_MAX_FUNCTIONS];
    int func_id_of[MIR_MAX_FUNCTIONS];     /* -1 until assigned */
    int n_funcs;
    /* enum constant table */
    char enum_const_names[64][HD_MAX_IDENT_LEN];
    int64_t enum_const_vals[64];
    int n_enum_consts;
} HDMirGen;

static wubu_vr_t mir_new_vr(HDMirGen *g) {
    return g->next_vr++;
}

/* symbol table: find a declared variable's vr, or -1 */
static int mir_find_var_is_float(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_float;
    return 0;
}

static wubu_vr_t mir_find_var(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].vr;
    return (wubu_vr_t)-1;
}

/* symbol table: find a declared variable's memory base addr, or 0 */
static wubu_vr_t mir_find_var_addr(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].addr;
    return 0;
}

/* struct helpers */
static mir_struct_t *mir_find_struct(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_structs; i++)
        if (strcmp(g->structs[i].name, name) == 0) return &g->structs[i];
    return NULL;
}
static int mir_struct_member_offset(HDMirGen *g, const char *struct_name, const char *member) {
    mir_struct_t *s = mir_find_struct(g, struct_name);
    if (!s) return -1;
    for (int i = 0; i < s->n_members; i++)
        if (strcmp(s->member_names[i], member) == 0) return s->member_offsets[i];
    return -1;
}

/* Forward declaration: defined later in this file. */
static mir_struct_t *mir_find_struct_by_size(HDMirGen *g, int size);

/* Resolve the struct type name for the left side of a DOT expression.
 * Handles: direct struct-typed nodes, IDENT vars (via var table), and
 * CALL/FUNC_CALL nodes (by looking up the callee function's return type). */
static const char *mir_dot_struct_type(HDMirGen *g, const HDASTNode *left) {
    if (!left) return NULL;
    /* Direct type annotation on the node */
    if (left->type && left->type->kind == HD_TYPE_STRUCT && left->type->name[0])
        return left->type->name;
    /* IDENT: look up in var table */
    if (left->kind == HD_AST_IDENT && left->ident[0]) {
        for (int i = 0; i < g->n_vars; i++)
            if (strcmp(g->vars[i].name, left->ident) == 0 && (g->vars[i].is_struct || g->vars[i].is_ptr_struct))
                return g->vars[i].struct_name;
    }
    /* CALL/FUNC_CALL: look up callee function's return type */
    if (left->kind == HD_AST_CALL || left->kind == HD_AST_FUNC_CALL) {
        if (left->callee && left->callee->kind == HD_AST_IDENT && left->callee->ident[0]) {
            for (int i = 0; i < g->n_funcs; i++) {
                if (strcmp(g->func_ast[i]->ident, left->callee->ident) == 0) {
                    HDASTNode *fn = (HDASTNode *)g->func_ast[i];
                    /* fn->type IS the return type for HD_AST_FUNC_DECL */
                    if (fn && fn->type && fn->type->kind == HD_TYPE_STRUCT && fn->type->name[0])
                        return fn->type->name;
                    break;
                }
            }
        }
    }
    /* DOT/MEMBER: q.p — look up p's type within q's struct type */
    if (left->kind == HD_AST_DOT || left->kind == HD_AST_MEMBER) {
        if (left->ident[0]) {
            const char *outer_type = mir_dot_struct_type(g, left->left);
            if (outer_type && outer_type[0]) {
                mir_struct_t *st = mir_find_struct(g, outer_type);
                if (st) {
                    for (int mi = 0; mi < st->n_members; mi++) {
                        if (strcmp(st->member_names[mi], left->ident) == 0) {
                            /* Return member type name from mir_struct_t if available */
                            if (st->member_type_names[mi][0])
                                return st->member_type_names[mi];
                            /* Fallback: try parser type annotations */
                            if (left->type && left->type->kind == HD_TYPE_STRUCT && left->type->name[0])
                                return left->type->name;
                            if (left->left && left->left->type && left->left->type->kind == HD_TYPE_STRUCT) {
                                HDType *parent_type = left->left->type;
                                for (int pi = 0; pi < parent_type->n_members; pi++) {
                                    if (strcmp(parent_type->members[pi].name, left->ident) == 0) {
                                        if (parent_type->members[pi].type &&
                                            parent_type->members[pi].type->kind == HD_TYPE_STRUCT &&
                                            parent_type->members[pi].type->name[0])
                                            return parent_type->members[pi].type->name;
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    /* INDEX: arr[i] or p[i] — look up the element type from var table */
    if (left->kind == HD_AST_INDEX) {
        /* Traverse to find the root IDENT */
        const HDASTNode *root = left;
        while (root && root->kind == HD_AST_INDEX) root = root->left;
        if (root && root->kind == HD_AST_IDENT) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, root->ident) == 0) {
                    /* Struct array or pointer-to-struct: return element struct type */
                    if (g->vars[i].struct_name[0])
                        return g->vars[i].struct_name;
                    break;
                }
            }
        }
    }
    /* ADD/SUB: pointer arithmetic (p+1, p-1) — get struct type from left operand */
    if (left->kind == HD_AST_ADD || left->kind == HD_AST_SUB) {
        return mir_dot_struct_type(g, left->left);
    }
    /* DEREF: *p — get struct type from the pointer's pointee type */
    if (left->kind == HD_AST_DEREF) {
        if (left->child && left->child->type && left->child->type->kind == HD_TYPE_PTR
            && left->child->type->base && left->child->type->base->kind == HD_TYPE_STRUCT
            && left->child->type->base->name[0])
            return left->child->type->base->name;
        /* Also check IDENT vars */
        if (left->child && left->child->kind == HD_AST_IDENT) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, left->child->ident) == 0 && g->vars[i].is_ptr_struct)
                    return g->vars[i].struct_name;
            }
        }
    }
    return NULL;
}

/* Find a struct by its total_size (for struct array element lookup). */
static mir_struct_t *mir_find_struct_by_size(HDMirGen *g, int size) {
    for (int i = 0; i < g->n_structs; i++)
        if (g->structs[i].total_size == size) return &g->structs[i];
    return NULL;
}

/* Get the byte offset of the e-th member of a struct by position. */
static int mir_struct_member_offset_by_index(HDMirGen *g, const char *struct_name, int index) {
    mir_struct_t *s = mir_find_struct(g, struct_name);
    if (!s || index < 0 || index >= s->n_members) return -1;
    return s->member_offsets[index];
}
static int mir_struct_member_is_unsigned(HDMirGen *g, const char *struct_name, const char *member) {
    mir_struct_t *s = mir_find_struct(g, struct_name);
    if (!s) return 0;
    for (int i = 0; i < s->n_members; i++)
        if (strcmp(s->member_names[i], member) == 0) return s->member_is_unsigned[i];
    return 0;
}
static const char *mir_find_var_struct_name(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0 && g->vars[i].is_struct)
            return g->vars[i].struct_name;
    return NULL;
}

/* symbol table: is the named variable an array? */
static int mir_var_is_array(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_array;
    return 0;
}

/* symbol table: is the named variable a struct? */
static int mir_var_is_struct(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_struct;
    return 0;
}

/* symbol table: is the named variable declared unsigned? */
static int mir_is_unsigned_var(HDMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_unsigned;
    return 0;
}

/* declare (or re-bind) a variable name -> vr; record unsigned-ness + float */
static wubu_vr_t mir_decl_var_unsigned(HDMirGen *g, const char *name, int is_unsigned) {
    /* Always allocate a new VR for shadowed variables — reusing would
     * corrupt the outer variable's address when the scope pops */
    wubu_vr_t vr = mir_new_vr(g);
    if (g->n_vars < MIRGEN_MAX_VARS) {
        strncpy(g->vars[g->n_vars].name, name, HD_MAX_IDENT_LEN - 1);
        g->vars[g->n_vars].name[HD_MAX_IDENT_LEN - 1] = '\0';
        g->vars[g->n_vars].vr = vr;
        g->vars[g->n_vars].addr = 0;
        g->vars[g->n_vars].is_unsigned = is_unsigned;
        g->vars[g->n_vars].is_float = 0;
        g->vars[g->n_vars].is_array = 0;
        g->vars[g->n_vars].array_size = 0;
        g->vars[g->n_vars].fn_ptr_func_id = -1;
        g->n_vars++;
    }
    return vr;
}

static wubu_vr_t mir_decl_var_float(HDMirGen *g, const char *name) {
    /* Always allocate a new VR — see mir_decl_var_unsigned */
    wubu_vr_t vr = mir_new_vr(g);
    if (g->n_vars < MIRGEN_MAX_VARS) {
        strncpy(g->vars[g->n_vars].name, name, HD_MAX_IDENT_LEN - 1);
        g->vars[g->n_vars].name[HD_MAX_IDENT_LEN - 1] = '\0';
        g->vars[g->n_vars].vr = vr;
        g->vars[g->n_vars].addr = 0;
        g->vars[g->n_vars].is_unsigned = 0;
        g->vars[g->n_vars].is_float = 1;
        g->vars[g->n_vars].is_array = 0;
        g->vars[g->n_vars].array_size = 0;
        g->vars[g->n_vars].fn_ptr_func_id = -1;
        g->n_vars++;
    }
    return vr;
}

/* Register a variable name -> (vr, addr) directly, WITHOUT allocating a new
 * virtual register. Used for function parameters, whose value lives in an
 * incoming call-register slot (v1..vN) already copied into a memory cell. */
static void mir_bind_var(HDMirGen *g, const char *name, wubu_vr_t vr, wubu_vr_t addr, int is_unsigned) {
    if (g->n_vars < MIRGEN_MAX_VARS) {
        strncpy(g->vars[g->n_vars].name, name, HD_MAX_IDENT_LEN - 1);
        g->vars[g->n_vars].name[HD_MAX_IDENT_LEN - 1] = '\0';
        g->vars[g->n_vars].vr = vr;
        g->vars[g->n_vars].addr = addr;
        g->vars[g->n_vars].is_unsigned = is_unsigned;
        g->vars[g->n_vars].is_float = 0;
        g->vars[g->n_vars].is_array = 0;
        g->vars[g->n_vars].array_size = 0;
        g->vars[g->n_vars].fn_ptr_func_id = -1;
        g->n_vars++;
    }
}

/* Walk a top-level translation unit and register every function definition
 * into the compiler's func table, assigning a stable id (so CALL sites can
 * resolve the callee regardless of definition order). */
static void mir_collect_funcs(HDMirGen *g, const HDASTNode *ast) {
    if (!ast) return;
    if (ast->kind == HD_AST_BLOCK) {
        for (uint32_t i = 0; i < ast->n_stmts; i++) {
            const HDASTNode *s = ast->stmts[i];
            if (s && s->kind == HD_AST_FUNC_DECL && g->n_funcs < MIR_MAX_FUNCTIONS) {
                int id = g->n_funcs++;
                g->func_ast[id] = s;
                strncpy(g->prog->funcs[id].name, s->ident, HD_MAX_IDENT_LEN - 1);
                g->prog->funcs[id].name[HD_MAX_IDENT_LEN - 1] = '\0';
                g->prog->funcs[id].start = 0;
                g->prog->funcs[id].end = 0;
            }
        }
    } else if (ast->kind == HD_AST_FUNC_DECL && g->n_funcs < MIR_MAX_FUNCTIONS) {
        int id = g->n_funcs++;
        g->func_ast[id] = ast;
        strncpy(g->prog->funcs[id].name, ast->ident, HD_MAX_IDENT_LEN - 1);
        g->prog->funcs[id].name[HD_MAX_IDENT_LEN - 1] = '\0';
    }
    g->prog->n_funcs = g->n_funcs;   /* make the table visible to the interpreter/drivers */
}

/* Recursively count the maximum number of arguments in any function call.
 * Returns the max n_args found (clamped to MIR_MAX_CALL_ARGS). */
static int mir_count_max_args(const HDASTNode *ast) {
    if (!ast) return 0;
    int max_args = 0;
    switch (ast->kind) {
    case HD_AST_BLOCK:
        for (uint32_t i = 0; i < ast->n_stmts; i++) {
            int child_max = mir_count_max_args(ast->stmts[i]);
            if (child_max > max_args) max_args = child_max;
        }
        break;
    case HD_AST_FUNC_DECL:
        /* Recurse into function body */
        if (ast->body) {
            int child_max = mir_count_max_args(ast->body);
            if (child_max > max_args) max_args = child_max;
        }
        break;
    case HD_AST_FUNC_CALL:
        /* Count arguments in this call */
        if (ast->n_args > (uint32_t)max_args) max_args = ast->n_args;
        /* Also check the callee expression */
        if (ast->child) {
            int callee_max = mir_count_max_args(ast->child);
            if (callee_max > max_args) max_args = callee_max;
        }
        /* Count arguments too */
        for (int i = 0; i < ast->n_args; i++)
            if (ast->args[i]) {
                int arg_max = mir_count_max_args(ast->args[i]);
                if (arg_max > max_args) max_args = arg_max;
            }
        break;
    default:
        /* For binary/unary ops, check children */
        if (ast->left) {
            int left_max = mir_count_max_args(ast->left);
            if (left_max > max_args) max_args = left_max;
        }
        if (ast->right) {
            int right_max = mir_count_max_args(ast->right);
            if (right_max > max_args) max_args = right_max;
        }
        if (ast->child) {
            int child_max = mir_count_max_args(ast->child);
            if (child_max > max_args) max_args = child_max;
        }
        if (ast->cond) {
            int cond_max = mir_count_max_args(ast->cond);
            if (cond_max > max_args) max_args = cond_max;
        }
        if (ast->body) {
            int body_max = mir_count_max_args(ast->body);
            if (body_max > max_args) max_args = body_max;
        }
        if (ast->init) {
            int init_max = mir_count_max_args(ast->init);
            if (init_max > max_args) max_args = init_max;
        }
        if (ast->then_branch) {
            int then_max = mir_count_max_args(ast->then_branch);
            if (then_max > max_args) max_args = then_max;
        }
        if (ast->else_branch) {
            int else_max = mir_count_max_args(ast->else_branch);
            if (else_max > max_args) max_args = else_max;
        }
        if (ast->stmts) {
            for (uint32_t i = 0; i < ast->n_stmts; i++) {
                int stmt_max = mir_count_max_args(ast->stmts[i]);
                if (stmt_max > max_args) max_args = stmt_max;
            }
        }
        break;
    }
    if (max_args > MIR_MAX_CALL_ARGS) max_args = MIR_MAX_CALL_ARGS;
    return max_args;
}

/* Is a comparison operand unsigned? Check the symbol table (var decl) or
 * the node's own type annotation (mirrors the golden JIT's expr_static_type). */
static int mir_operand_is_unsigned(HDMirGen *g, const HDASTNode *operand) {
    if (!operand) return 0;
    if (operand->kind == HD_AST_IDENT) {
        if (mir_is_unsigned_var(g, operand->ident)) return 1;
    }
    if (operand->kind == HD_AST_DOT || operand->kind == HD_AST_MEMBER || operand->kind == HD_AST_ARROW) {
        /* Check struct/union member type for unsigned */
        const char *var_name = (operand->left && operand->left->kind == HD_AST_IDENT) ? operand->left->ident : NULL;
        const char *struct_type = NULL;
        if (var_name) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, var_name) == 0 &&
                    (g->vars[i].is_struct || g->vars[i].is_ptr_struct)) {
                    struct_type = g->vars[i].struct_name;
                    break;
                }
            }
        }
        if (!struct_type && operand->type && operand->type->kind == HD_TYPE_PTR && operand->type->base)
            struct_type = operand->type->base->name;
        if (struct_type && mir_struct_member_is_unsigned(g, struct_type, operand->ident))
            return 1;
    }
    if (operand->type) {
        HDTypeKind k = operand->type->kind;
        if (k == HD_TYPE_U8 || k == HD_TYPE_U16 || k == HD_TYPE_U32 || k == HD_TYPE_U64)
            return 1;
    }
    return 0;
}

/* Check if an AST node produces a float (f32 bits) value.
 * Float literals (HD_AST_FLOAT_LIT) and float variables are float;
 * unary negation of a float child is also float. */
static bool mir_is_float_node(HDMirGen *g, const HDASTNode *n) {
    if (!n) return false;
    if (n->kind == HD_AST_FLOAT_LIT) return true;
    if (n->kind == HD_AST_IDENT)
        return mir_find_var_is_float(g, n->ident);
    if (n->kind == HD_AST_NEG || n->kind == HD_AST_ADD ||
        n->kind == HD_AST_SUB || n->kind == HD_AST_MUL ||
        n->kind == HD_AST_DIV) {
        if (n->left && mir_is_float_node(g, n->left)) return true;
        if (n->right && mir_is_float_node(g, n->right)) return true;
        if (n->child && mir_is_float_node(g, n->child)) return true;
    }
    if (n->type && n->type->kind == HD_TYPE_F64) return true;
    return false;
}

/* declare (or re-bind) a variable name -> vr */
static wubu_vr_t mir_decl_var(HDMirGen *g, const char *name) {
    return mir_decl_var_unsigned(g, name, 0);
}

static wubu_vr_t mir_gen_expr(HDMirGen *g, const HDASTNode *n);

/* Evaluate the address-of-first-element of an lvalue. Returns a vr holding the
 * memory address (a value), suitable as the index into mem[].
 * - array IDENT  -> base address (a decays to &a[0])
 * - scalar/pointer IDENT -> the value held in the var (pointer value)
 * - INDEX  -> address_of(left) + index * stride
 * - DEREF  -> the pointer's held value */

/* Find the stride for an INDEX node by traversing to the root IDENT
 * and looking up its array type. Returns the inner dimension size for
 * multi-dimensional arrays, or 1 for 1D arrays. */
static int mir_index_stride(HDMirGen *g, const HDASTNode *n) {
    /* Traverse left spine to find the root IDENT */
    const HDASTNode *root = n;
    while (root && root->kind == HD_AST_INDEX) root = root->left;
    if (!root || root->kind != HD_AST_IDENT) return 1;
    /* Look up the variable in the symbol table */
    for (int i = 0; i < g->n_vars; i++) {
        if (strcmp(g->vars[i].name, root->ident) == 0 && g->vars[i].is_array) {
            if (g->vars[i].array_stride > 1)
                return g->vars[i].array_stride;
            /* For arrays with element size > 1 cell, compute stride from element type */
            /* Check if this is a struct array (base type is struct) */
            if (g->vars[i].is_struct) {
                mir_struct_t *st = mir_find_struct(g, g->vars[i].struct_name);
                if (st && st->total_size > 1) return st->total_size;
            }
            /* Check the node type for array base type */
            if (root->type && root->type->kind == HD_TYPE_ARRAY && root->type->base) {
                if (root->type->base->kind == HD_TYPE_STRUCT) {
                    mir_struct_t *st = mir_find_struct(g, root->type->base->name);
                    if (st && st->total_size > 1) return st->total_size;
                }
                /* For non-struct arrays, element size is 1 cell */
            }
            return 1;
        }
    }
    return 1;
}
static wubu_vr_t mir_lvalue_addr(HDMirGen *g, const HDASTNode *n); /* forward decl */

static wubu_vr_t mir_address_of(HDMirGen *g, const HDASTNode *n) {
    if (!n) return 0;
    if (n->kind == HD_AST_STRING_LIT)
        return mir_gen_expr(g, n);  /* returns the allocated address */
    if (n->kind == HD_AST_IDENT) {
        wubu_vr_t addr = mir_find_var_addr(g, n->ident);
        if (addr == 0) return 0;
        if (mir_var_is_struct(g, n->ident))
            return addr;                        /* structs: addr IS the address */
        if (mir_var_is_array(g, n->ident))
            return addr;                       /* &a[0] == a's base */
        return wubu_mir_load(g->prog, addr);   /* a[0] is a pointer var: load it */
    }
    if (n->kind == HD_AST_DEREF)
        return mir_gen_expr(g, n->child);      /* *p: address is p's value */
    if (n->kind == HD_AST_CALL || n->kind == HD_AST_FUNC_CALL) {
        /* A call returning a struct yields the struct address in its result vr */
        wubu_vr_t rv = mir_gen_expr(g, n);
        if (rv == 0) return 0;
        /* The vr holds the address; the address IS the value */
        return rv;
    }
    if (n->kind == HD_AST_INDEX) {
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx  = mir_gen_expr(g, n->right);
        int stride = mir_index_stride(g, n);
        if (stride > 1) {
            idx = wubu_mir_binop(g->prog, MIR_MUL, idx, wubu_mir_const(g->prog, (int64_t)stride));
        }
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
    }
    if (n->kind == HD_AST_DOT || n->kind == HD_AST_MEMBER) {
        /* s.a: return the ADDRESS of the member (for &s.a, lvalue addr, etc.) */
        const char *varname = n->left && n->left->kind == HD_AST_IDENT ? n->left->ident : NULL;
        if (varname) {
            wubu_vr_t base = mir_find_var_addr(g, varname);
            const char *struct_type = mir_find_var_struct_name(g, varname);
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            if (offset < 0) return 0;
            wubu_vr_t member_addr = wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)offset));
            return member_addr;
        }
        /* Fallback: left side is a CALL, DOT, or other expr returning a struct address */
        if (n->left && n->ident[0]) {
            wubu_vr_t base;
            if (n->left->kind == HD_AST_CALL || n->left->kind == HD_AST_FUNC_CALL) {
                /* Generate the call once — its return vr holds the struct address */
                base = mir_gen_expr(g, n->left);
            } else if (n->left->kind == HD_AST_DOT || n->left->kind == HD_AST_MEMBER) {
                /* q.p.x: nested DOT — recursively compute address of q.p, then add x's offset.
                 * Walk the DOT chain to find the root IDENT and accumulate offsets. */
                const HDASTNode *dot = n;
                int total_offset = 0;
                /* Collect the chain of (struct_type, member_name) pairs from right to left */
                char chain_types[16][HD_MAX_IDENT_LEN];
                char chain_names[16][HD_MAX_IDENT_LEN];
                int chain_offsets[16];
                int chain_len = 0;
                while (dot && (dot->kind == HD_AST_DOT || dot->kind == HD_AST_MEMBER)) {
                    if (chain_len >= 16) break;
                    strncpy(chain_names[chain_len], dot->ident, HD_MAX_IDENT_LEN - 1);
                    chain_names[chain_len][HD_MAX_IDENT_LEN - 1] = '\0';
                    chain_offsets[chain_len] = 0; /* will be filled in reverse */
                    chain_len++;
                    dot = dot->left;
                }
                /* Now dot should be the root IDENT (or CALL) */
                const char *root_type = NULL;
                if (dot && dot->kind == HD_AST_IDENT) {
                    base = mir_find_var_addr(g, dot->ident);
                    root_type = mir_find_var_struct_name(g, dot->ident);
                } else if (dot && (dot->kind == HD_AST_CALL || dot->kind == HD_AST_FUNC_CALL)) {
                    base = mir_gen_expr(g, dot);
                    root_type = mir_dot_struct_type(g, dot);
                } else if (dot && dot->kind == HD_AST_INDEX) {
                    /* arr[0].a: compute address of arr[0], look up struct type from array */
                    base = mir_address_of(g, dot);
                    root_type = mir_dot_struct_type(g, dot);
                } else {
                    return 0;
                }
                /* Walk the chain from right to left, looking up each member's offset */
                for (int ci = chain_len - 1; ci >= 0; ci--) {
                    if (!root_type || !root_type[0]) return 0;
                    mir_struct_t *st = mir_find_struct(g, root_type);
                    if (!st) return 0;
                    int off = -1;
                    for (int mi = 0; mi < st->n_members; mi++) {
                        if (strcmp(st->member_names[mi], chain_names[ci]) == 0) {
                            off = st->member_offsets[mi];
                            /* Update root_type for next iteration */
                            if (st->member_type_names[mi][0])
                                root_type = st->member_type_names[mi];
                            else
                                root_type = NULL;
                            break;
                        }
                    }
                    if (off < 0) return 0;
                    total_offset += off;
                }
                if (total_offset > 0) {
                    base = wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)total_offset));
                }
                return base;
            } else {
                base = mir_address_of(g, n->left);
            }
            if (base == 0) return 0;
            const char *struct_type = mir_dot_struct_type(g, n->left);
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            if (offset < 0) return 0;
            wubu_vr_t member_addr = wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)offset));
            return member_addr;
        }
        return 0;
    }
    if (n->kind == HD_AST_ARROW) {
        /* ps->a as a value: load pointer ps, then load member a from pointed struct */
        if (!n->left || !n->ident[0]) return 0;
        wubu_vr_t ptr_val = mir_gen_expr(g, n->left);
        if (ptr_val == 0) return 0;
        /* Look up struct type: from symbol table if left is IDENT, else from node type annotation */
        const char *struct_type = NULL;
        if (n->left->kind == HD_AST_IDENT) {
            struct_type = mir_find_var_struct_name(g, n->left->ident);
        }
        if (!struct_type && n->left->type && n->left->type->kind == HD_TYPE_PTR && n->left->type->base)
            struct_type = n->left->type->base->name;
        int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
        if (offset < 0) return 0;
        return wubu_mir_binop(g->prog, MIR_ADD, ptr_val, wubu_mir_const(g->prog, (int64_t)offset));
    }
    return 0;
}

/* Compute the memory address of an lvalue (a value suitable for MIR_STORE). */
static wubu_vr_t mir_lvalue_addr(HDMirGen *g, const HDASTNode *n) {
    if (!n) return 0;
    if (n->kind == HD_AST_IDENT)
        return mir_find_var_addr(g, n->ident);  /* store into a's own cell */
    if (n->kind == HD_AST_DEREF)
        return mir_gen_expr(g, n->child);      /* *p: address == p's value */
    if (n->kind == HD_AST_INDEX) {
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx = mir_gen_expr(g, n->right);
        int stride = mir_index_stride(g, n);
        if (stride > 1) {
            idx = wubu_mir_binop(g->prog, MIR_MUL, idx, wubu_mir_const(g->prog, (int64_t)stride));
        }
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
    }
    if (n->kind == HD_AST_DOT || n->kind == HD_AST_MEMBER) {
        /* s.a = val — compute address of struct member */
        if (n->left && n->left->kind == HD_AST_IDENT && n->ident[0]) {
            const char *varname = n->left->ident;
            wubu_vr_t base = mir_find_var_addr(g, varname);
            if (base == 0) return 0;
            const char *struct_type = mir_find_var_struct_name(g, varname);
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            if (offset < 0) return 0;
            return wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)offset));
        }
        /* Fallback: left side is a CALL or other expr returning struct addr */
        if (n->left && n->ident[0]) {
            wubu_vr_t base = mir_address_of(g, n->left);
            if (base == 0) return 0;
            const char *struct_type = mir_dot_struct_type(g, n->left);
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            if (offset < 0) return 0;
            return wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)offset));
        }
    }
    if (n->kind == HD_AST_ARROW) {
        /* ps->a = val — compute address: ptr_value + member_offset */
        if (!n->left || !n->ident[0]) return 0;
        wubu_vr_t ptr_val = mir_gen_expr(g, n->left);
        if (ptr_val == 0) return 0;
        const char *struct_type = NULL;
        if (n->left->kind == HD_AST_IDENT) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, n->left->ident) == 0 &&
                    (g->vars[i].is_struct || g->vars[i].is_ptr_struct)) {
                    struct_type = g->vars[i].struct_name;
                    break;
                }
            }
        }
        if (!struct_type && n->left->type && n->left->type->kind == HD_TYPE_PTR && n->left->type->base)
            struct_type = n->left->type->base->name;
        int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
        if (offset < 0) return 0;
        return wubu_mir_binop(g->prog, MIR_ADD, ptr_val, wubu_mir_const(g->prog, (int64_t)offset));
    }
    return 0;  /* not an lvalue we can take address of */
}

static wubu_vr_t mir_gen_stmt(HDMirGen *g, const HDASTNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case HD_AST_BLOCK: {
        /* Push scope unless we're in a function body (scope already pushed before params)
         * or the block has no_scope_pop set (e.g. multi-declarator wrapper) */
        int pushed = 0;
        if (!g->in_function_body && g->n_scopes < MIRGEN_MAX_VARS && !n->no_scope_pop) {
            g->scope_var_start[g->n_scopes++] = g->n_vars;
            pushed = 1;
        }
        wubu_vr_t last = 0;
        for (uint32_t i = 0; i < n->n_stmts; i++) {
            last = mir_gen_stmt(g, n->stmts[i]);
            /* If this top-level statement is a RETURN, subsequent
             * statements are unreachable. Stop generating them. */
            if (n->stmts[i]->kind == HD_AST_RETURN) break;
            /* Note: GOTO should NOT break — the label target may be
                 * defined later in the same block, and we still need to
                 * emit those statements. */
        }
        /* Pop scope: remove vars added in this block so outer scope is restored */
        if (pushed)
            g->n_vars = g->scope_var_start[--g->n_scopes];
        return last;
    }
    case HD_AST_STRUCT_DECL: {
        /* Register the struct type and its member offsets.
         * Member info is in n->type (HDType) which has members[]. */
        if (g->n_structs < MAX_STRUCTS && n->ident[0]) {
            mir_struct_t *s = &g->structs[g->n_structs++];
            strncpy(s->name, n->ident, HD_MAX_IDENT_LEN - 1);
            s->name[HD_MAX_IDENT_LEN - 1] = '\0';
            s->n_members = 0;
            s->total_size = 0;
            if (n->type && (n->type->kind == HD_TYPE_STRUCT || n->type->kind == HD_TYPE_UNION)) {
                for (int i = 0; i < n->type->n_members && s->n_members < MAX_MEMBERS; i++) {
                    strncpy(s->member_names[s->n_members], n->type->members[i].name, HD_MAX_IDENT_LEN - 1);
                    s->member_names[s->n_members][HD_MAX_IDENT_LEN - 1] = '\0';
                    s->member_offsets[s->n_members] = (int)n->type->members[i].offset;
                    s->member_is_unsigned[s->n_members] = (n->type->members[i].type && (n->type->members[i].type->kind == HD_TYPE_U8 || n->type->members[i].type->kind == HD_TYPE_U16 || n->type->members[i].type->kind == HD_TYPE_U32 || n->type->members[i].type->kind == HD_TYPE_U64)) ? 1 : 0;
                    if (n->type->members[i].type && n->type->members[i].type->kind == HD_TYPE_STRUCT && n->type->members[i].type->name[0])
                        strncpy(s->member_type_names[s->n_members], n->type->members[i].type->name, HD_MAX_IDENT_LEN - 1);
                    s->n_members++;
                }
            }
            s->total_size = (int)n->type->size;  /* size in int64 cells */
            if (s->total_size <= 0) s->total_size = 1;
        }
        return 0;
    }
    case HD_AST_VAR_DECL: {
        int is_uns = 0;
        int arr_size = 0;
        int is_struct_var = 0;
        char struct_type_name[HD_MAX_IDENT_LEN] = {0};
        if (n->type) {
            HDTypeKind k = n->type->kind;
            if (k == HD_TYPE_U8 || k == HD_TYPE_U16 || k == HD_TYPE_U32 || k == HD_TYPE_U64)
                is_uns = 1;
            /* array type carries an element count in n->type->array_size */
            if (k == HD_TYPE_ARRAY && n->type->array_size > 0) {
                arr_size = (int)n->type->array_size;
                /* For struct arrays, multiply by struct element size */
                if (n->type->base && n->type->base->kind == HD_TYPE_STRUCT) {
                    int elem_cells = 0;
                    mir_struct_t *elem_st = mir_find_struct(g, n->type->base->name);
                    if (elem_st) elem_cells = elem_st->total_size;
                    if (elem_cells > 1) arr_size *= elem_cells;
                    else arr_size *= 1;
                }
            }
            if (k == HD_TYPE_PTR && n->type->base &&
                (n->type->base->kind == HD_TYPE_STRUCT || n->type->base->kind == HD_TYPE_UNION)) {
                /* Pointer to struct — track struct name for -> member access */
                if (n->type->base->name[0])
                    strncpy(struct_type_name, n->type->base->name, HD_MAX_IDENT_LEN - 1);
            }
            /* struct/union type: allocate memory for all members */
            if (k == HD_TYPE_STRUCT || k == HD_TYPE_UNION) {
                is_struct_var = 1;
                /* Look up struct size from registered structs */
                if (n->type->name[0])
                    strncpy(struct_type_name, n->type->name, HD_MAX_IDENT_LEN - 1);
                /* For unnamed structs, register with a synthetic name based
                 * on the variable name so member offsets are accessible later. */
                if (!struct_type_name[0] && n->ident[0]) {
                    snprintf(struct_type_name, HD_MAX_IDENT_LEN, "__anon_%s", n->ident);
                }
                int struct_size = 0;
                for (int si = 0; si < g->n_structs; si++) {
                    if (strcmp(g->structs[si].name, struct_type_name) == 0) {
                        struct_size = g->structs[si].total_size;
                        break;
                    }
                }
                if (struct_size > 0) arr_size = struct_size;
                /* If struct size not found, use member count from AST type */
                if (struct_size == 0 && n->type && n->type->n_members > 0) {
                    arr_size = (int)n->type->size;  /* size in int64 cells */
                    if (arr_size <= 0) arr_size = 1;
                    /* Register unnamed struct type so member lookups work */
                    if (g->n_structs < MAX_STRUCTS) {
                        mir_struct_t *s = &g->structs[g->n_structs++];
                        snprintf(s->name, HD_MAX_IDENT_LEN, "%s", struct_type_name);
                        s->n_members = 0;
                        for (int mi = 0; mi < n->type->n_members && mi < MAX_MEMBERS; mi++) {
                            strncpy(s->member_names[mi], n->type->members[mi].name, HD_MAX_IDENT_LEN - 1);
                            s->member_names[mi][HD_MAX_IDENT_LEN - 1] = '\0';
                            s->member_offsets[mi] = (int)n->type->members[mi].offset;
                            s->member_is_unsigned[mi] = (n->type->members[mi].type && (n->type->members[mi].type->kind == HD_TYPE_U8 || n->type->members[mi].type->kind == HD_TYPE_U16 || n->type->members[mi].type->kind == HD_TYPE_U32 || n->type->members[mi].type->kind == HD_TYPE_U64)) ? 1 : 0;
                            if (n->type->members[mi].type && n->type->members[mi].type->kind == HD_TYPE_STRUCT && n->type->members[mi].type->name[0])
                                strncpy(s->member_type_names[mi], n->type->members[mi].type->name, HD_MAX_IDENT_LEN - 1);
                            s->n_members++;
                        }
                        s->total_size = (int)n->type->size;  /* size in int64 cells */
                        if (s->total_size <= 0) s->total_size = 1;
                    }
                }
            }
        }
        wubu_vr_t vr;
        if (n->type && n->type->kind == HD_TYPE_F64)
            vr = mir_decl_var_float(g, n->ident);
        else
            vr = mir_decl_var_unsigned(g, n->ident, is_uns);
        /* Allocate memory for the variable (arrays get arr_size cells, scalars 1, structs = total_size).
         * Use a HIGH VR for the address so it never collides with argument registers
         * (v1..vN) or instruction-index VRs. */
        wubu_vr_t addr = mir_new_vr(g);  /* high VR for the address */
        int64_t mem_addr = (int64_t)(g->prog->total_mem + 1);
        g->prog->total_mem = mem_addr + (arr_size > 0 ? arr_size : 1) - 1;
        wubu_mir_const_to(g->prog, addr, mem_addr);  /* addr VR = constant mem_addr */
        /* Search from end to find the most recent declaration (shadowing) */
        for (int i = g->n_vars - 1; i >= 0; i--)
            if (strcmp(g->vars[i].name, n->ident) == 0) {
                g->vars[i].addr = addr;
                g->vars[i].is_array = (arr_size > 0 && !is_struct_var);
                g->vars[i].array_size = arr_size;
                /* For multi-dimensional arrays, compute the stride (inner dimension).
                 * For int w[M][N], stride = N. For int w[N], stride = 1. */
                g->vars[i].array_stride = 1;
                if (n->type && n->type->kind == HD_TYPE_ARRAY && n->type->base) {
                    if (n->type->base->kind == HD_TYPE_ARRAY) {
                        g->vars[i].array_stride = n->type->base->array_size;
                    } else if (n->type->base->kind == HD_TYPE_STRUCT) {
                        /* Struct array: stride = struct size in cells */
                        mir_struct_t *elem_st = mir_find_struct(g, n->type->base->name);
                        if (elem_st && elem_st->total_size > 1)
                            g->vars[i].array_stride = elem_st->total_size;
                    }
                }
                g->vars[i].is_struct = is_struct_var;
                if (is_struct_var)
                    strncpy(g->vars[i].struct_name, struct_type_name, HD_MAX_IDENT_LEN - 1);
                /* For struct arrays, also store the element struct type name */
                if (n->type && n->type->kind == HD_TYPE_ARRAY && n->type->base
                    && n->type->base->kind == HD_TYPE_STRUCT && n->type->base->name[0]) {
                    strncpy(g->vars[i].struct_name, n->type->base->name, HD_MAX_IDENT_LEN - 1);
                }
                break;
            }
        /* If this is a pointer to a struct, mark the var record for -> lookup */
        if (n->type && n->type->kind == HD_TYPE_PTR && n->type->base &&
            (n->type->base->kind == HD_TYPE_STRUCT || n->type->base->kind == HD_TYPE_UNION)) {
            for (int i = g->n_vars - 1; i >= 0; i--) {
                if (strcmp(g->vars[i].name, n->ident) == 0) {
                    g->vars[i].is_ptr_struct = 1;
                    if (n->type->base->name[0])
                        strncpy(g->vars[i].struct_name, n->type->base->name, HD_MAX_IDENT_LEN - 1);
                    break;
                }
            }
        }
        if (n->init) {
            if (n->init->kind == HD_AST_BRACE_INIT) {
                /* array/struct initializer list: store each element.
                 * Designated initializers (HD_AST_DESIG_INIT) store at a specific offset. */
                int n_elems = (int)n->init->n_args;
                /* Reallocate if the initializer determines the size */
                if (arr_size == 0 && n_elems > 0) {
                    arr_size = n_elems;
                    addr = wubu_mir_alloc(g->prog, arr_size);
                    for (int i = 0; i < g->n_vars; i++)
                        if (strcmp(g->vars[i].name, n->ident) == 0) {
                            g->vars[i].addr = addr;
                            g->vars[i].is_array = 1;
                            g->vars[i].array_size = arr_size;
                            g->vars[i].array_stride = 1;
                            break;
                        }
                }
                for (uint32_t e = 0; e < n->init->n_args && e < (uint32_t)arr_size; e++) {
                    int offset = (int)e; /* default: sequential */
                    /* For structs, use the actual member byte offset, not sequential. */
                    if (is_struct_var) {
                        if (struct_type_name[0]) {
                            int moff = mir_struct_member_offset_by_index(g, struct_type_name, (int)e);
                            if (moff >= 0) offset = moff;  /* cell offset of this member */
                        } else if (n->type && n->type->kind == HD_TYPE_STRUCT
                                   && e < (uint32_t)n->type->n_members) {
                            /* Unnamed struct: AST member offset is in cells */
                            offset = (int)n->type->members[e].offset;
                        }
                    }
                    HDASTNode *elem = n->init->args[e];
                    wubu_vr_t ev;
                    if (elem->kind == HD_AST_DESIG_INIT) {
                        /* Designated initializer: compute offset */
                        if (elem->ident[0] == '@') {
                            /* Array index designator: [index] */
                            offset = (int)elem->int_val;
                        } else {
                            /* Field designator: .field — look up struct member offset */
                            offset = mir_struct_member_offset(g, struct_type_name, elem->ident);
                            if (offset < 0) offset = (int)e; /* fallback */
                        }
                        ev = mir_gen_expr(g, elem->child);
                    } else {
                        /* If element is a struct IDENT and the corresponding member
                         * type is a struct, copy all cells of the source struct. */
                        int member_is_struct = (e < (uint32_t)n->type->n_members &&
                            n->type->members[e].type &&
                            n->type->members[e].type->kind == HD_TYPE_STRUCT);
                        if (member_is_struct && elem->kind == HD_AST_IDENT) {
                            int src_cells = (int)n->type->members[e].type->size; /* already in cells */
                            int dst_offset = offset;
                            wubu_vr_t src_addr = mir_find_var_addr(g, elem->ident);
                            if (src_addr > 0 && src_cells > 0) {
                                for (int m = 0; m < src_cells; m++) {
                                    wubu_vr_t src_elem_addr = wubu_mir_binop(g->prog, MIR_ADD, src_addr, wubu_mir_const(g->prog, (int64_t)m));
                                    wubu_vr_t dst_elem_addr = wubu_mir_binop(g->prog, MIR_ADD, addr, wubu_mir_const(g->prog, (int64_t)(dst_offset + m)));
                                    wubu_mir_store(g->prog, dst_elem_addr, wubu_mir_load(g->prog, src_elem_addr));
                                }
                                continue;
                            }
                        }
                        ev = mir_gen_expr(g, elem);
                    }
                    wubu_vr_t elem_addr = wubu_mir_binop(g->prog, MIR_ADD, addr,
                                                          wubu_mir_const(g->prog, (int64_t)offset));
                    wubu_mir_store(g->prog, elem_addr, ev);
                }
            } else {
                /* For struct variables initialized from another struct/expr,
                 * copy all members cell-by-cell (each member = 1 int64 cell). */
                if (is_struct_var && n->init) {
                    const char *src_name = (n->init->kind == HD_AST_IDENT)
                        ? n->init->ident : NULL;
                    if (src_name) {
                        wubu_vr_t src_addr = mir_find_var_addr(g, src_name);
                        if (src_addr > 0 && arr_size > 0) {
                            for (int m = 0; m < arr_size; m++) {
                                wubu_vr_t src_elem = wubu_mir_binop(g->prog, MIR_ADD, src_addr, wubu_mir_const(g->prog, (int64_t)m));
                                wubu_vr_t dst_elem = wubu_mir_binop(g->prog, MIR_ADD, addr, wubu_mir_const(g->prog, (int64_t)m));
                                wubu_mir_store(g->prog, dst_elem, wubu_mir_load(g->prog, src_elem));
                            }
                        } else if (src_addr == 0) {
                            /* No source var — just store single value */
                            wubu_vr_t val = mir_gen_expr(g, n->init);
                            wubu_mir_store(g->prog, addr, val);
                        }
                    } else {
                        wubu_vr_t val = mir_gen_expr(g, n->init);
                        if (arr_size > 0 && val != 0) {
                            for (int m = 0; m < arr_size; m++) {
                                wubu_vr_t src_elem = wubu_mir_binop(g->prog, MIR_ADD, val, wubu_mir_const(g->prog, (int64_t)m));
                                wubu_vr_t dst_elem = wubu_mir_binop(g->prog, MIR_ADD, addr, wubu_mir_const(g->prog, (int64_t)m));
                                wubu_mir_store(g->prog, dst_elem, wubu_mir_load(g->prog, src_elem));
                            }
                        } else {
                            wubu_mir_store(g->prog, addr, val);
                        }
                    }
                } else {
                    /* Scalar/array init from expr */
                    wubu_vr_t val = mir_gen_expr(g, n->init);
                    wubu_mir_store(g->prog, addr, val);
                }
            }
        }
        return vr;
    }
    case HD_AST_EXPR_STMT:
        return mir_gen_expr(g, n->child);
    case HD_AST_RETURN: {
        /* For struct returns, v0 must hold the struct's address, not its first cell value */
        int is_struct_ret = 0;
        if (n->child && n->child->type && n->child->type->kind == HD_TYPE_STRUCT)
            is_struct_ret = 1;
        else if (n->child && n->child->kind == HD_AST_IDENT && n->child->ident[0])
            is_struct_ret = mir_var_is_struct(g, n->child->ident);
        if (is_struct_ret) {
            /* Struct return: allocate a persistent static buffer, copy the
             * struct into it, and return the buffer's address in v0. The
             * buffer outlives the callee's stack frame, so the caller can
             * safely read from it. For multi-cell structs, we serialize into
             * a global static array indexed by g->struct_ret_count. */
            wubu_vr_t src_addr = mir_address_of(g, n->child);
            if (src_addr != 0) {
                const char *ret_struct_name = NULL;
                if (n->child && n->child->type && n->child->type->kind == HD_TYPE_STRUCT && n->child->type->name[0])
                    ret_struct_name = n->child->type->name;
                else if (n->child && n->child->kind == HD_AST_IDENT && n->child->ident[0])
                    ret_struct_name = mir_find_var_struct_name(g, n->child->ident);
                int ret_struct_size = 1;
                if (ret_struct_name) {
                    mir_struct_t *rs = mir_find_struct(g, ret_struct_name);
                    if (rs && rs->total_size > 0) ret_struct_size = rs->total_size;
                }
                /* Allocate a persistent buffer in the MIR's global memory space.
                 * Unlike stack-allocated locals, this survives the callee's return. */
                wubu_vr_t buf_addr = wubu_mir_alloc(g->prog, ret_struct_size);
                /* Copy ret_struct_size cells from src_addr to buf_addr. */
                wubu_vr_t src_base = mir_new_vr(g);
                wubu_mir_mov_to(g->prog, src_base, src_addr);
                wubu_vr_t dst_base = mir_new_vr(g);
                wubu_mir_mov_to(g->prog, dst_base, buf_addr);
                for (int m = 0; m < ret_struct_size; m++) {
                    wubu_vr_t src_p = wubu_mir_binop(g->prog, MIR_ADD, src_base, wubu_mir_const(g->prog, (int64_t)m));
                    wubu_vr_t dst_p = wubu_mir_binop(g->prog, MIR_ADD, dst_base, wubu_mir_const(g->prog, (int64_t)m));
                    wubu_mir_store(g->prog, dst_p, wubu_mir_load(g->prog, src_p));
                }
                wubu_mir_mov_to(g->prog, g->fn_ret_vr, buf_addr);
                wubu_mir_mov_to(g->prog, 0, buf_addr);
                /* For struct returns, bypass the epilogue (which would
                 * overwrite v0 with the default fn_ret_vr=0). Emit a direct
                 * ret so the caller's v0 holds buf_addr. */
                wubu_mir_ret(g->prog, 0);
                return buf_addr;
            }
        }
        wubu_vr_t val = n->child ? mir_gen_expr(g, n->child) : wubu_mir_const(g->prog, 0);
        /* Always emit a direct RET. This ensures that code after a return
         * (e.g. in a block) doesn't overwrite the return value. */
        wubu_mir_mov_to(g->prog, 0, val);
        wubu_mir_ret(g->prog, 0);
        return val;
    }
    case HD_AST_IF: {
        /* if (cond) then_branch else else_branch
         * MIR: [cond] jz else [then -> merge] jmp end [else -> merge] [end] */
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        uint32_t else_label = wubu_mir_new_label(g->prog);
        uint32_t end_label = wubu_mir_new_label(g->prog);
        wubu_mir_jz(g->prog, cond, else_label);
        wubu_vr_t merge = mir_new_vr(g);
        wubu_vr_t then_val = mir_gen_stmt(g, n->then_branch);
        wubu_mir_mov_to(g->prog, merge, then_val);
        wubu_mir_jmp(g->prog, end_label);
        wubu_mir_place_label(g->prog, else_label);
        wubu_vr_t else_val = n->else_branch ? mir_gen_stmt(g, n->else_branch) : wubu_mir_const(g->prog, 0);
        wubu_mir_mov_to(g->prog, merge, else_val);
        wubu_mir_place_label(g->prog, end_label);
        return merge;
    }
    case HD_AST_WHILE: {
        uint32_t top = wubu_mir_new_label(g->prog);
        uint32_t done = wubu_mir_new_label(g->prog);
        int lvl = g->n_loops++;
        g->loop_top[lvl] = top;
        g->loop_done[lvl] = done;
        wubu_mir_place_label(g->prog, top);
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        wubu_mir_jz(g->prog, cond, done);
        mir_gen_stmt(g, n->body);
        wubu_mir_jmp(g->prog, top);
        wubu_mir_place_label(g->prog, done);
        g->n_loops--;
        return 0;
    }
    case HD_AST_DO_WHILE: {
        uint32_t top = wubu_mir_new_label(g->prog);
        uint32_t done = wubu_mir_new_label(g->prog);
        int lvl = g->n_loops++;
        g->loop_top[lvl] = top;
        g->loop_done[lvl] = done;
        wubu_mir_place_label(g->prog, top);
        mir_gen_stmt(g, n->body);
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        wubu_mir_jz(g->prog, cond, done);
        wubu_mir_jmp(g->prog, top);
        wubu_mir_place_label(g->prog, done);
        g->n_loops--;
        return 0;
    }
    case HD_AST_FOR: {
        /* for (init; cond; update) body
         * continue jumps to the update step (like C), then re-checks cond. */
        if (n->init_expr) mir_gen_stmt(g, n->init_expr);
        uint32_t top = wubu_mir_new_label(g->prog);
        uint32_t cont = wubu_mir_new_label(g->prog);
        uint32_t done = wubu_mir_new_label(g->prog);
        int lvl = g->n_loops++;
        g->loop_top[lvl] = cont;   /* continue -> update step */
        g->loop_done[lvl] = done;  /* break -> after loop */
        wubu_mir_place_label(g->prog, top);
        if (n->cond) {
            wubu_vr_t cond = mir_gen_expr(g, n->cond);
            wubu_mir_jz(g->prog, cond, done);
        }
        mir_gen_stmt(g, n->body);
        wubu_mir_place_label(g->prog, cont);
        if (n->update) mir_gen_expr(g, n->update);
        wubu_mir_jmp(g->prog, top);
        wubu_mir_place_label(g->prog, done);
        g->n_loops--;
        return 0;
    }
    case HD_AST_BREAK:
        if (g->n_loops > 0)
            wubu_mir_jmp(g->prog, g->loop_done[g->n_loops - 1]);
        else
            wubu_mir_jmp(g->prog, 0);
        return 0;
    case HD_AST_CONTINUE:
        if (g->n_loops > 0)
            wubu_mir_jmp(g->prog, g->loop_top[g->n_loops - 1]);
        else
            wubu_mir_jmp(g->prog, 0);
        return 0;
    default:
        return mir_gen_expr(g, n);
    }
}

static wubu_vr_t mir_gen_expr(HDMirGen *g, const HDASTNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case HD_AST_INT_LIT:
    case HD_AST_CHAR_LIT: {
        wubu_vr_t r = wubu_mir_const(g->prog, n->int_val);
        return r;
    }
    case HD_AST_IDENT: {
        wubu_vr_t addr = mir_find_var_addr(g, n->ident);
        if (addr == 0) {
            /* Check if it's an enum constant */
            for (int i = 0; i < g->n_enum_consts; i++) {
                if (strcmp(g->enum_const_names[i], n->ident) == 0)
                    return wubu_mir_const(g->prog, g->enum_const_vals[i]);
            }
            return wubu_mir_const(g->prog, 0);
        }
        /* In C, an array name used as a value decays to a pointer to its
         * first element — return the array's base address (not a[0]). */
        if (mir_var_is_array(g, n->ident))
            return addr;
        return wubu_mir_load(g->prog, addr);
    }
    case HD_AST_DOT:
    case HD_AST_MEMBER:
    case HD_AST_ARROW: {
        /* s.a — struct member access (DOT/MEMBER: struct value; ARROW: pointer to struct) */
        if (n->ident[0] && n->left) {
            wubu_vr_t base = 0;
            const char *struct_type = NULL;
            if (n->kind == HD_AST_ARROW) {
                /* ps->a: evaluate ps to get the pointer value (struct address) */
                base = mir_gen_expr(g, n->left);
                if (base == 0) return wubu_mir_const(g->prog, 0);
                /* Look up the struct type from the pointer expression */
                const char *ptr_var = (n->left->kind == HD_AST_IDENT) ? n->left->ident : NULL;
                for (int i = 0; i < g->n_vars; i++) {
                    if (ptr_var && strcmp(g->vars[i].name, ptr_var) == 0 && g->vars[i].is_ptr_struct) {
                        struct_type = g->vars[i].struct_name;
                        break;
                    }
                }
                /* If we couldn't find struct type from symbol table, try the node's type annotation */
                if (!struct_type && n->left->type && n->left->type->kind == HD_TYPE_PTR && n->left->type->base)
                    struct_type = n->left->type->base->name;
                /* Also try mir_dot_struct_type for complex expressions like (p+1)->a */
                if (!struct_type)
                    struct_type = mir_dot_struct_type(g, n->left);
                /* Also handle &s->member where n->left is ADDR(s) */
                if (!struct_type && n->left->kind == HD_AST_ADDR && n->left->child) {
                    const HDASTNode *addr_child = n->left->child;
                    if (addr_child->kind == HD_AST_IDENT) {
                        for (int i = 0; i < g->n_vars; i++) {
                            if (strcmp(g->vars[i].name, addr_child->ident) == 0 && g->vars[i].is_struct) {
                                struct_type = g->vars[i].struct_name;
                                break;
                            }
                        }
                    }
                }
            } else {
                /* s.a: struct value — get the variable's memory address */
                if (n->left->kind == HD_AST_IDENT) {
                    const char *var_name = n->left->ident;
                    base = mir_find_var_addr(g, var_name);
                    if (base == 0) return wubu_mir_const(g->prog, 0);
                    for (int i = 0; i < g->n_vars; i++) {
                        if (strcmp(g->vars[i].name, var_name) == 0 && g->vars[i].is_struct) {
                            struct_type = g->vars[i].struct_name;
                            break;
                        }
                    }
                } else if (n->left->kind == HD_AST_CALL || n->left->kind == HD_AST_FUNC_CALL) {
                    /* f().a: call returns a struct address; use it as base */
                    base = mir_gen_expr(g, n->left);
                    if (base == 0) return wubu_mir_const(g->prog, 0);
                    /* Look up struct type from the call's return type */
                    struct_type = mir_dot_struct_type(g, n->left);
                } else if (n->left->kind == HD_AST_DOT || n->left->kind == HD_AST_MEMBER) {
                    /* q.p.x: nested struct member — recursively compute address of q.p,
                     * then add offset of x within p's struct type */
                    const char *inner_struct_type = mir_dot_struct_type(g, n->left);
                    int inner_offset = mir_struct_member_offset(g, inner_struct_type ? inner_struct_type : "", n->left->ident);
                    /* Get base address of the root variable */
                    /* Walk left spine to find root IDENT */
                    const HDASTNode *dot_node = n->left;
                    while (dot_node && (dot_node->kind == HD_AST_DOT || dot_node->kind == HD_AST_MEMBER))
                        dot_node = dot_node->left;
                    if (dot_node && dot_node->kind == HD_AST_IDENT) {
                        base = mir_find_var_addr(g, dot_node->ident);
                        if (base == 0) return wubu_mir_const(g->prog, 0);
                    } else {
                        base = mir_address_of(g, n->left);
                        if (base == 0) return wubu_mir_const(g->prog, 0);
                    }
                    if (inner_offset >= 0) {
                        base = wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)inner_offset));
                    }
                    struct_type = inner_struct_type;
                } else if (n->left->kind == HD_AST_INDEX) {
                    /* arr[i].member: compute address of arr[i], then add member offset */
                    base = mir_address_of(g, n->left);
                    if (base == 0) return wubu_mir_const(g->prog, 0);
                    struct_type = mir_dot_struct_type(g, n->left);
                } else if (n->left->kind == HD_AST_DEREF) {
                    /* (*p).member is equivalent to p->member.
                     * Evaluate the pointer (child of DEREF) to get base address. */
                    base = mir_gen_expr(g, n->left->child);
                    if (base == 0) return wubu_mir_const(g->prog, 0);
                    struct_type = mir_dot_struct_type(g, n->left->child);
                } else if (n->left->kind == HD_AST_ADDR) {
                    /* (&s)->member: evaluate &s to get address, use as base */
                    base = mir_address_of(g, n->left);
                    if (base == 0) return wubu_mir_const(g->prog, 0);
                    struct_type = mir_dot_struct_type(g, n->left);
                } else {
                    return wubu_mir_const(g->prog, 0);
                }
            }
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            if (offset < 0) return wubu_mir_const(g->prog, 0);
            wubu_vr_t member_addr = wubu_mir_binop(g->prog, MIR_ADD, base,
                                                    wubu_mir_const(g->prog, (int64_t)offset));
            return wubu_mir_load(g->prog, member_addr);
        }
        return 0;
    }
    case HD_AST_ASSIGN: {
        wubu_vr_t val = mir_gen_expr(g, n->right);
        wubu_vr_t addr = mir_lvalue_addr(g, n->left);
        /* Handle struct-to-struct assignment: need to copy memory, not scalar.
         * Check both the AST node type and the var table (for globals where
         * type info may not be propagated to the IDENT node). */
        int left_is_struct = 0;
        if (n->left && n->left->type && n->left->type->kind == HD_TYPE_STRUCT)
            left_is_struct = 1;
        else if (n->left && n->left->kind == HD_AST_IDENT && n->left->ident[0])
            left_is_struct = mir_var_is_struct(g, n->left->ident);
        if (addr && left_is_struct) {
            /* Get the size of the struct */
            int struct_size = 0;
            const char *struct_name = NULL;
            if (n->left->kind == HD_AST_IDENT) {
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->left->ident) == 0 && g->vars[i].is_struct) {
                        struct_name = g->vars[i].struct_name;
                        break;
                    }
                }
            } else if (n->left->kind == HD_AST_DOT || n->left->kind == HD_AST_MEMBER) {
                struct_name = mir_find_var_struct_name(g, n->left->left ? n->left->left->ident : "");
            }
            if (struct_name) {
                mir_struct_t *s = mir_find_struct(g, struct_name);
                if (s) struct_size = s->total_size;
            }
            /* Fallback: if struct_name lookup failed but we know it's a struct
             * from the var table, try to get size from the var's array_size */
            if (struct_size <= 0 && n->left && n->left->kind == HD_AST_IDENT && n->left->ident[0]) {
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->left->ident) == 0 && g->vars[i].is_struct) {
                        struct_size = g->vars[i].array_size;
                        break;
                    }
                }
            }
            if (struct_size > 0) {
                /* Get source address: if RHS is a call returning struct, val is the source addr.
                 * If RHS is a struct IDENT, we need its address instead. */
                wubu_vr_t src_addr;
                if (n->right->kind == HD_AST_IDENT) {
                    src_addr = mir_find_var_addr(g, n->right->ident);
                } else {
                    /* val IS the source address (from call return or struct expr) */
                    src_addr = val;
                }
                /* Copy struct_size words from src_addr to addr.
                 * Loop while i < struct_size: jz exits when (i < struct_size) == 0,
                 * i.e. when i >= struct_size. */
                wubu_vr_t src_base = mir_new_vr(g);
                wubu_mir_mov_to(g->prog, src_base, src_addr);
                wubu_vr_t dst_base = mir_new_vr(g);
                wubu_mir_mov_to(g->prog, dst_base, addr);
                wubu_vr_t i_vr = mir_new_vr(g);
                wubu_mir_const_to(g->prog, i_vr, 0);
                uint32_t copy_label = wubu_mir_new_label(g->prog);
                uint32_t end_label = wubu_mir_new_label(g->prog);
                wubu_mir_place_label(g->prog, copy_label);
                /* jnz exits when (i >= struct_size) — i.e. loop is done. */
                wubu_mir_jnz(g->prog, wubu_mir_binop(g->prog, MIR_GE, i_vr, wubu_mir_const(g->prog, (int64_t)struct_size)), end_label);
                wubu_vr_t src_ptr = wubu_mir_binop(g->prog, MIR_ADD, src_base, i_vr);
                wubu_vr_t dst_ptr = wubu_mir_binop(g->prog, MIR_ADD, dst_base, i_vr);
                wubu_vr_t data = wubu_mir_load(g->prog, src_ptr);
                wubu_mir_store(g->prog, dst_ptr, data);
                wubu_mir_mov_to(g->prog, i_vr, wubu_mir_binop(g->prog, MIR_ADD, i_vr, wubu_mir_const(g->prog, 1)));
                wubu_mir_jmp(g->prog, copy_label);
                wubu_mir_place_label(g->prog, end_label);
                /* Return source address (struct return convention) */
                wubu_mir_mov_to(g->prog, 0, src_addr);
                return src_addr;
            }
        }
        /* Function pointer assignment: s.fn = add — store func_id in var for later indirect calls.
         * Detects: LHS is a DOT/MEMBER on a struct var, RHS is a function IDENT. */
        if (n->left && (n->left->kind == HD_AST_DOT || n->left->kind == HD_AST_MEMBER) &&
            n->right && n->right->kind == HD_AST_IDENT && n->right->ident[0]) {
            /* Resolve the root variable and the function */
            const HDASTNode *lhs = n->left;
            while (lhs && (lhs->kind == HD_AST_DOT || lhs->kind == HD_AST_MEMBER))
                lhs = lhs->left;
            if (lhs && lhs->kind == HD_AST_IDENT && lhs->ident[0]) {
                int fid = -1;
                for (int i = 0; i < g->prog->n_funcs; i++)
                    if (strcmp(g->prog->funcs[i].name, n->right->ident) == 0) { fid = i; break; }
                if (fid >= 0) {
                    for (int i = 0; i < g->n_vars; i++)
                        if (strcmp(g->vars[i].name, lhs->ident) == 0) {
                            g->vars[i].fn_ptr_func_id = fid;
                            break;
                        }
                }
            }
        }
        if (addr) { wubu_mir_store(g->prog, addr, val); return val; }
        return val;
    }
    case HD_AST_POST_INC:
    case HD_AST_POST_DEC: {
        /* tmp = v; v = v +/- 1; return tmp */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t tmp = mir_new_vr(g);
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_mir_mov_to(g->prog, tmp, v);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HD_AST_POST_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, tmp, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, tmp, one);
                wubu_mir_store(g->prog, addr, upd);
                return tmp;
            }
        }
        return mir_gen_expr(g, n->child);
    }
    case HD_AST_PRE_INC:
    case HD_AST_PRE_DEC: {
        /* v = v +/- 1; return v */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HD_AST_PRE_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, v, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, v, one);
                wubu_mir_store(g->prog, addr, upd);
                return upd;
            }
        }
        return mir_gen_expr(g, n->child);
    }
    case HD_AST_ADD_ASSIGN:
    case HD_AST_SUB_ASSIGN:
    case HD_AST_MUL_ASSIGN:
    case HD_AST_DIV_ASSIGN:
    case HD_AST_MOD_ASSIGN:
    case HD_AST_SHL_ASSIGN:
    case HD_AST_SHR_ASSIGN:
    case HD_AST_AMP_ASSIGN:
    case HD_AST_PIPE_ASSIGN:
    case HD_AST_CARET_ASSIGN: {
        /* left = left OP right — supports IDENT, INDEX (a[i]), DEREF (*p), MEMBER (s.f) */
        wubu_vr_t addr = mir_lvalue_addr(g, n->left);
        if (addr) {
            wubu_vr_t lhs = wubu_mir_load(g->prog, addr);
            wubu_vr_t rhs = mir_gen_expr(g, n->right);
            /* Detect float operands so compound assign uses float MIR ops */
            int is_float = (n->left && n->left->type && n->left->type->kind == HD_TYPE_F64) ||
                           (n->right && n->right->type && n->right->type->kind == HD_TYPE_F64) ||
                           (n->left && n->left->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->left->ident)) ||
                           (n->right && n->right->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->right->ident)) ||
                           (n->left && n->left->kind == HD_AST_FLOAT_LIT) ||
                           (n->right && n->right->kind == HD_AST_FLOAT_LIT);
            wubu_mir_op_t op = MIR_ADD;
            switch (n->kind) {
                case HD_AST_ADD_ASSIGN: op = is_float ? MIR_FADD : MIR_ADD; break;
                case HD_AST_SUB_ASSIGN: op = is_float ? MIR_FSUB : MIR_SUB; break;
                case HD_AST_MUL_ASSIGN: op = is_float ? MIR_FMUL : MIR_MUL; break;
                case HD_AST_DIV_ASSIGN: op = is_float ? MIR_FDIV : MIR_DIV; break;
                case HD_AST_MOD_ASSIGN: op = MIR_MOD; break;
                case HD_AST_SHL_ASSIGN: op = MIR_SHL; break;
                case HD_AST_SHR_ASSIGN: op = MIR_SHR; break;
                case HD_AST_AMP_ASSIGN: op = MIR_AND; break;
                case HD_AST_PIPE_ASSIGN: op = MIR_OR;  break;
                case HD_AST_CARET_ASSIGN: op = MIR_XOR; break;
                default: break;
            }
            wubu_vr_t upd = wubu_mir_binop(g->prog, op, lhs, rhs);
            wubu_mir_store(g->prog, addr, upd);
            return upd;
        }
        return mir_gen_expr(g, n->right);
    }
    case HD_AST_FLOAT_LIT: {
        /* Store as 32-bit float bits (matching test expectations for float add) */
        union { float f; uint32_t u; } u;
        u.f = (float)n->float_val;
        return wubu_mir_const(g->prog, (int64_t)u.u);
    }
    case HD_AST_BOOL_LIT:
        return wubu_mir_const(g->prog, n->int_val ? 1 : 0);
    case HD_AST_ADD: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        /* Pointer arithmetic: if left is a pointer to struct, scale right by struct size */
        if (n->left && n->left->type && n->left->type->kind == HD_TYPE_PTR
            && n->left->type->base && n->left->type->base->kind == HD_TYPE_STRUCT) {
            mir_struct_t *st = mir_find_struct(g, n->left->type->base->name);
            if (st && st->total_size > 1) {
                b = wubu_mir_binop(g->prog, MIR_MUL, b, wubu_mir_const(g->prog, (int64_t)st->total_size));
            }
        }
        /* Also check var table for pointer types (IDENT nodes don't have type annotations) */
        if (n->left && n->left->kind == HD_AST_IDENT && n->left->ident[0]) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, n->left->ident) == 0 && g->vars[i].is_ptr_struct) {
                    mir_struct_t *st = mir_find_struct(g, g->vars[i].struct_name);
                    if (st && st->total_size > 1) {
                        b = wubu_mir_binop(g->prog, MIR_MUL, b, wubu_mir_const(g->prog, (int64_t)st->total_size));
                    }
                    break;
                }
            }
        }
        /* Use float ops if either operand is F64 or a float variable */
        int is_float = (n->left && n->left->type && n->left->type->kind == HD_TYPE_F64) ||
                       (n->right && n->right->type && n->right->type->kind == HD_TYPE_F64) ||
                       (n->left && n->left->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->left->ident)) ||
                       (n->right && n->right->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->right->ident));
        return wubu_mir_binop(g->prog, is_float ? MIR_FADD : MIR_ADD, a, b);
    }
    case HD_AST_SUB: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        int is_float = (n->left && n->left->type && n->left->type->kind == HD_TYPE_F64) ||
                       (n->right && n->right->type && n->right->type->kind == HD_TYPE_F64) ||
                       (n->left && n->left->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->left->ident)) ||
                       (n->right && n->right->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->right->ident));
        return wubu_mir_binop(g->prog, is_float ? MIR_FSUB : MIR_SUB, a, b);
    }
    case HD_AST_MUL: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        int is_float = (n->left && n->left->type && n->left->type->kind == HD_TYPE_F64) ||
                       (n->right && n->right->type && n->right->type->kind == HD_TYPE_F64) ||
                       (n->left && n->left->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->left->ident)) ||
                       (n->right && n->right->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->right->ident));
        return wubu_mir_binop(g->prog, is_float ? MIR_FMUL : MIR_MUL, a, b);
    }
    case HD_AST_DIV: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        int is_float = (n->left && n->left->type && n->left->type->kind == HD_TYPE_F64) ||
                       (n->right && n->right->type && n->right->type->kind == HD_TYPE_F64) ||
                       (n->left && n->left->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->left->ident)) ||
                       (n->right && n->right->kind == HD_AST_IDENT && mir_find_var_is_float(g, n->right->ident));
        return wubu_mir_binop(g->prog, is_float ? MIR_FDIV : MIR_DIV, a, b);
    }
    case HD_AST_MOD: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_MOD, a, b);
    }
    case HD_AST_COMMA: {
        /* Evaluate left (discard), return right */
        wubu_vr_t left = mir_gen_expr(g, n->left);
        wubu_vr_t right = mir_gen_expr(g, n->right);
        return right;
    }
    case HD_AST_AND: {
        /* short-circuit: a && b  ==  (a!=0) ? (b!=0) : 0 */
        wubu_vr_t a = mir_gen_expr(g, n->left);
        uint32_t lbl_false = wubu_mir_new_label(g->prog);
        uint32_t lbl_end = wubu_mir_new_label(g->prog);
        wubu_vr_t za = wubu_mir_binop(g->prog, MIR_NE, a, wubu_mir_const(g->prog, 0));
        wubu_mir_jz(g->prog, za, lbl_false);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_vr_t zb = wubu_mir_binop(g->prog, MIR_NE, b, wubu_mir_const(g->prog, 0));
        wubu_vr_t merge = mir_new_vr(g);
        wubu_mir_mov_to(g->prog, merge, zb);
        wubu_mir_jmp(g->prog, lbl_end);
        wubu_mir_place_label(g->prog, lbl_false);
        wubu_mir_mov_to(g->prog, merge, wubu_mir_const(g->prog, 0));
        wubu_mir_place_label(g->prog, lbl_end);
        return merge;
    }
    case HD_AST_BITAND: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_AND, a, b);
    }
    case HD_AST_OR: {
        /* short-circuit: a || b  ==  (a!=0) ? 1 : (b!=0) */
        wubu_vr_t a = mir_gen_expr(g, n->left);
        uint32_t lbl_true = wubu_mir_new_label(g->prog);
        uint32_t lbl_end = wubu_mir_new_label(g->prog);
        wubu_vr_t za = wubu_mir_binop(g->prog, MIR_NE, a, wubu_mir_const(g->prog, 0));
        wubu_mir_jnz(g->prog, za, lbl_true);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_vr_t zb = wubu_mir_binop(g->prog, MIR_NE, b, wubu_mir_const(g->prog, 0));
        wubu_vr_t merge = mir_new_vr(g);
        wubu_mir_mov_to(g->prog, merge, zb);
        wubu_mir_jmp(g->prog, lbl_end);
        wubu_mir_place_label(g->prog, lbl_true);
        wubu_mir_mov_to(g->prog, merge, wubu_mir_const(g->prog, 1));
        wubu_mir_place_label(g->prog, lbl_end);
        return merge;
    }
    case HD_AST_BITOR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_OR, a, b);
    }
    case HD_AST_BITXOR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_XOR, a, b);
    }
    case HD_AST_SHL: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_SHL, a, b);
    }
    case HD_AST_SHR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_SHR, a, b);
    }
    case HD_AST_NEG: {
        wubu_vr_t a = mir_gen_expr(g, n->child);
        /* Float negation uses MIR_FNEG (f32 bit-flip); integer uses MIR_NEG */
        bool is_float = mir_is_float_node(g, n->child);
        if (is_float)
            return wubu_mir_unop(g->prog, MIR_FNEG, a);
        return wubu_mir_unop(g->prog, MIR_NEG, a);
    }
    case HD_AST_BITNOT: {
        wubu_vr_t a = mir_gen_expr(g, n->child);
        return wubu_mir_unop(g->prog, MIR_NOT, a);
    }
    case HD_AST_NOT: {
        /* !x = (x == 0) ? 1 : 0 */
        wubu_vr_t a = mir_gen_expr(g, n->child);
        wubu_vr_t zero = wubu_mir_const(g->prog, 0);
        return wubu_mir_binop(g->prog, MIR_EQ, a, zero);
    }
    case HD_AST_EQ: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_EQ, a, b);
    }
    case HD_AST_NE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_NE, a, b);
    }
    case HD_AST_LT: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_ULT : MIR_LT;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HD_AST_LE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_ULE : MIR_LE;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HD_AST_GT: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_UGT : MIR_GT;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HD_AST_GE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_UGE : MIR_GE;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HD_AST_TERNARY: {
        /* cond ? then : else  ->  merge = (cond!=0) ? then : else */
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        uint32_t else_label = wubu_mir_new_label(g->prog);
        uint32_t end_label = wubu_mir_new_label(g->prog);
        wubu_mir_jz(g->prog, cond, else_label);
        wubu_vr_t merge = mir_new_vr(g);
        wubu_vr_t then_val = mir_gen_expr(g, n->then_branch);
        wubu_mir_mov_to(g->prog, merge, then_val);
        wubu_mir_jmp(g->prog, end_label);
        wubu_mir_place_label(g->prog, else_label);
        wubu_vr_t else_val = mir_gen_expr(g, n->else_branch);
        wubu_mir_mov_to(g->prog, merge, else_val);
        wubu_mir_place_label(g->prog, end_label);
        return merge;
    }
    case HD_AST_CAST: {
        wubu_vr_t val = mir_gen_expr(g, n->child);
        if (!n->type) return val;
        /* Determine if the source is f64: check the AST node type, or
         * look up the variable type from the symbol table for IDENT nodes */
        bool from_f64 = mir_is_float_node(g, n->child);
        if (!from_f64 && n->child && n->child->kind == HD_AST_IDENT) {
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].is_float) {
                    from_f64 = true;
                    break;
                }
            }
        }
        bool to_f64 = (n->type->kind == HD_TYPE_F64);
        if (to_f64 && !from_f64) {
            /* int → f32 (use ITOF for 64-bit int to float) */
            return wubu_mir_unop(g->prog, MIR_ITOF, val);
        } else if (!to_f64 && from_f64) {
            /* f32 → int (use FTOI for float to 64-bit int) */
            return wubu_mir_unop(g->prog, MIR_FTOI, val);
        }
        /* Integer-to-integer cast: truncate to target type width.
         * In the MIR model, all values are int64 cells. We need to mask
         * the value to the target type's bit width and sign-extend for signed types. */
        if (!to_f64 && !from_f64 && n->type) {
            switch (n->type->kind) {
            case HD_TYPE_I8:
                /* (char)x: keep low 8 bits, sign-extend from bit 7 */
                val = wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFF));
                val = wubu_mir_binop(g->prog, MIR_SHL, val, wubu_mir_const(g->prog, 56)); /* shift left to sign bit */
                val = wubu_mir_binop(g->prog, MIR_SHR, val, wubu_mir_const(g->prog, 56)); /* arithmetic shift right for sign-extension */
                return val;
            case HD_TYPE_U8:
                /* (unsigned char)x: keep low 8 bits */
                return wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFF));
            case HD_TYPE_I16:
                /* (short)x: keep low 16 bits, sign-extend from bit 15 */
                val = wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFF));
                val = wubu_mir_binop(g->prog, MIR_SHL, val, wubu_mir_const(g->prog, 48));
                val = wubu_mir_binop(g->prog, MIR_SHR, val, wubu_mir_const(g->prog, 48));
                return val;
            case HD_TYPE_U16:
                /* (unsigned short)x: keep low 16 bits */
                return wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFF));
            default:
                break;
            }
        }
        /* Same-width integer cast: no-op */
        return val;
    }
    case HD_AST_SIZEOF: {
        /* sizeof(type) or sizeof(expr) — emit the type size in BYTES as a constant.
         * n->type->size for structs is in int64 cells; multiply by 8 for bytes.
         * For arrays, n->type->size is 0 (not set by parser); use hd_type_size instead.
         * For sizeof(expr) without type annotation, derive from child. */
        int size = 8; /* default: pointer */
        /* If n->type is not set (sizeof expr), try to derive from child */
        if (!n->type && n->child) {
            if (n->child->kind == HD_AST_IDENT) {
                /* Look up the variable's type from the symbol table */
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->child->ident) == 0) {
                        if (g->vars[i].is_struct) {
                            /* Compute packed byte size from member offsets */
                            mir_struct_t *st = mir_find_struct(g, g->vars[i].struct_name);
                            if (st && st->n_members > 0) {
                                /* Last member's offset + its size */
                                int last_mi = st->n_members - 1;
                                int last_off = st->member_offsets[last_mi];
                                size = (last_off + 1) * 8; /* each cell = 8 bytes */
                                /* But this over-counts; use heuristic: if all members are int (1 cell each),
                                 * packed size = n_members * 4 bytes */
                                size = st->n_members * 4; /* assume all int members */
                            } else {
                                size = 8;
                            }
                            if (size <= 0) size = 8;
                        } else if (g->vars[i].is_array) {
                            /* Array: element_size * element_count in bytes */
                            size = g->vars[i].array_size * 4; /* assume int elements = 4B each */
                            if (size <= 0) size = 4;
                        } else {
                            /* Scalar: int = 4 bytes, pointer = 8 bytes */
                            size = 4; /* int is 4 bytes */
                        }
                        break;
                    }
                }
                /* Also check the child's type annotation if available */
                if (n->child->type) {
                    size = (int)hd_type_size(n->child->type);
                    if (size <= 0) size = 8;
                }
            } else if (n->child->type) {
                /* Use the child's type annotation */
                size = (int)hd_type_size(n->child->type);
                if (size <= 0) size = 8;
            } else if (n->child->kind == HD_AST_DOT || n->child->kind == HD_AST_MEMBER) {
                /* sizeof(s.a): look up member type from struct */
                const char *varname = (n->child->left && n->child->left->kind == HD_AST_IDENT)
                    ? n->child->left->ident : NULL;
                if (varname && n->child->ident[0]) {
                    const char *struct_type = mir_find_var_struct_name(g, varname);
                    if (struct_type[0]) {
                        int moff = mir_struct_member_offset(g, struct_type, n->child->ident);
                        if (moff >= 0) {
                            /* Find the member's type from the struct definition */
                            mir_struct_t *st = mir_find_struct(g, struct_type);
                            if (st) {
                                /* Look up member type from parser type info */
                                /* For now, use heuristic: if member offset matches, assume int (4) */
                                size = 4; /* default for int members */
                                /* Check if it's a long long member (8 bytes) by looking at next member offset */
                                for (int mi = 0; mi < st->n_members; mi++) {
                                    if (strcmp(st->member_names[mi], n->child->ident) == 0) {
                                        /* Found the member; assume int size (4 bytes).
                                         * TODO: look up actual member type for long long, pointer, etc. */
                                        size = 4;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (n->type) {
            switch (n->type->kind) {
            case HD_TYPE_VOID:   size = 1; break;
            case HD_TYPE_I8:     size = 1; break;
            case HD_TYPE_U8:     size = 1; break;
            case HD_TYPE_I16:    size = 2; break;
            case HD_TYPE_U16:    size = 2; break;
            case HD_TYPE_I32:    size = 4; break;
            case HD_TYPE_U32:    size = 4; break;
            case HD_TYPE_I64:    size = 8; break;
            case HD_TYPE_U64:    size = 8; break;
            case HD_TYPE_F64:    size = 8; break;
            case HD_TYPE_BOOL:   size = 1; break;
            case HD_TYPE_PTR:    size = 8; break;
            case HD_TYPE_ARRAY: {
                /* n->type->size is 0 for arrays; compute total bytes */
                size = (int)hd_type_size(n->type);
                if (size <= 0) size = 4;
                break;
            }
            case HD_TYPE_STRUCT:
                /* Use hd_type_size for accurate packed byte size */
                size = (int)hd_type_size(n->type);
                if (size <= 0) size = 8;
                break;
            default:             size = 8; break;
            }
        }
        return wubu_mir_const(g->prog, (int64_t)size);
    }
    case HD_AST_STRING_LIT: {
        /* Store string in memory and return address. */
        size_t len = strlen(n->str_val) + 1; /* include NUL */
        wubu_vr_t addr = wubu_mir_alloc(g->prog, (uint32_t)len);
        for (size_t i = 0; i < len; i++)
            wubu_mir_store(g->prog, wubu_mir_binop(g->prog, MIR_ADD, addr, wubu_mir_const(g->prog, (int64_t)i)),
                           wubu_mir_const(g->prog, (int64_t)(unsigned char)n->str_val[i]));
        return addr;
    }
    case HD_AST_INDEX: {
        /* a[i] -> load from address_of(a) + i * stride. array name decays to base;
         * pointer var loads its held value. */
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx = mir_gen_expr(g, n->right);
        int stride = mir_index_stride(g, n);
        if (stride > 1) {
            idx = wubu_mir_binop(g->prog, MIR_MUL, idx, wubu_mir_const(g->prog, (int64_t)stride));
        }
        wubu_vr_t addr = wubu_mir_binop(g->prog, MIR_ADD, base, idx);
        return wubu_mir_load(g->prog, addr);
    }
    case HD_AST_ADDR: {
        /* &x -> the memory address of x (lvalue address-of) */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) return addr;
        }
        /* For &s[i], &*p, &s->member, &s.member: delegate to mir_address_of */
        return mir_address_of(g, n->child);
    }
    case HD_AST_DEREF: {
        /* *p -> load from address held in p */
        wubu_vr_t addr = mir_gen_expr(g, n->child);
        return wubu_mir_load(g->prog, addr);
    }
    case HD_AST_CALL:
    case HD_AST_FUNC_CALL: {
        /* Resolve callee name -> func_id via the collected func table. */
        int fid = -1;
        if (n->callee && n->callee->kind == HD_AST_IDENT) {
            for (int i = 0; i < g->prog->n_funcs; i++)
                if (strcmp(g->prog->funcs[i].name, n->callee->ident) == 0) { fid = i; break; }
        }
        /* Function pointer member call: s.fn(args) where callee is a DOT/MEMBER expr.
         * The function pointer was previously stored as a func_id in the var.
         * Resolve the root variable name and look up fn_ptr_func_id. */
        if (fid < 0 && n->callee && (n->callee->kind == HD_AST_DOT || n->callee->kind == HD_AST_MEMBER)) {
            const HDASTNode *callee = n->callee;
            /* Walk down to find the root IDENT */
            while (callee && (callee->kind == HD_AST_DOT || callee->kind == HD_AST_MEMBER))
                callee = callee->left;
            if (callee && callee->kind == HD_AST_IDENT && callee->ident[0]) {
                /* Look up the variable's fn_ptr_func_id */
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, callee->ident) == 0 &&
                        g->vars[i].fn_ptr_func_id >= 0) {
                        fid = g->vars[i].fn_ptr_func_id;
                        break;
                    }
                }
            }
        }
        /* Place arguments in v1..vN (calling convention).
         * For struct-by-value arguments, the argument is an address pointing
         * to the struct data; we store that address directly so the callee
         * can copy from it. For scalar arguments, we store the value. */
        for (uint32_t a = 0; a < n->n_args && a < MIR_MAX_CALL_ARGS; a++) {
            int arg_is_struct = 0;
            if (n->args[a]) {
                if (n->args[a]->type && n->args[a]->type->kind == HD_TYPE_STRUCT)
                    arg_is_struct = 1;
                else if (n->args[a]->kind == HD_AST_IDENT && n->args[a]->ident[0] &&
                         mir_var_is_struct(g, n->args[a]->ident))
                    arg_is_struct = 1;
            }
            wubu_vr_t av;
            if (arg_is_struct) {
                av = mir_address_of(g, n->args[a]);
            } else {
                av = mir_gen_expr(g, n->args[a]);
            }
            if (av != (wubu_vr_t)(a + 1)) {
                wubu_mir_mov_to(g->prog, a + 1, av);
            }
        }
        /* For unknown functions (fid < 0), use func_id 0xFFFF that the JIT
         * recognizes as "external" and handles by returning 0 instead of crashing.
         * Without this, unknown func calls resolve to main (func_id 0),
         * causing infinite recursion or crashes. */
        uint32_t call_fid = (fid >= 0) ? (uint32_t)fid : 0xFFFF;
        wubu_mir_call(g->prog, call_fid);
        /* Callee returns in vr0; capture it into a fresh vr for the caller. */
        wubu_vr_t rv = mir_new_vr(g);
        wubu_mir_mov_to(g->prog, rv, 0);
        return rv;
    }
    case HD_AST_LABEL: {
        /* label: — place a MIR_LABEL with the label name */
        /* Check if this label already has a placeholder (from a forward goto) */
        uint32_t lbl = 0;
        for (int i = 0; i < g->n_labels; i++) {
            if (strcmp(g->label_names[i], n->ident) == 0) { lbl = g->label_ids[i]; break; }
        }
        if (lbl == 0) {
            lbl = wubu_mir_new_label(g->prog);
            if (n->ident[0] && g->n_labels < 64) {
                strncpy(g->label_names[g->n_labels], n->ident, HD_MAX_IDENT_LEN - 1);
                g->label_names[g->n_labels][HD_MAX_IDENT_LEN - 1] = '\0';
                g->label_ids[g->n_labels] = lbl;
                g->n_labels++;
            }
        }
        wubu_mir_place_label(g->prog, lbl);
        return 0;
    }
    case HD_AST_GOTO: {
        /* goto label; — jump to the named label */
        /* Find the label */
        uint32_t lbl = 0;
        for (int i = 0; i < g->n_labels; i++) {
            if (strcmp(g->label_names[i], n->ident) == 0) { lbl = g->label_ids[i]; break; }
        }
        if (lbl == 0) {
            /* Forward reference — create a placeholder label */
            lbl = wubu_mir_new_label(g->prog);
            if (n->ident[0] && g->n_labels < 64) {
                strncpy(g->label_names[g->n_labels], n->ident, HD_MAX_IDENT_LEN - 1);
                g->label_ids[g->n_labels] = lbl;
                g->n_labels++;
            }
        }
        wubu_mir_jmp(g->prog, lbl);
        return 0;
    }
    case HD_AST_SWITCH: {
        /* switch(expr) { case VAL: ... break; default: ... }
         * Lowered as a chain of if-else:
         *   if (expr == VAL1) goto case1;
         *   if (expr == VAL2) goto case2;
         *   goto default_label;
         *   case1: ... break_label;
         *   case2: ... break_label;
         *   default_label: ...
         *   break_label: ...
         */
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        uint32_t break_label = wubu_mir_new_label(g->prog);
        uint32_t default_label = wubu_mir_new_label(g->prog);
        /* We need to collect case labels first, then emit them */
        /* For now, emit a simplified version: chain of comparisons */
        /* Each case: if (cond == val) { body; jmp break_label } */
        /* Track case values and their labels */
        typedef struct { int64_t val; uint32_t label; } case_entry_t;
        case_entry_t cases[64];
        int ncases = 0;
        uint32_t cl = default_label;

        /* First pass: create labels for each case and the default */
        /* We emit: for each case, cmp + jz to body; after body, jmp break */
        /* At the end: default label (if any), then break label */

        /* Simple approach: emit comparisons inline */
        HDASTNode *body = n->body; /* BLOCK of CASE nodes */
        if (body && body->kind == HD_AST_BLOCK) {
            /* Collect case values (skip default case — it has cond==NULL) */
            for (uint32_t i = 0; i < body->n_stmts; i++) {
                HDASTNode *stmt = body->stmts[i];
                if (stmt->kind == HD_AST_CASE && stmt->cond != NULL && ncases < 64) {
                    /* Evaluate case value */
                    wubu_vr_t cval = mir_gen_expr(g, stmt->cond);
                    uint32_t case_label = wubu_mir_new_label(g->prog);
                    /* Emit: if (cond == cval) goto case_label */
                    wubu_vr_t cmp = wubu_mir_binop(g->prog, MIR_EQ, cond, cval);
                    wubu_mir_jnz(g->prog, cmp, case_label);
                    cases[ncases].val = 0; /* unused */
                    cases[ncases].label = case_label;
                    ncases++;
                }
            }
            /* Jump to default if no case matched */
            wubu_mir_jmp(g->prog, default_label);

            /* Second pass: emit case bodies */
            int ci = 0;
            for (uint32_t i = 0; i < body->n_stmts; i++) {
                HDASTNode *stmt = body->stmts[i];
                if (stmt->kind == HD_AST_CASE && stmt->cond != NULL && ci < ncases) {
                    wubu_mir_place_label(g->prog, cases[ci].label);
                    /* Emit case body statements */
                    if (stmt->body && stmt->body->kind == HD_AST_BLOCK) {
                        for (uint32_t j = 0; j < stmt->body->n_stmts; j++) {
                            mir_gen_stmt(g, stmt->body->stmts[j]);
                        }
                    } else if (stmt->body) {
                        mir_gen_stmt(g, stmt->body);
                    }
                    /* Note: no automatic JMP break_label — fall-through is the
                     * default C semantics. Break statements emit their own JMP. */
                    ci++;
                } else if (stmt->kind == HD_AST_CASE && stmt->cond == NULL) {
                    /* Default case (cond == NULL) */
                    wubu_mir_place_label(g->prog, default_label);
                    if (stmt->body && stmt->body->kind == HD_AST_BLOCK) {
                        for (uint32_t j = 0; j < stmt->body->n_stmts; j++) {
                            mir_gen_stmt(g, stmt->body->stmts[j]);
                        }
                    } else if (stmt->body) {
                        mir_gen_stmt(g, stmt->body);
                    }
                }
            }
            /* If no default case, place the default label here */
            int has_default = 0;
            for (uint32_t i = 0; i < body->n_stmts; i++) {
                if (body->stmts[i]->kind == HD_AST_CASE && body->stmts[i]->cond == NULL) {
                    has_default = 1; break;
                }
            }
            if (!has_default) {
                wubu_mir_place_label(g->prog, default_label);
            }
        }
        wubu_mir_place_label(g->prog, break_label);
        /* Push break label for any break statements inside */
        int prev_break = -1;
        if (g->n_loops < MIRGEN_MAX_VARS) {
            prev_break = g->loop_done[g->n_loops];
            g->loop_done[g->n_loops] = break_label;
            g->n_loops++;
        }
        return 0;
    }
    default:
        /* Unsupported: emit 0 */
        return wubu_mir_const(g->prog, 0);
    }
}

/*
 * hd_build_mir: parse HolyD source -> canonical, optimized MIR.
 * Returns 0 on success (prog initialized, caller must wubu_mir_free it),
 * or -1 on parse/lower error (prog left uninitialized). The SAME prog is
 * consumed by every ISA driver, so building it ONCE and running it through
 * all N backends is both correct and fast (the differential battery).
 */
int hd_build_mir(const char *source, wubu_mir_prog_t *prog) {
    /* Preprocess #define / strip directives — must match hd_eval exactly so
     * the MIR path parses the same sources the x86-64 JIT (golden) does. */
    char *pp = wubu_preprocess(source);
    const char *effective = pp ? pp : source;

    HDLexer lex;
    hd_lex_init(&lex, effective);
    if (lex.has_error) { free(pp); return -1; }

    HDParser parse;
    hd_parse_init(&parse, &lex);

    HDASTNode *ast;
    if (*source == '{') {
        ast = hd_parse_stmt(&parse);
    } else {
        ast = hd_parse_expr(&parse);

        if (parse.has_error || (hd_parse_peek(&parse) != HD_TOK_EOF && hd_parse_peek(&parse) != HD_TOK_SEMI)) {
            hd_ast_free(ast);
            parse.has_error = false;
            parse.n_errors = 0;
            hd_lex_init(&lex, effective);
            hd_parse_init(&parse, &lex);
            const char *p = source;
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
            bool starts_with_keyword = false;
            if (strncmp(p, "if ", 3) == 0 || strncmp(p, "while ", 6) == 0 ||
                strncmp(p, "for ", 4) == 0 || strncmp(p, "do ", 3) == 0 ||
                strncmp(p, "return", 6) == 0 || strncmp(p, "break", 5) == 0 ||
                strncmp(p, "continue", 8) == 0 || *p == '{') {
                starts_with_keyword = true;
            }
            bool has_semicolon = false;
            p = source;
            while (*p) { if (*p == ';') { has_semicolon = true; break; } p++; }
            if (has_semicolon && !starts_with_keyword) {
                size_t len = strlen(effective);
                char *wrapped = malloc(len + 5);
                sprintf(wrapped, "{ %s }", effective);
                hd_lex_init(&lex, wrapped);
                hd_parse_init(&parse, &lex);
                ast = hd_parse_block(&parse);
                free(wrapped);
            } else {
                ast = hd_parse_stmt(&parse);
            }
        }
    }

    if (parse.has_error || !ast) {
        hd_ast_free(ast);
        free(pp);
        return -1;
    }

    wubu_mir_init(prog);
    HDMirGen g;
    memset(&g, 0, sizeof(g));
    g.prog = prog;
    g.next_vr = 1u << 16;  /* 65536 — far above any instr-index vr */
    g.has_error = 0;
    prog->next_vr_hi = g.next_vr;  /* reserve mem up to the high-vr param-slot range */

    /* Register struct/union types from the parser's named_types table.
     * Parser stores sizes and offsets in BYTES; MIR uses int64 CELLS.
     * Convert: cells = (bytes + 7) / 8 */
    for (int i = 0; i < parse.n_named_types && g.n_structs < MAX_STRUCTS; i++) {
        HDType *t = parse.named_types[i];
        if (t && (t->kind == HD_TYPE_STRUCT || t->kind == HD_TYPE_UNION) && t->n_members > 0) {
            mir_struct_t *s = &g.structs[g.n_structs++];
            strncpy(s->name, parse.named_type_names[i], HD_MAX_IDENT_LEN - 1);
            s->name[HD_MAX_IDENT_LEN - 1] = '\0';
            s->n_members = 0;
            s->total_size = 0;
            for (int j = 0; j < t->n_members && s->n_members < MAX_MEMBERS; j++) {
                strncpy(s->member_names[s->n_members], t->members[j].name, HD_MAX_IDENT_LEN - 1);
                s->member_names[s->n_members][HD_MAX_IDENT_LEN - 1] = '\0';
                s->member_offsets[s->n_members] = (int)t->members[j].offset;
                s->member_is_unsigned[s->n_members] = (t->members[j].type && (t->members[j].type->kind == HD_TYPE_U8 || t->members[j].type->kind == HD_TYPE_U16 || t->members[j].type->kind == HD_TYPE_U32 || t->members[j].type->kind == HD_TYPE_U64)) ? 1 : 0;
                if (t->members[j].type && t->members[j].type->kind == HD_TYPE_STRUCT && t->members[j].type->name[0])
                    strncpy(s->member_type_names[s->n_members], t->members[j].type->name, HD_MAX_IDENT_LEN - 1);
                s->n_members++;
            }
            s->total_size = (int)t->size;  /* size in int64 cells */
        }
    }

    /* Phase 1: collect top-level function definitions into the func table
     * (assign stable ids so CALL sites resolve regardless of order). */
    mir_collect_funcs(&g, ast);

    /* Phase 1b: determine max args in the program and pre-allocate argument VRs.
     * v1..vN are pre-assigned to physical regs 1..N (capped at 6 per ABI). */
    int max_args = mir_count_max_args(ast);
    wubu_mir_set_n_args(prog, (uint32_t)max_args);

    /* Phase 2: generate top-level (module) statements, skipping function
     * definitions (their bodies are emitted separately in Phase 3).
     * Track the last expression result so the top-level RETURN carries it.
     * Global variables must persist across all functions: set no_scope_pop
     * on the top-level block so the BLOCK handler doesn't pop their scope. */
    wubu_vr_t top_val = 0;
    if (ast->kind == HD_AST_BLOCK) {
        ((HDASTNode *)ast)->no_scope_pop = 1;
        for (uint32_t i = 0; i < ast->n_stmts; i++)
            if (ast->stmts[i]->kind != HD_AST_FUNC_DECL)
                top_val = mir_gen_stmt(&g, ast->stmts[i]);
    } else {
        top_val = mir_gen_expr(&g, ast);
    }

    /* Phase 3+4: emit the ENTRY point FIRST (so the interpreter, which starts
     * at pc=0, runs main), then emit each function body into the MIR recording
     * its start/end in the func table. Parameters are bound to v1..vN. */
    int main_id = -1;
    for (int i = 0; i < prog->n_funcs; i++)
        if (strcmp(prog->funcs[i].name, "main") == 0) { main_id = i; break; }
    if (main_id >= 0) {
        wubu_mir_call(prog, (uint32_t)main_id);
        wubu_mir_ret(prog, 0);   /* vr0 holds main()'s return */
    } else {
        wubu_mir_ret(prog, top_val);
    }

    for (int fi = 0; fi < g.n_funcs; fi++) {
        const HDASTNode *fn = g.func_ast[fi];
        prog->funcs[fi].start = prog->n;
        /* Push scope BEFORE binding params so pop removes them too */
        if (g.n_scopes < MIRGEN_MAX_VARS)
            g.scope_var_start[g.n_scopes++] = g.n_vars;
        /* bind parameters: use HIGH virtual registers for both the address
         * slot and the value so they NEVER collide with v1..vN (the argument
         * registers). slot_addr is a high-vr holding the memory address; base
         * is a high-vr that holds slot_addr; copy the incoming arg (v1..vN)
         * into the slot, and register the param name so references load it. */
        for (int pi = 0; pi < fn->n_params; pi++) {
            /* Check if this parameter is a struct by value */
            int param_is_struct = 0;
            int param_struct_size = 1;
            if (fn->param_types[pi] && fn->param_types[pi]->kind == HD_TYPE_STRUCT) {
                param_is_struct = 1;
                /* Look up struct size (in cells) */
                if (fn->param_types[pi]->name[0]) {
                    mir_struct_t *ps = mir_find_struct(&g, fn->param_types[pi]->name);
                    if (ps && ps->total_size > 0) param_struct_size = ps->total_size;
                }
                if (param_struct_size <= 0) param_struct_size = 1;
            }
            /* Allocate memory for this parameter: struct params get
             * param_struct_size cells; scalar params get 1 cell. */
            wubu_vr_t addr = mir_new_vr(&g);  /* high VR for the address */
            int64_t mem_addr = (int64_t)(prog->total_mem + 1);
            prog->total_mem = mem_addr + param_struct_size - 1;
            wubu_mir_const_to(prog, addr, mem_addr);  /* addr VR = memory address */
            if (param_is_struct) {
                /* Struct-by-value: v(pi+1) holds the source address.
                 * Copy param_struct_size cells from src addr to param slot. */
                wubu_vr_t src_addr_vr = mir_new_vr(&g);
                wubu_mir_mov_to(prog, src_addr_vr, (wubu_vr_t)(pi + 1));
                for (int m = 0; m < param_struct_size; m++) {
                    wubu_vr_t src_p = wubu_mir_binop(prog, MIR_ADD, src_addr_vr, wubu_mir_const(prog, (int64_t)m));
                    wubu_vr_t dst_p = wubu_mir_binop(prog, MIR_ADD, addr, wubu_mir_const(prog, (int64_t)m));
                    wubu_mir_store(prog, dst_p, wubu_mir_load(prog, src_p));
                }
            } else {
                /* Scalar parameter: copy the argument value directly */
                wubu_mir_store(prog, addr, (wubu_vr_t)(pi + 1));
            }
            int param_is_unsigned = 0;
            if (fn->param_types[pi]) {
                HDTypeKind pk = fn->param_types[pi]->kind;
                if (pk == HD_TYPE_U8 || pk == HD_TYPE_U16 || pk == HD_TYPE_U32 || pk == HD_TYPE_U64)
                    param_is_unsigned = 1;
            }
            mir_bind_var(&g, fn->param_names[pi], addr, addr, param_is_unsigned);
            /* If the parameter is a struct by value, mark is_struct so member access works */
            if (param_is_struct && fn->param_types[pi]->name[0]) {
                for (int i = 0; i < g.n_vars; i++) {
                    if (strcmp(g.vars[i].name, fn->param_names[pi]) == 0) {
                        g.vars[i].is_struct = 1;
                        strncpy(g.vars[i].struct_name, fn->param_types[pi]->name, HD_MAX_IDENT_LEN - 1);
                        g.vars[i].struct_name[HD_MAX_IDENT_LEN - 1] = '\0';
                        g.vars[i].array_size = param_struct_size;
                        break;
                    }
                }
            }
            /* For pointer-to-struct parameters, mark the variable so -> lookups work */
            if (fn->param_types[pi] && fn->param_types[pi]->kind == HD_TYPE_PTR && fn->param_types[pi]->base &&
                (fn->param_types[pi]->base->kind == HD_TYPE_STRUCT || fn->param_types[pi]->base->kind == HD_TYPE_UNION)) {
                for (int i = 0; i < g.n_vars; i++) {
                    if (strcmp(g.vars[i].name, fn->param_names[pi]) == 0) {
                        g.vars[i].is_ptr_struct = 1;
                        if (fn->param_types[pi]->base->name[0])
                            strncpy(g.vars[i].struct_name, fn->param_types[pi]->base->name, HD_MAX_IDENT_LEN - 1);
                        break;
                    }
                }
            }
        }
        /* Set up early-return: RETURN emits `result_vr = expr; jmp ret_label`,
         * and the epilogue (placed after the body) moves result_vr into vr0
         * and returns. This makes `if(c) return x; return y;` correct. */
        g.fn_ret_vr = mir_new_vr(&g);
        wubu_mir_const_to(prog, g.fn_ret_vr, 0);  /* default return = 0 */
        g.fn_ret_label = wubu_mir_new_label(prog);
        g.in_function_body = 1;
        mir_gen_stmt(&g, fn->body);
        g.in_function_body = 0;
        /* Pop the function body scope to clean up params/locals */
        if (g.n_scopes > 0)
            g.n_vars = g.scope_var_start[--g.n_scopes];
        wubu_mir_place_label(prog, g.fn_ret_label);
        wubu_mir_mov_to(prog, 0, g.fn_ret_vr);   /* callee returns in vr0 */
        wubu_mir_ret(prog, 0);
        g.fn_ret_label = 0;
        prog->funcs[fi].end = prog->n;
    }

    /* Record the final high-vr water mark so the interpreter sizes its
     * memory array to cover the high-vr parameter slot addresses. */
    if (g.next_vr > prog->next_vr_hi) prog->next_vr_hi = g.next_vr;

    hd_ast_free(ast);

    if (g.has_error || prog->n == 0) {
        wubu_mir_free(prog);
        free(pp);
        return -1;
    }

    /* DEBUG: dump MIR before optimization */
    if (getenv("WUBU_DEBUG_MIR_BEFORE")) {
        fprintf(stderr, "[DEBUG_MIR_BEFORE] ===\n");
        wubu_mir_dump(prog);
    }

    /* Optimize the canonical MIR once (benefits ALL backends). Side-effect
     * safe passes only; the optimizer preserves semantics, so the differential
     * battery (every driver agrees with the portable interp oracle) proves
     * cross-target correctness. */
    wubu_mir_optimize(prog,
                      MIR_OPT_FOLD | MIR_OPT_STRENGTH | MIR_OPT_DCE |
                      MIR_OPT_COMBINE | MIR_OPT_CSE);

    /* DEBUG: dump MIR after optimization */
    if (getenv("WUBU_DEBUG_MIR_AFTER")) {
        fprintf(stderr, "[DEBUG_MIR_AFTER] ===\n");
        wubu_mir_dump(prog);
    }

    free(pp);
    return 0;
}

/*
 * hd_run_prog: compile the canonical MIR through the given driver and run its
 * emitted code. If driver is NULL or its encoder fails, fall back to the
 * portable interpreter (the golden oracle) so a result is always produced.
 * A result disagreement between two drivers on the same prog is a
 * frontend/lowering bug, never an encoder artifact (all drivers consume the
 * identical MIR).
 */
int64_t hd_run_prog(const wubu_mir_prog_t *prog, const wubu_isa_driver_t *driver) {
    /* Interpreter-family backends execute the canonical MIR directly via the
     * reference interpreter (wubu_mir_interp) — that IS their execution
     * engine, and it is the correctness oracle for canonical MIR. Native
     * backends (x86-64/arm64/wasm/ptx/mips) use their own encoder+runner.
     * This guarantees every target returns the same canonical-MIR result. */
    if (driver && driver->exec == WUBU_ISA_INTERPRETED)
        return wubu_mir_interp(prog);

    /* Ensure mem is allocated before JIT compile (JIT embeds mem pointer
     * as immediate in movabs instructions). The interpreter allocates it
     * lazily, but the JIT needs it at compile time. */
    int64_t *mem_ptr = prog->mem;
    if (mem_ptr == NULL) {
        int64_t mem_hi = prog->total_mem;
        if ((int64_t)(prog->next_vr_hi) - 1 > mem_hi) mem_hi = (int64_t)(prog->next_vr_hi) - 1;
        int64_t mem_size = (mem_hi < 1) ? 1 : (mem_hi + 1);
        mem_ptr = (int64_t *)calloc((size_t)mem_size, sizeof(int64_t));
    }

    /* Build a mutable copy of prog with mem set for the JIT compiler */
    wubu_mir_prog_t prog_copy = *prog;
    prog_copy.mem = mem_ptr;

    uint8_t *code = NULL;
    size_t csize = 0;
    if (driver && driver->compile && driver->compile(&prog_copy, &code, &csize) == 0 && code) {
        int64_t result = driver->run(code, csize, (int64_t)mem_ptr);
        free(code);
        if (mem_ptr != prog->mem) free(mem_ptr);
        return result;
    }
    if (mem_ptr != prog->mem) free(mem_ptr);
    return wubu_mir_interp(prog);
}

/*
 * hd_eval_mir: parse HolyD source, emit MIR, compile + run via driver.
 * Convenience wrapper = hd_build_mir + hd_run_prog. Returns the result (0 on
 * error). Kept for callers that only need a single driver result.
 */
int64_t hd_eval_mir(const char *source, const wubu_isa_driver_t *driver) {
    wubu_mir_prog_t prog;
    if (hd_build_mir(source, &prog) != 0) return 0;
    int64_t result = hd_run_prog(&prog, driver);
    wubu_mir_free(&prog);
    return result;
}

/*
 * hd_build_mir_ex: like hd_build_mir, but on failure fills errbuf (size
 * errcap) with the parser's first diagnostic message. Used by the gauntlet
 * to classify WHY a test failed to parse (no guessing — the real reason).
 * Returns 0 on success, -1 on error (errbuf valid then).
 */
int hd_build_mir_ex(const char *source, wubu_mir_prog_t *prog,
                    char *errbuf, size_t errcap) {
    if (errbuf && errcap) errbuf[0] = '\0';
    char *pp = wubu_preprocess(source);
    const char *effective = pp ? pp : source;

    HDLexer lex;
    hd_lex_init(&lex, effective);
    if (lex.has_error) {
        if (errbuf && errcap) snprintf(errbuf, errcap, "lex error: %s", lex.error);
        free(pp);
        return -1;
    }

    HDParser parse;
    hd_parse_init(&parse, &lex);

    HDASTNode *ast;
    if (*source == '{') {
        ast = hd_parse_stmt(&parse);
    } else {
        ast = hd_parse_expr(&parse);
        if (parse.has_error || (hd_parse_peek(&parse) != HD_TOK_EOF && hd_parse_peek(&parse) != HD_TOK_SEMI)) {
            hd_ast_free(ast);
            parse.has_error = false;
            parse.n_errors = 0;
            hd_lex_init(&lex, effective);
            hd_parse_init(&parse, &lex);
            const char *p = source;
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
            bool starts_with_keyword = (strncmp(p, "if ", 3) == 0 || strncmp(p, "while ", 6) == 0 ||
                strncmp(p, "for ", 4) == 0 || strncmp(p, "do ", 3) == 0 ||
                strncmp(p, "return", 6) == 0 || strncmp(p, "break", 5) == 0 ||
                strncmp(p, "continue", 8) == 0 || *p == '{');
            bool has_semicolon = false;
            p = source;
            while (*p) { if (*p == ';') { has_semicolon = true; break; } p++; }
            if (has_semicolon && !starts_with_keyword) {
                size_t len = strlen(effective);
                char *wrapped = malloc(len + 5);
                sprintf(wrapped, "{ %s }", effective);
                hd_lex_init(&lex, wrapped);
                hd_parse_init(&parse, &lex);
                ast = hd_parse_block(&parse);
                free(wrapped);
            } else {
                ast = hd_parse_stmt(&parse);
            }
        }
    }

    if (parse.has_error || !ast) {
        if (errbuf && errcap && parse.n_errors > 0)
            snprintf(errbuf, errcap, "%s", parse.errors[0]);
        else if (errbuf && errcap)
            snprintf(errbuf, errcap, "parse error");
        hd_ast_free(ast);
        free(pp);
        return -1;
    }

    /* Delegate to the shared builder (which now emits full TUs with
     * function definitions + a `main` entry point). On failure we can't
     * cheaply recover the exact parse message here, but hd_build_mir
     * already validated the AST above, so a failure here is a lowering
     * issue; report honestly. */
    int rc = hd_build_mir(source, prog);
    free(pp);
    return rc;
}
