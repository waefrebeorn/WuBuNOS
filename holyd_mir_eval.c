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
    int array_size;
    int is_struct;                  /* 1 if this variable is a struct instance */
    char struct_name[HD_MAX_IDENT_LEN]; /* struct type name */
} mir_var_t;

/* struct member offset table */
#define MAX_STRUCTS 64
#define MAX_MEMBERS 32
typedef struct {
    char name[HD_MAX_IDENT_LEN];
    char member_names[MAX_MEMBERS][HD_MAX_IDENT_LEN];
    int member_offsets[MAX_MEMBERS];
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
    /* function table for MIR_CALL (collected during TU generation) */
    const HDASTNode *func_ast[MIR_MAX_FUNCTIONS];
    int func_id_of[MIR_MAX_FUNCTIONS];     /* -1 until assigned */
    int n_funcs;
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

/* Is a comparison operand unsigned? Check the symbol table (var decl) or
 * the node's own type annotation (mirrors the golden JIT's expr_static_type). */
static int mir_operand_is_unsigned(HDMirGen *g, const HDASTNode *operand) {
    if (!operand) return 0;
    if (operand->kind == HD_AST_IDENT) {
        if (mir_is_unsigned_var(g, operand->ident)) return 1;
    }
    if (operand->type) {
        HDTypeKind k = operand->type->kind;
        if (k == HD_TYPE_U8 || k == HD_TYPE_U16 || k == HD_TYPE_U32 || k == HD_TYPE_U64)
            return 1;
    }
    return 0;
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
 * - INDEX  -> address_of(left) + index
 * - DEREF  -> the pointer's held value */
static wubu_vr_t mir_address_of(HDMirGen *g, const HDASTNode *n) {
    if (!n) return 0;
    if (n->kind == HD_AST_STRING_LIT)
        return mir_gen_expr(g, n);  /* returns the allocated address */
    if (n->kind == HD_AST_IDENT) {
        wubu_vr_t addr = mir_find_var_addr(g, n->ident);
        if (addr == 0) return 0;
        if (mir_var_is_array(g, n->ident))
            return addr;                       /* &a[0] == a's base */
        return wubu_mir_load(g->prog, addr);   /* a[0] is a pointer var: load it */
    }
    if (n->kind == HD_AST_DEREF)
        return mir_gen_expr(g, n->child);      /* *p: address is p's value */
    if (n->kind == HD_AST_INDEX) {
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx  = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
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
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
    }
    if (n->kind == HD_AST_DOT || n->kind == HD_AST_MEMBER) {
        /* s.a = val — compute address of struct member */
        if (n->left && n->left->kind == HD_AST_IDENT && n->ident[0]) {
            const char *varname = n->left->ident;
            wubu_vr_t base = mir_find_var_addr(g, varname);
            fprintf(stderr, "DOT lvalue: var=%s base=%d\n", varname, base);
            if (base == 0) return 0;
            const char *struct_type = mir_find_var_struct_name(g, varname);
            fprintf(stderr, "DOT lvalue: struct_type=%s\n", struct_type ? struct_type : "NULL");
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", n->ident);
            fprintf(stderr, "DOT lvalue: offset=%d\n", offset);
            if (offset < 0) return 0;
            return wubu_mir_binop(g->prog, MIR_ADD, base, wubu_mir_const(g->prog, (int64_t)offset));
        }
    }
    return 0;  /* not an lvalue we can take address of */
}

static wubu_vr_t mir_gen_expr(HDMirGen *g, const HDASTNode *n);

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
        for (uint32_t i = 0; i < n->n_stmts; i++)
            last = mir_gen_stmt(g, n->stmts[i]);
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
            if (n->type && n->type->kind == HD_TYPE_STRUCT) {
                fprintf(stderr, "STRUCT_DECL: %s n_members=%d\n", n->ident, n->type->n_members);
                for (int i = 0; i < n->type->n_members && s->n_members < MAX_MEMBERS; i++) {
                    fprintf(stderr, "  member[%d] name='%s' offset=%lld\n", i, n->type->members[i].name, (long long)n->type->members[i].offset);
                    strncpy(s->member_names[s->n_members], n->type->members[i].name, HD_MAX_IDENT_LEN - 1);
                    s->member_names[s->n_members][HD_MAX_IDENT_LEN - 1] = '\0';
                    s->member_offsets[s->n_members] = (int)n->type->members[i].offset;
                    s->n_members++;
                }
            }
            s->total_size = s->n_members;
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
            if (k == HD_TYPE_STRUCT) fprintf(stderr, "STRUCT_VAR_DECL: %s type=%s n_members=%d\n", n->ident, n->type->name, n->type->n_members);
            if (k == HD_TYPE_U8 || k == HD_TYPE_U16 || k == HD_TYPE_U32 || k == HD_TYPE_U64)
                is_uns = 1;
            /* array type carries an element count in n->type->array_size */
            if (k == HD_TYPE_ARRAY && n->type->array_size > 0) arr_size = (int)n->type->array_size;
            /* struct type: allocate memory for all members */
            if (k == HD_TYPE_STRUCT) {
                is_struct_var = 1;
                if (n->type->name[0])
                    strncpy(struct_type_name, n->type->name, HD_MAX_IDENT_LEN - 1);
                /* Look up struct size from registered structs */
                int struct_size = 0;
                for (int si = 0; si < g->n_structs; si++) {
                    if (strcmp(g->structs[si].name, struct_type_name) == 0) {
                        struct_size = g->structs[si].total_size;
                        break;
                    }
                }
                if (struct_size > 0) arr_size = struct_size;
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
                g->vars[i].is_struct = is_struct_var;
                if (is_struct_var)
                    strncpy(g->vars[i].struct_name, struct_type_name, HD_MAX_IDENT_LEN - 1);
                break;
            }
        if (n->init) {
            if (n->init->kind == HD_AST_BRACE_INIT) {
                /* array initializer list: store each element */
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
                            break;
                        }
                }
                for (uint32_t e = 0; e < n->init->n_args && e < (uint32_t)arr_size; e++) {
                    wubu_vr_t ev = mir_gen_expr(g, n->init->args[e]);
                    wubu_vr_t elem_addr = wubu_mir_binop(g->prog, MIR_ADD, addr,
                                                          wubu_mir_const(g->prog, (int64_t)e));
                    wubu_mir_store(g->prog, elem_addr, ev);
                }
            } else {
                wubu_vr_t val = mir_gen_expr(g, n->init);
                wubu_mir_store(g->prog, addr, val);
            }
        }
        return vr;
    }
    case HD_AST_EXPR_STMT:
        return mir_gen_expr(g, n->child);
    case HD_AST_RETURN: {
        wubu_vr_t val = n->child ? mir_gen_expr(g, n->child) : wubu_mir_const(g->prog, 0);
        if (g->fn_ret_label != 0) {
            /* early return: store the value and jump to the function epilogue */
            wubu_mir_mov_to(g->prog, g->fn_ret_vr, val);
            wubu_mir_jmp(g->prog, g->fn_ret_label);
            return g->fn_ret_vr;
        }
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
        if (addr == 0) return wubu_mir_const(g->prog, 0);
        /* In C, an array name used as a value decays to a pointer to its
         * first element — return the array's base address (not a[0]). */
        if (mir_var_is_array(g, n->ident))
            return addr;
        return wubu_mir_load(g->prog, addr);
    }
    case HD_AST_DOT:
    case HD_AST_MEMBER: {
        /* s.a — struct member access */
        if (n->left && n->left->kind == HD_AST_IDENT && n->ident[0]) {
            const char *var_name = n->left->ident;
            const char *member_name = n->ident;
            wubu_vr_t base = mir_find_var_addr(g, var_name);
            fprintf(stderr, "DOT expr: var=%s member=%s base=%d\n", var_name, member_name, base);
            if (base == 0) return wubu_mir_const(g->prog, 0);
            const char *struct_type = NULL;
            for (int i = 0; i < g->n_vars; i++) {
                if (strcmp(g->vars[i].name, var_name) == 0 && g->vars[i].is_struct) {
                    struct_type = g->vars[i].struct_name;
                    break;
                }
            }
            int offset = mir_struct_member_offset(g, struct_type ? struct_type : "", member_name);
            fprintf(stderr, "DOT expr: struct=%s offset=%d\n", struct_type ? struct_type : "NULL", offset);
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
        if (n->left && n->left->kind == HD_AST_MEMBER)
            fprintf(stderr, "ASSIGN to MEMBER: val=%d addr=%d\n", val, addr);
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
        /* left = left OP right — supports IDENT, INDEX (a[i]), DEREF (*p) */
        wubu_vr_t addr = mir_lvalue_addr(g, n->left);
        if (addr) {
            wubu_vr_t lhs = wubu_mir_load(g->prog, addr);
            wubu_vr_t rhs = mir_gen_expr(g, n->right);
            wubu_mir_op_t op = MIR_ADD;
            switch (n->kind) {
                case HD_AST_ADD_ASSIGN: op = MIR_ADD; break;
                case HD_AST_SUB_ASSIGN: op = MIR_SUB; break;
                case HD_AST_MUL_ASSIGN: op = MIR_MUL; break;
                case HD_AST_DIV_ASSIGN: op = MIR_DIV; break;
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
        /* Store as 32-bit float (softfloat uses f32 for MIR_FADD) */
        union { float f; int32_t i; } u;
        u.f = (float)n->float_val;
        return wubu_mir_const(g->prog, (int64_t)u.i);
    }
    case HD_AST_BOOL_LIT:
        return wubu_mir_const(g->prog, n->int_val ? 1 : 0);
    case HD_AST_ADD: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
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
        /* (type)expr — cast. For integer-to-integer casts, this is a no-op
         * (the value is already in the right bit pattern). For float↔int,
         * emit conversion. */
        wubu_vr_t val = mir_gen_expr(g, n->child);
        if (getenv("WUBU_DEBUG_MIR")) {
            fprintf(stderr, "[DEBUG_MIR] CAST: type=%d child_type=%d val=%d\n",
                    n->type ? (int)n->type->kind : -1,
                    (n->child && n->child->type) ? (int)n->child->type->kind : -1,
                    val);
        }
        if (!n->type) return val;
        bool to_f64 = (n->type->kind == HD_TYPE_F64);
        bool from_f64 = n->child && n->child->type && (n->child->type->kind == HD_TYPE_F64);
        if (to_f64 && !from_f64) {
            /* int → float */
            return wubu_mir_unop(g->prog, MIR_ITOF, val);
        } else if (!to_f64 && from_f64) {
            /* float → int */
            return wubu_mir_unop(g->prog, MIR_FTOI, val);
        }
        /* Integer-to-integer cast: no-op (same bit width) */
        return val;
    }
    case HD_AST_SIZEOF: {
        /* sizeof(type) or sizeof(expr) — emit the type size as a constant. */
        int size = 8; /* default: pointer */
        if (n->type) {
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
            case HD_TYPE_ARRAY:  size = (int)(n->type->size > 0 ? n->type->size : 4); break;
            case HD_TYPE_STRUCT: size = (int)(n->type->size > 0 ? n->type->size : 4); break;
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
        /* a[i] -> load from address_of(a) + i. array name decays to base;
         * pointer var loads its held value. */
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx = mir_gen_expr(g, n->right);
        wubu_vr_t addr = wubu_mir_binop(g->prog, MIR_ADD, base, idx);
        return wubu_mir_load(g->prog, addr);
    }
    case HD_AST_ADDR: {
        /* &x -> the memory base address of x */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) return addr;
        }
        return wubu_mir_const(g->prog, 0);
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
        /* Place arguments in v1..vN (calling convention). */
        for (uint32_t a = 0; a < n->n_args && a < MIR_MAX_CALL_ARGS; a++) {
            wubu_vr_t av = mir_gen_expr(g, n->args[a]);
            if (av != (wubu_vr_t)(a + 1)) {
                wubu_mir_mov_to(g->prog, a + 1, av);
            }
        }
        wubu_mir_call(g->prog, (uint32_t)(fid >= 0 ? fid : 0));
        /* Callee returns in vr0; capture it into a fresh vr for the caller. */
        wubu_vr_t rv = mir_new_vr(g);
        wubu_mir_mov_to(g->prog, rv, 0);
        return rv;
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

    /* Register struct types from the parser's named_types table */
    for (int i = 0; i < parse.n_named_types && g.n_structs < MAX_STRUCTS; i++) {
        HDType *t = parse.named_types[i];
        if (t && t->kind == HD_TYPE_STRUCT && t->n_members > 0) {
            mir_struct_t *s = &g.structs[g.n_structs++];
            strncpy(s->name, parse.named_type_names[i], HD_MAX_IDENT_LEN - 1);
            s->name[HD_MAX_IDENT_LEN - 1] = '\0';
            s->n_members = 0;
            s->total_size = 0;
            for (int j = 0; j < t->n_members && s->n_members < MAX_MEMBERS; j++) {
                strncpy(s->member_names[s->n_members], t->members[j].name, HD_MAX_IDENT_LEN - 1);
                s->member_names[s->n_members][HD_MAX_IDENT_LEN - 1] = '\0';
                s->member_offsets[s->n_members] = (int)t->members[j].offset;
                s->n_members++;
            }
            s->total_size = s->n_members;
        }
    }

    /* Phase 1: collect top-level function definitions into the func table
     * (assign stable ids so CALL sites resolve regardless of order). */
    mir_collect_funcs(&g, ast);

    /* Phase 2: generate top-level (module) statements, skipping function
     * definitions (their bodies are emitted separately in Phase 3).
     * Track the last expression result so the top-level RETURN carries it. */
    wubu_vr_t top_val = 0;
    if (ast->kind == HD_AST_BLOCK) {
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
        fprintf(stderr, "FUNC_BODY[%d]: %s body_kind=%d n_params=%d body_n_stmts=%d\n", fi, fn->ident, fn->body ? fn->body->kind : -1, fn->n_params, fn->body && fn->body->kind == HD_AST_BLOCK ? (int)fn->body->n_stmts : 0);
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
            /* Allocate memory for this parameter and get its address
             * as a high VR (same pattern as global/local var allocation). */
            wubu_vr_t addr = mir_new_vr(&g);  /* high VR for the address */
            int64_t mem_addr = (int64_t)(g.prog->total_mem + 1);
            g.prog->total_mem = mem_addr + 1 - 1;
            wubu_mir_const_to(g.prog, addr, mem_addr);  /* addr VR = memory address */
            /* Copy the argument (in v1..vN) to the parameter's memory slot */
            wubu_mir_store(prog, addr, (wubu_vr_t)(pi + 1));
            mir_bind_var(&g, fn->param_names[pi], addr, addr, 0);
        }
        /* Set up early-return: RETURN emits `result_vr = expr; jmp ret_label`,
         * and the epilogue (placed after the body) moves result_vr into vr0
         * and returns. This makes `if(c) return x; return y;` correct. */
        g.fn_ret_vr = mir_new_vr(&g);
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
    if (getenv("WUBU_DEBUG_MIR")) {
        fprintf(stderr, "[DEBUG_MIR] Before optimization:\n");
        for (size_t i = 0; i < prog->n; i++) {
            const wubu_mir_instr_t *in = &prog->ins[i];
            fprintf(stderr, "  [%zu] op=%d dst=%d a=%d b=%d imm=%lld\n",
                    i, in->op, in->dst, in->a, in->b, (long long)in->imm);
        }
    }

    /* Optimize the canonical MIR once (benefits ALL backends). Side-effect
     * safe passes only; the optimizer preserves semantics, so the differential
     * battery (every driver agrees with the portable interp oracle) proves
     * cross-target correctness. */
    wubu_mir_optimize(prog,
                      MIR_OPT_FOLD | MIR_OPT_STRENGTH | MIR_OPT_DCE |
                      MIR_OPT_COMBINE | MIR_OPT_CSE);

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
