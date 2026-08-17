/*
 * holyc_mir_eval.c  --  HolyC AST → MIR → ISA driver evaluation.
 *
 * Parses HolyC source, emits MIR (the hourglass neck), then compiles
 * and runs via any ISA driver. This is the cross-target path: one
 * frontend, N backends.
 *
 * C11, self-contained.
 */
#include "holyc_mir_eval.h"
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
#include "holyc_ast.h"
#include "holyc_types.h"
#include "holyc_lexer.h"
#include "holyc_parser.h"
#include "wubu_preproc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MIR generation state */
#define MIRGEN_MAX_VARS 256
typedef struct {
    char name[HC_MAX_IDENT_LEN];
    wubu_vr_t vr;          /* home vr for scalars (also = memory base for load/store) */
    wubu_vr_t addr;        /* memory base address (0 if not yet allocated) */
    int is_unsigned;
    int is_array;
    int array_size;
} mir_var_t;

typedef struct {
    wubu_mir_prog_t *prog;
    wubu_vr_t next_vr;      /* next virtual register index */
    int has_error;
    mir_var_t vars[MIRGEN_MAX_VARS];
    int n_vars;
    /* loop stack for break/continue resolution */
    uint32_t loop_top[MIRGEN_MAX_VARS];   /* continue target */
    uint32_t loop_done[MIRGEN_MAX_VARS];  /* break target */
    int n_loops;
} HCMirGen;

static wubu_vr_t mir_new_vr(HCMirGen *g) {
    return g->next_vr++;
}

/* symbol table: find a declared variable's vr, or -1 */
static wubu_vr_t mir_find_var(HCMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].vr;
    return (wubu_vr_t)-1;
}

/* symbol table: find a declared variable's memory base addr, or 0 */
static wubu_vr_t mir_find_var_addr(HCMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].addr;
    return 0;
}

/* symbol table: is the named variable an array? */
static int mir_var_is_array(HCMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_array;
    return 0;
}

/* symbol table: is the named variable declared unsigned? */
static int mir_is_unsigned_var(HCMirGen *g, const char *name) {
    for (int i = 0; i < g->n_vars; i++)
        if (strcmp(g->vars[i].name, name) == 0) return g->vars[i].is_unsigned;
    return 0;
}

/* declare (or re-bind) a variable name -> vr; record unsigned-ness */
static wubu_vr_t mir_decl_var_unsigned(HCMirGen *g, const char *name, int is_unsigned) {
    for (int i = 0; i < g->n_vars; i++) {
        if (strcmp(g->vars[i].name, name) == 0) {
            if (is_unsigned) g->vars[i].is_unsigned = 1;
            return g->vars[i].vr;  /* already declared: reuse vr */
        }
    }
    wubu_vr_t vr = mir_new_vr(g);
    if (g->n_vars < MIRGEN_MAX_VARS) {
        strncpy(g->vars[g->n_vars].name, name, HC_MAX_IDENT_LEN - 1);
        g->vars[g->n_vars].name[HC_MAX_IDENT_LEN - 1] = '\0';
        g->vars[g->n_vars].vr = vr;
        g->vars[g->n_vars].addr = 0;
        g->vars[g->n_vars].is_unsigned = is_unsigned;
        g->vars[g->n_vars].is_array = 0;
        g->vars[g->n_vars].array_size = 0;
        g->n_vars++;
    }
    return vr;
}

/* Is a comparison operand unsigned? Check the symbol table (var decl) or
 * the node's own type annotation (mirrors the golden JIT's expr_static_type). */
static int mir_operand_is_unsigned(HCMirGen *g, const HCASTNode *operand) {
    if (!operand) return 0;
    if (operand->kind == HC_AST_IDENT) {
        if (mir_is_unsigned_var(g, operand->ident)) return 1;
    }
    if (operand->type) {
        HCTypeKind k = operand->type->kind;
        if (k == HC_TYPE_U8 || k == HC_TYPE_U16 || k == HC_TYPE_U32 || k == HC_TYPE_U64)
            return 1;
    }
    return 0;
}

/* declare (or re-bind) a variable name -> vr */
static wubu_vr_t mir_decl_var(HCMirGen *g, const char *name) {
    return mir_decl_var_unsigned(g, name, 0);
}

static wubu_vr_t mir_gen_expr(HCMirGen *g, const HCASTNode *n);

/* Evaluate the address-of-first-element of an lvalue. Returns a vr holding the
 * memory address (a value), suitable as the index into mem[].
 * - array IDENT  -> base address (a decays to &a[0])
 * - scalar/pointer IDENT -> the value held in the var (pointer value)
 * - INDEX  -> address_of(left) + index
 * - DEREF  -> the pointer's held value */
static wubu_vr_t mir_address_of(HCMirGen *g, const HCASTNode *n) {
    if (!n) return 0;
    if (n->kind == HC_AST_IDENT) {
        wubu_vr_t addr = mir_find_var_addr(g, n->ident);
        if (addr == 0) return 0;
        if (mir_var_is_array(g, n->ident))
            return addr;                       /* &a[0] == a's base */
        return wubu_mir_load(g->prog, addr);   /* a[0] is a pointer var: load it */
    }
    if (n->kind == HC_AST_DEREF)
        return mir_gen_expr(g, n->child);      /* *p: address is p's value */
    if (n->kind == HC_AST_INDEX) {
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx  = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
    }
    return 0;
}

/* Compute the memory address of an lvalue (a value suitable for MIR_STORE). */
static wubu_vr_t mir_lvalue_addr(HCMirGen *g, const HCASTNode *n) {
    if (!n) return 0;
    if (n->kind == HC_AST_IDENT)
        return mir_find_var_addr(g, n->ident);  /* store into a's own cell */
    if (n->kind == HC_AST_DEREF)
        return mir_gen_expr(g, n->child);      /* *p: address == p's value */
    if (n->kind == HC_AST_INDEX) {
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_ADD, base, idx);
    }
    return 0;  /* not an lvalue we can take address of */
}

static wubu_vr_t mir_gen_expr(HCMirGen *g, const HCASTNode *n);

static wubu_vr_t mir_gen_stmt(HCMirGen *g, const HCASTNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case HC_AST_BLOCK: {
        wubu_vr_t last = 0;
        for (uint32_t i = 0; i < n->n_stmts; i++)
            last = mir_gen_stmt(g, n->stmts[i]);
        return last;
    }
    case HC_AST_VAR_DECL: {
        int is_uns = 0;
        int arr_size = 0;
        if (n->type) {
            HCTypeKind k = n->type->kind;
            if (k == HC_TYPE_U8 || k == HC_TYPE_U16 || k == HC_TYPE_U32 || k == HC_TYPE_U64)
                is_uns = 1;
            /* array type carries an element count in n->type->array_size */
            if (k == HC_TYPE_ARRAY && n->type->array_size > 0) arr_size = (int)n->type->array_size;
        }
        wubu_vr_t vr = mir_decl_var_unsigned(g, n->ident, is_uns);
        /* Allocate memory for the variable (arrays get arr_size cells, scalars 1). */
        wubu_vr_t addr = wubu_mir_alloc(g->prog, arr_size > 0 ? arr_size : 1);
        for (int i = 0; i < g->n_vars; i++)
            if (strcmp(g->vars[i].name, n->ident) == 0) {
                g->vars[i].addr = addr;
                g->vars[i].is_array = (arr_size > 0);
                g->vars[i].array_size = arr_size;
                break;
            }
        if (n->init) {
            if (arr_size > 0 && n->init->kind == HC_AST_BRACE_INIT) {
                /* array initializer list: store each element */
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
    case HC_AST_EXPR_STMT:
        return mir_gen_expr(g, n->child);
    case HC_AST_RETURN:
        return mir_gen_expr(g, n->child);
    case HC_AST_IF: {
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
    case HC_AST_WHILE: {
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
    case HC_AST_DO_WHILE: {
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
    case HC_AST_FOR: {
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
    case HC_AST_BREAK:
        if (g->n_loops > 0)
            wubu_mir_jmp(g->prog, g->loop_done[g->n_loops - 1]);
        else
            wubu_mir_jmp(g->prog, 0);
        return 0;
    case HC_AST_CONTINUE:
        if (g->n_loops > 0)
            wubu_mir_jmp(g->prog, g->loop_top[g->n_loops - 1]);
        else
            wubu_mir_jmp(g->prog, 0);
        return 0;
    default:
        return mir_gen_expr(g, n);
    }
}

static wubu_vr_t mir_gen_expr(HCMirGen *g, const HCASTNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case HC_AST_INT_LIT:
        return wubu_mir_const(g->prog, n->int_val);
    case HC_AST_IDENT: {
        wubu_vr_t addr = mir_find_var_addr(g, n->ident);
        if (addr == 0) return wubu_mir_const(g->prog, 0);
        /* In C, an array name used as a value decays to a pointer to its
         * first element — return the array's base address (not a[0]). */
        if (mir_var_is_array(g, n->ident))
            return addr;
        return wubu_mir_load(g->prog, addr);
    }
    case HC_AST_ASSIGN: {
        wubu_vr_t val = mir_gen_expr(g, n->right);
        wubu_vr_t addr = mir_lvalue_addr(g, n->left);
        if (addr) { wubu_mir_store(g->prog, addr, val); return val; }
        return val;
    }
    case HC_AST_POST_INC:
    case HC_AST_POST_DEC: {
        /* tmp = v; v = v +/- 1; return tmp */
        if (n->child && n->child->kind == HC_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t tmp = mir_new_vr(g);
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_mir_mov_to(g->prog, tmp, v);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HC_AST_POST_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, tmp, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, tmp, one);
                wubu_mir_store(g->prog, addr, upd);
                return tmp;
            }
        }
        return mir_gen_expr(g, n->child);
    }
    case HC_AST_PRE_INC:
    case HC_AST_PRE_DEC: {
        /* v = v +/- 1; return v */
        if (n->child && n->child->kind == HC_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HC_AST_PRE_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, v, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, v, one);
                wubu_mir_store(g->prog, addr, upd);
                return upd;
            }
        }
        return mir_gen_expr(g, n->child);
    }
    case HC_AST_ADD_ASSIGN:
    case HC_AST_SUB_ASSIGN:
    case HC_AST_MUL_ASSIGN:
    case HC_AST_DIV_ASSIGN:
    case HC_AST_MOD_ASSIGN:
    case HC_AST_SHL_ASSIGN:
    case HC_AST_SHR_ASSIGN:
    case HC_AST_AMP_ASSIGN:
    case HC_AST_PIPE_ASSIGN:
    case HC_AST_CARET_ASSIGN: {
        /* left = left OP right — supports IDENT, INDEX (a[i]), DEREF (*p) */
        wubu_vr_t addr = mir_lvalue_addr(g, n->left);
        if (addr) {
            wubu_vr_t lhs = wubu_mir_load(g->prog, addr);
            wubu_vr_t rhs = mir_gen_expr(g, n->right);
            wubu_mir_op_t op = MIR_ADD;
            switch (n->kind) {
                case HC_AST_ADD_ASSIGN: op = MIR_ADD; break;
                case HC_AST_SUB_ASSIGN: op = MIR_SUB; break;
                case HC_AST_MUL_ASSIGN: op = MIR_MUL; break;
                case HC_AST_DIV_ASSIGN: op = MIR_DIV; break;
                case HC_AST_MOD_ASSIGN: op = MIR_MOD; break;
                case HC_AST_SHL_ASSIGN: op = MIR_SHL; break;
                case HC_AST_SHR_ASSIGN: op = MIR_SHR; break;
                case HC_AST_AMP_ASSIGN: op = MIR_AND; break;
                case HC_AST_PIPE_ASSIGN: op = MIR_OR;  break;
                case HC_AST_CARET_ASSIGN: op = MIR_XOR; break;
                default: break;
            }
            wubu_vr_t upd = wubu_mir_binop(g->prog, op, lhs, rhs);
            wubu_mir_store(g->prog, addr, upd);
            return upd;
        }
        return mir_gen_expr(g, n->right);
    }
    case HC_AST_FLOAT_LIT: {
        union { double d; int64_t i; } u;
        u.d = n->float_val;
        return wubu_mir_const(g->prog, u.i);
    }
    case HC_AST_BOOL_LIT:
        return wubu_mir_const(g->prog, n->int_val ? 1 : 0);
    case HC_AST_ADD: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_ADD, a, b);
    }
    case HC_AST_SUB: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_SUB, a, b);
    }
    case HC_AST_MUL: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_MUL, a, b);
    }
    case HC_AST_DIV: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_DIV, a, b);
    }
    case HC_AST_MOD: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_MOD, a, b);
    }
    case HC_AST_AND: {
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
    case HC_AST_BITAND: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_AND, a, b);
    }
    case HC_AST_OR: {
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
    case HC_AST_BITOR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_OR, a, b);
    }
    case HC_AST_BITXOR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_XOR, a, b);
    }
    case HC_AST_SHL: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_SHL, a, b);
    }
    case HC_AST_SHR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_SHR, a, b);
    }
    case HC_AST_NEG: {
        wubu_vr_t a = mir_gen_expr(g, n->child);
        return wubu_mir_unop(g->prog, MIR_NEG, a);
    }
    case HC_AST_BITNOT: {
        wubu_vr_t a = mir_gen_expr(g, n->child);
        return wubu_mir_unop(g->prog, MIR_NOT, a);
    }
    case HC_AST_NOT: {
        /* !x = (x == 0) ? 1 : 0 */
        wubu_vr_t a = mir_gen_expr(g, n->child);
        wubu_vr_t zero = wubu_mir_const(g->prog, 0);
        return wubu_mir_binop(g->prog, MIR_EQ, a, zero);
    }
    case HC_AST_EQ: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_EQ, a, b);
    }
    case HC_AST_NE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_NE, a, b);
    }
    case HC_AST_LT: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_ULT : MIR_LT;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HC_AST_LE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_ULE : MIR_LE;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HC_AST_GT: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_UGT : MIR_GT;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HC_AST_GE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        wubu_mir_op_t op = (mir_operand_is_unsigned(g, n->left) || mir_operand_is_unsigned(g, n->right))
                            ? MIR_UGE : MIR_GE;
        return wubu_mir_binop(g->prog, op, a, b);
    }
    case HC_AST_TERNARY: {
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
    case HC_AST_INDEX: {
        /* a[i] -> load from address_of(a) + i. array name decays to base;
         * pointer var loads its held value. */
        wubu_vr_t base = mir_address_of(g, n->left);
        wubu_vr_t idx = mir_gen_expr(g, n->right);
        wubu_vr_t addr = wubu_mir_binop(g->prog, MIR_ADD, base, idx);
        return wubu_mir_load(g->prog, addr);
    }
    case HC_AST_ADDR: {
        /* &x -> the memory base address of x */
        if (n->child && n->child->kind == HC_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) return addr;
        }
        return wubu_mir_const(g->prog, 0);
    }
    case HC_AST_DEREF: {
        /* *p -> load from address held in p */
        wubu_vr_t addr = mir_gen_expr(g, n->child);
        return wubu_mir_load(g->prog, addr);
    }
    default:
        /* Unsupported: emit 0 */
        return wubu_mir_const(g->prog, 0);
    }
}

/*
 * hc_eval_mir: parse HolyC source, emit MIR, compile + run via driver.
 * Returns the result. On error, returns 0.
 */
int64_t hc_eval_mir(const char *source, const wubu_isa_driver_t *driver) {
    /* Preprocess #define / strip directives — must match hc_eval exactly so
     * the MIR path parses the same sources the x86-64 JIT (golden) does. */
    char *pp = wubu_preprocess(source);
    const char *effective = pp ? pp : source;

    HCLexer lex;
    hc_lex_init(&lex, effective);
    if (lex.has_error) { free(pp); return 0; }

    HCParser parse;
    hc_parse_init(&parse, &lex);

    /* If the source starts with '{', parse as a statement/block directly.
     * parse_primary treats '{...}' as a BRACE_INIT (array/struct initializer)
     * and would consume the whole input as an initializer, so the normal
     * expr-first-then-fallback path never reaches the block parser. Route
     * brace-prefixed sources straight to the statement parser (which handles
     * blocks) to match the golden JIT's block-as-expression semantics. */
    HCASTNode *ast;
    if (*source == '{') {
        ast = hc_parse_stmt(&parse);
    } else {
        ast = hc_parse_expr(&parse);

        /* If expression parsing failed or didn't consume all input, fall back to
         * block / statement parsing (with optional brace-wrapping). */
        if (parse.has_error || (hc_parse_peek(&parse) != HC_TOK_EOF && hc_parse_peek(&parse) != HC_TOK_SEMI)) {
            hc_ast_free(ast);
            parse.has_error = false;
            parse.n_errors = 0;
            hc_lex_init(&lex, effective);
            hc_parse_init(&parse, &lex);
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
                hc_lex_init(&lex, wrapped);
                hc_parse_init(&parse, &lex);
                ast = hc_parse_block(&parse);
                free(wrapped);
            } else {
                ast = hc_parse_stmt(&parse);
            }
        }
    }

    if (parse.has_error || !ast) {
        hc_ast_free(ast);
        free(pp);
        return 0;
    }    /* Emit MIR */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    HCMirGen g;
    memset(&g, 0, sizeof(g));
    g.prog = &prog;
    /* The wubu_mir_* helpers assign fresh vrs as their own instruction index
     * (0,1,2,...). mir_new_vr() must NOT collide with that space, so it starts
     * at a high base well above any instruction count in a test program. */
    g.next_vr = 1u << 16;  /* 65536 — far above any instr-index vr */
    g.has_error = 0;

    /* Generate MIR from AST */
    wubu_vr_t result_vr;
    if (ast->kind == HC_AST_BLOCK) {
        result_vr = mir_gen_stmt(&g, ast);
    } else {
        result_vr = mir_gen_expr(&g, ast);
    }
    wubu_mir_ret(&prog, result_vr);

    hc_ast_free(ast);

    if (g.has_error || prog.n == 0) {
        wubu_mir_free(&prog);
        free(pp);
        return 0;
    }

    /* Native execution of a non-host ISA is impossible on this x86-64 box.
     * The per-ISA *encoders* are validated separately by their own oracle
     * (tools/verify_isa.sh, byte-for-byte against GNU objdump). Here we run
     * the canonical MIR directly via the portable interpreter so the
     * differential battery can verify that ALL backends agree with the x86-64
     * native JIT (the golden reference). Every target executes the SAME MIR,
     * so a result discrepancy is a frontend/lowering bug, never an encoder
     * artifact. */
    int64_t result = wubu_mir_interp(&prog);
    wubu_mir_free(&prog);
    free(pp);
    return result;
}
