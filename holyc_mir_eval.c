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
#include <stdlib.h>
#include <string.h>

/* MIR generation state */
typedef struct {
    wubu_mir_prog_t *prog;
    wubu_vr_t next_vr;      /* next virtual register index */
    int has_error;
} HCMirGen;

static wubu_vr_t mir_new_vr(HCMirGen *g) {
    return g->next_vr++;
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
        if (n->init)
            return mir_gen_expr(g, n->init);
        return mir_new_vr(g);  /* uninitialized: emit 0 */
    }
    case HC_AST_EXPR_STMT:
        return mir_gen_expr(g, n->child);
    case HC_AST_RETURN:
        return mir_gen_expr(g, n->child);
    case HC_AST_IF: {
        /* if (cond) then_branch else else_branch
         * MIR: [cond] jz else [then] jmp end: ... [else]: ... [end]: */
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        uint32_t else_label = wubu_mir_new_label(g->prog);
        uint32_t end_label = wubu_mir_new_label(g->prog);
        wubu_mir_jz(g->prog, cond, else_label);
        wubu_vr_t then_val = mir_gen_stmt(g, n->then_branch);
        wubu_mir_jmp(g->prog, end_label);
        wubu_mir_place_label(g->prog, else_label);
        wubu_vr_t else_val = n->else_branch ? mir_gen_stmt(g, n->else_branch) : 0;
        wubu_mir_place_label(g->prog, end_label);
        /* Return whichever branch value (simplified: just return then_val) */
        (void)else_val;
        return then_val;
    }
    case HC_AST_WHILE: {
        uint32_t top = wubu_mir_new_label(g->prog);
        uint32_t done = wubu_mir_new_label(g->prog);
        wubu_mir_place_label(g->prog, top);
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        wubu_mir_jz(g->prog, cond, done);
        mir_gen_stmt(g, n->body);
        wubu_mir_jmp(g->prog, top);
        wubu_mir_place_label(g->prog, done);
        return 0;
    }
    case HC_AST_FOR: {
        /* for (init; cond; update) body */
        if (n->init) mir_gen_stmt(g, n->init);
        uint32_t top = wubu_mir_new_label(g->prog);
        uint32_t done = wubu_mir_new_label(g->prog);
        wubu_mir_place_label(g->prog, top);
        if (n->cond) {
            wubu_vr_t cond = mir_gen_expr(g, n->cond);
            wubu_mir_jz(g->prog, cond, done);
        }
        mir_gen_stmt(g, n->body);
        if (n->update) mir_gen_expr(g, n->update);
        wubu_mir_jmp(g->prog, top);
        wubu_mir_place_label(g->prog, done);
        return 0;
    }
    default:
        return mir_gen_expr(g, n);
    }
}

static wubu_vr_t mir_gen_expr(HCMirGen *g, const HCASTNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case HC_AST_INT_LIT:
        return wubu_mir_const(g->prog, n->int_val);
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
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_AND, a, b);
    }
    case HC_AST_BITAND: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_AND, a, b);
    }
    case HC_AST_OR: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_OR, a, b);
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
        return wubu_mir_binop(g->prog, MIR_LT, a, b);
    }
    case HC_AST_LE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_LE, a, b);
    }
    case HC_AST_GT: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_GT, a, b);
    }
    case HC_AST_GE: {
        wubu_vr_t a = mir_gen_expr(g, n->left);
        wubu_vr_t b = mir_gen_expr(g, n->right);
        return wubu_mir_binop(g->prog, MIR_GE, a, b);
    }
    case HC_AST_TERNARY: {
        /* cond ? then : else */
        wubu_vr_t cond = mir_gen_expr(g, n->cond);
        uint32_t else_label = wubu_mir_new_label(g->prog);
        uint32_t end_label = wubu_mir_new_label(g->prog);
        wubu_mir_jz(g->prog, cond, else_label);
        wubu_vr_t then_val = mir_gen_expr(g, n->then_branch);
        wubu_mir_jmp(g->prog, end_label);
        wubu_mir_place_label(g->prog, else_label);
        wubu_vr_t else_val = mir_gen_expr(g, n->else_branch);
        wubu_mir_place_label(g->prog, end_label);
        (void)else_val;
        return then_val;
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
    /* Parse */
    HCLexer lex;
    hc_lex_init(&lex, source);
    if (lex.has_error) return 0;

    HCParser parse;
    hc_parse_init(&parse, &lex);
    HCASTNode *ast = hc_parse_compilation_unit(&parse);
    if (parse.has_error || !ast) {
        hc_ast_free(ast);
        return 0;
    }

    /* Emit MIR */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    HCMirGen g;
    g.prog = &prog;
    g.next_vr = 1;  /* vr 0 = return value */
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
        return 0;
    }

    /* Compile via driver */
    uint8_t *code = NULL;
    size_t code_size = 0;
    int rc = driver->compile(&prog, &code, &code_size);
    wubu_mir_free(&prog);
    if (rc != 0 || !code) return 0;

    /* Run via driver */
    int64_t result = driver->run(code, code_size, 0);
    free(code);
    return result;
}
