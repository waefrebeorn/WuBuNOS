/*
 * holyd_ast.h  --  HolyD AST Node Definitions and Utilities
 * Self-contained, C11, minimal includes.
 */
#ifndef WUBUNOS_HOLYC_AST_H
#define WUBUNOS_HOLYC_AST_H

#include "holyd_types.h"

/* AST node definition (opaque in types, concrete here) */
struct HDASTNode {
    HDASTKind kind;
    HDASTNode *child;          /* For unary/ternary/return/expr_stmt */
    HDASTNode *left;           /* For binary */
    HDASTNode *right;          /* For binary */
    HDASTNode *cond;           /* For if/while/for/ternary */
    HDASTNode *then_branch;    /* For if */
    HDASTNode *else_branch;    /* For if */
    HDASTNode *init_expr;      /* For for */
    HDASTNode *update;         /* For for */
    HDASTNode *body;           /* For while/for/do_while/func_decl/block */
    HDASTNode **stmts;         /* For block */
    int n_stmts;
    int stmts_cap;
    int args_cap;

    /* For var_decl / func_decl */
    char ident[HD_MAX_IDENT_LEN];
    HDType *ast_type;
    HDASTNode *init;           /* For var_decl */
    HDASTNode **params;        /* For func_decl */
    char param_names[HD_MAX_PARAMS][HD_MAX_IDENT_LEN];
    HDType *param_types[HD_MAX_PARAMS];
    int n_params;

    /* For extern_decl */
    char extern_c_name[HD_MAX_IDENT_LEN];
    HDType *extern_ret_type;
    HDType *extern_param_types[HD_MAX_PARAMS];
    int extern_n_params;

    /* For call */
    HDASTNode **args;
    int n_args;

    /* For string/char literals */
    char str_val[HD_MAX_STRING_LEN];

    /* For int/float literals */
    int64_t int_val;
    double float_val;

    /* Type info for codegen */
    HDType *type;

    /* Function call target */
    void *func_ptr;
    HDASTNode *callee;

    /* Scope control: if set, BLOCK does not pop scope on exit */
    int no_scope_pop;
};

/* AST utilities */
HDASTNode *hd_ast_new(HDASTKind kind);
void hd_ast_free(HDASTNode *node);
void hd_ast_add_stmt(HDASTNode *block, HDASTNode *stmt);
void hd_ast_add_arg(HDASTNode *call, HDASTNode *arg);
void hd_ast_add_param(HDASTNode *func, HDASTNode *param, const char *name, HDType *type);
void hd_ast_print(const HDASTNode *node, int indent);

#endif /* WUBUNOS_HOLYC_AST_H */