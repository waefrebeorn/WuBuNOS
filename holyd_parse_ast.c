/* holyd_parse_ast.c -- HolyD AST construction/utility helpers (self-contained).
 *
 * hd_ast_new / hd_ast_free / hd_ast_add_stmt / hd_ast_add_arg. Uses HDASTNode
 * (holyd_parse.h). Minimal includes.
 */

#include "holyd_parse_internal.h"
#include <stdio.h>

HDASTNode *hd_ast_new(HDASTKind kind) {
    HDASTNode *n = (HDASTNode *)calloc(1, sizeof(HDASTNode));
    if (!n) return NULL;
    n->kind = kind;
    return n;
}

void hd_ast_free(HDASTNode *node) {
    if (!node) return;
    hd_ast_free(node->child);
    hd_ast_free(node->left);
    hd_ast_free(node->right);
    hd_ast_free(node->cond);
    hd_ast_free(node->then_branch);
    hd_ast_free(node->else_branch);
    hd_ast_free(node->init);
    hd_ast_free(node->callee);
    hd_ast_free(node->body);
    hd_ast_free(node->init_expr);
    hd_ast_free(node->update);

    if (node->stmts) {
        for (int i = 0; i < node->n_stmts; i++)
            hd_ast_free(node->stmts[i]);
        free(node->stmts);
    }
    if (node->args) {
        for (int i = 0; i < node->n_args; i++)
            hd_ast_free(node->args[i]);
        free(node->args);
    }
    free(node);
}

void hd_ast_add_stmt(HDASTNode *block, HDASTNode *stmt) {
    if (!block || !stmt) return;
    if (block->n_stmts >= block->stmts_cap) {
        block->stmts_cap = block->stmts_cap ? block->stmts_cap * 2 : 8;
        block->stmts = (HDASTNode **)realloc(block->stmts, block->stmts_cap * sizeof(HDASTNode *));
    }
    block->stmts[block->n_stmts++] = stmt;
}

void hd_ast_add_arg(HDASTNode *call, HDASTNode *arg) {
    if (!call || !arg) return;
    if (call->n_args >= call->args_cap) {
        call->args_cap = call->args_cap ? call->args_cap * 2 : 4;
        call->args = (HDASTNode **)realloc(call->args, call->args_cap * sizeof(HDASTNode *));
    }
    call->args[call->n_args++] = arg;
}

/* -- AST print + type-size (moved from holyd_parse.c to consolidate AST utils) -- */

static const char *ast_kind_name(HDASTKind k) {
    switch (k) {
        case HD_AST_INT_LIT:    return "INT";
        case HD_AST_FLOAT_LIT:  return "FLOAT";
        case HD_AST_STRING_LIT: return "STRING";
        case HD_AST_IDENT:      return "IDENT";
        case HD_AST_NEG:        return "NEG";
        case HD_AST_NOT:        return "NOT";
        case HD_AST_ADD:        return "ADD";
        case HD_AST_SUB:        return "SUB";
        case HD_AST_MUL:        return "MUL";
        case HD_AST_DIV:        return "DIV";
        case HD_AST_MOD:        return "MOD";
        case HD_AST_EQ:         return "EQ";
        case HD_AST_NE:         return "NE";
        case HD_AST_LT:         return "LT";
        case HD_AST_LE:         return "LE";
        case HD_AST_GT:         return "GT";
        case HD_AST_GE:         return "GE";
        case HD_AST_AND:        return "AND";
        case HD_AST_OR:         return "OR";
        case HD_AST_ASSIGN:     return "ASSIGN";
        case HD_AST_IF:         return "IF";
        case HD_AST_WHILE:      return "WHILE";
        case HD_AST_FOR:        return "FOR";
        case HD_AST_RETURN:     return "RETURN";
        case HD_AST_BLOCK:      return "BLOCK";
        case HD_AST_EXPR_STMT:  return "EXPR_STMT";
        case HD_AST_VAR_DECL:   return "VAR_DECL";
        case HD_AST_FUNC_DECL:  return "FUNC_DECL";
        case HD_AST_FUNC_CALL:  return "FUNC_CALL";
        default:                return "?";
    }
}

void hd_ast_print(const HDASTNode *node, int indent) {
    if (!node) return;
    for (int i = 0; i < indent; i++) printf("  ");
    printf("%s", ast_kind_name(node->kind));
    if (node->kind == HD_AST_INT_LIT)    printf(" val=%lld", (long long)node->int_val);
    if (node->kind == HD_AST_FLOAT_LIT)  printf(" val=%g", node->float_val);
    if (node->kind == HD_AST_IDENT)      printf(" name='%s'", node->ident);
    if (node->kind == HD_AST_STRING_LIT) printf(" str=\"%s\"", node->str_val);
    printf("\n");
    if (node->child) hd_ast_print(node->child, indent + 1);
    if (node->left)  hd_ast_print(node->left, indent + 1);
    if (node->right) hd_ast_print(node->right, indent + 1);
    if (node->cond)        hd_ast_print(node->cond, indent + 1);
    if (node->then_branch) hd_ast_print(node->then_branch, indent + 1);
    if (node->else_branch) hd_ast_print(node->else_branch, indent + 1);
    if (node->callee)      hd_ast_print(node->callee, indent + 1);
    if (node->body)        hd_ast_print(node->body, indent + 1);
    if (node->init)        hd_ast_print(node->init, indent + 1);
    if (node->init_expr)   hd_ast_print(node->init_expr, indent + 1);
    if (node->update)      hd_ast_print(node->update, indent + 1);
    for (int i = 0; i < node->n_stmts; i++) hd_ast_print(node->stmts[i], indent + 1);
    for (int i = 0; i < node->n_args; i++)  hd_ast_print(node->args[i], indent + 1);
}

size_t hd_type_size(const HDType *t) {
    if (!t) return 8;
    switch (t->kind) {
        case HD_TYPE_VOID: return 0;
        case HD_TYPE_I8:   case HD_TYPE_U8:  return 1;
        case HD_TYPE_I16:  case HD_TYPE_U16: return 2;
        case HD_TYPE_I32:  case HD_TYPE_U32: case HD_TYPE_BOOL: return 4;
        case HD_TYPE_I64:  case HD_TYPE_U64: case HD_TYPE_F64:  return 8;
        case HD_TYPE_PTR:  return 8;
        case HD_TYPE_ARRAY: return t->base ? hd_type_size(t->base) * (size_t)t->array_size : 0;
        case HD_TYPE_STRUCT: return t->size > 0 ? t->size : 8;
        default: return 8;
    }
}
