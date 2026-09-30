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
        case HD_TYPE_STRUCT: {
            /* TRUE C byte size -- this is what `sizeof` must report, so
             * `sizeof(struct S{int a;int b;})` == 8, not 16.
             *
             * Do NOT use t->size here: the parser advances t->size in CELLS
             * (one cell per member) to match the member offsets codegen
             * addresses as `offset * 8`. Allocation needs that inflated
             * figure -- see hd_type_alloc_size() -- but sizeof needs the
             * real C size. Mixing them up is what made
             * `struct S s; s.b=20; int* p=&s.b; *p;` read garbage (allocated
             * 8 bytes but addressed members[1] at byte 8).
             */
            if (t->n_members > 0) {
                int64_t total_bytes = 0, max_align = 1;
                for (int i = 0; i < t->n_members; i++) {
                    size_t msz = hd_type_size(t->members[i].type);
                    total_bytes += (int64_t)msz;
                    if ((int64_t)msz > max_align) max_align = (int64_t)msz;
                }
                if (max_align > 0 && total_bytes % max_align != 0)
                    total_bytes += max_align - (total_bytes % max_align);
                return (size_t)(total_bytes > 0 ? total_bytes : 8);
            }
            return t->size > 0 ? (size_t)(t->size * 8) : 8;
        }
        default: return 8;
    }
}

/* Bytes to RESERVE when storing a value of this type in the data section or
 * on the stack.
 *
 * This differs from hd_type_size() for structs and unions on purpose. The
 * parser lays a struct out in CELLS -- t->size grows by
 * ceil(hd_type_size(member)/8) per member and members[].offset records that
 * running cell count -- and codegen addresses a member as `offset * 8`. So a
 * struct must be reserved at `t->size * 8` bytes even though its C size is
 * smaller, or the last members land past the end and alias whatever was
 * allocated next.
 *
 * hd_type_size() stays the true C size because `sizeof` must report that.
 * Use this function for allocation/reservation, never for sizeof.
 */
size_t hd_type_alloc_size(const HDType *t) {
    if (!t) return 8;
    if ((t->kind == HD_TYPE_STRUCT || t->kind == HD_TYPE_UNION) &&
        t->n_members > 0 && t->size > 0)
        return (size_t)t->size * 8;
    size_t sz = hd_type_size(t);
    return sz > 0 ? sz : 8;
}
