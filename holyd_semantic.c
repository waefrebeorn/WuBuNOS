/* holyd_semantic.c -- narrow semantic pass: unresolved identifiers.
 *
 * The parser accepts every identifier; codegen then fails anonymously. This
 * pass walks the AST *after parsing* and reports an identifier that is not a
 * local, parameter, global, function or type name.
 *
 * Deliberately conservative:
 *   - scopes are flattened per function (a name declared anywhere in the body
 *     counts for the whole body), which can only MISS errors, never invent them;
 *   - only IDENT nodes in expression position are checked; a member name after
 *     '.'/'->' is carried on the MEMBER node, not as a free IDENT.
 */
#include <stdio.h>
#include <string.h>
#include "holyd_types.h"
#include "holyd_ast.h"
#include "holyd_parser.h"

#define SEM_MAX_NAMES 1024
#define SEM_MAX_LEN 64

typedef struct {
    char names[SEM_MAX_NAMES][SEM_MAX_LEN];
    int  n;
} SemNames;

static int sem_has(const SemNames *s, const char *name)
{
    for (int i = 0; i < s->n; i++)
        if (strcmp(s->names[i], name) == 0) return 1;
    return 0;
}

static void sem_add(SemNames *s, const char *name)
{
    if (!name || !name[0] || s->n >= SEM_MAX_NAMES) return;
    if (sem_has(s, name)) return;
    snprintf(s->names[s->n], SEM_MAX_LEN, "%s", name);
    s->n++;
}

typedef struct { char name[SEM_MAX_LEN]; int n_params; int variadic; } SemFn;
typedef struct { SemFn fns[256]; int n; } SemFns;

static void sem_add_fn(SemFns *t, const char *name, int n_params, int variadic)
{
    if (!name || !name[0] || t->n >= 256) return;
    for (int i = 0; i < t->n; i++) if (!strcmp(t->fns[i].name, name)) {
        t->fns[i].n_params = n_params; t->fns[i].variadic = variadic; return; }
    snprintf(t->fns[t->n].name, SEM_MAX_LEN, "%s", name);
    t->fns[t->n].n_params = n_params; t->fns[t->n].variadic = variadic; t->n++;
}

static SemFn *sem_find_fn(SemFns *t, const char *name)
{
    for (int i = 0; i < t->n; i++) if (!strcmp(t->fns[i].name, name)) return &t->fns[i];
    return NULL;
}

static void sem_collect_fns(HDASTNode *n, SemFns *t)
{
    if (!n) return;
    if (n->kind == HD_AST_FUNC_DECL && n->ident[0])
        sem_add_fn(t, n->ident, n->n_params, n->is_variadic);
    if (n->kind == HD_AST_BLOCK || n->kind == HD_AST_FUNC_DECL) {
        HDASTNode **list = n->stmts; int count = n->n_stmts;
        if (n->kind == HD_AST_FUNC_DECL && n->body) { list = n->body->stmts; count = n->body->n_stmts; }
        for (int i = 0; i < count; i++) sem_collect_fns(list[i], t);
    }
    sem_collect_fns(n->child, t);
    sem_collect_fns(n->left, t);
    sem_collect_fns(n->right, t);
    sem_collect_fns(n->body, t);
}

static void sem_collect_decls(HDASTNode *n, SemNames *out)
{
    if (!n) return;
    if (n->kind == HD_AST_VAR_DECL || n->kind == HD_AST_FUNC_DECL ||
        n->kind == HD_AST_STRUCT_DECL || n->kind == HD_AST_LABEL)
        sem_add(out, n->ident);
    if (n->kind == HD_AST_BLOCK || n->kind == HD_AST_FUNC_DECL) {
        HDASTNode **list = n->stmts;
        int count = n->n_stmts;
        if (n->kind == HD_AST_FUNC_DECL && n->body) { list = n->body->stmts; count = n->body->n_stmts; }
        for (int i = 0; i < count; i++) sem_collect_decls(list[i], out);
        if (n->kind == HD_AST_FUNC_DECL && n->body) sem_collect_decls(n->body, out);
    }
    if (n->kind == HD_AST_FOR) {
        /* `for (int i = 0; ...)` introduces i for the loop body. */
        sem_collect_decls(n->init_expr, out);
        sem_collect_decls(n->cond, out);
        sem_collect_decls(n->update, out);
        sem_collect_decls(n->body, out);
    }
    sem_collect_decls(n->child, out);
    sem_collect_decls(n->left, out);
    sem_collect_decls(n->right, out);
    sem_collect_decls(n->then_branch, out);
    sem_collect_decls(n->else_branch, out);
    sem_collect_decls(n->body, out);
    sem_collect_decls(n->init, out);
}

/* Check expression-position identifiers. `stmt_ctx` suppresses the declaration
 * name itself (a VAR_DECL's ident is not a use). */
static void sem_check_expr(HDParser *p, HDASTNode *n, const SemNames *scope,
                           const SemNames *globals, SemFns *fns);

static void sem_check_stmt(HDParser *p, HDASTNode *n, const SemNames *scope,
                           const SemNames *globals, SemFns *fns)
{
    if (!n) return;
    switch (n->kind) {
    case HD_AST_BLOCK:
        for (int i = 0; i < n->n_stmts; i++)
            sem_check_stmt(p, n->stmts[i], scope, globals, fns);
        return;
    case HD_AST_VAR_DECL:
        if (n->init) sem_check_expr(p, n->init, scope, globals, fns);
        return;
    case HD_AST_RETURN:
    case HD_AST_EXPR_STMT:
        if (n->child) sem_check_expr(p, n->child, scope, globals, fns);
        return;
    case HD_AST_IF:
        if (n->cond) sem_check_expr(p, n->cond, scope, globals, fns);
        sem_check_stmt(p, n->then_branch, scope, globals, fns);
        sem_check_stmt(p, n->else_branch, scope, globals, fns);
        return;
    case HD_AST_WHILE:
    case HD_AST_DO_WHILE:
        if (n->cond) sem_check_expr(p, n->cond, scope, globals, fns);
        sem_check_stmt(p, n->body, scope, globals, fns);
        return;
    case HD_AST_FOR:
        if (n->init_expr) sem_check_expr(p, n->init_expr, scope, globals, fns);
        if (n->cond) sem_check_expr(p, n->cond, scope, globals, fns);
        if (n->update) sem_check_expr(p, n->update, scope, globals, fns);
        sem_check_stmt(p, n->body, scope, globals, fns);
        return;
    default:
        return;
    }
}

static void sem_check_expr(HDParser *p, HDASTNode *n, const SemNames *scope,
                           const SemNames *globals, SemFns *fns)
{
    if (!n) return;
    if (n->kind == HD_AST_VAR_DECL) {   /* declaration, not a use */
        sem_check_expr(p, n->init, scope, globals, fns);
        return;
    }
    if (n->kind == HD_AST_IDENT && n->ident[0] == '_' && n->ident[1] == '_')
        return;                        /* __VA_ARGS__ and friends */
    if (n->kind == HD_AST_IDENT) {
        if (n->ident[0] && !sem_has(scope, n->ident) && !sem_has(globals, n->ident))
            parse_error_at(p, n->line, n->col, n->ident, "use of undeclared identifier");
        return;
    }
    if (n->kind == HD_AST_CALL || n->kind == HD_AST_FUNC_CALL) {
        const char *callee = n->callee && n->callee->kind == HD_AST_IDENT
                                 ? n->callee->ident : n->ident;
        SemFn *f = callee && callee[0] ? sem_find_fn(fns, callee) : NULL;
        if (f && !f->variadic && n->n_args != f->n_params)
            parse_error_at(p, n->line, n->col, callee, "wrong number of arguments");
    }
    /* A callee name is deliberately NOT checked here: calling a function that
     * was never declared is a different diagnostic, and K&R-era tests rely on
     * it (implicit library functions). */
    sem_check_expr(p, n->left, scope, globals, fns);
    sem_check_expr(p, n->right, scope, globals, fns);
    sem_check_expr(p, n->child, scope, globals, fns);
    sem_check_expr(p, n->cond, scope, globals, fns);
    sem_check_expr(p, n->then_branch, scope, globals, fns);
    sem_check_expr(p, n->else_branch, scope, globals, fns);
    for (int i = 0; i < n->n_args; i++) sem_check_expr(p, n->args[i], scope, globals, fns);
}

void hd_semantic_check(HDParser *p, HDASTNode *unit)
{
    if (!p || !unit) return;

    SemNames globals = {0};
    /* Names a hosted C program is allowed to use without a declaration in the
     * translation unit: the preprocessor strips #include, so these never appear
     * as declarations even though the source legitimately uses them. */
    static const char *const builtins[] = {
        "stdout", "stderr", "stdin", "EOF", "NULL", "HUGE_VAL", "HUGE_VALF",
        "EXIT_SUCCESS", "EXIT_FAILURE", "SEEK_SET", "SEEK_CUR", "SEEK_END",
        "errno", "size_t", "ssize_t", "FILE", "va_list", NULL
    };
    for (int bi = 0; builtins[bi]; bi++) sem_add(&globals, builtins[bi]);
    SemFns fns = {0};
    sem_collect_decls(unit, &globals);
    sem_collect_fns(unit, &fns);

    for (int i = 0; i < unit->n_stmts; i++) {
        HDASTNode *d = unit->stmts[i];
        if (!d || d->kind != HD_AST_FUNC_DECL || !d->body) continue;

        SemNames scope = {0};
        for (int pi = 0; pi < d->n_params; pi++) sem_add(&scope, d->param_names[pi]);
        sem_collect_decls(d->body, &scope);

        for (int si = 0; si < d->body->n_stmts; si++)
            sem_check_stmt(p, d->body->stmts[si], &scope, &globals, &fns);
    }
}
