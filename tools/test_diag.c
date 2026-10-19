/* test_diag.c -- diagnostics harness for the HolyD parser.
 *
 * The battery proves evaluator behaviour; this proves what the compiler
 * REPORTS. HDParser already carries errors[][], n_errors and has_error, so the
 * goal criteria that name diagnostics (125, 131, 184) can be asserted directly
 * instead of being assumed.
 */
#include <stdio.h>
#include <string.h>
#include "holyd_types.h"
#include "holyd_lexer.h"
#include "holyd_parser.h"

static int fails;

#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", #cond, __LINE__); fails++; } \
} while (0)

static int count_errors(const char *src, char lines_out[][256], int max_out)
{
    HDLexer lex;
    hd_lex_init(&lex, src);
    HDParser p;
    hd_parse_init(&p, &lex);
    (void)hd_parse_compilation_unit(&p);
    int n = p.n_errors;
    for (int i = 0; i < n && i < max_out; i++)
        snprintf(lines_out[i], 256, "%s", p.errors[i]);
    return n;
}

int main(void)
{
    char errs[16][256];

    /* Goal 184: three independent syntax errors must produce three diagnostics,
     * not one fatal stop. */
    int n = count_errors("int a = ;\nint b = ;\nint c = ;\n", errs, 16);
    printf("[184] %d diagnostics for 3 bad initialisers\n", n);
    for (int i = 0; i < n && i < 4; i++) printf("      %s\n", errs[i]);
    CHECK(n >= 3);

    /* Goal 125: an undeclared identifier must be reported before codegen. */
    n = count_errors("int main(void){ return nope; }\n", errs, 16);
    printf("[125] %d diagnostics for an undeclared identifier\n", n);
    for (int i = 0; i < n && i < 3; i++) printf("      %s\n", errs[i]);
    CHECK(n >= 1);   /* currently red: no semantic pass exists yet */

    /* Goal 131: wrong arity at a call site. */
    n = count_errors("void f(int a, int b);\nint main(void){ return f(1); }\n", errs, 16);
    printf("[131] %d diagnostics for a 1-arg call to a 2-param prototype\n", n);
    for (int i = 0; i < n && i < 3; i++) printf("      %s\n", errs[i]);
    CHECK(n >= 1);   /* currently red: no arity checking exists yet */

    printf("\n%s: %d failure(s)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
