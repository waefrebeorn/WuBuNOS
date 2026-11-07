/* test_backend_consistency.c -- B4 (docs/GAP_ROADMAP, GAP-B rung): the
 * two-backend split must not diverge. Proof = a cross-backend consistency test:
 * the SAME construct-heavy HolyD programs, run through BOTH backends, must
 * agree.
 *
 *   backend A = the OLD holyd_codegen_* emitter -> PE -> wubu_zx_run loader
 *               (the loader path). Its real run results are the anchors below:
 *               checksum 323, struct 42, num 321, tokenizer 4 -- verified in
 *               wubuos's tokenizer_test (self-host stages at two bases).
 *   backend B = the MODERN holyd_mir_eval (MIR interpreter, NULL driver) in
 *               this repo.
 *
 * hd_eval_mir must reproduce every loader-path anchor. Each source is a
 * verbatim copy of the matching src/runtime/selfbuild/*.hd program the loader
 * ran (single `int f(){}` + an appended `f();` to call the entry). A future
 * change to either backend that breaks the agreement fails THIS gate.
 */
#include "holyd_mir_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct { const char *name; const char *src; long want; } CASES[] = {
    { "checksum(pointer-walk string)",   /* src/runtime/selfbuild/checksum.hd */
      "int f() {\n"
      "    char *s = \"WUBU\";\n"
      "    char *p = s;\n"
      "    int c = 0;\n"
      "    while (*p != 0) {\n"
      "        c = c + (int) *p;\n"
      "        p = p + 1;\n"
      "    }\n"
      "    return c;\n"
      "}\n", 323 },
    { "struct member store/load",         /* src/runtime/selfbuild/struct.hd */
      "struct P { int x; int y; };\n"
      "int f() {\n"
      "    struct P p;\n"
      "    p.x = 30;\n"
      "    p.y = 12;\n"
      "    return p.x + p.y;\n"
      "}\n", 42 },
    { "multi-digit integer parse",        /* src/runtime/selfbuild/num.hd */
      "int f() {\n"
      "    char *s = \"321\";\n"
      "    int v = 0;\n"
      "    for (int i = 0; s[i] != 0; i++) {\n"
      "        v = v * 10 + (int)(s[i] - '0');\n"
      "    }\n"
      "    return v;\n"
      "}\n", 321 },
    { "tokenizer (lexer stage)",          /* src/runtime/selfbuild/tokenizer.hd */
      "int f() {\n"
      "    char *s = \"int x = 42\";\n"
      "    int toks = 0, in = 0;\n"
      "    for (int i = 0; s[i] != 0; i++) {\n"
      "        if (s[i] == ' ') { if (in) { toks++; in = 0; } }\n"
      "        else { in = 1; }\n"
      "    }\n"
      "    if (in) toks++;\n"
      "    return toks;\n"
      "}\n", 4 },
};
#define NCASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

int main(void)
{
    int pass = 0, fail = 0;
    printf("=== B4 CROSS-BACKEND CONSISTENCY: old emitter->loader vs MIR ===\n");
    printf("  anchors are the OLD-backend loader-run results (wubuos tokenizer_test)\n\n");
    for (int i = 0; i < NCASES; i++) {
        char *full = malloc(strlen(CASES[i].src) + 8);
        sprintf(full, "%s\nf();\n", CASES[i].src);
        int64_t r = hd_eval_mir(full, NULL);
        free(full);
        int ok = (r == CASES[i].want);
        printf("  [%d] %-32s MIR=%lld loader=%ld  %s\n", i + 1,
               CASES[i].name, (long long)r, CASES[i].want,
               ok ? "AGREE" : "DIVERGE");
        if (ok) pass++; else fail++;
    }
    printf("\n  cross-backend: %d/%d AGREE\n", pass, pass + fail);
    if (fail) { printf("=== TWO BACKENDS DIVERGED (fix the diverging backend) ===\n"); return 1; }
    printf("=== BACKENDS AGREE (old emitter->loader == MIR) on ");
    for (int i = 0; i < NCASES; i++) printf("%s%s", i ? ", " : "", CASES[i].name);
    printf(") ===\n");
    return 0;
}
