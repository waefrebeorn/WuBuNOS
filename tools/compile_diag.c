/* compile_diag.c -- print parser/semantic diagnostics for a source file.
 *
 * The gauntlet harness swallows compiler stderr, so a rejected test gives no
 * reason. This reads a file, runs lex -> parse -> semantic, and prints every
 * diagnostic with its line, which is what classifying the semantic pass's
 * false rejections needs.
 *
 *   tools/compile_diag <file.c> [--no-semantic]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "holyd_types.h"
#include "holyd_lexer.h"
#include "wubu_preproc.h"
#include "holyd_parser.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <file.c> [--no-semantic]\n", argv[0]); return 2; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = malloc((size_t)n + 1);
    if (!src) { fclose(f); return 2; }
    size_t rd = fread(src, 1, (size_t)n, f);
    src[rd] = '\0';
    fclose(f);

    /* Run the real preprocessor first: the compiler does, so measuring the raw
     * text produced phantom diagnostics for anything beginning with '#'. */
    char *pp = wubu_preprocess(src);
    const char *text = pp ? pp : src;

    HDLexer lex;
    hd_lex_init(&lex, text);
    HDParser p;
    hd_parse_init(&p, &lex);
    HDASTNode *unit = hd_parse_compilation_unit(&p);

    int parse_errors = p.n_errors;
    if (parse_errors == 0 && !(argc > 2 && !strcmp(argv[2], "--no-semantic")))
        hd_semantic_check(&p, unit);

    printf("parse_errors=%d total_diagnostics=%d\n", parse_errors, p.n_errors);
    for (int i = 0; i < p.n_errors; i++)
        printf("  %s\n", i < parse_errors ? "parse: " : "semantic: ");
    for (int i = 0; i < p.n_errors; i++)
        printf("  [%d] %s\n", i, p.errors[i]);
    return p.n_errors ? 1 : 0;
}
