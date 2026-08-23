/*
 * holyd_bf_stub.c -- minimal bf_run stub so holyd.c links without
 * pulling in brainfuck.c's own main(). The -brainfuck flag still
 * works at the CLI level; this stub reports it is built without the
 * embedded interpreter. Replace with -DHOLYC_BF_EMBEDDED brainfuck.c
 * if the meme needs to actually run.
 *
 * C11, self-contained.
 */
#include <stdio.h>

int bf_run(const char *src) {
    (void)src;
    fprintf(stderr, "holyd: -brainfuck built without embedded interpreter "
                    "(link brainfuck.c with -DHOLYC_BF_EMBEDDED to enable)\n");
    return 1;
}
