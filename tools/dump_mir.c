/* dump_mir.c — standalone MIR dump tool */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mir.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <source>\n", argv[0]);
        return 1;
    }

    const char *source = argv[1];
    wubu_mir_prog_t prog;
    extern int hd_build_mir(const char *source, wubu_mir_prog_t *prog);

    if (hd_build_mir(source, &prog) != 0) {
        printf("build failed\n");
        return 1;
    }

    wubu_mir_dump(&prog);
    return 0;
}
