/* test_direct.c - Direct test of x86-64 JIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

extern const wubu_isa_driver_t wubu_x86_64_driver;

int main(int argc, char **argv) {
    const char *src = "int w[10]; void f(void) { int i = 0; w[i] = 42; } int main(void) { f(); return w[0]; }";
    if (argc > 1) src = argv[1];

    wubu_mir_prog_t prog = {0};
    if (hd_build_mir(src, &prog) != 0) {
        fprintf(stderr, "Failed to build MIR\n");
        return 1;
    }

    /* Dump MIR */
    printf("MIR:\n");
    wubu_mir_dump(&prog);

    /* Run on interpreter */
    int64_t interp_result = wubu_mir_interp(&prog);
    printf("Interpreter result: %lld\n", (long long)interp_result);

    return 0;
}