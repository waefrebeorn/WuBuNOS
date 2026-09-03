/* interp_test.c — test interpreter vs JIT on MIR programs */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

extern int64_t wubu_mir_interp(const wubu_mir_prog_t *p);
extern int hd_build_mir(const char *source, wubu_mir_prog_t *prog);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <source>\n", argv[0]);
        return 1;
    }

    const char *source = argv[1];
    wubu_mir_prog_t prog;

    if (hd_build_mir(source, &prog) != 0) {
        printf("build failed\n");
        return 1;
    }

    /* Allocate mem */
    int64_t *mem_ptr = prog.mem;
    if (mem_ptr == NULL) {
        int64_t mem_hi = prog.total_mem;
        if ((int64_t)(prog.next_vr_hi) - 1 > mem_hi)
            mem_hi = (int64_t)(prog.next_vr_hi) - 1;
        int64_t mem_size = (mem_hi < 1) ? 1 : (mem_hi + 1);
        mem_ptr = (int64_t *)calloc((size_t)mem_size, sizeof(int64_t));
    }

    /* Run via interpreter */
    int64_t result = wubu_mir_interp(&prog);
    printf("interp result=%lld\n", (long long)result);

    /* Run via JIT */
    const wubu_isa_driver_t *drv = &wubu_isa_x86_64;
    wubu_mir_prog_t prog_copy = prog;
    prog_copy.mem = mem_ptr;
    uint8_t *code; size_t sz;
    if (drv->compile(&prog_copy, &code, &sz) == 0) {
        int64_t jit_result = drv->run(code, sz, (int64_t)mem_ptr);
        printf("jit result=%lld\n", (long long)jit_result);
        free(code);
    } else {
        printf("jit compile failed\n");
    }

    if (mem_ptr != prog.mem) free(mem_ptr);
    wubu_mir_free(&prog);
    return 0;
}
