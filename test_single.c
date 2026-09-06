/*
 * test_single.c — Run a single gauntlet test.
 * Usage: test_single <source> <expected>
 * Compiles source with HolyD, runs on x86-64 JIT, prints result.
 * Exit code: 0 = PASS, 1 = FAIL, 2 = ERROR
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <source> <expected>\n", argv[0]);
        return 2;
    }
    
    const char *source = argv[1];
    int64_t expected = atoll(argv[2]);
    
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    
    int build_result = hd_build_mir(source, &prog);
    if (build_result != 0) {
        return 2; /* ERROR */
    }
    
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    int64_t result = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
    
    wubu_mir_free(&prog);
    
    if (result == expected) {
        printf("PASS %lld\n", (long long)result);
        return 0;
    } else {
        printf("FAIL expected=%lld got=%lld\n", (long long)expected, (long long)result);
        return 1;
    }
}
