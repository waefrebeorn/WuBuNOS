/* jit_stubs_arm64.c — arm64-only stub (x86-64 uses the REAL driver now). */
#include "wubu_isa_driver.h"
#include <stdio.h>

static int arm64_stub_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *size) {
    (void)p; (void)out; (void)size;
    return -1;
}
static int64_t arm64_stub_run(const uint8_t *code, size_t size, int64_t arg) {
    (void)code; (void)size; (void)arg;
    return 0;
}
static void arm64_stub_describe(void) {
    printf("Native JIT driver (requires OS repo build)\n");
}

const wubu_isa_driver_t wubu_isa_arm64 = {
    .name = "arm64", .family = "native-jit",
    .exec = WUBU_ISA_NATIVE,
    .compile = arm64_stub_compile, .run = arm64_stub_run, .describe = arm64_stub_describe,
};
