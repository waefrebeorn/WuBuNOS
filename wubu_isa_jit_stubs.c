/*
 * wubu_isa_jit_stubs.c — stubs for native JIT drivers (standalone build only).
 *
 * The x86-64 and ARM64 drivers need the JIT encoder from the OS repo.
 * When building standalone, we provide minimal stubs that report
 * "cannot run on this host" so the registry is complete but tests skip.
 */
#include "wubu_isa_driver.h"
#include <stdio.h>

static int stub_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *size) {
    (void)p; (void)out; (void)size;
    return -1;  /* JIT not available in standalone build */
}

static int64_t stub_run(const uint8_t *code, size_t size, int64_t arg) {
    (void)code; (void)size; (void)arg;
    return 0;
}

static void stub_describe(void) {
    printf("Native JIT driver (requires OS repo build)\n");
}

const wubu_isa_driver_t wubu_isa_x86_64 = {
    .name = "x86-64", .family = "native-jit",
    .exec = WUBU_ISA_NATIVE,
    .compile = stub_compile, .run = stub_run, .describe = stub_describe,
};

const wubu_isa_driver_t wubu_isa_arm64 = {
    .name = "arm64", .family = "native-jit",
    .exec = WUBU_ISA_NATIVE,
    .compile = stub_compile, .run = stub_run, .describe = stub_describe,
};
