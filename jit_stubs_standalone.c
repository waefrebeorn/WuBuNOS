/* jit_stubs_standalone.c — minimal stubs for non-essential ISA backends
 * so the standalone compiler can link without MIPS/RISC-V/ARM64/8086/etc.
 * Each backend returns "not available in standalone mode" from describe().
 * The real implementations live in the OS repo (src/) alongside the kernel. */
#include "wubu_isa_driver.h"
#include <stdio.h>

/* -- MIPS stub -- */
static int stub_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *size) {
    (void)p; (void)out; (void)size; return -1;
}
static int64_t stub_run(const uint8_t *code, size_t size, int64_t arg) {
    (void)code; (void)size; (void)arg; return 0;
}
static void stub_describe(void) {
    printf("standalone stub (backend not built)\n");
}

/* Each real backend exports a wubu_isa_<name> symbol. We provide
 * stubs so wubu_isa_driver.c's registry links cleanly in standalone mode. */
const wubu_isa_driver_t wubu_isa_mips = {
    .name="mips", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_m68k = {
    .name="m68k", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_i8086 = {
    .name="8086", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_riscv = {
    .name="riscv", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_6502 = {
    .name="6502", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_z80 = {
    .name="z80", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_8051 = {
    .name="8051", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_avr = {
    .name="avr", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_pic = {
    .name="pic", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
const wubu_isa_driver_t wubu_isa_amdgpu = {
    .name="amdgpu", .family="interp", .exec=0,
    .compile=stub_compile, .run=stub_run, .describe=stub_describe
};
