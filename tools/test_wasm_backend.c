/*
 * test_wasm_backend.c — Validate WASM backend emits correct binaries.
 *
 * Tests the WASM emitter directly with hand-crafted MIR programs.
 * This is a compile+valid test; WASM can't be run natively on x86-64.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_wasm.h"
#include "wubu_isa_driver.h"

/* WASM opcodes we use */
#define OP_I32_ADD    0x6A
#define OP_I32_CONST  0x41
#define OP_END        0x0B
#define SEC_TYPE      1

static int total = 0, pass = 0;
#define CHECK(c, m) do { total++; if (c) pass++; else printf("  FAIL: %s\n", m); } while(0)

int main(void)
{
    printf("=== WASM BACKEND VALIDATION ===\n\n");

    /* Test 1: Driver is in registry */
    const wubu_isa_driver_t *d = wubu_isa_find("wasm");
    CHECK(d != NULL, "driver 'wasm' found in registry");
    if (!d) return 1;

    /* Test 2: Compile simple MIR to WASM */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);

    /* CONST 42 ; RET */
    wubu_vr_t v = wubu_mir_const(&prog, 42);
    wubu_mir_ret(&prog, v);

    uint8_t *buf = NULL;
    size_t sz = 0;
    int rc = d->compile(&prog, &buf, &sz);

    CHECK(rc == 0, "WASM compilation succeeded");
    if (rc != 0) {
        wubu_mir_free(&prog);
        printf("\n=== FAIL (%d/%d) ===\n", pass, total);
        return 1;
    }

    /* Test 3: Validate WASM binary structure */
    CHECK(buf != NULL, "WASM output buffer is non-null");
    if (!buf) {
        wubu_mir_free(&prog);
        printf("\n=== FAIL (%d/%d) ===\n", pass, total);
        return 1;
    }

    CHECK(sz >= 16, "WASM binary has minimum size");
    CHECK(memcmp(buf, "\0asm", 4) == 0, "WASM magic header correct");
    CHECK(buf[4] == 1, "WASM version is 1");

    /* Test 4: Check for expected opcodes */
    int found_const = 0, found_end = 0;
    for (size_t i = 0; i < sz; i++) {
        if (buf[i] == OP_I32_CONST) found_const = 1;
        if (buf[i] == OP_END) found_end = 1;
    }
    CHECK(found_const, "i32.const opcode emitted");
    CHECK(found_end, "END opcode emitted");

    free(buf);
    wubu_mir_free(&prog);

    printf("\n=== %s ===\n", pass == total ? "PASS" : "FAIL");
    printf("(PASS: %d, TOTAL: %d)\n", pass, total);
    return (pass == total) ? 0 : 1;
}