/*
 * test_wasm_backend.c — Validate WASM backend emits correct binaries.
 *
 * Tests the WASM emitter directly without the driver registry.
 * Self-contained: only links wubu_mir + wubu_isa_wasm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "jit/wubu_isa_wasm.h"

/* WASM opcodes we validate */
#define OP_I32_ADD    0x6A
#define OP_I32_CONST  0x41
#define OP_I32_MUL    0x6C
#define OP_END        0x0B

static int total = 0, pass_count = 0;
#define CHECK(c, m) do { total++; if (c) { pass_count++; } else printf("  FAIL: %s\n", m); } while(0)

int main(void)
{
    printf("=== WASM BACKEND VALIDATION ===\n\n");

    /* Test 1: Compile CONST 42 ; RET to WASM */
    {
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        wubu_vr_t v = wubu_mir_const(&prog, 42);
        wubu_mir_ret(&prog, v);

        uint8_t *buf = NULL;
        size_t sz = 0;
        int rc = wubu_isa_wasm_compile(&prog, &buf, &sz);
        CHECK(rc == 0, "WASM compilation succeeded");
        CHECK(buf != NULL, "WASM output buffer non-null");
        CHECK(sz > 0, "WASM output non-empty");

        /* Validate WASM magic number and version */
        if (buf && sz >= 8) {
            CHECK(buf[0] == 0x00 && buf[1] == 0x61 && buf[2] == 0x73 && buf[3] == 0x6D,
                  "WASM magic number (\\0asm)");
            CHECK(buf[4] == 0x01 && buf[5] == 0x00 && buf[6] == 0x00 && buf[7] == 0x00,
                  "WASM version 1");
        }

        /* Validate structure: check for expected content */
        if (buf && sz > 8) {
            /* Check for functype (0x60) which appears in type section */
            int found_functype = 0, found_i32_const = 0;
            for (size_t i = 8; i < sz; i++) {
                if (buf[i] == 0x60) found_functype = 1;
                if (buf[i] == OP_I32_CONST) found_i32_const = 1;
            }
            CHECK(found_functype, "Function type (0x60) present");
            CHECK(found_i32_const, "i32.const opcode present");
        }

        if (buf) free(buf);
        wubu_mir_free(&prog);
        printf("Test 1 (CONST 42; RET): %s\n", rc == 0 ? "PASS" : "FAIL");
    }

    /* Test 2: Compile ADD to WASM */
    {
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        wubu_vr_t a = wubu_mir_const(&prog, 20);
        wubu_vr_t b = wubu_mir_const(&prog, 22);
        wubu_vr_t sum = wubu_mir_binop(&prog, MIR_ADD, a, b);
        wubu_mir_ret(&prog, sum);

        uint8_t *buf = NULL;
        size_t sz = 0;
        int rc = wubu_isa_wasm_compile(&prog, &buf, &sz);
        CHECK(rc == 0, "WASM ADD compilation succeeded");
        CHECK(buf != NULL && sz > 8, "WASM ADD output valid");

        /* Check for i32.add opcode in the binary */
        if (buf && sz > 8) {
            int found_add = 0;
            for (size_t i = 8; i < sz - 1; i++) {
                if (buf[i] == OP_I32_ADD) { found_add = 1; break; }
            }
            CHECK(found_add, "i32.add opcode present in WASM binary");
        }

        if (buf) free(buf);
        wubu_mir_free(&prog);
        printf("Test 2 (ADD): %s\n", rc == 0 ? "PASS" : "FAIL");
    }

    /* Test 3: Compile MUL to WASM */
    {
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        wubu_vr_t a = wubu_mir_const(&prog, 6);
        wubu_vr_t b = wubu_mir_const(&prog, 7);
        wubu_vr_t prod = wubu_mir_binop(&prog, MIR_MUL, a, b);
        wubu_mir_ret(&prog, prod);

        uint8_t *buf = NULL;
        size_t sz = 0;
        int rc = wubu_isa_wasm_compile(&prog, &buf, &sz);
        CHECK(rc == 0, "WASM MUL compilation succeeded");

        if (buf && sz > 8) {
            int found_mul = 0;
            for (size_t i = 8; i < sz - 1; i++) {
                if (buf[i] == OP_I32_MUL) { found_mul = 1; break; }
            }
            CHECK(found_mul, "i32.mul opcode present in WASM binary");
        }

        if (buf) free(buf);
        wubu_mir_free(&prog);
        printf("Test 3 (MUL): %s\n", rc == 0 ? "PASS" : "FAIL");
    }

    printf("\n=== WASM Backend: %d/%d PASS ===\n", pass_count, total);
    return (pass_count == total) ? 0 : 1;
}
