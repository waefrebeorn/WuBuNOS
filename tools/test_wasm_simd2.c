/*
 * tools/test_wasm_simd2.c — WASM SIMD binary validation (direct emission).
 *
 * Emits a complete WASM module with v128 type and validates the binary
 * structure without going through the full MIR pipeline.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* WASM opcodes */
#define OP_I32_CONST    0x41
#define OP_F32_CONST    0x43
#define OP_LOCAL_GET    0x20
#define OP_LOCAL_SET    0x21
#define OP_I32_ADD      0x6A
#define OP_F32_ADD      0x92
#define OP_RETURN       0x0F
#define OP_END          0x0B
#define OP_CALL         0x10

/* Types */
#define TYPE_I32        0x7F
#define TYPE_F32        0x7D
#define TYPE_V128       0x7B

/* Sections */
#define SEC_TYPE        1
#define SEC_FUNC        3
#define SEC_CODE        10

/* SIMD sub-opcodes */
#define OP_F32X4_ADD    0xD4
#define OP_F32X4_MUL    0xD6
#define OP_V128_LOAD    0x00
#define OP_V128_STORE   0x0B
#define OP_V128_CONST   0x0C
#define OP_F32X4_SPLAT  0x13

static uint8_t buf[4096];
static size_t pos = 0;

static void emit_byte(uint8_t b) { buf[pos++] = b; }

static void emit_leb_u(uint32_t v) {
    while (v > 0x7F) { emit_byte((v & 0x7F) | 0x80); v >>= 7; }
    emit_byte(v & 0x7F);
}

static void emit_leb_s(int64_t v) {
    int more = 1;
    while (more) {
        uint8_t byte = v & 0x7F;
        v >>= 7;
        if ((v == 0 && !(byte & 0x40)) || (v == -1 && (byte & 0x40)))
            more = 0;
        else byte |= 0x80;
        emit_byte(byte);
    }
}

static void emit_f32(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    for (int i = 0; i < 4; i++) emit_byte((u >> (i*8)) & 0xFF);
}

static void emit_simd(uint32_t op) {
    emit_byte(0xFD);
    emit_leb_u(op);
}

static size_t sec_start(uint8_t id) {
    emit_byte(id);
    size_t patch = pos;
    emit_leb_u(0); /* placeholder */
    return patch;
}

static void sec_end(size_t patch) {
    size_t payload = pos - patch - 1;
    /* Patch the size (simplified: assume 1 byte) */
    buf[patch] = (uint8_t)payload;
}

static void emit_magic(void) {
    emit_byte(0x00); emit_byte(0x61); emit_byte(0x73); emit_byte(0x6D);
    emit_byte(0x01); emit_byte(0x00); emit_byte(0x00); emit_byte(0x00);
}

int main(void) {
    printf("=== WASM SIMD Binary Test ===\n\n");
    int pass = 0, fail = 0;

    emit_magic();

    /* Type section: 2 types */
    size_t tpatch = sec_start(SEC_TYPE);
    emit_leb_u(2);
    /* Type 0: (i32) -> (i32) */
    emit_byte(0x60); emit_byte(1); emit_byte(TYPE_I32);
    emit_byte(1); emit_byte(TYPE_I32);
    /* Type 1: (v128, v128) -> (v128) */
    emit_byte(0x60); emit_byte(2); emit_byte(TYPE_V128); emit_byte(TYPE_V128);
    emit_byte(1); emit_byte(TYPE_V128);
    sec_end(tpatch);

    /* Function section: 2 functions */
    size_t fpatch = sec_start(SEC_FUNC);
    emit_leb_u(2);
    emit_leb_u(0); /* func 0: type 0 (scalar) */
    emit_leb_u(1); /* func 1: type 1 (SIMD) */
    sec_end(fpatch);

    /* Code section: 2 bodies */
    size_t cpatch = sec_start(SEC_CODE);
    emit_leb_u(2);

    /* Body 0: scalar add (i32.const 1, i32.const 2, i32.add, return) */
    size_t b0 = pos; emit_leb_u(0); /* placeholder size */
    emit_leb_u(0); /* no locals */
    emit_byte(OP_I32_CONST); emit_leb_s(1);
    emit_byte(OP_I32_CONST); emit_leb_s(2);
    emit_byte(OP_I32_ADD);
    emit_byte(OP_RETURN);
    emit_byte(OP_END);
    buf[b0] = (uint8_t)(pos - b0 - 1); /* patch size */

    /* Body 1: SIMD f32x4.add (v128.const [1,2,3,4], v128.const [5,6,7,8], f32x4.add, return) */
    size_t b1 = pos; emit_leb_u(0); /* placeholder size */
    emit_leb_u(0); /* no locals */
    emit_simd(OP_V128_CONST);
    for (int i = 0; i < 16; i++) emit_byte(i); /* 16 bytes of v128.const */
    emit_simd(OP_V128_CONST);
    for (int i = 0; i < 16; i++) emit_byte(i + 16);
    emit_simd(OP_F32X4_ADD);
    emit_byte(OP_RETURN);
    emit_byte(OP_END);
    buf[b1] = (uint8_t)(pos - b1 - 1); /* patch size */

    sec_end(cpatch);

    size_t total = pos;
    printf("Generated %zu bytes\n", total);

    /* Validate */
    if (total >= 8 && buf[0]==0x00 && buf[1]==0x61 && buf[2]==0x73 && buf[3]==0x6D) {
        printf("PASS: WASM magic\n"); pass++;
    } else { printf("FAIL: WASM magic\n"); fail++; }

    /* Check for v128 type (0x7B) */
    int has_v128 = 0;
    for (size_t i = 0; i < total; i++) if (buf[i] == TYPE_V128) { has_v128 = 1; break; }
    if (has_v128) { printf("PASS: v128 type present\n"); pass++; }
    else { printf("FAIL: v128 type missing\n"); fail++; }

    /* Check for SIMD opcodes (0xFD prefix) */
    int has_simd = 0;
    for (size_t i = 0; i < total - 1; i++) {
        if (buf[i] == 0xFD) {
            uint32_t sub = 0, shift = 0;
            for (int j = 0; j < 4 && i+1+j < total; j++) {
                sub |= (uint32_t)(buf[i+1+j] & 0x7F) << shift;
                shift += 7;
                if (!(buf[i+1+j] & 0x80)) break;
            }
            if (sub == OP_F32X4_ADD || sub == OP_F32X4_MUL || sub == OP_V128_CONST) {
                has_simd = 1; break;
            }
        }
    }
    if (has_simd) { printf("PASS: SIMD opcodes present\n"); pass++; }
    else { printf("FAIL: SIMD opcodes missing\n"); fail++; }

    /* Check for f32x4.add specifically */
    int has_f32x4_add = 0;
    for (size_t i = 0; i < total - 1; i++) {
        if (buf[i] == 0xFD && buf[i+1] == OP_F32X4_ADD) { has_f32x4_add = 1; break; }
    }
    if (has_f32x4_add) { printf("PASS: f32x4.add opcode\n"); pass++; }
    else { printf("FAIL: f32x4.add missing\n"); fail++; }

    /* Check section structure */
    if (buf[8] == SEC_TYPE) { printf("PASS: type section\n"); pass++; }
    else { printf("FAIL: type section\n"); fail++; }

    printf("\n=== WASM SIMD: %d PASS, %d FAIL ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
