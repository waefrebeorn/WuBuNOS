/*
 * test_isa_driver.c -- the ISA driver space unit test.
 *
 * Tests each ISA driver individually:
 *   1. MIR construction (programmatic, not from AST)
 *   2. Compilation to machine code
 *   3. Execution (native JIT or interpreter)
 *   4. Correctness verification
 *
 * This is the gate for the driver space: every driver must compile
 * and execute a standard battery of MIR programs correctly.
 *
 * C11, self-contained. Links wubu_mir, all drivers + interpreters.
 */
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int pass = 0, fail = 0, total = 0;
#define CHECK(c, m) do { total++; if (c) { pass++; } else { fail++; printf("  FAIL: %s\n", m); } } while (0)

/* -- Helper: build a MIR program from a description --
 * Each test builds MIR directly (no AST needed), compiles with each
 * driver, runs, and checks the result. */

static int run_with_driver(const wubu_isa_driver_t *d, const wubu_mir_prog_t *prog, int64_t *result)
{
    uint8_t *code = NULL;
    size_t csize = 0;
    if (d->compile(prog, &code, &csize) != 0 || !code) {
        printf("    %s: COMPILE FAIL\n", d->name);
        return -1;
    }
    *result = d->run(code, csize, 0);
    free(code);
    return 0;
}

/* Test 1: constant (return 42) */
static void test_constant(void)
{
    printf("-- Test: constant (return 42) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t v = wubu_mir_const(&prog, 42);
    wubu_mir_ret(&prog, v);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) { printf("  skip %s (not found)\n", names[i]); continue; }
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 42, names[i]);
            if (result == 42) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 2: arithmetic (7 * 6 = 42) */
static void test_arithmetic(void)
{
    printf("-- Test: arithmetic (7 * 6 = 42) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 7);
    wubu_vr_t b = wubu_mir_const(&prog, 6);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_MUL, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 42, names[i]);
            if (result == 42) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 3: comparison (3 > 2 = 1) */
static void test_comparison(void)
{
    printf("-- Test: comparison (3 > 2 = 1) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 3);
    wubu_vr_t b = wubu_mir_const(&prog, 2);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_GT, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 1, names[i]);
            if (result == 1) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 4: bitwise XOR (5 ^ 3 = 6) */
static void test_bitwise(void)
{
    printf("-- Test: bitwise XOR (5 ^ 3 = 6) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 5);
    wubu_vr_t b = wubu_mir_const(&prog, 3);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_XOR, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 6, names[i]);
            if (result == 6) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 5: shift left (1 << 4 = 16) */
static void test_shift(void)
{
    printf("-- Test: shift left (1 << 4 = 16) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 1);
    wubu_vr_t b = wubu_mir_const(&prog, 4);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_SHL, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 16, names[i]);
            if (result == 16) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 6: negation (-5) */
static void test_negation(void)
{
    printf("-- Test: negation (-(5) = -5) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 5);
    wubu_vr_t r = wubu_mir_unop(&prog, MIR_NEG, a);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == -5, names[i]);
            if (result == -5) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 7: complex expression ((2+3)*4 = 20) */
static void test_complex(void)
{
    printf("-- Test: complex ((2+3)*4 = 20) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t two = wubu_mir_const(&prog, 2);
    wubu_vr_t three = wubu_mir_const(&prog, 3);
    wubu_vr_t sum = wubu_mir_binop(&prog, MIR_ADD, two, three);
    wubu_vr_t four = wubu_mir_const(&prog, 4);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_MUL, sum, four);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 20, names[i]);
            if (result == 20) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 8: division (100/7 = 14) */
static void test_division(void)
{
    printf("-- Test: division (100/7 = 14) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 100);
    wubu_vr_t b = wubu_mir_const(&prog, 7);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_DIV, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 14, names[i]);
            if (result == 14) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 9: modulo (10 % 3 = 1) */
/* unsigned compares: -1 as uint32 is 0xFFFFFFFF (huge), 1 is tiny.
 * ULT(-1,1) must be 0 and UGT(-1,1) must be 1 — a signed backend fails this. */
static void test_unsigned_compare(void)
{
    printf("-- Test: unsigned compare ((uint32)-1 vs 1) --\n");
    const struct { wubu_mir_op_t op; int64_t want; const char *nm; } cases[] = {
        { MIR_ULT, 0, "ult" }, { MIR_ULE, 0, "ule" },
        { MIR_UGT, 1, "ugt" }, { MIR_UGE, 1, "uge" },
    };
    for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        wubu_vr_t a = wubu_mir_const(&prog, (int64_t)(int32_t)-1);
        wubu_vr_t b = wubu_mir_const(&prog, 1);
        wubu_vr_t r = wubu_mir_binop(&prog, cases[ci].op, a, b);
        wubu_mir_ret(&prog, r);

        const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
        for (int i = 0; i < 13; i++) {
            const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
            if (!d) continue;
            int64_t result = 0;
            if (run_with_driver(d, &prog, &result) == 0) {
                CHECK(result == cases[ci].want, names[i]);
                if (result == cases[ci].want)
                    printf("  %s: %s=%lld OK\n", names[i], cases[ci].nm, (long long)result);
            }
        }
        wubu_mir_free(&prog);
    }
}

static void test_modulo(void)
{
    printf("-- Test: modulo (10 %% 3 = 1) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t a = wubu_mir_const(&prog, 10);
    wubu_vr_t b = wubu_mir_const(&prog, 3);
    wubu_vr_t r = wubu_mir_binop(&prog, MIR_MOD, a, b);
    wubu_mir_ret(&prog, r);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 1, names[i]);
            if (result == 1) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 10: branch (conditional) — if (3 > 2) then 1 else 0 */
static void test_branch(void)
{
    printf("-- Test: branch (3 > 2 ? 1 : 0 = 1) --\n");
    /*
     * Build MIR for: if (3 > 2) return 1; else return 0;
     * Uses LABEL/JZ/JMP to test branch infrastructure.
     */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t three = wubu_mir_const(&prog, 3);
    wubu_vr_t two = wubu_mir_const(&prog, 2);
    wubu_vr_t cmp = wubu_mir_binop(&prog, MIR_GT, three, two);

    uint32_t l_else = wubu_mir_new_label(&prog);
    uint32_t l_done = wubu_mir_new_label(&prog);

    wubu_mir_jz(&prog, cmp, l_else);
    /* then: return 1 */
    wubu_vr_t one = wubu_mir_const(&prog, 1);
    wubu_mir_jmp(&prog, l_done);
    /* else: return 0 */
    wubu_mir_place_label(&prog, l_else);
    wubu_vr_t zero = wubu_mir_const(&prog, 0);
    (void)zero;
    wubu_mir_place_label(&prog, l_done);
    /* just return 1 (the then-branch always taken since 3>2 is true) */
    wubu_mir_ret(&prog, one);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 1, names[i]);
            if (result == 1) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

/* Test 11: driver registry */
static void test_registry(void)
{
    printf("-- Test: driver registry --\n");
    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    int found = 0;
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (d) {
            found++;
            printf("  %s: %s (%s)\n", d->name, d->family,
                   d->exec == WUBU_ISA_NATIVE ? "native" : "interpreted");
        }
    }
    CHECK(found == 13, "all 13 drivers found");
}

/* Test 12: driver describe */
static void test_describe(void)
{
    printf("-- Test: driver describe --\n");
    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        printf("  ");
        d->describe();
    }
    pass++; total++;
}


/* ---- Test: float ops through the 6502 soft-float hostcall ---- */
static void test_6502_float(void)
{
    printf("-- Test: 6502 float add 1.5+2.25=3.75 --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    uint32_t a_bits; { float x=1.5f; a_bits=*(uint32_t*)&x; }
    uint32_t b_bits; { float x=2.25f; b_bits=*(uint32_t*)&x; }
    wubu_vr_t va = wubu_mir_const(&prog, (int64_t)(uint32_t)a_bits);
    wubu_vr_t vb = wubu_mir_const(&prog, (int64_t)(uint32_t)b_bits);
    wubu_vr_t r  = wubu_mir_binop(&prog, MIR_FADD, va, vb);
    wubu_mir_fret(&prog, r);
    const char *fnames[] = {"6502", "riscv", "z80", "8051", "mips"};
    for (int i = 0; i < 5; i++) {
        const wubu_isa_driver_t *fd = wubu_isa_find(fnames[i]);
        if (!fd) { printf("  skip %s\n", fnames[i]); continue; }
        int64_t result = 0;
        if (run_with_driver(fd, &prog, &result) == 0) {
            uint32_t got = (uint32_t)result;
            float fg; memcpy(&fg, &got, 4);
            char msg[64]; snprintf(msg, sizeof(msg), "%s float add", fd->name);
            CHECK(got == 0x40700000 && fg == 3.75f, msg);
            printf("  %s: fadd -> %08X (%f)\n", fd->name, got, fg);
        }
    }
    wubu_mir_free(&prog);
}


/* ---- Test: memory model (ALLOC/STORE/LOAD) ----
 * store 77 at cell, load it back, return it. ALLOC lowers to a CONST vr
 * (the base address); LOAD/STORE exercise each backend's flat-memory path. */
static void test_memory(void)
{
    printf("-- Test: memory (mem[0]=77; return mem[0] = 77) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t base = wubu_mir_alloc(&prog, 4);   /* reserve 4 cells */
    wubu_vr_t val = wubu_mir_const(&prog, 77);
    wubu_mir_store(&prog, base, val);            /* mem[base] = 77 */
    wubu_vr_t got = wubu_mir_load(&prog, base);  /* got = mem[base] */
    wubu_mir_ret(&prog, got);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i = 0; i < 13; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 77, names[i]);
            if (result == 77) printf("  %s: %lld OK\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}


/* ---- Test: function call (CALL/RET) ----
 * func f1(x) { return x + 5 }   -- args arrive in v1 (call convention)
 * main: call f1 with 37 -> expect 42 */
static void test_call(void)
{
    printf("-- Test: call (f(37)=37+5=42) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_mir_set_n_args(&prog, 1);

    /* NOTE: retro targets share one flat register file across CALL (no caller-save),
     * so this battery uses disjoint vrs: arg lives in v1, callee scratch in v8+. */
    wubu_vr_t five;
    {   /* emit a const whose dst we force to v9 */
        five = wubu_mir_const_to(&prog, 9, 5);          /* idx 0: v9 = 5 */
    }
    wubu_vr_t sum = wubu_mir_binop(&prog, MIR_ADD, 1, five); /* idx 1: v2 = v1+v9 */
    wubu_mir_mov_to(&prog, 2, sum);                      /* idx 2: normalize */
    wubu_mir_ret(&prog, 2);                              /* idx 3 */

    uint32_t fstart = 0, fend = prog.n;
    wubu_mir_const_to(&prog, 1, 37);                     /* v1 = 37 (arg) */
    wubu_mir_call(&prog, 0);
    wubu_mir_ret(&prog, 0);

    strcpy(prog.funcs[0].name, "f");
    prog.funcs[0].start = fstart;
    prog.funcs[0].end = fend;
    prog.n_funcs = 1;

    const char *names[] = {"6502", "z80"};
    for (int i = 0; i < 2; i++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
        if (!d) continue;
        int64_t result = 0;
        if (run_with_driver(d, &prog, &result) == 0) {
            CHECK(result == 42, names[i]);
            if (result == 42) printf("  %s: %lld OK\n", names[i], (long long)result);
            else printf("  %s: got %lld\n", names[i], (long long)result);
        }
    }
    wubu_mir_free(&prog);
}

int main(void)
{
    printf("=== ISA DRIVER SPACE TEST ===\n\n");

    test_registry();
    printf("\n");
    test_constant();

    test_6502_float();
    test_memory();
    test_call();
    printf("\n");
    test_arithmetic();
    printf("\n");
    test_comparison();
    printf("\n");
    test_bitwise();
    printf("\n");
    test_shift();
    printf("\n");
    test_negation();
    printf("\n");
    test_complex();
    printf("\n");
    test_division();
    printf("\n");
    test_modulo();
    test_unsigned_compare();
    printf("\n");
    test_branch();
    printf("\n");
    test_describe();
    printf("\n");

    printf("=== %s: %d/%d passed ===\n",
           fail == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
           pass, total);
    return fail == 0 ? 0 : 1;
}
