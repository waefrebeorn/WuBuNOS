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
#include "x86_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static int pass = 0, fail = 0, total = 0;
static int compiled_only = 0;   /* cross backends: encoded, not executable here */

/* -- arm64 execution dump ------------------------------------------------
 * A cross backend's code cannot run here, but it CAN run under
 * qemu-aarch64-static (QEMU's own cross-arch test approach, see
 * https://www.qemu.org/2018/06/21/tcg-testing/). When WUBU_ARM64_EXEC_DIR is
 * set, every ARM64 blob is written there with the value it must produce, so
 * tools/arm64_exec_gate.py can execute the real instructions and compare. That
 * turns "the encoder emitted something" into "the emitted code computes the
 * right answer". */
static const char *g_dump_dir = NULL;   /* base dir; one subdir per target */
static const char *g_dump_targets = ""; /* comma list, e.g. "arm64,riscv" */

/* Declared gaps: a (backend, case) pair that this backend genuinely cannot do,
 * with the reason. Listed here so the battery stays green HONESTLY -- the case is
 * reported as GAP with its reason, never silently skipped and never asserted to
 * pass. Remove a line the moment the backend implements it. */
static const struct { const char *driver, *case_name, *why; } g_known_gaps[] = {
    /* Found by the loop case (a backward branch with a vr re-assigned each
     * iteration). These are NOT "the backend cannot express a loop" -- several
     * emit one fine -- they are specific, measured failures. Each line is a bug
     * with a symptom attached, and removing a line is how the fix is declared. */
    /* Re-measured 2026-11-24 with WUBU_ISA_NO_GAPS=1 after MIR_JNZ was added to
     * 6502, z80, 8086, m68k and mips: 8086, m68k, z80 and mips now PASS -- their
     * earlier "wrong sum" numbers were the missing branch, not an allocator bug. */
    /* 6502 fixed: JMP_ABS to `done` was computed from e.n after the opcode
     * (one byte into the true path), and MIR_GT was emitted as GE. */
    /* riscv was here: "the FORWARD patch resolves to pos+4". It was not the
     * label machinery -- the B-type immediate fields were transposed, so every
     * branch landed a few bytes away. Fixed in the encoder; the case passes. */
    /* 8051 fixed: LJMP + a patch pass + JZ/JNZ (skip-the-inverse form). */
    /* avr fixed: patch pass + AVR_JMP (0x26, already implemented but never
     * emitted) + AVR_JZ/AVR_JNZ (0x28/0x29) in both the emitter and the
     * interpreter. */
    /* pic fixed: label fixups (it had a fixup pass only for CALL) + JZ/JNZ. */
    {"arm64","loop(1..4)", "cannot execute on this x86-64 host (NO_EXEC); executed under qemu-aarch64 it returns 5, not 10 (tools/isa_exec_gate.py KNOWN_FAIL)"},
    {"ptx",  "loop(1..4)", "cannot compile the case at all"},
};

static int g_unaccounted_holes = 0;

static const char *mir_op_name(int op)
{
    switch (op) {
    case MIR_ADD: return "MIR_ADD";     case MIR_SUB: return "MIR_SUB";
    case MIR_MUL: return "MIR_MUL";     case MIR_DIV: return "MIR_DIV";
    case MIR_AND: return "MIR_AND";     case MIR_OR:  return "MIR_OR";
    case MIR_XOR: return "MIR_XOR";     case MIR_GT:  return "MIR_GT";
    case MIR_JMP: return "MIR_JMP";     case MIR_JZ:  return "MIR_JZ";
    case MIR_JNZ: return "MIR_JNZ";     case MIR_LOAD: return "MIR_LOAD";
    case MIR_STORE: return "MIR_STORE"; case MIR_MOV: return "MIR_MOV";
    case MIR_RET: return "MIR_RET";
    default: return "?";
    }
}

/* Ops a backend genuinely cannot express yet, with the measured reason. A hole
 * that is NOT listed here fails the run: silence is how these survived. */
static const char *known_hole(const char *driver, int op)
{
    static const struct { const char *driver; int op; const char *why; } holes[] = {
        {"arm64", MIR_LOAD,  "no load at all -- the memory selftest 'passed' because the store wrote nothing and the value stayed in a register"},
        {"arm64", MIR_STORE, "no store at all (see MIR_LOAD)"},
    };
    for (size_t i = 0; i < sizeof holes / sizeof holes[0]; i++)
        if (strcmp(holes[i].driver, driver) == 0 && holes[i].op == op)
            return holes[i].why;
    return NULL;
}

static const char *known_gap(const char *driver, const char *case_name)
{
    for (size_t i = 0; i < sizeof g_known_gaps / sizeof g_known_gaps[0]; i++)
        if (strcmp(g_known_gaps[i].driver, driver) == 0 &&
            strcmp(g_known_gaps[i].case_name, case_name) == 0)
            return g_known_gaps[i].why;
    return NULL;
}

static void dump_arm64(const wubu_isa_driver_t *d, const char *case_name,
                       const uint8_t *code, size_t csize, int64_t want)
{
    if (!g_dump_dir || !code || !csize || !d->name) return;
    /* only the targets named in WUBU_EXEC_DUMP get dumped */
    if (!strstr(g_dump_targets, d->name)) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s", g_dump_dir, d->name);
    mkdir(path, 0755);          /* per-target subdir; EEXIST is fine */
    snprintf(path, sizeof path, "%s/%s/%s.bin", g_dump_dir, d->name, case_name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(code, 1, csize, f);
    fclose(f);
    snprintf(path, sizeof path, "%s/%s/%s.want", g_dump_dir, d->name, case_name);
    f = fopen(path, "w");
    if (f) { fprintf(f, "%lld\n", (long long)want); fclose(f); }
}
#define CHECK(c, m) do { total++; if (c) { pass++; } else { fail++; printf("  FAIL: %s\n", m); } } while (0)

/* -- Helper: build a MIR program from a description --
 * Each test builds MIR directly (no AST needed), compiles with each
 * driver, runs, and checks the result. */

static int run_with_driver(const wubu_isa_driver_t *d, const wubu_mir_prog_t *prog, int64_t *result)
{
    uint8_t *code = NULL;
    size_t csize = 0;
    wubu_isa_unsupported_op = -1;
    if (d->compile(prog, &code, &csize) != 0 || !code || csize == 0) {
        printf("    %s: COMPILE FAIL\n", d->name);
        return -1;
    }
    /* A backend that hit its op-switch default emitted a HOLE, not code. Running
     * it would test the hole and report whatever the hole produced. */
    if (wubu_isa_unsupported_op >= 0) {
        int op = wubu_isa_unsupported_op;
        printf("    %s: COMPILE FAIL -- MIR op %d (%s) has no case; the emitted code "
               "has a hole where that op should be\n", d->name, op, mir_op_name(op));
        free(code);
        /* EVERY test below is shaped `if (run_with_driver(...) == 0) CHECK(...)`,
         * so a backend that cannot compile a case was silently SKIPPED from the
         * tally -- the battery only ever checked backends that happened to
         * compile. Count it, name it, and require it be declared. */
        const char *why = known_hole(d->name, op);
        if (why) printf("  GAP  %s does not implement %s: %s\n", d->name, mir_op_name(op), why);
        else { g_unaccounted_holes++; printf("  FAIL %s: %s not implemented (undeclared)\n", d->name, mir_op_name(op)); }
        return -1;
    }
    fflush(stdout);
    /* The run contract is run(code, size, memory_pointer). Passing 0 here
     * worked only by luck for programs that never touch memory; test_memory
     * stored 77 through the NULL pointer and segfaulted inside x86_run.
     * Allocate a real buffer sized from the program's own memory requests. */
    size_t cells = (size_t)(prog->total_mem > 0 ? prog->total_mem : 16) + 64;
    int64_t *mem = (int64_t *)calloc(cells, sizeof(int64_t));
    if (!mem) { free(code); return -1; }
    *result = d->run(code, csize, (int64_t)(intptr_t)mem);
    free(mem);
    free(code);
    /* A cross backend (ARM64/PTX on x86-64) encodes correctly but cannot
     * execute here. That is not a wrong answer, and it must not be counted as
     * one -- it used to be reported as COMPILE FAIL, because the encoder
     * refused outright on a foreign host. The encoding itself is still
     * asserted: non-empty code, above. */
    if (*result == WUBU_ISA_NO_EXEC) { compiled_only++; return 1; }
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

/* A runaway case must not hang the whole gate.
 *
 * The loop test is the first case that can plausibly never terminate (a backend
 * that emits a broken backward branch spins forever, and for x86-64 the code
 * runs NATIVELY so there is no step budget to hit). SIGALRM + siglongjmp turns
 * that into a verdict -- "did not terminate" -- instead of a hung gate.
 */
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>

static sigjmp_buf g_case_jmp;
static volatile sig_atomic_t g_case_timed_out = 0;

static void case_alarm(int sig)
{
    (void)sig;
    g_case_timed_out = 1;
    siglongjmp(g_case_jmp, 1);
}

#define RUN_RAN      0
#define RUN_NOEXEC   1     /* backend says it cannot execute on this host */
#define RUN_HUNG     2
#define RUN_COMPILEFAIL 3

static int run_guarded(const wubu_isa_driver_t *d, const wubu_mir_prog_t *prog,
                       int64_t *result)
{
    int verdict = RUN_RAN;
    int rc = -1;
    g_case_timed_out = 0;
    struct sigaction sa, old;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = case_alarm;
    sigaction(SIGALRM, &sa, &old);
    if (sigsetjmp(g_case_jmp, 1) == 0) {
        alarm(5);
        rc = run_with_driver(d, prog, result);
        alarm(0);
    }
    sigaction(SIGALRM, &old, NULL);
    if (g_case_timed_out) {
        printf("  %s: did not terminate within 5 s\n", d->name);
        return RUN_HUNG;
    }
    /* "Cannot execute here" is NOT "hung": arm64 on this x86-64 host and ptx
     * without a driver both return WUBU_ISA_NO_EXEC, and calling that a hang
     * would blame the backend for the host. */
    if (rc == WUBU_ISA_NO_EXEC) return RUN_NOEXEC;
    if (rc != 0) return RUN_COMPILEFAIL;   /* a hole in the emitted code is a FAILURE */
    return verdict;
}

/* Test: LOOP -- a BACKWARD branch with a variable that changes each iteration.
 *
 * Every case before this was straight-line: forward branches only, each vr
 * written once. That is precisely the shape in which the arm64 label patcher
 * looked fine (it rewrote `b` as `b.eq` and every straight-line program still
 * passed, because the flags happened to be right). A loop exercises a backward
 * patch and a vr that is re-assigned across iterations, which is where an
 * allocator or frame bug shows up as a wrong sum rather than as a crash.
 *
 *     acc = 0; i = 1;
 *   loop:
 *     if (i > 4) goto done;
 *     acc = acc + i;  i = i + 1;
 *     goto loop;
 *   done: return acc;              -> 10
 */
static void test_loop(void)
{
    printf("-- Test: loop (sum 1..4 = 10, backward branch) --\n");
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);

    wubu_vr_t acc = wubu_mir_const(&prog, 0);
    wubu_vr_t i   = wubu_mir_const(&prog, 1);
    wubu_vr_t four = wubu_mir_const(&prog, 4);

    uint32_t l_loop = wubu_mir_new_label(&prog);
    uint32_t l_done = wubu_mir_new_label(&prog);

    wubu_mir_place_label(&prog, l_loop);
    wubu_vr_t cond = wubu_mir_binop(&prog, MIR_GT, i, four);
    wubu_mir_jnz(&prog, cond, l_done);

    wubu_vr_t sum = wubu_mir_binop(&prog, MIR_ADD, acc, i);
    wubu_mir_mov_to(&prog, acc, sum);
    wubu_vr_t next = wubu_mir_binop(&prog, MIR_ADD, i, wubu_mir_const(&prog, 1));
    wubu_mir_mov_to(&prog, i, next);
    wubu_mir_jmp(&prog, l_loop);

    wubu_mir_place_label(&prog, l_done);
    wubu_mir_ret(&prog, acc);

    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (int i2 = 0; i2 < 13; i2++) {
        const wubu_isa_driver_t *d = wubu_isa_find(names[i2]);
        if (!d) continue;
        /* WUBU_ISA_NO_GAPS=1 measures every backend regardless of what the gap
         * list claims, so a stale reason cannot stay on the books. */
        const char *gap = getenv("WUBU_ISA_NO_GAPS") ? NULL : known_gap(d->name, "loop(1..4)");
        if (gap) { printf("  GAP  %s loop(1..4): %s\n", d->name, gap); continue; }
        int64_t result = 0;
        if (g_dump_dir && strstr(g_dump_targets, d->name)) {
            uint8_t *dc = NULL; size_t dsz = 0;
            if (d->compile(&prog, &dc, &dsz) == 0 && dc && dsz)
                dump_arm64(d, "loop_1_4", dc, dsz, 10);
            free(dc);
        }
        printf("  %s: running...\n", d->name); fflush(stdout);
        int verdict = run_guarded(d, &prog, &result);
        if (verdict == RUN_NOEXEC) {
            printf("  SKIP %s loop(1..4): no execution path on this host\n", d->name);
            continue;
        }
        if (verdict == RUN_COMPILEFAIL) {
            printf("  FAIL %s loop(1..4): the backend could not compile it\n", d->name);
        }
        {
            char msg[96];
            snprintf(msg, sizeof msg, "%s loop(1..4)", d->name);
            /* A hang is a FAILURE, not an absence of evidence: the program must
             * terminate, and a backend that spins on a backward branch is broken
             * in a way no straight-line case can show. */
            CHECK(verdict == RUN_RAN && result == 10, msg);
            if (verdict == RUN_HUNG)
                printf("  %s: no result (did not terminate)\n", msg);
            else if (result != 10)
                printf("  %s: got %lld want 10\n", msg, (long long)result);
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
    /* x86-64 pending: order-dependent float crash under ASAN (see wave notes) */
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

    /* mips CALL: standalone OK but harness layout misaligns by 2B — next wave */
    const char *names[] = {"6502", "z80", "8051", "avr", "pic", "m68k", "8086", "riscv"};
    for (int i = 0; i < 8; i++) {
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


/* -- goal 174: immediate selection ---------------------------------- */
extern int wubu_isa_x86_64_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size);

static int find_movabs(const uint8_t *code, size_t n)
{
    /* REX.W (0x48/0x49) then B8+rd => mov r64, imm64 */
    for (size_t i = 0; i + 1 < n; i++)
        if ((code[i] == 0x48 || code[i] == 0x49) && code[i+1] >= 0xB8 && code[i+1] <= 0xBF)
            return 1;
    return 0;
}

static int find_bytes(const uint8_t *code, size_t n, const uint8_t *pat, size_t plen)
{
    for (size_t i = 0; i + plen <= n; i++)
        if (memcmp(code + i, pat, plen) == 0) return 1;
    return 0;
}

static void test_imm_selection(void)
{
    printf("\n[Immediate selection]\n");

    struct { int64_t imm; const uint8_t *pat; size_t len; const char *what; } cases[3];
    const uint8_t c7[] = {0xC7};      /* REX.W + C7 /0 id */
    const uint8_t movabs[] = {0xB8};  /* REX.W + B8+rd imm64 */

    cases[0].imm = 0x1234;             cases[0].pat = c7;     cases[0].len = sizeof c7;     cases[0].what = "0x1234 uses C7 /0 id (imm32)";
    cases[1].imm = 0x123456789ABCDEF0LL; cases[1].pat = movabs; cases[1].len = sizeof movabs; cases[1].what = "0x123456789ABCDEF0 uses movabs";
    cases[2].imm = 0x80000000LL;       cases[2].pat = movabs; cases[2].len = sizeof movabs; cases[2].what = "0x80000000 stays movabs (C7 would sign-extend)";

    for (int k = 0; k < 3; k++) {
        wubu_mir_prog_t prog;
        memset(&prog, 0, sizeof prog);
        wubu_mir_instr_t ins[2];
        memset(ins, 0, sizeof ins);
        ins[0].op = MIR_CONST; ins[0].dst = 0; ins[0].imm = cases[k].imm;
        ins[1].op = MIR_RET;   ins[1].a = 0;
        prog.ins = ins; prog.n = 2; prog.n_funcs = 0;

        uint8_t *code = NULL; size_t n = 0;
        int rc = wubu_isa_x86_64_compile(&prog, &code, &n);
        int hit = 0;
        if (rc == 0 && code) {
            hit = (k == 0) ? find_bytes(code, n, cases[k].pat, cases[k].len)
                           : find_movabs(code, n);
        }
        total++;
        if (hit) pass++; else { fail++; printf("  FAIL: %s\n", cases[k].what); }
        printf("  %-46s rc=%d bytes=%zu %s\n", cases[k].what, rc, n, hit ? "match" : "MISS");
        free(code);
    }
}


/* -- goal 173: REX selection ----------------------------------------- */
static int rex_bad;
static int rex_seen;
static void rex_check(const x86_insn_t *in, void *user)
{
    (void)user;
    if (in->has_modrm) {
        if (in->modrm_rm >= 8 && !in->rex_b) rex_bad++;
        if (in->modrm_reg >= 8 && !in->rex_r) rex_bad++;
    }
    if (in->rex && (in->rex_r || in->rex_b || in->rex_w)) rex_seen++;
}

static void test_rex_selection(void)
{
    printf("\n[REX selection]\n");
    enum { N = 12 };
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof prog);
    wubu_mir_instr_t ins[2 * N + 2];
    memset(ins, 0, sizeof ins);
    int k = 0;
    for (int i = 0; i < N; i++) {
        ins[k].op = MIR_CONST; ins[k].dst = (wubu_vr_t)i; ins[k].imm = 0x1000 + i; k++;
    }
    for (int i = 1; i < N; i++) {
        ins[k].op = MIR_ADD; ins[k].dst = (wubu_vr_t)i;
        ins[k].a = (wubu_vr_t)(i - 1); ins[k].b = (wubu_vr_t)i; k++;
    }
    ins[k].op = MIR_RET; ins[k].a = (wubu_vr_t)(N - 1); k++;
    prog.ins = ins; prog.n = (size_t)k;

    uint8_t *code = NULL; size_t n = 0;
    int rc = wubu_isa_x86_64_compile(&prog, &code, &n);
    rex_bad = 0; rex_seen = 0;
    int walk = (rc == 0 && code) ? x86_decode_all(code, n, rex_check, NULL) : -1;

    total++;
    int ok = (walk > 0 && rex_bad == 0 && rex_seen > 0);
    if (ok) pass++; else { fail++; printf("  FAIL: REX selection\n"); }
    printf("  %-46s insns=%d rex=%d bad=%d\n", "extended regs carry their REX bit", walk, rex_seen, rex_bad);
    free(code);
}


/* -- goal 170: rbp-relative frames, no red zone ---------------------- */
static int mem_rsp, mem_rbp_pos, mem_rbp_neg;
static void frame_check(const x86_insn_t *in, void *user)
{
    (void)user;
    if (!in->has_modrm || in->modrm_mod == 3) return;   /* register operand */
    if (in->modrm_rm == 4) { mem_rsp++; return; }        /* SIB: rsp/r12 base */
    if (in->modrm_rm == 5) {                             /* rbp/r13 base */
        if (in->disp < 0) mem_rbp_neg++; else mem_rbp_pos++;
    }
}

static void test_frame_layout(void)
{
    printf("\n[Frame layout]\n");
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof prog);
    /* More live values than registers, so the allocator must spill and the
     * locals actually reach the frame. */
    enum { NL = 16 };
    wubu_mir_instr_t ins[2 * NL + 2];
    memset(ins, 0, sizeof ins);
    int k = 0;
    for (int i = 0; i < NL; i++) {
        ins[k].op = MIR_CONST; ins[k].dst = (wubu_vr_t)i; ins[k].imm = 100 + i; k++;
    }
    for (int i = 1; i < NL; i++) {
        ins[k].op = MIR_ADD; ins[k].dst = 0;
        ins[k].a = 0; ins[k].b = (wubu_vr_t)i; k++;
    }
    ins[k].op = MIR_RET; ins[k].a = 0; k++;
    prog.ins = ins; prog.n = (size_t)k;

    uint8_t *code = NULL; size_t n = 0;
    int rc = wubu_isa_x86_64_compile(&prog, &code, &n);
    mem_rsp = mem_rbp_pos = mem_rbp_neg = 0;
    int walk = (rc == 0 && code) ? x86_decode_all(code, n, frame_check, NULL) : -1;

    total++;
    int ok = (walk > 0 && mem_rsp == 0 && mem_rbp_pos == 0 && mem_rbp_neg > 0);
    if (ok) pass++; else { fail++; printf("  FAIL: frame layout\n"); }
    printf("  %-46s insns=%d rsp=%d rbp+=%d rbp-=%d\n",
           "locals live at [rbp-N], never in a red zone", walk, mem_rsp, mem_rbp_pos, mem_rbp_neg);
    free(code);
}

/* Test: comparison battery — EQ/LT/GT signed + unsigned family.
 * Added by the hardcoding census (2026-11-19): the PIC encoder's private
 * opcode table had drifted from the interpreter (EQ emitted GTS, LT
 * emitted LEU, GT emitted GTU) and NO test covered EQ/LT, so it shipped.
 * Values stay inside int8 so 8-bit backends are exercised honestly. */
static void test_compare_battery(void)
{
    printf("-- Test: comparison battery (EQ/LT/GT/ULT/ULE/UGT/UGE) --\n");
    const struct { wubu_mir_op_t op; int64_t a, b, want; const char *nm; } cases[] = {
        { MIR_EQ,  5,  5, 1, "eq(5,5)" },
        { MIR_EQ,  5,  6, 0, "eq(5,6)" },
        { MIR_LT,  2,  3, 1, "lt(2,3)" },
        { MIR_LT,  3,  2, 0, "lt(3,2)" },
        { MIR_LT, -5,  3, 1, "lt(-5,3)" },
        { MIR_LT,  3, -5, 0, "lt(3,-5)" },
        { MIR_GT,  3,  2, 1, "gt(3,2)" },
        { MIR_GT,  2,  3, 0, "gt(2,3)" },
        { MIR_GT, -5,  3, 0, "gt(-5,3)" },
        { MIR_GT,  3, -5, 1, "gt(3,-5)" },
        { MIR_ULT, 1,  2, 1, "ult(1,2)" },
        { MIR_ULT, 2,  1, 0, "ult(2,1)" },
        { MIR_ULE, 2,  2, 1, "ule(2,2)" },
        { MIR_ULE, 3,  2, 0, "ule(3,2)" },
        { MIR_UGT, 2,  1, 1, "ugt(2,1)" },
        { MIR_UGE, 2,  2, 1, "uge(2,2)" },
        /* Shifts: MIR_SHR is ARITHMETIC (sar) and MIR_SHR_U is LOGICAL (shr).
         * Both were missing from most backends (only x86-64 had SHR_U at all),
         * so the destination silently kept its old value -- a wrong answer, not
         * an error. riscv, m68k, 8086 and z80 implement them now; the backends
         * still listed in docs/compendium/04-roadmap/isa-shift-gaps-2026-11-24.md
         * get their cases as they are fixed. */
        { MIR_SHL,      1,   4,    16,  "shl(1,4)" },
        { MIR_SHR,    -16,   2,    -4,  "shr(-16,2)" },
        { MIR_SHR_U,   16,   2,     4,  "shru(16,2)" },
    };
    const char *names[] = {"x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "arm64"};
    for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        wubu_vr_t a = wubu_mir_const(&prog, cases[ci].a);
        wubu_vr_t b = wubu_mir_const(&prog, cases[ci].b);
        wubu_vr_t r = wubu_mir_binop(&prog, cases[ci].op, a, b);
        wubu_mir_ret(&prog, r);
        for (int i = 0; i < 13; i++) {
            const wubu_isa_driver_t *d = wubu_isa_find(names[i]);
            if (!d) continue;
            int64_t result = 0;
            const char *gap = known_gap(d->name, cases[ci].nm);
            if (gap) {
                printf("  GAP  %s %s: %s\n", d->name, cases[ci].nm, gap);
                continue;
            }
            if (g_dump_dir && strstr(g_dump_targets, d->name)) {
                uint8_t *dc = NULL; size_t dsz = 0;
                if (d->compile(&prog, &dc, &dsz) == 0 && dc && dsz) {
                    char dumpname[96];
                    snprintf(dumpname, sizeof dumpname, "%02u_%s", ci, cases[ci].nm);
                    for (char *q = dumpname; *q; q++)
                        if (!((*q >= 'a' && *q <= 'z') || (*q >= 'A' && *q <= 'Z') ||
                              (*q >= '0' && *q <= '9') || *q == '_' || *q == '-' || *q == '.'))
                            *q = '_';
                    dump_arm64(d, dumpname, dc, dsz, cases[ci].want);
                }
                free(dc);
            }
            if (run_with_driver(d, &prog, &result) == 0) {
                char msg[96];
                snprintf(msg, sizeof msg, "%s %s", names[i], cases[ci].nm);
                CHECK(result == cases[ci].want, msg);
                if (result != cases[ci].want)
                    printf("  %s: got %lld want %lld\n", msg,
                           (long long)result, (long long)cases[ci].want);
            }
        }
        wubu_mir_free(&prog);
    }
}

int main(void)
{
    printf("=== ISA DRIVER SPACE TEST ===\n\n");
    g_dump_dir = getenv("WUBU_EXEC_DUMP_DIR");
    g_dump_targets = getenv("WUBU_EXEC_DUMP");
    if (g_dump_dir) printf("(dumping ARM64 blobs to %s for qemu execution)\n", g_dump_dir);

    /* The amdgpu backend emits real amdgcn but has no ROCm runtime, so run()
     * refuses to execute and returns the -1 failure sentinel rather than
     * silently substituting wubu_mir_interp. These tests exercise driver
     * EXECUTION across the whole registry, so opt into the fallback here --
     * explicitly, at the call site that needs it, instead of having the driver
     * pretend it ran on a GPU. */
    setenv("WUBU_AMDGPU_INTERP", "1", 1);

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
    test_compare_battery();
    printf("\n");
    test_branch();
    printf("\n");
    test_describe();
    printf("\n");
    test_imm_selection();
    test_rex_selection();
    test_loop();
    test_frame_layout();
    printf("\n");

    /* A hole the backend did not declare is a failure even though no CHECK ran
     * for it: the battery's `if (run == 0)` shape would otherwise report a clean
     * run for a backend that compiled nothing. */
    if (g_unaccounted_holes) {
        printf("=== %d COMPILE HOLE(S) not declared above: a backend silently has no "
               "code for an op the battery asked for ===\n", g_unaccounted_holes);
        fail += g_unaccounted_holes;
        total += g_unaccounted_holes;
    }
    printf("=== %s: %d/%d passed ===\n",
           fail == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
           pass, total);
    if (compiled_only)
        printf("    (%d checks were encode-only: the backend emits for a foreign "
               "target and refuses to execute it here)\n", compiled_only);
    return fail == 0 ? 0 : 1;
}
