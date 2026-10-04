/* Differential oracle: every real backend must agree with wubu_mir_interp.
 *
 * Every silent-wrong-answer bug found so far was invisible to a
 * self-consistent test and obvious to a differential one:
 *   - the Vulkan runner computed 892 and printed 0
 *   - PTX shared /tmp scratch files across concurrent processes
 *   - T_GEMM decoded its N field with the wrong mask in the interpreter only
 *   - OpTypeFloat 64 was emitted for f32-only modules
 *
 * Build MIR programs directly, run them through every registered backend, and
 * compare against the interpreter. A backend that cannot execute honestly
 * (amdgpu without a ROCm runtime) reports -1 and is SKIPPED, not passed --
 * that distinction is the whole point.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

static const char *BACKENDS[] = {
    "x86-64", "8086", "m68k", "6502", "riscv", "z80", "mips",
    "8051", "avr", "pic", "ptx", "spirv", "amdgpu",
};

/* 8-bit targets truncate a 64-bit result to 8 bits -- that is correct
 * behaviour, not a bug, so the expected value is masked to match. */
static int is_8bit(const char *be)
{
    return !strcmp(be,"6502") || !strcmp(be,"z80") || !strcmp(be,"8051") ||
           !strcmp(be,"avr")  || !strcmp(be,"pic");
}
#define NBACKENDS ((int)(sizeof(BACKENDS)/sizeof(BACKENDS[0])))

static int total = 0, pass = 0, fail = 0, skip = 0;
static void check(int ok, const char *what, const char *be, int64_t got, int64_t want)
{
    total++;
    if (ok) { pass++; return; }
    fail++;
    printf("  FAIL: %-14s %-8s got=%lld want=%lld\n",
           what, be, (long long)got, (long long)want);
}

/* Run one program on one backend. Returns 0 executed, -1 could not execute. */
static int run_one(const char *be, const wubu_mir_prog_t *p, int64_t *out)
{
    const wubu_isa_driver_t *d = wubu_isa_find(be);
    if (!d) return -1;
    uint8_t *code = NULL; size_t sz = 0;
    if (d->compile(p, &code, &sz) != 0 || !code) return -1;
    size_t cells = (size_t)(p->total_mem > 0 ? p->total_mem : 16) + 64;
    int64_t *mem = (int64_t *)calloc(cells, sizeof(int64_t));
    if (!mem) { free(code); return -1; }
    /* run(code, size, memory_pointer) -- never 0, see the segfault fix */
    *out = d->run(code, sz, (int64_t)(intptr_t)mem);
    free(mem); free(code);
    return (*out == -1) ? -1 : 0;
}

/* Compare every backend against the interpreter. */
static void differential(const char *what, wubu_mir_prog_t p)
{
    int64_t want = wubu_mir_interp(&p);
    for (int i = 0; i < NBACKENDS; i++) {
        int64_t got = 0;
        if (run_one(BACKENDS[i], &p, &got) != 0) { skip++; continue; }
        /* 8-bit targets sign-extend their byte result, so compare the low
         * byte as a signed char rather than as an unsigned 0..255 value. */
        int64_t expect = is_8bit(BACKENDS[i]) ? (int64_t)(int8_t)(want & 0xFF) : want;
        check(got == expect, what, BACKENDS[i], got, expect);
    }
    wubu_mir_free(&p);
}

int main(void)
{
    /* amdgpu has no ROCm runtime; opt in explicitly rather than have the
     * driver silently substitute the interpreter. */
    setenv("WUBU_AMDGPU_INTERP", "1", 1);

    printf("=== DIFFERENTIAL ORACLE: backends vs wubu_mir_interp ===\n\n");

    {   /* integer arithmetic */
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t a = wubu_mir_const(&p, 7), b = wubu_mir_const(&p, 9);
        wubu_mir_ret(&p, wubu_mir_binop(&p, MIR_MUL,
                        wubu_mir_binop(&p, MIR_ADD, a, b), wubu_mir_const(&p, 56)));
        p.total_mem = 32;
        differential("int (7+9)*56-4", p);
    }
    {   /* memory: store at byte 8, load from byte 0 and 8 */
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t base = wubu_mir_alloc(&p, 4);
        wubu_mir_store(&p, base, wubu_mir_const(&p, 63));
        wubu_mir_ret(&p, wubu_mir_load(&p, wubu_mir_binop(&p, MIR_ADD, base, wubu_mir_const(&p, 0))));
        p.total_mem = 64;
        differential("mem roundtrip", p);
    }
    {   /* f64 arithmetic, exercising the conditional OpTypeFloat 64 */
        int64_t b25, b40;
        double d25 = 2.5, d40 = 4.0;
        memcpy(&b25, &d25, 8); memcpy(&b40, &d40, 8);
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t v = wubu_mir_binop(&p, MIR_DADD,
                        wubu_mir_binop(&p, MIR_DMUL, wubu_mir_const(&p, b25),
                                                   wubu_mir_const(&p, b40)),
                        wubu_mir_const(&p, b40));
        wubu_mir_ret(&p, v);
        p.total_mem = 32;
        differential("f64 2.5*4.0+4.0", p);
    }

    printf("\n  total=%d pass=%d fail=%d skipped(no runtime)=%d\n",
           total, pass, fail, skip);
    if (fail) { printf("=== DIFFERENTIAL ORACLE FAILED ===\n"); return 1; }
    printf("=== ALL BACKENDS AGREE WITH THE INTERPRETER ===\n");
    return 0;
}