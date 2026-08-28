#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Reference: dot product of two N-element vectors */
static float ref_dot(const float *a, const float *b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    printf("=== Dot Product Benchmark (N=%d) ===\n\n", N);

    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        a[i] = (float)(rand() % 100) / 100.0f;
        b[i] = (float)(rand() % 100) / 100.0f;
    }

    /* Reference result */
    float ref_result = ref_dot(a, b, N);
    printf("Reference result: %.6f\n\n", ref_result);

    /* Build MIR program for dot product:
     * sum = 0
     * for i in 0..N: sum += a[i] * b[i]
     * return sum
     *
     * MIR doesn't have loops, so we emit N iterations unrolled.
     * For N=1024 this is too large. Instead, use a smaller N for MIR
     * and compare against the same N for reference. */

    int N_mir = 16; /* Small enough to unroll */

    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);

    /* Allocate memory for vectors and accumulator */
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, N_mir);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, N_mir);
    wubu_vr_t addr_sum = wubu_mir_alloc(&prog, 1);

    /* Initialize sum = 0 */
    wubu_vr_t zero = wubu_mir_const(&prog, 0);
    wubu_mir_store(&prog, addr_sum, zero);

    /* Unrolled dot product */
    for (int i = 0; i < N_mir; i++) {
        wubu_vr_t addr_ai = wubu_mir_binop(&prog, MIR_ADD, addr_a, wubu_mir_const(&prog, i));
        wubu_vr_t addr_bi = wubu_mir_binop(&prog, MIR_ADD, addr_b, wubu_mir_const(&prog, i));
        wubu_vr_t ai = wubu_mir_load(&prog, addr_ai);
        wubu_vr_t bi = wubu_mir_load(&prog, addr_bi);
        wubu_vr_t prod = wubu_mir_binop(&prog, MIR_MUL, ai, bi);
        wubu_vr_t sum_val = wubu_mir_load(&prog, addr_sum);
        wubu_vr_t new_sum = wubu_mir_binop(&prog, MIR_ADD, sum_val, prod);
        wubu_mir_store(&prog, addr_sum, new_sum);
    }

    wubu_vr_t result_vr = wubu_mir_load(&prog, addr_sum);
    wubu_mir_ret(&prog, result_vr);

    /* Setup memory */
    int ncells = 3 * N_mir + 10;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));

    /* Store test data */
    float *a_small = (float*)malloc(N_mir * sizeof(float));
    float *b_small = (float*)malloc(N_mir * sizeof(float));
    for (int i = 0; i < N_mir; i++) {
        a_small[i] = a[i];
        b_small[i] = b[i];
        union { float f; int32_t i; } ua, ub;
        ua.f = a_small[i]; ub.f = b_small[i];
        mem[1 + i] = (int64_t)ua.i;
        mem[1 + N_mir + i] = (int64_t)ub.i;
    }
    prog.mem = mem;

    /* Interpreter timing */
    int runs = 10000;
    double t0 = now_sec();
    float interp_result = 0;
    for (int r = 0; r < runs; r++) {
        interp_result = (float)wubu_mir_interp(&prog);
    }
    double t_interp = (now_sec() - t0) / runs * 1e6;

    /* Reference timing for same N */
    t0 = now_sec();
    float ref_small = 0;
    for (int r = 0; r < runs; r++) {
        ref_small = ref_dot(a_small, b_small, N_mir);
    }
    double t_ref = (now_sec() - t0) / runs * 1e6;

    printf("N=%d (unrolled)\n", N_mir);
    printf("  Reference:  %.6f (%.3f us)\n", ref_small, t_ref);
    printf("  Interpreter: %.6f (%.3f us)\n", interp_result, t_interp);
    printf("  Ratio: %.1fx\n", t_interp / t_ref);

    /* JIT timing */
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    if (jit && jit->compile && jit->run) {
        uint8_t *code = NULL; size_t code_size = 0;
        if (jit->compile(&prog, &code, &code_size) == 0 && code) {
            /* Warmup */
            for (int w = 0; w < 100; w++) jit->run(code, code_size, 0);

            t0 = now_sec();
            int jit_runs = 100000;
            for (int r = 0; r < jit_runs; r++) {
                jit->run(code, code_size, 0);
            }
            double t_jit = (now_sec() - t0) / jit_runs * 1e6;
            printf("  JIT:        %.6f (%.3f us)\n", (float)(int32_t)mem[1], t_jit);
            printf("  JIT speedup vs interp: %.1fx\n", t_interp / t_jit);
            printf("  JIT speedup vs ref: %.1fx\n", t_ref / t_jit);
            free(code);
        }
    }

    free(a); free(b); free(a_small); free(b_small); free(mem);
    wubu_mir_free(&prog);
    return 0;
}
