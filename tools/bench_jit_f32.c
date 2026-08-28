/*
 * tools/bench_jit_f32.c — JIT float32 GEMM performance benchmark.
 *
 * Compiles a T_GEMM_F32 MIR program through the x86-64 JIT and measures
 * execution time. Compares against naive C and interpreter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int n2 = N * N;

    printf("=== JIT Float32 GEMM Benchmark (N=%d) ===\n\n", N);

    float *A = (float*)malloc(n2 * sizeof(float));
    float *B = (float*)malloc(n2 * sizeof(float));

    srand(42);
    for (int i = 0; i < n2; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    /* Naive C reference + timing */
    float *C_naive = (float*)calloc(n2, sizeof(float));
    double t0 = now_sec();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C_naive[i*N+j] = sum;
        }
    double t_naive = now_sec() - t0;
    printf("Naive C:  %.3f ms (%.2f GFLOPS)\n", t_naive*1000, 2.0*N*N*N/t_naive/1e9);

    /* Build MIR program */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);
    wubu_mir_tgemm_f32(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);

    /* Setup memory */
    int ncells = 3 * n2 + 1;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } ua, ub;
        ua.f = A[i];
        ub.f = B[i];
        mem[1 + i] = (int64_t)ua.i;
        mem[1 + n2 + i] = (int64_t)ub.i;
    }

    /* Interpreter timing */
    prog.mem = mem;
    t0 = now_sec();
    wubu_mir_interp(&prog);
    double t_interp = now_sec() - t0;
    printf("Interp:   %.3f ms (%.2f GFLOPS)\n", t_interp*1000, 2.0*N*N*N/t_interp/1e9);

    /* JIT compile */
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    if (!jit || !jit->compile || !jit->run) {
        printf("JIT: not available\n");
        return 1;
    }

    uint8_t *code = NULL;
    size_t code_size = 0;
    int rc = jit->compile(&prog, &code, &code_size);
    if (rc != 0 || !code) {
        printf("JIT compile failed: rc=%d\n", rc);
        return 1;
    }

    /* JIT timing */
    int iterations = (N <= 128) ? 100 : 10;
    t0 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < n2; i++) {
            union { float f; int32_t i; } ua, ub;
            ua.f = A[i];
            ub.f = B[i];
            mem[1 + i] = (int64_t)ua.i;
            mem[1 + n2 + i] = (int64_t)ub.i;
        }
        memset(mem + 1 + 2*n2, 0, n2 * sizeof(int64_t));
        jit->run(code, code_size, 0);
    }
    double t_jit = (now_sec() - t0) / iterations;
    printf("JIT:      %.3f ms (%.2f GFLOPS)\n", t_jit*1000, 2.0*N*N*N/t_jit/1e9);

    /* Verify correctness */
    double max_err = 0;
    int errors = 0;
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } r;
        r.i = (int32_t)mem[1 + 2*n2 + i];
        double err = fabs((double)r.f - (double)C_naive[i]);
        if (err > 1e-2) errors++;
        if (err > max_err) max_err = err;
    }
    printf("Errors:   %d / %d (max_err=%e)\n", errors, n2, max_err);
    printf("Speedup:  %.1fx vs naive, %.1fx vs interp\n",
           t_naive/t_jit, t_interp/t_jit);

    free(code); free(mem); free(A); free(B); free(C_naive);
    wubu_mir_free(&prog);
    return errors == 0 ? 0 : 1;
}
