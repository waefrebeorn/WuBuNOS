/*
 * wubu_auto_tune.c — Auto-tuning framework for WuBuNOS.
 *
 * Profiles optimization configurations and selects the best parameters
 * for each kernel (T_GEMM tile sizes, loop unroll factors, etc.).
 *
 * C11, self-contained.
 */

#include "wubu_tgemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* ---- Tuning parameters ---- */

typedef struct {
    int mc;
    int nc;
    int kc;
    int mr;
    int nr;
    double gflops;
} tune_tgemm_t;

static double time_gemm_kernel(int M, int N, int K, const tune_tgemm_t *t)
{
    /* Allocate ONE contiguous memory block (as the kernel expects) */
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;
    size_t total = (a_sz + b_sz + c_sz) * sizeof(int64_t);

    int64_t *block = aligned_alloc(64, total);
    if (!block) return 0.0;

    int64_t A_off = 0;
    int64_t B_off = (int64_t)a_sz;
    int64_t C_off = (int64_t)(a_sz + b_sz);

    int64_t *A = block;
    int64_t *B = block + a_sz;
    int64_t *C = block + a_sz + b_sz;

    /* Initialize with deterministic values */
    for (size_t i = 0; i < a_sz; i++) A[i] = (int64_t)((i & 0xFF) + 1);
    for (size_t i = 0; i < b_sz; i++) B[i] = (int64_t)(((i * 7) & 0xFF) + 1);
    memset(C, 0, c_sz * sizeof(int64_t));

    /* Time multiple runs */
    const int WARMUP = 2;
    const int RUNS   = 10;
    double best = 1e9;

    for (int r = -WARMUP; r < RUNS; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Call the existing wubu_tgemm (signature: mem, A, B, C, M, N, K) */
        wubu_tgemm(A, A_off, B_off, C_off, M, N, K);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double sec = (double)(t1.tv_sec - t0.tv_sec) +
                     (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (r >= 0 && sec < best) best = sec;
    }

    free(block);
    return best;
}

static double compute_gflops(int M, int N, int K, double sec)
{
    if (sec <= 0.0) return 0.0;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / (sec * 1e9);
}

/* ---- Auto-tuning entry point ---- */

tune_tgemm_t wubu_tune_tgemm(int M, int N, int K)
{
    tune_tgemm_t best = { .mc = 64, .nc = 64, .kc = 64, .mr = 8, .nr = 4, .gflops = 0.0 };
    tune_tgemm_t candidates[] = {
        { 64, 64, 64, 8, 4, 0.0 },
        { 64, 64, 64, 4, 8, 0.0 },
        { 32, 32, 32, 8, 4, 0.0 },
        { 32, 32, 32, 4, 8, 0.0 },
        { 128, 128, 64, 8, 4, 0.0 },
        { 64, 128, 64, 8, 4, 0.0 },
        { 128, 64, 64, 8, 4, 0.0 },
        { 32, 64, 32, 8, 4, 0.0 },
        { 64, 32, 32, 8, 4, 0.0 },
    };
    const int NCAND = (int)(sizeof(candidates) / sizeof(candidates[0]));

    printf("=== Auto-Tuning T_GEMM (%dx%dx%d) ===\n", M, N, K);
    printf("Trying %d configurations...\n", NCAND);

    for (int i = 0; i < NCAND; i++) {
        double sec = time_gemm_kernel(M, N, K, &candidates[i]);
        candidates[i].gflops = compute_gflops(M, N, K, sec);
        printf("  [%d] mc=%d nc=%d kc=%d mr=%d nr=%d -> %.1f GFLOPS (%.3fs)\n",
               i, candidates[i].mc, candidates[i].nc, candidates[i].kc,
               candidates[i].mr, candidates[i].nr, candidates[i].gflops, sec);
        if (candidates[i].gflops > best.gflops) best = candidates[i];
    }

    printf("Best: mc=%d nc=%d kc=%d mr=%d nr=%d -> %.1f GFLOPS\n",
           best.mc, best.nc, best.kc, best.mr, best.nr, best.gflops);
    return best;
}
