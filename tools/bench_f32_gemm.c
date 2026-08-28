/*
 * tools/bench_f32_gemm.c — Float32 GEMM benchmark.
 *
 * Compares our float32 GEMM kernel against naive C.
 * Measures GFLOPS at various sizes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "wubu_tgemm.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void gemm_naive_f32(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

int main(void) {
    printf("=== WuBuNOS Float32 GEMM Benchmark ===\n\n");

#if defined(__AVX2__)
    printf("AVX2: YES\n");
#else
    printf("AVX2: NO (scalar fallback)\n");
#endif
#if defined(__FMA__)
    printf("FMA:  YES\n");
#else
    printf("FMA:  NO\n");
#endif
    printf("\n");

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-8s %-12s %-12s %-10s %-8s\n", "Size", "Naive(ms)", "WuBu(ms)", "Speedup", "GFLOPS");
    printf("-----------------------------------------------------\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        int M = N, K = N;
        size_t n2 = (size_t)N * N;

        float *A = (float*)malloc(n2 * sizeof(float));
        float *B = (float*)malloc(n2 * sizeof(float));
        float *C_naive = (float*)malloc(n2 * sizeof(float));
        float *C_wubu = (float*)malloc(n2 * sizeof(float));

        if (!A || !B || !C_naive || !C_wubu) {
            printf("%-8d OOM\n", N);
            free(A); free(B); free(C_naive); free(C_wubu);
            continue;
        }

        srand(42);
        for (size_t i = 0; i < n2; i++) {
            A[i] = (float)(rand() % 100) / 100.0f;
            B[i] = (float)(rand() % 100) / 100.0f;
        }

        /* Naive */
        memset(C_naive, 0, n2 * sizeof(float));
        double t0 = now_sec();
        gemm_naive_f32(A, B, C_naive, M, N, K);
        double t_naive = now_sec() - t0;

        /* WuBuNOS */
        memset(C_wubu, 0, n2 * sizeof(float));
        t0 = now_sec();
        wubu_tgemm_f32(A, B, C_wubu, M, N, K);
        double t_wubu = now_sec() - t0;

        /* Verify */
        double max_err = 0;
        for (size_t i = 0; i < n2; i++) {
            double err = fabs((double)C_naive[i] - (double)C_wubu[i]);
            if (err > max_err) max_err = err;
        }

        double speedup = t_naive / t_wubu;
        double gflops = 2.0 * (double)M * N * K / t_wubu / 1e9;

        printf("%-8d %-12.3f %-12.3f %-10.2fx %-8.1f\n",
               N, t_naive*1000, t_wubu*1000, speedup, gflops);

        if (max_err > 1e-2) {
            printf("  WARN: max error = %f\n", max_err);
        }

        free(A); free(B); free(C_naive); free(C_wubu);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
