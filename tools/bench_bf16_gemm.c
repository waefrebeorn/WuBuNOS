#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "wubu_tgemm.h"

static float bf16_to_fp32(uint16_t b) {
    uint32_t v = (uint32_t)b << 16;
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static uint16_t fp32_to_bf16(float f) {
    uint32_t v;
    memcpy(&v, &f, 4);
    return (uint16_t)(v >> 16);
}

int main() {
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int nsizes = 6;

    printf("=== WuBuNOS BF16 GEMM Benchmark ===\n");
    printf("VDPBF16PS: 32 BF16 FMAs -> 16 FP32 accumulates per instruction\n\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        uint16_t *A = (uint16_t*)aligned_alloc(64, (size_t)N * N * 2);
        uint16_t *B = (uint16_t*)aligned_alloc(64, (size_t)N * N * 2);
        float *C = (float*)aligned_alloc(64, (size_t)N * N * 4);

        for (int i = 0; i < N*N; i++) {
            A[i] = fp32_to_bf16((float)(i % 50) / 50.0f);
            B[i] = fp32_to_bf16((float)(i % 50) / 50.0f);
        }
        memset(C, 0, (size_t)N * N * 4);

        /* Warmup */
        wubu_tgemm_bf16_avx512(A, B, C, N, N, N);

        /* Timed run */
        int ntrials = 3;
        double best_ms = 1e9;
        for (int t = 0; t < ntrials; t++) {
            memset(C, 0, (size_t)N * N * 4);
            double t0 = omp_get_wtime();
            wubu_tgemm_bf16_avx512(A, B, C, N, N, N);
            double t1 = omp_get_wtime();
            double ms = (t1 - t0) * 1000.0;
            if (ms < best_ms) best_ms = ms;
        }

        double gflops = (2.0 * N * N * N) / (best_ms / 1000.0) / 1e9;

        /* Correctness check on first row */
        double maxerr = 0;
        for (int j = 0; j < N; j++) {
            float acc = 0;
            for (int k = 0; k < N; k++)
                acc += bf16_to_fp32(A[k]) * bf16_to_fp32(B[(size_t)k * N + j]);
            double e = fabs((double)C[j] - (double)acc);
            if (e > maxerr) maxerr = e;
        }

        printf("%-6d  %10.3f ms  %8.1f GFLOPS  maxerr=%f %s\n",
               N, best_ms, gflops, maxerr, maxerr > 1.0 ? "WARN" : "OK");

        free(A); free(B); free(C);
    }

    printf("\n=== BF16 Benchmark Complete ===\n");
    return 0;
}
