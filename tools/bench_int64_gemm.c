#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "wubu_tgemm.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int sizes[] = {32, 64, 128, 256, 512};
    int nsizes = 5;
    
    printf("=== WuBuNOS int64 GEMM Benchmark ===\n\n");
    
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        int64_t *mem = (int64_t*)aligned_alloc(64, (size_t)3 * N * N * sizeof(int64_t));
        int64_t *A = &mem[0];
        int64_t *B = &mem[(size_t)N * N];
        int64_t *C = &mem[(size_t)2 * N * N];
        
        for (int i = 0; i < (size_t)N*N; i++) {
            A[i] = (int64_t)(i % 17);
            B[i] = (int64_t)(i % 13);
            C[i] = 0;
        }
        
        extern void tgemm_avx512(int64_t *mem, int64_t A, int64_t B, int64_t C, int M, int N, int K);
        
        /* Warmup */
        tgemm_avx512(mem, 0, (int64_t)N*N, (int64_t)2*N*N, N, N, N);
        
        double t0 = now_sec();
        int iterations = 5;
        for (int iter = 0; iter < iterations; iter++) {
            memset(C, 0, (size_t)N*N*8);
            tgemm_avx512(mem, 0, (int64_t)N*N, (int64_t)2*N*N, N, N, N);
        }
        double t1 = now_sec();
        
        double avg = (t1 - t0) / iterations;
        double gflops = 2.0 * N * N * N / (avg * 1e9);
        
        printf("N=%4d  avg=%.3fms  GFLOPS=%.1f\n", N, avg*1000, gflops);
        free(mem);
    }
    return 0;
}
