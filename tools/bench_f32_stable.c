
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

int main() {
    int sizes[] = {128, 256, 512, 1024};
    int nsizes = 4;
    
    printf("=== WuBuNOS f32 GEMM (warmup + 5 iters, median) ===\n\n");
    
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        float *A = (float*)aligned_alloc(64, (size_t)N*N*4);
        float *B = (float*)aligned_alloc(64, (size_t)N*N*4);
        float *C = (float*)aligned_alloc(64, (size_t)N*N*4);
        
        for (int i = 0; i < N*N; i++) {
            A[i] = (float)(i % 17) / 17.0f;
            B[i] = (float)(i % 13) / 13.0f;
            C[i] = 0;
        }
        
        /* Warmup */
        wubu_tgemm_f32(A, B, C, N, N, N);
        
        /* 5 iterations */
        double times[5];
        for (int iter = 0; iter < 5; iter++) {
            memset(C, 0, (size_t)N*N*4);
            double t0 = now_sec();
            wubu_tgemm_f32(A, B, C, N, N, N);
            double t1 = now_sec();
            times[iter] = t1 - t0;
        }
        
        /* Sort and take median */
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 5; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
        
        double median = times[2];
        double gflops = 2.0 * N * N * N / (median * 1e9);
        
        printf("N=%4d  median=%.3fms  GFLOPS=%.1f  [", N, median*1000, gflops);
        for (int i = 0; i < 5; i++) printf("%.1f ", times[i]*1000);
        printf("]\n");
        
        free(A); free(B); free(C);
    }
    
    return 0;
}
