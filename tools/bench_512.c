
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
    int N = 512;
    float *A = (float*)aligned_alloc(64, (size_t)N*N*4);
    float *B = (float*)aligned_alloc(64, (size_t)N*N*4);
    float *C = (float*)aligned_alloc(64, (size_t)N*N*4);
    
    for (int i = 0; i < N*N; i++) {
        A[i] = (float)(i % 17) / 17.0f;
        B[i] = (float)(i % 13) / 13.0f;
        C[i] = 0;
    }
    
    /* Extended warmup: 20 iterations to let CPU freq stabilize */
    for (int w = 0; w < 20; w++) {
        memset(C, 0, (size_t)N*N*4);
        wubu_tgemm_f32(A, B, C, N, N, N);
    }
    
    /* 10 measurements */
    double times[10];
    for (int iter = 0; iter < 10; iter++) {
        memset(C, 0, (size_t)N*N*4);
        double t0 = now_sec();
        wubu_tgemm_f32(A, B, C, N, N, N);
        double t1 = now_sec();
        times[iter] = t1 - t0;
        printf("  iter %2d: %.3f ms  (%.1f GFLOPS)\n", iter, times[iter]*1000, 2.0*N*N*N/(times[iter]*1e9));
    }
    
    double sum = 0, min_t = times[0], max_t = times[0];
    for (int i = 0; i < 10; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }
    printf("\n  avg=%.3f ms  min=%.3f ms  max=%.3f ms\n", sum/10*1000, min_t*1000, max_t*1000);
    printf("  avg GFLOPS=%.1f  peak GFLOPS=%.1f\n", 2.0*N*N*N/(sum/10*1e9), 2.0*N*N*N/(min_t*1e9));
    
    free(A); free(B); free(C);
    return 0;
}
