/*
 * tools/bench_speed.c — Speed benchmark for WuBuNOS.
 *
 * Measures MIR interpreter performance on matrix multiply.
 * Uses the HolyD→MIR→interpreter pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Build a HolyD program that computes matrix multiply result */
static void build_matmul_program(char *buf, int bufsize, int N) {
    snprintf(buf, bufsize,
        "int N = %d;\n"
        "int A[%d];\n"
        "int B[%d];\n"
        "int C[%d];\n"
        "\n"
        "int i = 0;\n"
        "while (i < N) {\n"
        "  int j = 0;\n"
        "  while (j < N) {\n"
        "    A[i*N+j] = (i*N+j) %% 10;\n"
        "    B[i*N+j] = (i*N+j*3+1) %% 10;\n"
        "    j++;\n"
        "  }\n"
        "  i++;\n"
        "}\n"
        "\n"
        "i = 0;\n"
        "while (i < N) {\n"
        "  int j = 0;\n"
        "  while (j < N) {\n"
        "    int s = 0;\n"
        "    int k = 0;\n"
        "    while (k < N) {\n"
        "      s = s + A[i*N+k] * B[k*N+j];\n"
        "      k++;\n"
        "    }\n"
        "    C[i*N+j] = s;\n"
        "    j++;\n"
        "  }\n"
        "  i++;\n"
        "}\n"
        "\n"
        "C[0];\n"
        , N, N*N, N*N, N*N);
}

/* Naive int64 GEMM for verification */
static int64_t gemm_naive(int N, int64_t *A, int64_t *B) {
    int64_t result = 0;
    for (int k = 0; k < N; k++)
        result += A[k] * B[k*N];
    return result;
}

int main(void) {
    printf("=== WuBuNOS Speed Benchmark ===\n\n");

    int sizes[] = {4, 8, 16, 32, 64};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    printf("%-6s %-12s %-12s %-8s\n", "Size", "Naive(ms)", "MIR(ms)", "Correct");
    printf("----------------------------------------------\n");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        int n2 = N * N;

        char src[8192];
        build_matmul_program(src, sizeof(src), N);

        int64_t *A = (int64_t*)malloc((size_t)n2 * sizeof(int64_t));
        int64_t *B = (int64_t*)malloc((size_t)n2 * sizeof(int64_t));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i*N+j] = (int64_t)((i*N+j) % 10);
                B[i*N+j] = (int64_t)((i*N+j*3+1) % 10);
            }
        int64_t expected = gemm_naive(N, A, B);

        double t0 = now_sec();
        volatile int64_t naive_result = gemm_naive(N, A, B);
        double t_naive = now_sec() - t0;
        (void)naive_result;

        double t_interp = 0;
        int64_t mir_result = 0;
        int rc = -1;

        int iterations = (N <= 16) ? 100 : 10;
        t0 = now_sec();
        for (int iter = 0; iter < iterations; iter++) {
            wubu_mir_prog_t prog;
            memset(&prog, 0, sizeof(prog));
            rc = hd_build_mir(src, &prog);
            if (rc == 0) {
                mir_result = hd_run_prog(&prog, NULL);
            }
            wubu_mir_free(&prog);
        }
        t_interp = (now_sec() - t0) / iterations;

        int correct = (mir_result == expected);
        printf("%-6d %-12.4f %-12.4f %-8s\n",
               N, t_naive*1000, t_interp*1000,
               correct ? "YES" : "NO");

        if (!correct) {
            printf("  expected=%lld got=%lld\n", (long long)expected, (long long)mir_result);
        }

        free(A);
        free(B);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
