/*
 * tools/bench_all.c — Cross-backend benchmark harness.
 * Measures GEMM performance across JIT and interpreted backends.
 */
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

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    printf("=== WuBuNOS Cross-Backend Benchmark (N=%d) ===\n\n", N);

    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    int64_t n2 = (int64_t)N * N;
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);
    wubu_mir_tgemm(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);

    int ncells = (int)(3 * n2 + 10);
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));
    srand(42);
    for (int64_t i = 0; i < n2; i++) {
        union { float f; int32_t i; } u;
        u.f = (float)(rand() % 100) / 100.0f;
        mem[1 + i] = (int64_t)u.i;
        mem[1 + n2 + i] = (int64_t)u.i;
    }
    prog.mem = mem;

    printf("%-12s %10s %10s\n", "Backend", "GFLOPS", "Time(ms)");
    printf("------------------------------------\n");

    /* Test x86-64 JIT */
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    if (jit && jit->compile && jit->run) {
        uint8_t *code = NULL; size_t code_size = 0;
        if (jit->compile(&prog, &code, &code_size) == 0 && code) {
            /* Warmup */
            for (int w = 0; w < 2; w++) {
                for (int64_t j = 0; j < n2; j++) {
                    union { float f; int32_t i; } u;
                    u.f = (float)(rand() % 100) / 100.0f;
                    mem[1 + j] = (int64_t)u.i;
                    mem[1 + n2 + j] = (int64_t)u.i;
                }
                jit->run(code, code_size, 0);
            }
            int runs = 5;
            double t0 = now_sec();
            for (int r = 0; r < runs; r++) {
                for (int64_t j = 0; j < n2; j++) {
                    union { float f; int32_t i; } u;
                    u.f = (float)(rand() % 100) / 100.0f;
                    mem[1 + j] = (int64_t)u.i;
                    mem[1 + n2 + j] = (int64_t)u.i;
                }
                jit->run(code, code_size, 0);
            }
            double t1 = now_sec();
            double ms = (t1 - t0) / runs * 1000.0;
            double gflops = 2.0 * (double)N * N * N / (ms / 1000.0) / 1e9;
            printf("%-12s %10.2f %10.3f\n", "x86-64", gflops, ms);
            free(code);
        }
    }

    /* Test interpreter */
    {
        /* Warmup */
        for (int w = 0; w < 2; w++) {
            for (int64_t j = 0; j < n2; j++) {
                union { float f; int32_t i; } u;
                u.f = (float)(rand() % 100) / 100.0f;
                mem[1 + j] = (int64_t)u.i;
                mem[1 + n2 + j] = (int64_t)u.i;
            }
            wubu_mir_interp(&prog);
        }
        int runs = 5;
        double t0 = now_sec();
        for (int r = 0; r < runs; r++) {
            for (int64_t j = 0; j < n2; j++) {
                union { float f; int32_t i; } u;
                u.f = (float)(rand() % 100) / 100.0f;
                mem[1 + j] = (int64_t)u.i;
                mem[1 + n2 + j] = (int64_t)u.i;
            }
            wubu_mir_interp(&prog);
        }
        double t1 = now_sec();
        double ms = (t1 - t0) / runs * 1000.0;
        double gflops = 2.0 * (double)N * N * N / (ms / 1000.0) / 1e9;
        printf("%-12s %10.2f %10.3f\n", "interpreter", gflops, ms);
    }

    free(mem);
    wubu_mir_free(&prog);
    return 0;
}
