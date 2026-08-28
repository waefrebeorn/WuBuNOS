/*
 * tools/test_mir_f32_gemm.c — Test float32 GEMM through the JIT pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

int main(void) {
    printf("=== MIR Float32 GEMM Test ===\n\n");

    int N = 32;
    int n2 = N * N;

    /* Initialize matrices */
    float *A = (float*)malloc(n2 * sizeof(float));
    float *B = (float*)malloc(n2 * sizeof(float));

    srand(42);
    for (int i = 0; i < n2; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    /* Naive reference */
    float *C_naive = (float*)calloc(n2, sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C_naive[i*N+j] = sum;
        }

    /* Build MIR program */
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);

    /* Allocate: each float in one int64 cell (lower 32 bits).
     * wubu_mir_alloc returns VR containing base cell index. */
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);

    wubu_mir_tgemm_f32(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);

    /* Setup memory: store floats in lower 32 bits of int64 cells */
    int ncells = 3 * n2 + 1;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } ua, ub;
        ua.f = A[i];
        ub.f = B[i];
        mem[1 + i] = (int64_t)ua.i;
        mem[1 + n2 + i] = (int64_t)ub.i;
    }

    /* Run via interpreter */
    prog.mem = mem;
    wubu_mir_interp(&prog);

    /* Read result */
    union { float f; int32_t i; } res;
    res.i = (int32_t)mem[1 + 2*n2];
    float diff = fabsf(res.f - C_naive[0]);

    printf("Interpreter: C[0] = %.6f (expected %.6f, diff=%e) %s\n",
           res.f, C_naive[0], diff, diff < 1e-3 ? "PASS" : "FAIL");

    /* Run via x86-64 JIT */
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    int jit_pass = 0;
    if (jit && jit->compile && jit->run) {
        /* Reset memory */
        for (int i = 0; i < n2; i++) {
            union { float f; int32_t i; } ua, ub;
            ua.f = A[i];
            ub.f = B[i];
            mem[1 + i] = (int64_t)ua.i;
            mem[1 + n2 + i] = (int64_t)ub.i;
        }
        memset(mem + 1 + 2*n2, 0, n2 * sizeof(int64_t));

        wubu_mir_prog_t prog2;
        wubu_mir_init(&prog2);
        wubu_vr_t a2 = wubu_mir_alloc(&prog2, n2);
        wubu_vr_t b2 = wubu_mir_alloc(&prog2, n2);
        wubu_vr_t c2 = wubu_mir_alloc(&prog2, n2);
        wubu_mir_tgemm_f32(&prog2, a2, b2, c2, N, N, N);
        wubu_vr_t r2 = wubu_mir_load(&prog2, c2);
        wubu_mir_ret(&prog2, r2);
        prog2.mem = mem;

        uint8_t *code = NULL;
        size_t code_size = 0;
        int rc = jit->compile(&prog2, &code, &code_size);
        if (rc == 0 && code) {
            jit->run(code, code_size, 0);
            union { float f; int32_t i; } res_jit;
            res_jit.i = (int32_t)mem[1 + 2*n2];
            float jit_diff = fabsf(res_jit.f - C_naive[0]);
            printf("JIT:         C[0] = %.6f (expected %.6f, diff=%e) %s\n",
                   res_jit.f, C_naive[0], jit_diff,
                   jit_diff < 1e-3 ? "PASS" : "FAIL");
            jit_pass = (jit_diff < 1e-3);
            free(code);
        } else {
            printf("JIT: compile failed (rc=%d)\n", rc);
        }
        wubu_mir_free(&prog2);
    } else {
        printf("JIT: not available\n");
    }

    free(mem);
    free(A);
    free(B);
    free(C_naive);
    wubu_mir_free(&prog);

    int pass = (diff < 1e-3) && (jit_pass || !jit);
    printf("\n=== %s ===\n", pass ? "ALL PASS" : "FAILURES");
    return pass ? 0 : 1;
}
