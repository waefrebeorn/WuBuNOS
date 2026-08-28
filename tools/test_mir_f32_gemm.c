#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 64;
    int n2 = N * N;

    float *A = (float*)malloc(n2 * sizeof(float));
    float *B = (float*)malloc(n2 * sizeof(float));

    srand(42);
    for (int i = 0; i < n2; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *C_naive = (float*)calloc(n2, sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C_naive[i*N+j] = sum;
        }

    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);
    wubu_mir_tgemm_f32(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);

    int ncells = 3 * n2 + 1;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } ua, ub;
        ua.f = A[i]; ub.f = B[i];
        mem[1 + i] = (int64_t)ua.i;
        mem[1 + n2 + i] = (int64_t)ub.i;
    }

    /* Interpreter */
    prog.mem = mem;
    wubu_mir_interp(&prog);

    double max_err = 0; int errors = 0;
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } r;
        r.i = (int32_t)mem[1 + 2*n2 + i];
        double err = fabs((double)r.f - (double)C_naive[i]);
        if (err > 1e-2) errors++;
        if (err > max_err) max_err = err;
    }
    printf("Interpreter: %d errors (max=%e) %s\n", errors, max_err, errors==0?"PASS":"FAIL");

    /* JIT */
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } ua, ub;
        ua.f = A[i]; ub.f = B[i];
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

    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    if (!jit || !jit->compile || !jit->run) { printf("JIT unavailable\n"); return 1; }

    uint8_t *code = NULL; size_t code_size = 0;
    int rc = jit->compile(&prog2, &code, &code_size);
    if (rc != 0) { printf("JIT compile failed: rc=%d\n", rc); return 1; }

    jit->run(code, code_size, 0);

    max_err = 0; errors = 0;
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } r;
        r.i = (int32_t)mem[1 + 2*n2 + i];
        double err = fabs((double)r.f - (double)C_naive[i]);
        if (err > 1e-2) errors++;
        if (err > max_err) max_err = err;
    }
    printf("JIT:        %d errors (max=%e) %s\n", errors, max_err, errors==0?"PASS":"FAIL");

    free(code); free(mem); free(A); free(B); free(C_naive);
    wubu_mir_free(&prog); wubu_mir_free(&prog2);
    return (errors == 0) ? 0 : 1;
}
