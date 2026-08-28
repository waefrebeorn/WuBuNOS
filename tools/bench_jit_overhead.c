#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    int N = 256;
    int n2 = N * N;
    
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);
    wubu_mir_tgemm_f32(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);
    
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    if (!jit || !jit->compile || !jit->run) {
        printf("JIT not available\n");
        return 1;
    }
    
    /* Measure compile time */
    double t0 = now_sec();
    uint8_t *code = NULL;
    size_t code_size = 0;
    int rc = jit->compile(&prog, &code, &code_size);
    double t_compile = now_sec() - t0;
    printf("JIT compile: %.3f ms (rc=%d, code_size=%zu bytes)\n", t_compile*1000, rc, code_size);
    
    /* Setup memory */
    int ncells = 3 * n2 + 1;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));
    srand(42);
    for (int i = 0; i < n2; i++) {
        union { float f; int32_t i; } ua, ub;
        ua.f = (float)(rand() % 100) / 100.0f;
        ub.f = (float)(rand() % 100) / 100.0f;
        mem[1 + i] = (int64_t)ua.i;
        mem[1 + n2 + i] = (int64_t)ub.i;
    }
    prog.mem = mem;
    
    /* Measure single-run time */
    t0 = now_sec();
    jit->run(code, code_size, 0);
    double t_run = now_sec() - t0;
    printf("JIT single run: %.3f ms (%.1f GFLOPS)\n", t_run*1000, 2.0*N*N*N/t_run/1e9);
    
    /* Measure interp time for comparison */
    t0 = now_sec();
    wubu_mir_interp(&prog);
    double t_interp = now_sec() - t0;
    printf("Interp: %.3f ms (%.1f GFLOPS)\n", t_interp*1000, 2.0*N*N*N/t_interp/1e9);
    printf("JIT/Interp ratio: %.2fx\n", t_interp / t_run);
    
    free(code); free(mem);
    wubu_mir_free(&prog);
    return 0;
}
