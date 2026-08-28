#include "wubu_mir.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
extern int64_t wubu_mir_interp(const wubu_mir_prog_t *p);
int main(void) {
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    int iters = 10000000;
    wubu_vr_t vr1 = wubu_mir_const(&prog, 1000000);
    wubu_vr_t vr2 = wubu_mir_const(&prog, 1);
    wubu_mir_label(&prog, 0);
    wubu_vr_t vr3 = wubu_mir_add(&prog, vr1, vr2);
    wubu_mir_mov(&prog, vr1, vr3);
    wubu_mir_jmp(&prog, 0);
    /* unreachable, but keep ret for safety */
    wubu_mir_ret(&prog, vr1);
    wubu_mir_finalize(&prog);
    printf("n_ops = %zu\n", prog.n);
    clock_t start = clock();
    int64_t r = wubu_mir_interp(&prog);
    clock_t end = clock();
    double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Result: %lld (expected %d)\n", (long long)r, iters);
    printf("Time: %.3f ms for %d dynamic ops\n", elapsed_ms, iters * 5);
    printf("Throughput: %.1f M ops/sec\n", (double)(iters * 5) / elapsed_ms);
    wubu_mir_free(&prog);

    /* Naive */
    start = clock();
    volatile int64_t s = 0;
    for (int i = 0; i < iters; i++) s += 1;
    end = clock();
    double naive_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Naive: %.6f ms (result %lld)\n", naive_ms, s);
    printf("Interp overhead: %.0fx\n", elapsed_ms / (naive_ms > 0 ? naive_ms : 0.001));
    return 0;
}
