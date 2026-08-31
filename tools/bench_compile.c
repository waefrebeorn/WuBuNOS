#include <stdio.h>
#include <string.h>
#include <time.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    /* Benchmark: compile + run stress_fib 1000 times */
    const char *src = "{int a=0; int b=1; for(int i=0;i<10;i++) {int t=a+b; a=b; b=t;} a}";
    
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    
    /* Benchmark compilation */
    int n = 1000;
    double t0 = now_sec();
    for (int i = 0; i < n; i++) {
        wubu_mir_prog_t prog;
        hd_build_mir(src, &prog);
        if (!prog.mem) prog.mem = (int64_t*)calloc(8192, sizeof(int64_t));
        uint8_t *code = NULL;
        size_t sz = 0;
        drv->compile(&prog, &code, &sz);
        free(code);
        wubu_mir_free(&prog);
    }
    double t1 = now_sec();
    printf("Compile + build: %d iterations in %.3f ms (%.1f us/iter)\n", n, (t1-t0)*1000, (t1-t0)/n*1e6);
    
    /* Benchmark JIT execution */
    wubu_mir_prog_t prog;
    hd_build_mir(src, &prog);
    if (!prog.mem) prog.mem = (int64_t*)calloc(8192, sizeof(int64_t));
    uint8_t *code = NULL;
    size_t sz = 0;
    drv->compile(&prog, &code, &sz);
    
    n = 1000000;
    double t2 = now_sec();
    volatile int64_t result;
    for (int i = 0; i < n; i++) {
        result = drv->run(code, sz, (int64_t)prog.mem);
    }
    double t3 = now_sec();
    printf("JIT execution: %d iterations in %.3f ms (%.3f us/iter) result=%lld\n", n, (t3-t2)*1000, (t3-t2)/n*1e6, (long long)result);
    
    /* Benchmark interpreter execution */
    double t4 = now_sec();
    for (int i = 0; i < n; i++) {
        result = wubu_mir_interp(&prog);
    }
    double t5 = now_sec();
    printf("Interpreter execution: %d iterations in %.3f ms (%.3f us/iter) result=%lld\n", n, (t5-t4)*1000, (t5-t4)/n*1e6, (long long)result);
    
    free(code);
    wubu_mir_free(&prog);
    return 0;
}
