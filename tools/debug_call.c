#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"
#include "wubu_isa_driver.h"

int main(void) {
    const char *src = "int add(int a,int b){return a+b;} add(20,22);";
    printf("=== Interpreter ===\n");
    int64_t r = hd_eval_mir(src, NULL);
    printf("result=%lld (expect 42)\n", (long long)r);

    printf("=== x86-64 JIT ===\n");
    const wubu_isa_driver_t *d = wubu_isa_find("x86-64");
    int64_t r2 = hd_eval_mir(src, d);
    printf("result=%lld (expect 42)\n", (long long)r2);
    return 0;
}
