#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"
#include "wubu_isa_driver.h"

int main(void) {
    /* Simple function call */
    const char *src = "int sq(int n){return n*n;} sq(6);";
    
    printf("=== Interpreter ===\n");
    int64_t r1 = hd_eval_mir(src, NULL);
    printf("result=%lld\n", (long long)r1);
    
    printf("=== x86-64 JIT ===\n");
    const wubu_isa_driver_t *d = wubu_isa_find("x86-64");
    int64_t r2 = hd_eval_mir(src, d);
    printf("result=%lld\n", (long long)r2);
    
    return (r1 == 36 && r2 == 36) ? 0 : 1;
}
