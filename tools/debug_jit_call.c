#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"
#include "wubu_isa_driver.h"

int main(void) {
    const char *src = "int add(int a,int b){return a+b;} add(20,22);";
    
    printf("=== Interpreter ===\n");
    int64_t r1 = hd_eval_mir(src, NULL);
    printf("result=%lld\n", (long long)r1);
    
    printf("=== x86-64 JIT ===\n");
    const wubu_isa_driver_t *d = wubu_isa_find("x86-64");
    printf("driver: %s exec=%d\n", d->name, d->exec);
    int64_t r2 = hd_eval_mir(src, d);
    printf("result=%lld\n", (long long)r2);
    
    return (r1 == 42 && r2 == 42) ? 0 : 1;
}
