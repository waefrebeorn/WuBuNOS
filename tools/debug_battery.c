#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"

int main(void) {
    struct { const char *name; const char *src; long long expect; } tests[] = {
        {"func 2 params", "int add(int a,int b){return a+b;} add(20,22);", 42},
        {"array init", "int a[]={1,2,3}; a[2];", 3},
        {"ptr arith", "int a[]={1,2,3}; int*p=a; p++; *p;", 2},
        {"float add", "float a=1.5; float b=2.5; a+b;", 4},
        {"multi-arg call", "int f(int a,int b,int c){return a+b+c;} f(10,20,12);", 42},
        {"scope shadow", "int x=1; {int x=2; x;} x;", 1},
        {"comma op", "int a=0,b=0; (a=1,b=2); a+b;", 3},
    };
    int n = sizeof(tests)/sizeof(tests[0]);
    for (int i = 0; i < n; i++) {
        int64_t r = hd_eval_mir(tests[i].src, NULL);
        const char *status = (r == tests[i].expect) ? "PASS" : "FAIL";
        printf("%s: %s (got %lld, expect %lld)\n", status, tests[i].name, (long long)r, (long long)tests[i].expect);
    }
    return 0;
}
