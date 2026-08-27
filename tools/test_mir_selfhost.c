#include "holyd_mir_eval.h"
#include <stdio.h>
int main() {
    printf("=== Self-Hosting via MIR Pipeline ===\n");
    const char *tests[] = {
        "3+4*5",
        "42-8",
        "6*7",
        "84/2",
        "int sq(int n){return n*n;} sq(6)+6;",
        "int fib(int n){if(n<2)return n;return fib(n-1)+fib(n-2);} fib(9);",
        "if(1){42;}else{0;}",
        "int f(int n){ int x=n*2; return x+2; } f(20);",
    };
    const int expected[] = {23, 34, 42, 42, 42, 34, 42, 42};
    int pass = 0, fail = 0;
    for (int i = 0; i < 8; i++) {
        int64_t r = hd_eval_mir(tests[i], NULL);
        printf("  [%d] %s -> %lld (want %d) %s\n", i+1, tests[i], (long long)r, expected[i],
               r == expected[i] ? "PASS" : "FAIL");
        if (r == expected[i]) pass++; else fail++;
    }
    printf("MIR self-hosting: %d/%d PASS\n", pass, pass+fail);
    return fail ? 1 : 0;
}