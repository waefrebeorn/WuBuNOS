/* Debug: dump MIR for specific cases */
#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

int main(void) {
    const char *cases[] = {
        "int f(int n){ int x=n*2; return x+2; } f(20);",
        "int v=84; v>>=1; v;",
        "int a[2]; a[0]=42; a[1]=7; a[0]+a[1];",
        "sizeof(int);",
    };
    for (int i = 0; i < 4; i++) {
        printf("=== case %d: %s ===\n", i, cases[i]);
        long long r = hd_eval_mir(cases[i], NULL);
        printf("result: %lld\n\n", r);
    }
    return 0;
}
