#include <stdio.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

int main() {
    const char *tests[] = {
        "{int sum=0; for(int i=0;i<10;i++) sum=sum+i; sum}",
        "{int sum=0; int i=0; while(i<10) {sum=sum+i; i=i+1;} sum}",
        "{int a=0; for(int i=0;i<5;i++) a=a+1; a}",
        "{int a=1; for(int i=0;i<10;i++) a=a*2; a}",
        "{int a=0; int b=1; for(int i=0;i<10;i++) {int t=a+b; a=b; b=t;} a}",
        NULL
    };
    for (int i = 0; tests[i]; i++) {
        long long r = hd_eval_mir(tests[i], NULL);
        printf("  [%d] %s -> %lld\n", i, tests[i], r);
    }
    return 0;
}
