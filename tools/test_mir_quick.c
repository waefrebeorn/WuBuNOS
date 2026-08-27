/* Quick test: run hd_eval_mir on failing cases */
#include <stdio.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"
int main() {
    const char *tests[] = {
        "int v=21; v<<=1; v;",
        "int v=84; v>>=1; v;",
        "int f(int n){ int x=n*2; return x+2; } f(20);",
        "int a[]={1,2,3}; a[2];",
        "struct S{int a;}; struct S s; s.a=42; s.a;",
        "struct S{int a;int b;}; struct S s; s.a=1; s.b=2; s.a+s.b;",
        "char s[4]; s[0]='h'; s[1]='i'; s[2]=0; s[0];",
        "int a[3]; a[0]=1;a[1]=2;a[2]=3; a[2];",
        "int a[]={1,2,3}; int*p=a; p++; *p;",
        NULL
    };
    for (int i = 0; tests[i]; i++) {
        long long r = hd_eval_mir(tests[i], NULL);
        printf("  %-50s -> %lld\n", tests[i], r);
    }
    return 0;
}
