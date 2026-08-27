#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

int main(void) {
    const char *cases[] = {
        "int v=21; v<<=1; v;",
        "int v=84; v>>=1; v;",
        "int f(int n){ int x=n*2; return x+2; } f(20);",
        "int a[]={1,2,3}; a[2];",
        "int a[]={1,2,3}; int*p=a; p++; *p;",
        "struct S{int a;}; struct S s; s.a=42; s.a;",
        "struct S{int a;int b;}; struct S s; s.a=1; s.b=2; s.a+s.b;",
        "char s[4]; s[0]='h'; s[1]='i'; s[2]=0; s[0];",
    };
    for (int i = 0; i < 8; i++) {
        long long r = hd_eval_mir(cases[i], NULL);
        printf("[%d] %s => %lld\n", i, cases[i], r);
    }
    return 0;
}
