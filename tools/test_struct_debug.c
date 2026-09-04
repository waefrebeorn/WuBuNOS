#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

int main(void) {
    struct { const char *name; const char *src; long long expect; } cases[] = {
        {"deref int*", "struct S{int a;int b;}; struct S s; s.a=10; s.b=20; int* p=&s.a; *p;", 10},
        {"sizeof struct", "struct S{int a;int b;}; sizeof(struct S);", 8},
        {"sizeof nested", "struct P{int a;}; struct Q{struct P p; int b;}; sizeof(struct Q);", 8},
        {"ret struct sum", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} struct S r; r=f(); r.a+r.b;", 49},
        {"call member a", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} f().a;", 42},
        {"call member sum", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} f().a+f().b;", 49},
        {"12B pass arg", "struct S{int a;int b;int c;}; int f(struct S x){return x.a+x.b+x.c;} struct S s; s.a=10; s.b=20; s.c=30; f(s);", 60},
        {"struct+int arg", "struct S{int a;int b;int c;}; int f(struct S x, int n){return x.a+x.b+x.c+n;} struct S s; s.a=10; s.b=20; s.c=30; f(s,-8);", 52},
        {"int+struct arg", "struct S{int a;int b;int c;}; int f(int n, struct S x){return n+x.a+x.b+x.c;} struct S s; s.a=10; s.b=20; s.c=30; f(-8,s);", 52},
        {"2 struct args", "struct S{int a;int b;int c;}; int f(struct S x, struct S y){return x.a+x.b+y.c;} struct S s; s.a=10; s.b=20; s.c=30; f(s,s);", 60},
        {"16B pass arg", "struct S{long long a;long long b;}; long long f(struct S x){return x.a+x.b;} struct S s; s.a=9; s.b=8; f(s);", 17},
        {"arg+ret nested", "struct S{int a;int b;int c;}; int f(struct S x){return x.a+x.b+x.c;} struct S g(){struct S s; s.a=10; s.b=20; s.c=30; return s;} f(g());", 60},
        {"fn ptr member", "struct S{int (*fn)(int,int); int n;}; int add(int x,int y){return x+y;} struct S s; s.fn=add; s.n=5; s.fn(3,4);", 7},
        {"array member", "struct S{int a[3]; int n;}; struct S s; s.a[0]=1; s.a[1]=2; s.a[2]=3; s.n=3; s.a[0]+s.a[1]+s.a[2];", 6},
        {"sizeof member", "struct S{int a;int b;}; struct S s; sizeof(s.a);", 4},
        {NULL, NULL, 0}
    };

    int pass = 0, fail = 0;
    for (int i = 0; cases[i].src; i++) {
        long long r = hd_eval_mir(cases[i].src, NULL);
        int ok = (r == cases[i].expect);
        printf("  %s %-25s got=%lld expect=%lld\n",
               ok ? "PASS" : "FAIL", cases[i].name, r, cases[i].expect);
        if (ok) pass++; else fail++;
    }
    printf("\n=== %d PASS, %d FAIL ===\n", pass, fail);
    return fail ? 1 : 0;
}
