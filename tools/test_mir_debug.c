#include <stdio.h>
#include "holyd_mir_eval.h"
int main() {
    printf("p[0]: %lld (expect 42)\n",
        hd_eval_mir("struct S{int a;}; struct S s; s.a=42; struct S* p=&s; p[0].a;", NULL));
    printf("p[0] via mir: %lld (expect 42)\n",
        hd_eval_mir("struct S{int a;}; struct S s; s.a=42; struct S* p=&s; struct S x=*p; x.a;", NULL));
    return 0;
}
