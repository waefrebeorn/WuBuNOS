#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

int main(void) {
    const char *src = "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} struct S r; r=f(); r.b;";
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int rc = hd_build_mir(src, &prog);
    if (rc != 0) { printf("BUILD ERROR\n"); return 1; }
    printf("MIR after optimization:\n");
    wubu_mir_dump(&prog);
    int64_t result = hd_run_prog(&prog, NULL);
    printf("Result: %lld\n", (long long)result);
    wubu_mir_free(&prog);
    return 0;
}
