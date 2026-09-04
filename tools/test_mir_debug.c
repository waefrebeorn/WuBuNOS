#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"
int main() {
    const char *src = "struct P{int x;}; struct Q{struct P p; int y;}; struct Q q; q.p.x=42; q.y=7; q.p.x+q.y;";
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int rc = hd_build_mir(src, &prog);
    if (rc != 0) { printf("BUILD ERROR\n"); return 1; }
    int64_t result = hd_run_prog(&prog, NULL);
    printf("Result: %lld (expect 49)\n", (long long)result);
    wubu_mir_free(&prog);
    return 0;
}
