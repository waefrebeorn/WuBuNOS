#include <stdio.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

int main(void) {
    const char *src = "int v=21; v<<=1; v;";
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    extern int hd_build_mir(const char *src, wubu_mir_prog_t *prog);
    int rc = hd_build_mir(src, &prog);
    printf("rc=%d n=%d\n", rc, prog.n);
    for (uint32_t i = 0; i < prog.n; i++) {
        wubu_mir_instr_t *in = &prog.ins[i];
        printf("  [%2d] op=%2d a=%d b=%d dst=%d imm=%lld\n",
               i, in->op, in->a, in->b, in->dst, (long long)in->imm);
    }
    wubu_mir_free(&prog);
    return 0;
}
