#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"

int main(void) {
    const char *src = "int add(int a,int b){return a+b;} add(20,22);";
    wubu_mir_prog_t prog;
    if (hd_build_mir(src, &prog) != 0) { printf("build failed\n"); return 1; }
    printf("n=%zu n_funcs=%d total_mem=%lld next_vr_hi=%d\n",
           prog.n, prog.n_funcs, (long long)prog.total_mem, prog.next_vr_hi);
    for (int f = 0; f < prog.n_funcs; f++)
        printf("  func[%d]: start=%zu end=%zu name=%s\n",
               f, prog.funcs[f].start, prog.funcs[f].end, prog.funcs[f].name);
    for (size_t i = 0; i < prog.n; i++)
        printf("  ins[%zu]: op=%d dst=%d a=%d b=%d imm=%lld\n",
               i, prog.ins[i].op, prog.ins[i].dst, prog.ins[i].a, prog.ins[i].b, (long long)prog.ins[i].imm);
    wubu_mir_free(&prog);
    return 0;
}
