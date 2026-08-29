#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"

/* Manually build the MIR for: int add(int a,int b){return a+b;} add(20,22); */
int main(void) {
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    
    /* Main code: add(20, 22) */
    wubu_vr_t v20 = wubu_mir_const(&prog, 20);  /* vr0 = 20 */
    wubu_vr_t v22 = wubu_mir_const(&prog, 22);  /* vr1 = 22 */
    wubu_mir_mov_to(&prog, 1, v20);             /* vr1 = vr0 (arg 0) */
    wubu_mir_mov_to(&prog, 2, v22);             /* vr2 = vr1 (arg 1) */
    
    printf("v20=%d v22=%d\n", v20, v22);
    printf("After args: n=%zu\n", prog.n);
    for (size_t i = 0; i < prog.n; i++)
        printf("  ins[%zu]: op=%d dst=%d a=%d b=%d imm=%lld\n",
               i, prog.ins[i].op, prog.ins[i].dst, prog.ins[i].a, prog.ins[i].b, (long long)prog.ins[i].imm);
    
    wubu_mir_free(&prog);
    return 0;
}
