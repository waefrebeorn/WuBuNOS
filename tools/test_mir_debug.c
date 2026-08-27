/* Debug harness: dump MIR for failing cases via hd_build_mir */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

extern int hd_build_mir(const char *src, wubu_mir_prog_t *prog);

int main(void) {
    const char *tests[] = {
        "20+22;",
        "int x=40; x+2;",
        "int v=30; v+=12; v;",
        "int x=10; x=42; x;",
        "if(1){42;}else{0;}",
        "int i=0; while(i<3){i++;} i;",
        "sizeof(int);",
        "\"hello\"[0];",
        NULL
    };
    for (int i = 0; tests[i]; i++) {
        wubu_mir_prog_t prog;
        int rc = hd_build_mir(tests[i], &prog);
        if (rc < 0) {
            printf("=== %s === BUILD FAILED (rc=%d)\n\n", tests[i]);
            continue;
        }
        printf("=== %s === (%d instrs, %d funcs)\n", tests[i], prog.n, prog.n_funcs);
        for (int j = 0; j < prog.n; j++) {
            const wubu_mir_instr_t *ins = &prog.ins[j];
            printf("  [%d] op=%d dst=%d a=%d b=%d imm=%lld label=%u func=%u\n",
                   j, ins->op, ins->dst, ins->a, ins->b, (long long)ins->imm,
                   ins->label, ins->func_id);
        }
        printf("\n");
    }
    return 0;
}
