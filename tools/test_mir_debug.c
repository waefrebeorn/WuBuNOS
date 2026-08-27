/* Debug harness: dump MIR for failing cases via hd_build_mir */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mir.h"
#include "holyd_mir_eval.h"

extern int hd_build_mir(const char *src, wubu_mir_prog_t *prog);

int main(void) {
    const char *tests[] = {
        "int v=42; v<<=1; v;",
        "int v=84; v>>=1; v;",
        "int f(int n){ int x=n*2; return x+2; } f(20);",
        "int a[]={1,2,3}; a[2];",
        "struct S{int a;}; struct S s; s.a=42; s.a;",
        "struct S{int a;int b;}; struct S s; s.a=1; s.b=2; s.a+s.b;",
        "char s[4]; s[0]='h'; s[1]='i'; s[2]=0; s[0];",
        NULL
    };
    for (int i = 0; tests[i]; i++) {
        wubu_mir_prog_t prog;
        int rc = hd_build_mir(tests[i], &prog);
        if (rc < 0) {
            printf("=== %s === BUILD FAILED (rc=%d)\n\n", tests[i]);
            continue;
        }
        printf("=== %s === (%d instrs)\n", tests[i], prog.n);
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
