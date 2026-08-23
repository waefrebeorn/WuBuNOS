
/* test_mir_float.c -- MIR-level float ops through the soft-float runtime. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "wubu_mir.h"

static int fails=0, tot=0;
static void chk(int ok, const char *m){ tot++; if(!ok){fails++; printf("FAIL %s\n",m);} }

int main(void){
    /* f32 add: 1.5 + 2.25 = 3.75 */
    uint32_t a, b, r;
    { float x=1.5f, y=2.25f; memcpy(&a,&x,4); memcpy(&b,&y,4); }
    wubu_mir_prog_t p; wubu_mir_init(&p);
    wubu_vr_t va = wubu_mir_const(&p, (int64_t)(uint32_t)a);
    wubu_vr_t vb = wubu_mir_const(&p, (int64_t)(uint32_t)b);
    wubu_vr_t vs = wubu_mir_binop(&p, MIR_FADD, va, vb);
    wubu_mir_ret(&p, vs);
    r = (uint32_t)wubu_mir_interp(&p);
    float fr; memcpy(&fr,&r,4);
    chk(fr==3.75f, "f32 add 1.5+2.25");
    wubu_mir_free(&p);

    /* f32 mul: 1.5 * 2 = 3 via FMUL; and ITOF/FTOI roundtrip */
    wubu_mir_init(&p);
    va = wubu_mir_const(&p, (int64_t)a); vb = wubu_mir_const(&p, (int64_t)b);
    vs = wubu_mir_binop(&p, MIR_FMUL, va, vb);
    wubu_mir_ret(&p, vs);
    r = (uint32_t)wubu_mir_interp(&p); memcpy(&fr,&r,4);
    chk(fr==3.375f, "f32 mul 1.5*2.25");   /* 1.5*2.25=3.375 */
    wubu_mir_free(&p);

    /* ITOF: 7 -> 7.0f ; FTOI back -> 7 */
    wubu_mir_init(&p);
    va = wubu_mir_const(&p, 7);
    wubu_vr_t vf = wubu_mir_unop(&p, MIR_ITOF, va);
    wubu_vr_t vi = wubu_mir_unop(&p, MIR_FTOI, vf);
    wubu_mir_ret(&p, vi);
    int64_t iv = wubu_mir_interp(&p);
    chk(iv==7, "itof/ftoi roundtrip 7");
    wubu_mir_free(&p);

    printf("=== %d/%d passed (MIR float interp) ===\n", tot-fails, tot);
    return fails ? 1 : 0;
}
