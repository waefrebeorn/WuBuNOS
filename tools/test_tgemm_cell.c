/* vktgp4.c — grid-stride proof: read C[M*N-1] (last cell).
 * For 32x32, cell 1023 is owned by invocation 1023%64=63 after many
 * grid-stride iterations. If the stride loop were broken, this cell
 * would still hold its init value 7 instead of 7+K*(K+1). */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
extern const wubu_isa_driver_t wubu_isa_vulkan;
int main(int argc, char **argv){
    int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]);
    const wubu_isa_driver_t *d = &wubu_isa_vulkan;
    wubu_mir_prog_t p; wubu_mir_init(&p);
    int offA=1; int offB=offA+M*K; int offC=offB+N*K;
    wubu_mir_alloc(&p, offC+M*N);
    for(int i=0;i<M*K;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offA+i)),wubu_mir_const(&p,(int64_t)i+1));
    for(int i=0;i<N*K;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offB+i)),wubu_mir_const(&p,(int64_t)2));
    for(int i=0;i<M*N;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offC+i)),wubu_mir_const(&p,(int64_t)7));
    wubu_mir_tgemm(&p,wubu_mir_const(&p,(int64_t)offA),
        wubu_mir_const(&p,(int64_t)offB),wubu_mir_const(&p,(int64_t)offC),M,N,K);
    wubu_vr_t v=wubu_mir_load(&p,wubu_mir_const(&p,(int64_t)(offC+M*N-1)));
    wubu_mir_ret(&p,v);
    uint8_t *c=NULL; size_t n=0;
    d->compile(&p,&c,&n);
    long long got = d->run(c,n,12345);
    long long want = 7 + (long long)K*(K+1);   /* every C cell gets same K-sum */
    printf("LASTCELL M=%d N=%d K=%d C=%lld (want %lld) %s\n", M,N,K,got,want,
           got==want?"PASS":"FAIL");
    return got==want?0:1;
}
