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
    wubu_vr_t v=wubu_mir_load(&p,wubu_mir_const(&p,(int64_t)offC));
    wubu_mir_ret(&p,v);
    uint8_t *c=NULL; size_t n=0;
    d->compile(&p,&c,&n);
    /* The driver writes /tmp/wubu_kernel.spv and run() calls vk_run on it */
    long long got = d->run(c,n,12345);
    long long want = 7 + (long long)K*(K+1);
    printf("M=%d N=%d K=%d C00=%lld (want %lld) rc_arg=%lld\n", M, N, K, got, want, 12345LL);
    return 0;
}
