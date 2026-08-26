#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
extern const wubu_isa_driver_t wubu_isa_vulkan;
int main(int argc,char**argv){
  int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]);
  const wubu_isa_driver_t *d=&wubu_isa_vulkan;
  wubu_mir_prog_t p; wubu_mir_init(&p);
  int offA=1,offB=offA+M*K,offC=offB+N*K;
  wubu_mir_alloc(&p,offC+M*N);
  for(int i=0;i<M*K;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offA+i)),wubu_mir_const(&p,(int64_t)i+1));
  for(int i=0;i<N*K;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offB+i)),wubu_mir_const(&p,(int64_t)2));
  for(int i=0;i<M*N;i++) wubu_mir_store(&p,wubu_mir_const(&p,(int64_t)(offC+i)),wubu_mir_const(&p,(int64_t)7));
  wubu_mir_tgemm(&p,wubu_mir_const(&p,(int64_t)offA),wubu_mir_const(&p,(int64_t)offB),wubu_mir_const(&p,(int64_t)offC),M,N,K);
  wubu_vr_t v=wubu_mir_load(&p,wubu_mir_const(&p,(int64_t)(offC+M*N-1)));
  wubu_mir_ret(&p,v);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  uint8_t *c=NULL; size_t n=0;
  d->compile(&p,&c,&n);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double ms=(t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;
  fprintf(stderr,"M=%d emit=%.1fms spv=%zu bytes\n",M,ms,n);
  (void)c;  return 0;
}
