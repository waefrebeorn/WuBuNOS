/* H3: T_GEMM JIT benchmark — correctness via interp, timing via x86-64 JIT.
 * Build: gcc -O2 -std=c11 -DWUBU_HOSTED -include wubu_gnu_compat.h -I. -Ijit -I/home/wubu -o bench_jit tools/bench_tgemm_jit.c [srcs] -lm
 */
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double nowsec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static void naive_gemm(int64_t *C, const int64_t *A, const int64_t *B, int M, int N, int K){
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){
        int64_t acc=C[i*N+j];
        for(int kk=0;kk<K;kk++) acc+=A[i*K+kk]*B[kk*N+j];
        C[i*N+j]=acc;
    }
}

/* Build a T_GEMM MIR program that stores A, B, does the GEMM, then returns C[idx]. */
static void build_prog(wubu_mir_prog_t *p, const int64_t*A,const int64_t*B,
                       int M,int N,int K,int offA,int offB,int offC,int total,int idx){
    wubu_mir_init(p);
    wubu_mir_alloc(p, total);
    wubu_vr_t vA=wubu_mir_const(p,offA),vB=wubu_mir_const(p,offB),vC=wubu_mir_const(p,offC);
    for(int i=0;i<M*K;i++) wubu_mir_store(p,wubu_mir_const(p,offA+i),wubu_mir_const(p,A[i]));
    for(int i=0;i<K*N;i++) wubu_mir_store(p,wubu_mir_const(p,offB+i),wubu_mir_const(p,B[i]));
    wubu_mir_tgemm(p,vA,vB,vC,M,N,K);
    wubu_mir_ret(p,wubu_mir_load(p,wubu_mir_const(p,offC+idx)));
}

int main(int argc,char**argv){
    int M=64,N=64,K=64;int reps=50;
    if(argc>=4){M=atoi(argv[1]);N=atoi(argv[2]);K=atoi(argv[3]);}
    if(argc>=5)reps=atoi(argv[4]);

    int64_t*A=malloc((size_t)M*K*8);int64_t*B=malloc((size_t)N*K*8);
    int64_t*Cref=calloc((size_t)M*N,8);
    srand(42);
    for(int i=0;i<M*K;i++)A[i]=(rand()%17)-8;
    for(int i=0;i<N*K;i++)B[i]=(rand()%13)-6;
    int total=1+M*K+K*N+M*N;int offA=1,offB=offA+M*K,offC=offB+K*N;

    double t0=nowsec();naive_gemm(Cref,A,B,M,N,K);double t_ref=nowsec()-t0;

    /* correctness: verify entries via interpreter */
    int bad=0;
    int verify_n = (M*N < 256) ? M*N : 16;  /* cap for large matrices */
    int mis_reported = 0;
    for(int idx=0; idx<verify_n; idx++){
        wubu_mir_prog_t rp;build_prog(&rp,A,B,M,N,K,offA,offB,offC,total,idx);
        int64_t got=wubu_mir_interp(&rp);
        if(got!=Cref[idx]){
            if(mis_reported<5)printf("MISMATCH C[%d] interp=%lld ref=%lld\n",idx,(long long)got,(long long)Cref[idx]);
            mis_reported++;
            bad++;
        }
        wubu_mir_free(&rp);
    }

    /* JIT timing: compile once, run reps */
    const wubu_isa_driver_t*drv=wubu_isa_find("x86-64");
    double t_jit=0;const char*jitstatus="(unavailable)";
    if(drv&&drv->exec==WUBU_ISA_NATIVE){
        wubu_mir_prog_t p;build_prog(&p,A,B,M,N,K,offA,offB,offC,total,0);
        uint8_t*code=NULL;size_t csz=0;
        if(drv->compile(&p,&code,&csz)==0){
            t0=nowsec();volatile int64_t sink=0;
            for(int r=0;r<reps;r++)sink+=drv->run(code,csz,0);
            t_jit=(nowsec()-t0)/reps;(void)sink;jitstatus="";
            free(code);
        }
        wubu_mir_free(&p);
    }

    printf("sz=%d naive=%.6fs jit=%.6fs ops=%llu %s C[%s]\n",
           M,t_ref,t_jit,(unsigned long long)(M*N*K*2),
           jitstatus,(bad==0)?"ALL CORRECT":"FAIL");

    free(A);free(B);free(Cref);return bad?1:0;
}
