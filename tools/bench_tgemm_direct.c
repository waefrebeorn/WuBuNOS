/* Parallel T_GEMM benchmark - row-chunked OpenMP with correctness */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern __thread int64_t *wubu_jit_mem_ptr;
extern void wubu_tgemm_scalar(int64_t *mem, int64_t A, int64_t B, int64_t C, int M, int N, int64_t K);
extern void wubu_tgemm_parallel(int64_t *stack_mem, int64_t A, int64_t B, int64_t C, int M, int N, int64_t K);

static double nowsec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(int argc, char**argv){
    int M=argc>1?atoi(argv[1]):256;
    int N=M, K=M;
    int reps=5; if(argc>2)reps=atoi(argv[2]);
    int nt=1;
#ifdef _OPENMP
    const char* ntc=getenv("OMP_NUM_THREADS");
    if(ntc) nt=atoi(ntc);
#endif

    int64_t offA=2, offB=2+M*K, offC=offB+K*N;
    size_t total=2+(size_t)M*K+(size_t)K*N+(size_t)M*N;
    
    int64_t *stack_mem=calloc(total,8);
    int64_t *A=&stack_mem[offA], *B=&stack_mem[offB], *C=&stack_mem[offC];
    
    void *heap_buf = mmap(NULL, total*8, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    wubu_jit_mem_ptr = (int64_t*)heap_buf;
    memcpy(wubu_jit_mem_ptr, stack_mem, total*8);

    printf("M=%d N=%d K=%d nt=%d reps=%d\n", M, N, K, nt, reps);
    
    /* Warmup */
    wubu_tgemm_parallel(stack_mem, offA, offB, offC, M, N, K);
    memcpy(&stack_mem[offC], &wubu_jit_mem_ptr[offC], sizeof(int64_t)*M*N);
    
    /* Timed */
    double t0=nowsec();
    for(int r=0;r<reps;r++){
        memset(&wubu_jit_mem_ptr[offC], 0, sizeof(int64_t)*M*N);
        wubu_tgemm_parallel(stack_mem, offA, offB, offC, M, N, K);
    }
    double t=(nowsec()-t0)/reps;

    memcpy(&stack_mem[offC], &wubu_jit_mem_ptr[offC], sizeof(int64_t)*M*N);
    
    /* Verify */
    int bad=0;
    for(int i=0;i<M*N && bad<3;i++){
        int64_t ref=0;
        int ri=i/N, ci=i%N;
        for(int k=0;k<K;k++) ref += A[ri*K+k]*B[k*N+ci];
        if(C[i]!=ref) bad++;
    }

    double gops=(double)(M*N*K*2)/1e9;
    printf("t=%.3fms GOPS=%.1f bad=%d %s\n", t*1000, gops/t, bad, bad?"FAIL":"PASS");
    
    if (wubu_jit_mem_ptr) munmap(wubu_jit_mem_ptr, total*8);
    free(stack_mem);
    return bad;
}