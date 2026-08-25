/* tgemm_gpu_checksum.c -- prove hw-aware parallel T_GEMM on the GPU:
 * every output row computed by a different %tid.x via grid-stride loop.
 * Program: C(3x3) = A*B, RET = sum of all 9 C cells (621). */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
#define M 3
#define N 3
#define K 3

int main(void) {
    int64_t A[M*K] = {1,2,3, 4,5,6, 7,8,9};
    int64_t B[K*N] = {9,8,7, 6,5,4, 3,2,1};

    const wubu_isa_driver_t *drv = wubu_isa_find("ptx");
    if (!drv) { fprintf(stderr,"no ptx\n"); return 1; }

    /* Layout: mem[1..]=A, then B, then C (cell offsets, NOT alloc VRs) */
    int offA = 1, offB = offA + M*K, offC = offB + K*N;
    wubu_mir_prog_t p; wubu_mir_init(&p);
    wubu_mir_alloc(&p, 1 + M*K + K*N + M*N);

    for (int i=0;i<M*K;i++)
        wubu_mir_store(&p, wubu_mir_const(&p,(int64_t)(offA+i)), wubu_mir_const(&p,A[i]));
    for (int i=0;i<K*N;i++)
        wubu_mir_store(&p, wubu_mir_const(&p,(int64_t)(offB+i)), wubu_mir_const(&p,B[i]));
    for (int i=0;i<M*N;i++)
        wubu_mir_store(&p, wubu_mir_const(&p,(int64_t)(offC+i)), wubu_mir_const(&p,0));

    wubu_mir_tgemm(&p,
                   wubu_mir_const(&p,(int64_t)offA),
                   wubu_mir_const(&p,(int64_t)offB),
                   wubu_mir_const(&p,(int64_t)offC),
                   M, N, K);

    /* checksum: sum all C cells */
    wubu_vr_t acc = wubu_mir_const(&p, 0);
    for (int i=0;i<M*N;i++)
        acc = wubu_mir_binop(&p, MIR_ADD, acc,
                             wubu_mir_load(&p, wubu_mir_const(&p,(int64_t)(offC+i))));
    wubu_mir_ret(&p, acc);

    uint8_t *code=NULL; size_t sz=0;
    if (drv->compile(&p,&code,&sz)!=0){fprintf(stderr,"compile fail\n");return 1;}
    int64_t got = drv->run(code, sz, 0);

    long long want = 0;
    for (int i=0;i<M;i++) for(int j=0;j<N;j++){
        long long s=0;
        for(int k=0;k<K;k++) s += A[i*K+k]*B[k*N+j];
        want += s;
    }
    printf("GPU parallel T_GEMM checksum: got=%lld want=%lld -> %s\n",
           (long long)got, want, got==want ? "PASS" : "FAIL");
    free(code); wubu_mir_free(&p);
    return got==want ? 0 : 1;
}
