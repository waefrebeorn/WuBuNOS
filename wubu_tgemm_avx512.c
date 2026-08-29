#include <omp.h>
#include <string.h>
#include <stdio.h>
/* wubu_tgemm_avx512.c — AVX-512DQ native int64 GEMM path.
 * Compiled ONLY with -mavx512dq -mavx512f -mavx512vl.
 * Uses native vpmullq (8 int64 products per 512-bit vector).
 * C18 pure, no external deps beyond immintrin.h. */
#include "wubu_tgemm.h"
#include <immintrin.h>

void tgemm_avx512(int64_t *mem, int64_t A, int64_t B,
                  int64_t C, int M, int N, int K)
{
    #pragma omp parallel for schedule(static) if(M >= 8)
    for (int i = 0; i < M; i++) {
        const int64_t *a = &mem[A + (int64_t)i * K];
        int64_t       *c = &mem[C + (int64_t)i * N];
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m512i acc = _mm512_loadu_si512(&c[j]);
            for (int k = 0; k < K; k++) {
                __m512i bv = _mm512_loadu_si512(&mem[B + (int64_t)k * N + j]);
                __m512i av = _mm512_set1_epi64(a[k]);
                __m512i prod = _mm512_mullo_epi64(av, bv);
                acc = _mm512_add_epi64(acc, prod);
            }
            _mm512_storeu_si512(&c[j], acc);
        }
        for (; j < N; j++) {
            int64_t s = c[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k < K; k++)
                s += a[k] * bk[(size_t)k * N];
            c[j] = s;
        }
    }
}

/* ---- Float32 GEMM (AVX-512) ---------------------------------------
 * wubu_tgemm_f32_avx512: C += A*B where A,B,C are float32 row-major.
 * Requires -mavx512f -mavx512dq -mavx512vl.
 *
 * SOTA micro-kernel: MR=6 x NR=16 (16 floats per ZMM register).
 * 6 accumulators + 1 B + 1 A = 8 ZMM regs (of 32 on Zen 4).
 * KC=64 fits L1d. No software prefetch (degrades Zen).
 * K-unroll by 4: load 4 B rows, broadcast 4 A values per row.
 * Maximizes FMA pipelining on Zen 4 (single 512-bit FMA unit).
 */
void wubu_tgemm_f32_avx512(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    const int MR = 6, NR = 16, KC = 64, NC = 256;
    int nthreads = (M * N * K >= 200000000) ? omp_get_max_threads() : 1;
    int mc = (M + nthreads - 1) / nthreads;
    mc = (mc + MR - 1) / MR * MR;
    if (mc < MR && M >= MR) mc = MR;

    #pragma omp parallel for schedule(dynamic) if(nthreads > 1)
    for (int i0 = 0; i0 < M; i0 += mc) {
        int imax = i0 + mc; if (imax > M) imax = M;
        float *bpack = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (!bpack) { fprintf(stderr, "malloc failed\n"); continue; }
        for (int j0 = 0; j0 < N; j0 += NC) {
            int jmax = j0 + NC; if (jmax > N) jmax = N;
            int nc = jmax - j0;
            for (int k0 = 0; k0 < K; k0 += KC) {
                int kmax = k0 + KC; if (kmax > K) kmax = K;
                for (int k = 0; k < kmax - k0; k++)
                    memcpy(&bpack[(size_t)k * nc],
                           &B[(int64_t)(k0 + k) * N + j0],
                           (size_t)nc * sizeof(float));
                for (int i = i0; i < imax; i += MR) {
                    int im = imax - i; if (im > MR) im = MR;
                    for (int j = 0; j + NR - 1 < nc; j += NR) {
                        __m512 c0=(im>0)?_mm512_loadu_ps(&C[(int64_t)(i+0)*N+j0+j]):_mm512_setzero_ps();
                        __m512 c1=(im>1)?_mm512_loadu_ps(&C[(int64_t)(i+1)*N+j0+j]):_mm512_setzero_ps();
                        __m512 c2=(im>2)?_mm512_loadu_ps(&C[(int64_t)(i+2)*N+j0+j]):_mm512_setzero_ps();
                        __m512 c3=(im>3)?_mm512_loadu_ps(&C[(int64_t)(i+3)*N+j0+j]):_mm512_setzero_ps();
                        __m512 c4=(im>4)?_mm512_loadu_ps(&C[(int64_t)(i+4)*N+j0+j]):_mm512_setzero_ps();
                        __m512 c5=(im>5)?_mm512_loadu_ps(&C[(int64_t)(i+5)*N+j0+j]):_mm512_setzero_ps();
                        int k = 0;
                        for (; k + 3 < kmax - k0; k += 4) {
                            __m512 b0=_mm512_loadu_ps(&bpack[(size_t)k*nc+j]);
                            __m512 b1=_mm512_loadu_ps(&bpack[(size_t)(k+1)*nc+j]);
                            __m512 b2=_mm512_loadu_ps(&bpack[(size_t)(k+2)*nc+j]);
                            __m512 b3=_mm512_loadu_ps(&bpack[(size_t)(k+3)*nc+j]);
                            if(im>0){c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+0)*K+k0+k]),b0,c0);
                                      c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+0)*K+k0+k+1]),b1,c0);
                                      c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+0)*K+k0+k+2]),b2,c0);
                                      c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+0)*K+k0+k+3]),b3,c0);}
                            if(im>1){c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+1)*K+k0+k]),b0,c1);
                                      c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+1)*K+k0+k+1]),b1,c1);
                                      c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+1)*K+k0+k+2]),b2,c1);
                                      c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+1)*K+k0+k+3]),b3,c1);}
                            if(im>2){c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+2)*K+k0+k]),b0,c2);
                                      c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+2)*K+k0+k+1]),b1,c2);
                                      c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+2)*K+k0+k+2]),b2,c2);
                                      c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+2)*K+k0+k+3]),b3,c2);}
                            if(im>3){c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+3)*K+k0+k]),b0,c3);
                                      c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+3)*K+k0+k+1]),b1,c3);
                                      c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+3)*K+k0+k+2]),b2,c3);
                                      c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+3)*K+k0+k+3]),b3,c3);}
                            if(im>4){c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+4)*K+k0+k]),b0,c4);
                                      c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+4)*K+k0+k+1]),b1,c4);
                                      c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+4)*K+k0+k+2]),b2,c4);
                                      c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+4)*K+k0+k+3]),b3,c4);}
                            if(im>5){c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+5)*K+k0+k]),b0,c5);
                                      c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+5)*K+k0+k+1]),b1,c5);
                                      c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+5)*K+k0+k+2]),b2,c5);
                                      c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+5)*K+k0+k+3]),b3,c5);}
                        }
                        for (; k < kmax - k0; k++) {
                            __m512 bpk=_mm512_loadu_ps(&bpack[(size_t)k*nc+j]);
                            if(im>0)c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+0)*K+k0+k]),bpk,c0);
                            if(im>1)c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+1)*K+k0+k]),bpk,c1);
                            if(im>2)c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+2)*K+k0+k]),bpk,c2);
                            if(im>3)c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+3)*K+k0+k]),bpk,c3);
                            if(im>4)c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+4)*K+k0+k]),bpk,c4);
                            if(im>5)c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i+5)*K+k0+k]),bpk,c5);
                        }
                        if(im>0)_mm512_storeu_ps(&C[(int64_t)(i+0)*N+j0+j],c0);
                        if(im>1)_mm512_storeu_ps(&C[(int64_t)(i+1)*N+j0+j],c1);
                        if(im>2)_mm512_storeu_ps(&C[(int64_t)(i+2)*N+j0+j],c2);
                        if(im>3)_mm512_storeu_ps(&C[(int64_t)(i+3)*N+j0+j],c3);
                        if(im>4)_mm512_storeu_ps(&C[(int64_t)(i+4)*N+j0+j],c4);
                        if(im>5)_mm512_storeu_ps(&C[(int64_t)(i+5)*N+j0+j],c5);
                    }
                    for (int j = (nc/NR)*NR; j < nc; j++) {
                        for (int ii = i; ii < imax; ii++) {
                            float acc = C[(int64_t)ii*N+j0+j];
                            for (int k = k0; k < kmax; k++)
                                acc += A[(int64_t)ii*K+k]*B[(int64_t)k*N+j0+j];
                            C[(int64_t)ii*N+j0+j] = acc;
                        }
                    }
                }
            }
        }
        free(bpack);
    }
}

