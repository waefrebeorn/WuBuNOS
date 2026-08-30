#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
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

/*
 * Optimized f32 GEMM for Zen 4 AVX-512.
 *
 * Zen 4: 512-bit FMA throughput = 1/cycle, latency = 5 cycles.
 *
 * Parallelization: over (j0,k0) panel pairs for load balance.
 * Each (j0,k0) pair is an independent unit of work.
 * For N=1024,K=1024,NC=256,KC=128: 4*8=32 panels -> 12 threads get ~3 each.
 *
 * K-unroll by 4: 4 independent FMAs per iteration hide 5-cycle latency.
 * NC=256: B-panel = KC*NC*4 = 128*256*4 = 128KB fits L2.
 */
void wubu_tgemm_f32_avx512(const float * __restrict__ A,
                            const float * __restrict__ B,
                            float * __restrict__ C,
                            int M, int N, int K)
{
    const int MR = 8, NR = 16, KC = 128, NC = 256;

    int nthreads = (M * (long long)N * K >= 200000000LL) ? omp_get_max_threads() : 1;

    /* Parallelize over (j0,k0) panel pairs for load balance.
     * Linear index jp -> (j0, k0) decomposition. */
    int n_jpanels = (N + NC - 1) / NC;
    int n_kpanels = (K + KC - 1) / KC;
    int n_panels = n_jpanels * n_kpanels;

    #pragma omp parallel for schedule(static) if(nthreads > 1)
    for (int jp = 0; jp < n_panels; jp++) {
        int jp_idx = jp / n_kpanels;
        int kp_idx = jp % n_kpanels;
        int j0 = jp_idx * NC;
        int k0 = kp_idx * KC;

            int jmax = j0 + NC; if (jmax > N) jmax = N;
            int nc = jmax - j0;
            int kmax = k0 + KC; if (kmax > K) kmax = K;
            int kc = kmax - k0;

            float *bpack = (float*)aligned_alloc(64, (size_t)KC * nc * sizeof(float));
            if (!bpack) continue;

            /* Pack B panel */
            for (int k = 0; k < kc; k++)
                memcpy(&bpack[(size_t)k * nc],
                       &B[(long long)(k0 + k) * N + j0],
                       (size_t)nc * sizeof(float));

            for (int i0 = 0; i0 < M; i0 += MR) {
                int imax = i0 + MR; if (imax > M) imax = M;
                int im = imax - i0;

                __m512 c0, c1, c2, c3, c4, c5, c6, c7;

                for (int j = 0; j + NR - 1 < nc; j += NR) {
                    c0=(im>0)?_mm512_loadu_ps(&C[(long long)(i0+0)*N+j0+j]):_mm512_setzero_ps();
                    c1=(im>1)?_mm512_loadu_ps(&C[(long long)(i0+1)*N+j0+j]):_mm512_setzero_ps();
                    c2=(im>2)?_mm512_loadu_ps(&C[(long long)(i0+2)*N+j0+j]):_mm512_setzero_ps();
                    c3=(im>3)?_mm512_loadu_ps(&C[(long long)(i0+3)*N+j0+j]):_mm512_setzero_ps();
                    c4=(im>4)?_mm512_loadu_ps(&C[(long long)(i0+4)*N+j0+j]):_mm512_setzero_ps();
                    c5=(im>5)?_mm512_loadu_ps(&C[(long long)(i0+5)*N+j0+j]):_mm512_setzero_ps();
                    c6=(im>6)?_mm512_loadu_ps(&C[(long long)(i0+6)*N+j0+j]):_mm512_setzero_ps();
                    c7=(im>7)?_mm512_loadu_ps(&C[(long long)(i0+7)*N+j0+j]):_mm512_setzero_ps();

                    const float *a_row[8];
                    for (int ii = 0; ii < im; ii++)
                        a_row[ii] = &A[(long long)(i0 + ii) * K + k0];

                    int k = 0;
                    for (; k + 3 < kc; k += 4) {
                        __m512 b0=_mm512_loadu_ps(&bpack[(size_t)(k+0)*nc+j]);
                        __m512 b1=_mm512_loadu_ps(&bpack[(size_t)(k+1)*nc+j]);
                        __m512 b2=_mm512_loadu_ps(&bpack[(size_t)(k+2)*nc+j]);
                        __m512 b3=_mm512_loadu_ps(&bpack[(size_t)(k+3)*nc+j]);

                        if(im>0){__m512 a0=_mm512_set1_ps(a_row[0][k+0]);__m512 a1=_mm512_set1_ps(a_row[0][k+1]);__m512 a2=_mm512_set1_ps(a_row[0][k+2]);__m512 a3=_mm512_set1_ps(a_row[0][k+3]);
                            c0=_mm512_fmadd_ps(a0,b0,c0);c0=_mm512_fmadd_ps(a1,b1,c0);c0=_mm512_fmadd_ps(a2,b2,c0);c0=_mm512_fmadd_ps(a3,b3,c0);}
                        if(im>1){__m512 a0=_mm512_set1_ps(a_row[1][k+0]);__m512 a1=_mm512_set1_ps(a_row[1][k+1]);__m512 a2=_mm512_set1_ps(a_row[1][k+2]);__m512 a3=_mm512_set1_ps(a_row[1][k+3]);
                            c1=_mm512_fmadd_ps(a0,b0,c1);c1=_mm512_fmadd_ps(a1,b1,c1);c1=_mm512_fmadd_ps(a2,b2,c1);c1=_mm512_fmadd_ps(a3,b3,c1);}
                        if(im>2){__m512 a0=_mm512_set1_ps(a_row[2][k+0]);__m512 a1=_mm512_set1_ps(a_row[2][k+1]);__m512 a2=_mm512_set1_ps(a_row[2][k+2]);__m512 a3=_mm512_set1_ps(a_row[2][k+3]);
                            c2=_mm512_fmadd_ps(a0,b0,c2);c2=_mm512_fmadd_ps(a1,b1,c2);c2=_mm512_fmadd_ps(a2,b2,c2);c2=_mm512_fmadd_ps(a3,b3,c2);}
                        if(im>3){__m512 a0=_mm512_set1_ps(a_row[3][k+0]);__m512 a1=_mm512_set1_ps(a_row[3][k+1]);__m512 a2=_mm512_set1_ps(a_row[3][k+2]);__m512 a3=_mm512_set1_ps(a_row[3][k+3]);
                            c3=_mm512_fmadd_ps(a0,b0,c3);c3=_mm512_fmadd_ps(a1,b1,c3);c3=_mm512_fmadd_ps(a2,b2,c3);c3=_mm512_fmadd_ps(a3,b3,c3);}
                        if(im>4){__m512 a0=_mm512_set1_ps(a_row[4][k+0]);__m512 a1=_mm512_set1_ps(a_row[4][k+1]);__m512 a2=_mm512_set1_ps(a_row[4][k+2]);__m512 a3=_mm512_set1_ps(a_row[4][k+3]);
                            c4=_mm512_fmadd_ps(a0,b0,c4);c4=_mm512_fmadd_ps(a1,b1,c4);c4=_mm512_fmadd_ps(a2,b2,c4);c4=_mm512_fmadd_ps(a3,b3,c4);}
                        if(im>5){__m512 a0=_mm512_set1_ps(a_row[5][k+0]);__m512 a1=_mm512_set1_ps(a_row[5][k+1]);__m512 a2=_mm512_set1_ps(a_row[5][k+2]);__m512 a3=_mm512_set1_ps(a_row[5][k+3]);
                            c5=_mm512_fmadd_ps(a0,b0,c5);c5=_mm512_fmadd_ps(a1,b1,c5);c5=_mm512_fmadd_ps(a2,b2,c5);c5=_mm512_fmadd_ps(a3,b3,c5);}
                        if(im>6){__m512 a0=_mm512_set1_ps(a_row[6][k+0]);__m512 a1=_mm512_set1_ps(a_row[6][k+1]);__m512 a2=_mm512_set1_ps(a_row[6][k+2]);__m512 a3=_mm512_set1_ps(a_row[6][k+3]);
                            c6=_mm512_fmadd_ps(a0,b0,c6);c6=_mm512_fmadd_ps(a1,b1,c6);c6=_mm512_fmadd_ps(a2,b2,c6);c6=_mm512_fmadd_ps(a3,b3,c6);}
                        if(im>7){__m512 a0=_mm512_set1_ps(a_row[7][k+0]);__m512 a1=_mm512_set1_ps(a_row[7][k+1]);__m512 a2=_mm512_set1_ps(a_row[7][k+2]);__m512 a3=_mm512_set1_ps(a_row[7][k+3]);
                            c7=_mm512_fmadd_ps(a0,b0,c7);c7=_mm512_fmadd_ps(a1,b1,c7);c7=_mm512_fmadd_ps(a2,b2,c7);c7=_mm512_fmadd_ps(a3,b3,c7);}
                    }
                    for (; k < kc; k++) {
                        __m512 bpk=_mm512_loadu_ps(&bpack[(size_t)k*nc+j]);
                        if(im>0){__m512 a=_mm512_set1_ps(a_row[0][k]);c0=_mm512_fmadd_ps(a,bpk,c0);}
                        if(im>1){__m512 a=_mm512_set1_ps(a_row[1][k]);c1=_mm512_fmadd_ps(a,bpk,c1);}
                        if(im>2){__m512 a=_mm512_set1_ps(a_row[2][k]);c2=_mm512_fmadd_ps(a,bpk,c2);}
                        if(im>3){__m512 a=_mm512_set1_ps(a_row[3][k]);c3=_mm512_fmadd_ps(a,bpk,c3);}
                        if(im>4){__m512 a=_mm512_set1_ps(a_row[4][k]);c4=_mm512_fmadd_ps(a,bpk,c4);}
                        if(im>5){__m512 a=_mm512_set1_ps(a_row[5][k]);c5=_mm512_fmadd_ps(a,bpk,c5);}
                        if(im>6){__m512 a=_mm512_set1_ps(a_row[6][k]);c6=_mm512_fmadd_ps(a,bpk,c6);}
                        if(im>7){__m512 a=_mm512_set1_ps(a_row[7][k]);c7=_mm512_fmadd_ps(a,bpk,c7);}
                    }

                    if (im > 0) _mm512_storeu_ps(&C[(long long)(i0+0)*N+j0+j], c0);
                    if (im > 1) _mm512_storeu_ps(&C[(long long)(i0+1)*N+j0+j], c1);
                    if (im > 2) _mm512_storeu_ps(&C[(long long)(i0+2)*N+j0+j], c2);
                    if (im > 3) _mm512_storeu_ps(&C[(long long)(i0+3)*N+j0+j], c3);
                    if (im > 4) _mm512_storeu_ps(&C[(long long)(i0+4)*N+j0+j], c4);
                    if (im > 5) _mm512_storeu_ps(&C[(long long)(i0+5)*N+j0+j], c5);
                    if (im > 6) _mm512_storeu_ps(&C[(long long)(i0+6)*N+j0+j], c6);
                    if (im > 7) _mm512_storeu_ps(&C[(long long)(i0+7)*N+j0+j], c7);
                }

                /* Scalar remainder */
                for (int j = (nc/NR)*NR; j < nc; j++) {
                    for (int ii = i0; ii < imax && ii < i0+8; ii++) {
                        float acc = C[(long long)ii*N+j0+j];
                        for (int kk = k0; kk < kmax; kk++)
                            acc += A[(long long)ii*K+kk] * B[(long long)kk*N+j0+j];
                        C[(long long)ii*N+j0+j] = acc;
                    }
                }
            }
            free(bpack);
        }
    }
/*
 * BF16 GEMM: convert to FP32 then use optimized FP32 path.
 */
void wubu_tgemm_bf16_avx512(const uint16_t *A_bf16, const uint16_t *B_bf16, float *C,
                             int M, int N, int K)
{
    float *A_fp32 = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *B_fp32 = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    if (!A_fp32 || !B_fp32) {
        if (A_fp32) free(A_fp32);
        if (B_fp32) free(B_fp32);
        return;
    }

    for (int i = 0; i < M * K; i++) {
        uint32_t v = (uint32_t)A_bf16[i] << 16;
        memcpy(&A_fp32[i], &v, 4);
    }
    for (int i = 0; i < K * N; i++) {
        uint32_t v = (uint32_t)B_bf16[i] << 16;
        memcpy(&B_fp32[i], &v, 4);
    }

    wubu_tgemm_f32_avx512(A_fp32, B_fp32, C, M, N, K);

    free(A_fp32);
    free(B_fp32);
}
