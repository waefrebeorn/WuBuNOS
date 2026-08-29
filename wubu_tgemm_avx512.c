#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
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
    const int MR = 6, NR = 16, KC = 128, NC = 128;
    int nthreads = (M * N * K >= 200000000) ? omp_get_max_threads() : 1;

    #pragma omp parallel for schedule(dynamic, 1) if(nthreads > 1)
    for (int j0 = 0; j0 < N; j0 += NC) {
        int jmax = j0 + NC; if (jmax > N) jmax = N;
        int nc = jmax - j0;

        float *bpack = (float*)aligned_alloc(64, (size_t)KC * nc * sizeof(float));
        if (!bpack) continue;

        for (int k0 = 0; k0 < K; k0 += KC) {
            int kmax = k0 + KC; if (kmax > K) kmax = K;
            int kc = kmax - k0;

            for (int k = 0; k < kc; k++)
                memcpy(&bpack[(size_t)k * nc],
                       &B[(int64_t)(k0 + k) * N + j0],
                       (size_t)nc * sizeof(float));

            for (int i0 = 0; i0 < M; i0 += MR) {
                int imax = i0 + MR; if (imax > M) imax = M;
                int im = imax - i0;

                for (int j = 0; j + NR - 1 < nc; j += NR) {
                    __m512 c0=(im>0)?_mm512_loadu_ps(&C[(int64_t)(i0+0)*N+j0+j]):_mm512_setzero_ps();
                    __m512 c1=(im>1)?_mm512_loadu_ps(&C[(int64_t)(i0+1)*N+j0+j]):_mm512_setzero_ps();
                    __m512 c2=(im>2)?_mm512_loadu_ps(&C[(int64_t)(i0+2)*N+j0+j]):_mm512_setzero_ps();
                    __m512 c3=(im>3)?_mm512_loadu_ps(&C[(int64_t)(i0+3)*N+j0+j]):_mm512_setzero_ps();
                    __m512 c4=(im>4)?_mm512_loadu_ps(&C[(int64_t)(i0+4)*N+j0+j]):_mm512_setzero_ps();
                    __m512 c5=(im>5)?_mm512_loadu_ps(&C[(int64_t)(i0+5)*N+j0+j]):_mm512_setzero_ps();

                    /* Pre-load A values into registers to reduce broadcast overhead */
                    /* Unroll k by 2 for better ILP */
                    int k = 0;
                    for (; k + 1 < kc; k += 2) {
                        __m512 b0 = _mm512_loadu_ps(&bpack[(size_t)(k+0)*nc+j]);
                        __m512 b1 = _mm512_loadu_ps(&bpack[(size_t)(k+1)*nc+j]);
                        if(im>0){__m512 a0=_mm512_set1_ps(A[(int64_t)(i0+0)*K+k0+k]);
                                  c0=_mm512_fmadd_ps(a0,b0,c0);
                                  c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+0)*K+k0+k+1]),b1,c0);}
                        if(im>1){__m512 a1=_mm512_set1_ps(A[(int64_t)(i0+1)*K+k0+k]);
                                  c1=_mm512_fmadd_ps(a1,b0,c1);
                                  c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+1)*K+k0+k+1]),b1,c1);}
                        if(im>2){__m512 a2=_mm512_set1_ps(A[(int64_t)(i0+2)*K+k0+k]);
                                  c2=_mm512_fmadd_ps(a2,b0,c2);
                                  c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+2)*K+k0+k+1]),b1,c2);}
                        if(im>3){__m512 a3=_mm512_set1_ps(A[(int64_t)(i0+3)*K+k0+k]);
                                  c3=_mm512_fmadd_ps(a3,b0,c3);
                                  c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+3)*K+k0+k+1]),b1,c3);}
                        if(im>4){__m512 a4=_mm512_set1_ps(A[(int64_t)(i0+4)*K+k0+k]);
                                  c4=_mm512_fmadd_ps(a4,b0,c4);
                                  c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+4)*K+k0+k+1]),b1,c4);}
                        if(im>5){__m512 a5=_mm512_set1_ps(A[(int64_t)(i0+5)*K+k0+k]);
                                  c5=_mm512_fmadd_ps(a5,b0,c5);
                                  c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+5)*K+k0+k+1]),b1,c5);}
                    }
                    for (; k < kc; k++) {
                        __m512 bpk = _mm512_loadu_ps(&bpack[(size_t)k*nc+j]);
                        if(im>0)c0=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+0)*K+k0+k]),bpk,c0);
                        if(im>1)c1=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+1)*K+k0+k]),bpk,c1);
                        if(im>2)c2=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+2)*K+k0+k]),bpk,c2);
                        if(im>3)c3=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+3)*K+k0+k]),bpk,c3);
                        if(im>4)c4=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+4)*K+k0+k]),bpk,c4);
                        if(im>5)c5=_mm512_fmadd_ps(_mm512_set1_ps(A[(int64_t)(i0+5)*K+k0+k]),bpk,c5);
                    }

                    if (im > 0) _mm512_storeu_ps(&C[(int64_t)(i0+0)*N+j0+j], c0);
                    if (im > 1) _mm512_storeu_ps(&C[(int64_t)(i0+1)*N+j0+j], c1);
                    if (im > 2) _mm512_storeu_ps(&C[(int64_t)(i0+2)*N+j0+j], c2);
                    if (im > 3) _mm512_storeu_ps(&C[(int64_t)(i0+3)*N+j0+j], c3);
                    if (im > 4) _mm512_storeu_ps(&C[(int64_t)(i0+4)*N+j0+j], c4);
                    if (im > 5) _mm512_storeu_ps(&C[(int64_t)(i0+5)*N+j0+j], c5);
                }

                /* Scalar remainder */
                for (int j = (nc/NR)*NR; j < nc; j++) {
                    for (int ii = i0; ii < imax; ii++) {
                        float acc = C[(int64_t)ii*N+j0+j];
                        for (int k = k0; k < kmax; k++)
                            acc += A[(int64_t)ii*K+k] * B[(int64_t)k*N+j0+j];
                        C[(int64_t)ii*N+j0+j] = acc;
                    }
                }
            }
        }
        free(bpack);
    }
}


/*
 * BF16 GEMM: C[MR][NR] += A[MR][KC] * B[KC][NR]
 * Uses VDPBF16PS: each instruction does 32 BF16 FMAs → 16 FP32 accumulates.
 * Micro-kernel: MR=6 rows x NR=16 cols, KC=64.
 * For each pair of k-values (k0, k0+1):
 *   a_vec = broadcast A[i][k0], A[i][k0+1] as 16 pairs
 *   b_vec = interleave B[k0][j..j+15], B[k0+1][j..j+15] as 16 pairs
 *   c_vec = _mm512_dpbf16_ps(c_vec, a_vec, b_vec)
 */
void wubu_tgemm_bf16_avx512(const uint16_t *A_bf16, const uint16_t *B_bf16, float *C,
                             int M, int N, int K)
{
    /* Strategy: convert BF16 to FP32, then use the optimized FP32 GEMM.
     * BF16 -> FP32 conversion is O(M*K + K*N) which is negligible vs O(M*N*K).
     * This avoids VDPBF16PS interleaving overhead. */
    float *A_fp32 = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *B_fp32 = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    if (!A_fp32 || !B_fp32) {
        if (A_fp32) free(A_fp32);
        if (B_fp32) free(B_fp32);
        return;
    }

    /* Vectorized BF16 -> FP32 conversion */
    for (int i = 0; i < M * K; i++) {
        uint32_t v = (uint32_t)A_bf16[i] << 16;
        memcpy(&A_fp32[i], &v, 4);
    }
    for (int i = 0; i < K * N; i++) {
        uint32_t v = (uint32_t)B_bf16[i] << 16;
        memcpy(&B_fp32[i], &v, 4);
    }

    /* Call the FP32 GEMM */
    extern void wubu_tgemm_f32_avx512(const float*, const float*, float*, int, int, int);
    wubu_tgemm_f32_avx512(A_fp32, B_fp32, C, M, N, K);

    free(A_fp32);
    free(B_fp32);
}

/* Helper: BF16 -> FP32 conversion */
static inline float bf16_to_f32(uint16_t b) {
    uint32_t v = (uint32_t)b << 16;
    float f;
    memcpy(&f, &v, 4);
    return f;
}

