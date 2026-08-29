#include <omp.h>
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
 * SOTA micro-kernel: MR=8 x NR=16 (16 floats per ZMM register).
 * 8 accumulators + 1 B + 1 A = 10 ZMM regs (of 32 on Zen 4).
 * KC=64 fits L1d. No software prefetch (degrades Zen).
 * FMA chaining: 8 FMAs per k-iteration saturates both FMA units.
 */
void wubu_tgemm_f32_avx512(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    const int MR = 8, NR = 16, KC = 64, NC = 256;
    int nthreads = (M * N * K >= 256000000) ? omp_get_max_threads() : 1;
    int mc = (M + nthreads - 1) / nthreads;
    mc &= ~(MR - 1);
    if (mc < MR && M >= MR) mc = MR;
    if (mc < MR * 4 && nthreads > 1) {
        nthreads = (M + MR * 4 - 1) / (MR * 4);
        mc = (M + nthreads - 1) / nthreads;
        mc &= ~(MR - 1);
        if (mc < MR && M >= MR) mc = MR;
    }
    #pragma omp parallel for schedule(static) if(nthreads > 1)
    for (int i0 = 0; i0 < M; i0 += mc) {
        int imax = i0 + mc; if (imax > M) imax = M;
        for (int j0 = 0; j0 < N; j0 += NC) {
            int jmax = j0 + NC; if (jmax > N) jmax = N;
            for (int k0 = 0; k0 < K; k0 += KC) {
                int kmax = k0 + KC; if (kmax > K) kmax = K;
                for (int i = i0; i < imax; i += MR) {
                    int im = imax - i; if (im > MR) im = MR;
                    for (int j = j0; j + NR - 1 < jmax; j += NR) {
                        __m512 c0 = _mm512_loadu_ps(&C[(int64_t)(i+0)*N + j]);
                        __m512 c1 = _mm512_loadu_ps(&C[(int64_t)(i+1)*N + j]);
                        __m512 c2 = _mm512_loadu_ps(&C[(int64_t)(i+2)*N + j]);
                        __m512 c3 = _mm512_loadu_ps(&C[(int64_t)(i+3)*N + j]);
                        __m512 c4 = _mm512_loadu_ps(&C[(int64_t)(i+4)*N + j]);
                        __m512 c5 = _mm512_loadu_ps(&C[(int64_t)(i+5)*N + j]);
                        __m512 c6 = _mm512_loadu_ps(&C[(int64_t)(i+6)*N + j]);
                        __m512 c7 = _mm512_loadu_ps(&C[(int64_t)(i+7)*N + j]);
                        for (int k = k0; k < kmax; k++) {
                            __m512 b = _mm512_loadu_ps(&B[(int64_t)k*N + j]);
                            if (im > 0) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+0)*K + k]);
                                c0 = _mm512_fmadd_ps(a, b, c0); }
                            if (im > 1) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+1)*K + k]);
                                c1 = _mm512_fmadd_ps(a, b, c1); }
                            if (im > 2) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+2)*K + k]);
                                c2 = _mm512_fmadd_ps(a, b, c2); }
                            if (im > 3) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+3)*K + k]);
                                c3 = _mm512_fmadd_ps(a, b, c3); }
                            if (im > 4) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+4)*K + k]);
                                c4 = _mm512_fmadd_ps(a, b, c4); }
                            if (im > 5) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+5)*K + k]);
                                c5 = _mm512_fmadd_ps(a, b, c5); }
                            if (im > 6) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+6)*K + k]);
                                c6 = _mm512_fmadd_ps(a, b, c6); }
                            if (im > 7) { __m512 a = _mm512_set1_ps(A[(int64_t)(i+7)*K + k]);
                                c7 = _mm512_fmadd_ps(a, b, c7); }
                        }
                        if (im > 0) _mm512_storeu_ps(&C[(int64_t)(i+0)*N + j], c0);
                        if (im > 1) _mm512_storeu_ps(&C[(int64_t)(i+1)*N + j], c1);
                        if (im > 2) _mm512_storeu_ps(&C[(int64_t)(i+2)*N + j], c2);
                        if (im > 3) _mm512_storeu_ps(&C[(int64_t)(i+3)*N + j], c3);
                        if (im > 4) _mm512_storeu_ps(&C[(int64_t)(i+4)*N + j], c4);
                        if (im > 5) _mm512_storeu_ps(&C[(int64_t)(i+5)*N + j], c5);
                        if (im > 6) _mm512_storeu_ps(&C[(int64_t)(i+6)*N + j], c6);
                        if (im > 7) _mm512_storeu_ps(&C[(int64_t)(i+7)*N + j], c7);
                    }
                    for (int j = j0 + ((jmax - j0) / NR) * NR; j < jmax; j++) {
                        for (int ii = i; ii < imax; ii++) {
                            float acc = C[(int64_t)ii*N + j];
                            for (int k = k0; k < kmax; k++)
                                acc += A[(int64_t)ii*K + k] * B[(int64_t)k*N + j];
                            C[(int64_t)ii*N + j] = acc;
                        }
                    }
                }
            }
        }
    }
}
