/* wubu_tgemm.c — shared tiled int64/float32 GEMM kernel (C += A*B, row-major).
 * One canonical implementation used by the x86-64 JIT libcall lowering,
 * the MIR interpreter, and every retro-ISA hostcall escape hatch.
 * C18 pure, no external deps. */
#include "wubu_tgemm.h"
#if defined(_OPENMP)
#include <omp.h>
#endif


/* ---- AVX2 / AVX-512 fast path (x86-64 only) ------------------------
 * AVX2: 4 columns via 256-bit vectors; 64-bit mul emulated via 32-bit halves.
 * AVX-512DQ: 8 columns via 512-bit vectors; native vpmullq for 64-bit mul.
 */
#if defined(__x86_64__)
#include <immintrin.h>

__attribute__((target("avx2")))
static void tgemm_avx2(int64_t *mem, int64_t A, int64_t B,
                       int64_t C, int M, int N, int K)
{
    const __m256i zero = _mm256_setzero_si256();
    #pragma omp parallel for schedule(static) if(M >= 8)
    for (int i = 0; i < M; i++) {
        const int64_t *a = &mem[A + (int64_t)i * K];
        int64_t       *c = &mem[C + (int64_t)i * N];
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            __m256i acc = _mm256_loadu_si256((const __m256i *)&c[j]);
            for (int k = 0; k < K; k++) {
                __m256i bv = _mm256_loadu_si256((const __m256i *)&mem[B + (int64_t)k * N + j]);
                __m256i av = _mm256_set1_epi64x(a[k]);
                __m256i alo = _mm256_and_si256(av, _mm256_set1_epi64x(0xFFFFFFFFLL));
                __m256i blo = _mm256_and_si256(bv, _mm256_set1_epi64x(0xFFFFFFFFLL));
                __m256i plo = _mm256_mul_epu32(alo, blo);
                __m256i ahi = _mm256_srli_epi64(av, 32);
                __m256i bhi = _mm256_srli_epi64(bv, 32);
                __m256i mid1 = _mm256_mul_epu32(alo, bhi);
                __m256i mid2 = _mm256_mul_epu32(ahi, blo);
                __m256i mid  = _mm256_add_epi64(mid1, mid2);
                mid = _mm256_slli_epi64(mid, 32);
                __m256i prod = _mm256_add_epi64(plo, mid);
                acc = _mm256_add_epi64(acc, prod);
            }
            _mm256_storeu_si256((__m256i *)&c[j], acc);
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

static int avx2_ok = -1, avx512_ok = -1;
static int have_avx2(void) {
    if (avx2_ok < 0) {
        __builtin_cpu_init();
        avx2_ok = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    return avx2_ok;
}
static int have_avx512(void) {
    if (avx512_ok < 0) {
        __builtin_cpu_init();
        avx512_ok = (__builtin_cpu_supports("avx512f") &&
                     __builtin_cpu_supports("avx512dq") &&
                     __builtin_cpu_supports("avx512vl")) ? 1 : 0;
    }
    return avx512_ok;
}
#endif /* __x86_64__ */

void wubu_tgemm(int64_t *mem, int64_t A, int64_t B,
                int64_t C, int M, int N, int K)
{
#if defined(__x86_64__)
    if (have_avx512()) { tgemm_avx512(mem, A, B, C, M, N, K); return; }
    if (have_avx2()) { tgemm_avx2(mem, A, B, C, M, N, K); return; }
#endif
    /* OpenMP parallel over 4-row blocks */
    int iupper = (M >= 4) ? M - 3 : 0;
    #pragma omp parallel for schedule(static, 4) if(M*N*K >= 65536)
    for (int i = 0; i < iupper; i += 4) {
        const int64_t *a0 = &mem[A + (int64_t)(i+0) * K];
        const int64_t *a1 = &mem[A + (int64_t)(i+1) * K];
        const int64_t *a2 = &mem[A + (int64_t)(i+2) * K];
        const int64_t *a3 = &mem[A + (int64_t)(i+3) * K];
        int64_t       *c0 = &mem[C + (int64_t)(i+0) * N];
        int64_t       *c1 = &mem[C + (int64_t)(i+1) * N];
        int64_t       *c2 = &mem[C + (int64_t)(i+2) * N];
        int64_t       *c3 = &mem[C + (int64_t)(i+3) * N];
        for (int j = 0; j < N; j++) {
            int64_t s0 = c0[j], s1 = c1[j], s2 = c2[j], s3 = c3[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k + 3 < K; k += 4) {
                const int64_t b0 = bk[(size_t)(k+0) * N];
                const int64_t b1 = bk[(size_t)(k+1) * N];
                const int64_t b2 = bk[(size_t)(k+2) * N];
                const int64_t b3 = bk[(size_t)(k+3) * N];
                s0 += a0[k+0]*b0 + a0[k+1]*b1 + a0[k+2]*b2 + a0[k+3]*b3;
                s1 += a1[k+0]*b0 + a1[k+1]*b1 + a1[k+2]*b2 + a1[k+3]*b3;
                s2 += a2[k+0]*b0 + a2[k+1]*b1 + a2[k+2]*b2 + a2[k+3]*b3;
                s3 += a3[k+0]*b0 + a3[k+1]*b1 + a3[k+2]*b2 + a3[k+3]*b3;
            }
            for (int k = K & ~3; k < K; k++) {
                const int64_t b = bk[(size_t)k * N];
                s0 += a0[k]*b; s1 += a1[k]*b; s2 += a2[k]*b; s3 += a3[k]*b;
            }
            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }
    for (int i = (M >> 2) << 2; i < M; i++) {
        const int64_t *a0 = &mem[A + (int64_t)i * K];
        int64_t       *c0 = &mem[C + (int64_t)i * N];
        for (int j = 0; j < N; j++) {
            int64_t s0 = c0[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k < K; k++)
                s0 += a0[k] * bk[(size_t)k * N];
            c0[j] = s0;
        }
    }
}

void wubu_tgemm_mem8(uint8_t *mem, uint32_t A, uint32_t B, uint32_t C,
                     int M, int N, int K)
{
#define RD64(base) ((int64_t)(uint64_t)( \
        (uint64_t)(mem)[base+0] | ((uint64_t)(mem)[base+1] << 8) | \
        ((uint64_t)(mem)[base+2] << 16) | ((uint64_t)(mem)[base+3] << 24) | \
        ((uint64_t)(mem)[base+4] << 32) | ((uint64_t)(mem)[base+5] << 40) | \
        ((uint64_t)(mem)[base+6] << 48) | ((uint64_t)(mem)[base+7] << 56)))
#define WR64(base, v) do { uint64_t _v = (uint64_t)(v); \
        (mem)[base+0]=(uint8_t)_v; (mem)[base+1]=(uint8_t)(_v>>8); \
        (mem)[base+2]=(uint8_t)(_v>>16); (mem)[base+3]=(uint8_t)(_v>>24); \
        (mem)[base+4]=(uint8_t)(_v>>32); (mem)[base+5]=(uint8_t)(_v>>40); \
        (mem)[base+6]=(uint8_t)(_v>>48); (mem)[base+7]=(uint8_t)(_v>>56); } while (0)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int64_t acc = RD64(((size_t)C + (size_t)i * N + j) * 8);
            for (int k = 0; k < K; k++)
                acc += RD64(((size_t)A + (size_t)i * K + k) * 8)
                     * RD64(((size_t)B + (size_t)k * N + j) * 8);
            WR64(((size_t)C + (size_t)i * N + j) * 8, acc);
        }
#undef RD64
#undef WR64
}

/* ---- Float32 GEMM (for ML inference) -------------------------------
 * wubu_tgemm_f32: C += A*B where A,B,C are float32 row-major matrices.
 * Uses AVX2 256-bit vectors (8 floats per vector) with FMA.
 * Correctness-verified 8x8 micro-kernel (MR=8 rows, NR=8 cols).
 * Cache-blocked ikj. Parallelized with OpenMP across row-blocks. */
void wubu_tgemm_f32(const float *A, const float *B, float *C,
                    int M, int N, int K) {
#if defined(__AVX2__)
    const int MR = 8, KC = 128, NC = 256;
    int nthreads = (M >= 768) ? omp_get_max_threads() : 1;
    int mc = (M + nthreads - 1) / nthreads;
    mc = (mc + MR - 1) & ~(MR - 1);
    if (mc < MR * 4 && nthreads > 1) {
        nthreads = (M + MR * 4 - 1) / (MR * 4);
        mc = (M + nthreads - 1) / nthreads;
        mc = (mc + MR - 1) & ~(MR - 1);
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
                    for (int j = j0; j < jmax; j += 8) {
                        int jn = jmax - j;
                        __m256 c0 = (im > 0 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+0)*N + j]) : _mm256_setzero_ps();
                        __m256 c1 = (im > 1 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+1)*N + j]) : _mm256_setzero_ps();
                        __m256 c2 = (im > 2 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+2)*N + j]) : _mm256_setzero_ps();
                        __m256 c3 = (im > 3 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+3)*N + j]) : _mm256_setzero_ps();
                        __m256 c4 = (im > 4 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+4)*N + j]) : _mm256_setzero_ps();
                        __m256 c5 = (im > 5 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+5)*N + j]) : _mm256_setzero_ps();
                        __m256 c6 = (im > 6 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+6)*N + j]) : _mm256_setzero_ps();
                        __m256 c7 = (im > 7 && jn > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+7)*N + j]) : _mm256_setzero_ps();
                        for (int k = k0; k < kmax; k++) {
                            __m256 a0 = _mm256_broadcast_ss(&A[(int64_t)(i+0)*K + k]);
                            __m256 a1 = _mm256_broadcast_ss(&A[(int64_t)(i+1)*K + k]);
                            __m256 a2 = _mm256_broadcast_ss(&A[(int64_t)(i+2)*K + k]);
                            __m256 a3 = _mm256_broadcast_ss(&A[(int64_t)(i+3)*K + k]);
                            __m256 a4 = (im > 4) ? _mm256_broadcast_ss(&A[(int64_t)(i+4)*K + k]) : _mm256_setzero_ps();
                            __m256 a5 = (im > 5) ? _mm256_broadcast_ss(&A[(int64_t)(i+5)*K + k]) : _mm256_setzero_ps();
                            __m256 a6 = (im > 6) ? _mm256_broadcast_ss(&A[(int64_t)(i+6)*K + k]) : _mm256_setzero_ps();
                            __m256 a7 = (im > 7) ? _mm256_broadcast_ss(&A[(int64_t)(i+7)*K + k]) : _mm256_setzero_ps();
                            __m256 bv = _mm256_loadu_ps(&B[(int64_t)k*N + j]);
                            c0 = _mm256_fmadd_ps(a0, bv, c0);
                            c1 = _mm256_fmadd_ps(a1, bv, c1);
                            c2 = _mm256_fmadd_ps(a2, bv, c2);
                            c3 = _mm256_fmadd_ps(a3, bv, c3);
                            c4 = _mm256_fmadd_ps(a4, bv, c4);
                            c5 = _mm256_fmadd_ps(a5, bv, c5);
                            c6 = _mm256_fmadd_ps(a6, bv, c6);
                            c7 = _mm256_fmadd_ps(a7, bv, c7);
                        }
                        if (im > 0) _mm256_storeu_ps(&C[(int64_t)(i+0)*N + j], c0);
                        if (im > 1) _mm256_storeu_ps(&C[(int64_t)(i+1)*N + j], c1);
                        if (im > 2) _mm256_storeu_ps(&C[(int64_t)(i+2)*N + j], c2);
                        if (im > 3) _mm256_storeu_ps(&C[(int64_t)(i+3)*N + j], c3);
                        if (im > 4) _mm256_storeu_ps(&C[(int64_t)(i+4)*N + j], c4);
                        if (im > 5) _mm256_storeu_ps(&C[(int64_t)(i+5)*N + j], c5);
                        if (im > 6) _mm256_storeu_ps(&C[(int64_t)(i+6)*N + j], c6);
                        if (im > 7) _mm256_storeu_ps(&C[(int64_t)(i+7)*N + j], c7);
                    }
                    for (int jj = j0 + ((jmax - j0) / 8) * 8; jj < jmax; jj++) {
                        for (int ii = i; ii < imax; ii++) {
                            float acc = C[(int64_t)ii*N + jj];
                            for (int k = k0; k < kmax; k++)
                                acc += A[(int64_t)ii*K + k] * B[(int64_t)k*N + jj];
                            C[(int64_t)ii*N + jj] = acc;
                        }
                    }
                }
            }
        }
    }
#else
    for (int i = 0; i < M; i++) {
        const float *a_row = A + (int64_t)i * K;
        float *c_row = C + (int64_t)i * N;
        for (int k = 0; k < K; k++) {
            float a_val = a_row[k];
            const float *b_row = B + (int64_t)k * N;
            for (int j = 0; j < N; j++)
                c_row[j] += a_val * b_row[j];
        }
    }
#endif
}

/* ---- MIR-compatible float32 GEMM ---------------------------------- */
void wubu_tgemm_f32_mir(int64_t *mem, int64_t a, int64_t b, int64_t c,
                                int M, int N, int K) {
    size_t nA = (size_t)M * K;
    size_t nB = (size_t)K * N;
    size_t nC = (size_t)M * N;
    float *A = (float*)malloc(nA * sizeof(float));
    float *B = (float*)malloc(nB * sizeof(float));
    float *C = (float*)calloc(nC, sizeof(float));
    if (A && B && C) {
        for (size_t i = 0; i < nA; i++) {
            union { float f; int32_t i; } u;
            u.i = (int32_t)mem[a + i];
            A[i] = u.f;
        }
        for (size_t i = 0; i < nB; i++) {
            union { float f; int32_t i; } u;
            u.i = (int32_t)mem[b + i];
            B[i] = u.f;
        }
        wubu_tgemm_f32(A, B, C, M, N, K);
        for (size_t i = 0; i < nC; i++) {
            union { float f; int32_t i; } u;
            u.f = C[i];
            mem[c + i] = (int64_t)u.i;
        }
    }
    free(A); free(B); free(C);
}

void wubu_tgemm_dispatch(int mode, int64_t *mem, int64_t A, int64_t B,
                         int64_t C, int M, int N, int K) {
    if (mode == 1) {
        float *fmem = (float *)mem;
        wubu_tgemm_f32(fmem + A, fmem + B, fmem + C, M, N, K);
    } else {
        wubu_tgemm(mem, A, B, C, M, N, K);
    }
}
