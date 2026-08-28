/* wubu_tgemm.c — shared tiled int64/float32 GEMM kernel (C += A*B, row-major).
 * One canonical implementation used by the x86-64 JIT libcall lowering,
 * the MIR interpreter, and every retro-ISA hostcall escape hatch.
 * C18 pure, no external deps. */
#include "wubu_tgemm.h"
#include <alloca.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>

/* ---- AVX2 fast path ------------------------------------------------
 * 4 columns of B are broadcast; each j-iteration accumulates 4 int64
 * products per row. AVX2 lacks 64-bit integer mul; emulate via 32-bit halves.
 */
__attribute__((target("avx2")))
static void tgemm_avx2(int64_t *mem, int64_t A, int64_t B,
                       int64_t C, int M, int N, int K)
{
    const __m256i zero = _mm256_setzero_si256();
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

extern void tgemm_avx512(int64_t *mem, int64_t A, int64_t B,
                              int64_t C, int M, int N, int K);

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
    if (have_avx512()) { extern void tgemm_avx512(int64_t*,int64_t,int64_t,int64_t,int,int,int); tgemm_avx512(mem, A, B, C, M, N, K); return; }
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

void wubu_tgemm_f32(const float *A, const float *B, float *C,
                    int M, int N, int K) {
#if defined(__AVX2__)
    /* SOTA micro-kernel: MR=4 x NR=16 (salykova.github.io, BLIS, EXO).
     * 4 rows x 16 cols = 8 YMM accumulators (c0_lo, c0_hi, ..., c3_lo, c3_hi).
     * 8 C + 2 B + 1 A = 11 YMM registers (of 16 available), no spill on Zen.
     * MR=4 divides all power-of-2 sizes evenly (no remainder pathology).
     * NR=16 = two full YMM loads of B, saturates both FMA pipes.
     * KC=64 fits L1d (32KB). No software prefetch (degrades Zen, rs-10345403).
     * -frename-registers -funroll-loops critical for reg alloc.
     *
     * Cache blocking: 3-level (MC/KC/NC) with OpenMP parallel over MC.
     * Inner loop: ikj order for B-panel reuse across k-iterations. */
    const int MR = 4, NR = 16, KC = 64, NC = 256;
    int nthreads = (M * N * K >= 64000000) ? omp_get_max_threads() : 1;
    int mc = (M + nthreads - 1) / nthreads;
    /* Round down to multiple of MR (MR=4 is power of 2, bitwise works) */
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
                    /* Vectorized inner loop: 16 columns per iteration (NR=16) */
                    for (int j = j0; j + NR - 1 < jmax; j += NR) {
                        /* 8 YMM accumulators: 4 rows x 2 halves (cols j..j+7, j+8..j+15) */
                        __m256 c0l = (im > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+0)*N + j]) : _mm256_setzero_ps();
                        __m256 c0h = (im > 0) ? _mm256_loadu_ps(&C[(int64_t)(i+0)*N + j + 8]) : _mm256_setzero_ps();
                        __m256 c1l = (im > 1) ? _mm256_loadu_ps(&C[(int64_t)(i+1)*N + j]) : _mm256_setzero_ps();
                        __m256 c1h = (im > 1) ? _mm256_loadu_ps(&C[(int64_t)(i+1)*N + j + 8]) : _mm256_setzero_ps();
                        __m256 c2l = (im > 2) ? _mm256_loadu_ps(&C[(int64_t)(i+2)*N + j]) : _mm256_setzero_ps();
                        __m256 c2h = (im > 2) ? _mm256_loadu_ps(&C[(int64_t)(i+2)*N + j + 8]) : _mm256_setzero_ps();
                        __m256 c3l = (im > 3) ? _mm256_loadu_ps(&C[(int64_t)(i+3)*N + j]) : _mm256_setzero_ps();
                        __m256 c3h = (im > 3) ? _mm256_loadu_ps(&C[(int64_t)(i+3)*N + j + 8]) : _mm256_setzero_ps();
                        /* Inner k-loop: broadcast A[i][k], FMA with B[k][j..j+15] */
                        for (int k = k0; k < kmax; k++) {
                            __m256 b0 = _mm256_loadu_ps(&B[(int64_t)k*N + j]);
                            __m256 b1 = _mm256_loadu_ps(&B[(int64_t)k*N + j + 8]);
                            if (im > 0) { __m256 a = _mm256_broadcast_ss(&A[(int64_t)(i+0)*K + k]);
                                c0l = _mm256_fmadd_ps(a, b0, c0l); c0h = _mm256_fmadd_ps(a, b1, c0h); }
                            if (im > 1) { __m256 a = _mm256_broadcast_ss(&A[(int64_t)(i+1)*K + k]);
                                c1l = _mm256_fmadd_ps(a, b0, c1l); c1h = _mm256_fmadd_ps(a, b1, c1h); }
                            if (im > 2) { __m256 a = _mm256_broadcast_ss(&A[(int64_t)(i+2)*K + k]);
                                c2l = _mm256_fmadd_ps(a, b0, c2l); c2h = _mm256_fmadd_ps(a, b1, c2h); }
                            if (im > 3) { __m256 a = _mm256_broadcast_ss(&A[(int64_t)(i+3)*K + k]);
                                c3l = _mm256_fmadd_ps(a, b0, c3l); c3h = _mm256_fmadd_ps(a, b1, c3h); }
                        }
                        /* No branch in store path — all 4 rows always stored */
                        _mm256_storeu_ps(&C[(int64_t)(i+0)*N + j], c0l); _mm256_storeu_ps(&C[(int64_t)(i+0)*N + j + 8], c0h);
                        _mm256_storeu_ps(&C[(int64_t)(i+1)*N + j], c1l); _mm256_storeu_ps(&C[(int64_t)(i+1)*N + j + 8], c1h);
                        _mm256_storeu_ps(&C[(int64_t)(i+2)*N + j], c2l); _mm256_storeu_ps(&C[(int64_t)(i+2)*N + j + 8], c2h);
                        _mm256_storeu_ps(&C[(int64_t)(i+3)*N + j], c3l); _mm256_storeu_ps(&C[(int64_t)(i+3)*N + j + 8], c3h);
                    }
                    /* Scalar remainder for columns not divisible by NR=16 */
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
#endif
}

/* ---- MIR-compatible float32 GEMM ---------------------------------- */
/* The MIR memory model stores f32 bit patterns in the lower 32 bits of
 * int64 cells (upper 32 bits are zero). wubu_tgemm_f32 expects contiguous
 * 4-byte floats, so we need to unpack into a temporary buffer.
 *
 * Optimization: use a single stack-allocated buffer for C (the result)
 * instead of 3 malloc/calloc calls. We unpack A and B in-place via
 * memcpy (no malloc), and accumulate C in a stack buffer.
 *
 * SOTA reference: BLIS micro-kernel packing achieves 90-95% of peak
 * by avoiding heap allocation in the inner loop (EXO framework, arxiv:2310.17408). */
void wubu_tgemm_f32_mir(int64_t *mem, int64_t a, int64_t b, int64_t c,
                                int M, int N, int K) {
    size_t nA = (size_t)M * K;
    size_t nB = (size_t)K * N;
    size_t nC = (size_t)M * N;
    /* Use a single stack buffer to hold all three matrices if small enough,
     * otherwise fall back to heap. Stack buffer avoids malloc/free overhead
     * which dominates for small GEMMs (< 64x64). */
    if (nA + nB + nC <= 65536) {
        /* Stack path: single allocation, no malloc/free overhead */
        float *buf = (float*)alloca((nA + nB + nC) * sizeof(float));
        float *A = buf;
        float *B = A + nA;
        float *C = B + nB;
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
        for (size_t i = 0; i < nC; i++) {
            union { float f; int32_t i; } u;
            u.i = (int32_t)mem[c + i];
            C[i] = u.f;
        }
        wubu_tgemm_f32(A, B, C, M, N, K);
        for (size_t i = 0; i < nC; i++) {
            union { float f; int32_t i; } u;
            u.f = C[i];
            mem[c + i] = (int64_t)u.i;
        }
    } else {
        /* Heap path for large GEMMs */
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
            for (size_t i = 0; i < nC; i++) {
                union { float f; int32_t i; } u;
                u.i = (int32_t)mem[c + i];
                C[i] = u.f;
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