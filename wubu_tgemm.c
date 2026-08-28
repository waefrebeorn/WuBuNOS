/* wubu_tgemm.c — shared tiled int64 GEMM kernel (C += A*B, row-major).
 * One canonical implementation used by the x86-64 JIT libcall lowering,
 * the MIR interpreter, and every retro-ISA hostcall escape hatch.
 * C18 pure, no external deps. */
#include "wubu_tgemm.h"


/* ---- AVX2 fast path (x86-64 only) ------------------------------------
 * 4 columns of B are broadcast; each j-iteration accumulates 4 int64
 * products per row via vpmullq+vpaddd-style 64-bit lanes. Requires
 * AVX2 (vpmullq is really AVX512DQ! — so use scalar mul into vectors:
 * we instead vectorize the j dimension with vpaddq after scalar muls).
 * Simpler correct approach: keep k-loop scalar but process 4 j's at
 * once with gathered B values. Actually int64 mul needs AVX512DQ for
 * full-width vpmullq; on plain AVX2 we emulate with 32-bit halves.
 * For correctness-first: vectorize the N dimension accumulation using
 * vpbroadcastq of a-products and 64-bit integer mul via two 32-bit
 * multiplies (mulx emulation). */
#if defined(__x86_64__)
#include <immintrin.h>

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
                /* broadcast a[k], multiply by 4 b-lanes, accumulate.
                 * 64x64->64 low-half multiply = _mm256_mullo_epi64
                 * (AVX512DQ only). Emulate: split into hi/lo 32-bit. */
                __m256i bv = _mm256_loadu_si256((const __m256i *)&mem[B + (int64_t)k * N + j]);
                __m256i av = _mm256_set1_epi64x(a[k]);
                /* lo32 parts */
                __m256i alo = _mm256_and_si256(av, _mm256_set1_epi64x(0xFFFFFFFFLL));
                __m256i blo = _mm256_and_si256(bv, _mm256_set1_epi64x(0xFFFFFFFFLL));
                __m256i plo = _mm256_mul_epu32(alo, blo);
                __m256i phi = _mm256_mul_epu32(_mm256_srli_epi64(av, 32),
                                               _mm256_srli_epi64(bv, 32));
                /* combine: (phi << 32) + ((alo*bhi + blo*ahi) << 32)? Full
                 * 64-bit product needs cross terms. Standard trick: */
                __m256i ahi = _mm256_srli_epi64(av, 32);
                __m256i bhi = _mm256_srli_epi64(bv, 32);
                __m256i mid1 = _mm256_mul_epu32(alo, bhi);
                __m256i mid2 = _mm256_mul_epu32(ahi, blo);
                /* result low 64 = plo + ((mid1+mid2) << 32) */
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

static int avx2_ok = -1;
static int have_avx2(void) {
    if (avx2_ok < 0) {
        __builtin_cpu_init();
        avx2_ok = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    return avx2_ok;
}
#endif /* __x86_64__ */

void wubu_tgemm(int64_t *mem, int64_t A, int64_t B,
                int64_t C, int M, int N, int K)
{
#if defined(__x86_64__)
    if (have_avx2()) { tgemm_avx2(mem, A, B, C, M, N, K); return; }
#endif
    int i = 0;
    for (; i + 3 < M; i += 4) {
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
    for (; i < M; i++) {                       /* tail rows */
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
    /* Byte-addressed variant for 8-bit retro ISAs: cells are little-endian
     * int64 at byte offset cell*8. Delegates through a staging buffer per
     * row-block would be slow; instead index directly via LE accessors. */
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

/* ---- Float32 GEMM (for ML inference) ------------------------------- */
/* wubu_tgemm_f32: C += A*B where A,B,C are float32 row-major matrices.
 * Uses AVX2 256-bit vectors (8 floats per vector) with FMA when available.
 * Implements tiled GEMM with 32-column micro-kernel for cache efficiency.
 * This is the hot path for transformer inference. */
void wubu_tgemm_f32(const float *A, const float *B, float *C,
                    int M, int N, int K) {
#if defined(__AVX2__)
    #include <immintrin.h>
    /* Tile sizes: MC=16 rows, NC=128 cols, KC=32 k-iterations.
     * L1: 32KB = 8192 floats. 16*32 + 32*128 + 16*128 = 512+4096+2048 = 6656 < 8192 */
    const int MC = 16, NC = 128, KC = 32;
    for (int i0 = 0; i0 < M; i0 += MC) {
        int imax = i0 + MC; if (imax > M) imax = M;
        for (int k0 = 0; k0 < K; k0 += KC) {
            int kmax = k0 + KC; if (kmax > K) kmax = K;
            for (int j0 = 0; j0 < N; j0 += NC) {
                int jmax = j0 + NC; if (jmax > N) jmax = N;
                /* Micro-kernel: process 8 columns per AVX2 vector */
                for (int i = i0; i < imax; i++) {
                    const float *a_row = A + (int64_t)i * K;
                    float *c_row = C + (int64_t)i * N;
                    int j = j0;
#if defined(__FMA__)
                    /* 32-column unrolled FMA path */
                    for (; j + 31 < jmax; j += 32) {
                        __m256 c0 = _mm256_loadu_ps(&c_row[j]);
                        __m256 c1 = _mm256_loadu_ps(&c_row[j+8]);
                        __m256 c2 = _mm256_loadu_ps(&c_row[j+16]);
                        __m256 c3 = _mm256_loadu_ps(&c_row[j+24]);
                        for (int k = k0; k < kmax; k++) {
                            __m256 a_val = _mm256_set1_ps(a_row[k]);
                            c0 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(&B[(int64_t)k*N+j]), c0);
                            c1 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(&B[(int64_t)k*N+j+8]), c1);
                            c2 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(&B[(int64_t)k*N+j+16]), c2);
                            c3 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(&B[(int64_t)k*N+j+24]), c3);
                        }
                        _mm256_storeu_ps(&c_row[j], c0);
                        _mm256_storeu_ps(&c_row[j+8], c1);
                        _mm256_storeu_ps(&c_row[j+16], c2);
                        _mm256_storeu_ps(&c_row[j+24], c3);
                    }
#endif
                    /* 8-column standard path */
                    for (; j + 7 < jmax; j += 8) {
                        __m256 acc = _mm256_loadu_ps(&c_row[j]);
                        for (int k = k0; k < kmax; k++) {
                            __m256 a_val = _mm256_set1_ps(a_row[k]);
                            __m256 b_val = _mm256_loadu_ps(&B[(int64_t)k * N + j]);
#if defined(__FMA__)
                            acc = _mm256_fmadd_ps(a_val, b_val, acc);
#else
                            acc = _mm256_add_ps(acc, _mm256_mul_ps(a_val, b_val));
#endif
                        }
                        _mm256_storeu_ps(&c_row[j], acc);
                    }
                    /* Scalar tail */
                    for (; j < jmax; j++) {
                        float acc = c_row[j];
                        for (int k = k0; k < kmax; k++)
                            acc += a_row[k] * B[(int64_t)k * N + j];
                        c_row[j] = acc;
                    }
                }
            }
        }
    }
#else
    /* Scalar fallback with cache-friendly k-j ordering */
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
/* wubu_tgemm_f32_mir: GEMM for MIR memory layout where each float is
 * stored as a 32-bit IEEE-754 value in the lower half of an int64 cell.
 * Extracts floats, computes, and stores back. */
void wubu_tgemm_f32_mir(int64_t *mem, int64_t a, int64_t b, int64_t c,
                                int M, int N, int K) {
    /* Allocate temporary float buffers */
    size_t nA = (size_t)M * K;
    size_t nB = (size_t)K * N;
    size_t nC = (size_t)M * N;
    float *A = (float*)malloc(nA * sizeof(float));
    float *B = (float*)malloc(nB * sizeof(float));
    float *C = (float*)calloc(nC, sizeof(float));
    if (!A || !B || C) {
        /* Extract floats from MIR memory (lower 32 bits of each cell) */
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
        /* Compute */
        wubu_tgemm_f32(A, B, C, M, N, K);
        /* Store back to MIR memory */
        for (size_t i = 0; i < nC; i++) {
            union { float f; int32_t i; } u;
            u.f = C[i];
            mem[c + i] = (int64_t)u.i;
        }
    }
    free(A); free(B); free(C);
}

/* ---- Hybrid int64/float32 dispatch ---------------------------------- */
/* wubu_tgemm_dispatch: picks the right kernel based on a mode flag.
 * mode=0: int64 (original), mode=1: float32 (ML inference) */
void wubu_tgemm_dispatch(int mode, int64_t *mem, int64_t A, int64_t B,
                         int64_t C, int M, int N, int K) {
    if (mode == 1) {
        /* Float32: treat memory as float arrays */
        float *fmem = (float *)mem;
        int64_t fA = A * 2; /* int64 cells -> float cells (2 floats per int64) */
        int64_t fB = B * 2;
        int64_t fC = C * 2;
        /* Actually, float32 uses 1 cell per element, not 2.
         * The caller should pass float-indexed offsets. */
        wubu_tgemm_f32(fmem + A, fmem + B, fmem + C, M, N, K);
    } else {
        wubu_tgemm(mem, A, B, C, M, N, K);
    }
}
