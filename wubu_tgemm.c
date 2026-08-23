/* wubu_tgemm.c — shared tiled int64 GEMM kernel (C += A*B, row-major).
 * One canonical implementation used by the x86-64 JIT libcall lowering,
 * the MIR interpreter, and every retro-ISA hostcall escape hatch.
 * C18 pure, no external deps. */
#include "wubu_tgemm.h"

void wubu_tgemm(int64_t *mem, int64_t A, int64_t B,
                int64_t C, int M, int N, int K)
{
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
