/* wubu_tgemm.h — shared tiled int64 GEMM kernel interface.
 * C += A*B, row-major, int64 cells. One canonical implementation for the
 * x86-64 JIT libcall, MIR interpreter, and retro-ISA hostcalls.
 * C18 pure, no external deps. */
#ifndef WUBU_TGEMM_H
#define WUBU_TGEMM_H

#include <stdint.h>
#include <stddef.h>

/* Pointer-addressed variant (host backends). */
void wubu_tgemm(int64_t *mem, int64_t A, int64_t B,
                int64_t C, int M, int N, int K);

/* Byte-addressed variant (8-bit retro ISAs): cells are little-endian
 * int64 at byte offset cell*8 within mem. */
void wubu_tgemm_mem8(uint8_t *mem, uint32_t A, uint32_t B, uint32_t C,
                     int M, int N, int K);

/* Float32 GEMM for ML inference: C += A*B, float32 row-major.
 * Uses AVX2+FMA when available, scalar fallback otherwise. */
void wubu_tgemm_f32(const float *A, const float *B, float *C,
                    int M, int N, int K);

/* Dispatch: mode=0 int64, mode=1 float32 */
void wubu_tgemm_dispatch(int mode, int64_t *mem, int64_t A, int64_t B,
                         int64_t C, int M, int N, int K);

#endif /* WUBU_TGEMM_H */
