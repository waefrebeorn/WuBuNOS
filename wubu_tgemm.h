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

/* AVX-512DQ native path — separate TU compiled with -mavx512dq (x86-64 only). */
#if defined(__x86_64__)
void tgemm_avx512(int64_t *mem, int64_t A, int64_t B,
                  int64_t C, int M, int N, int K);
#endif

/* Byte-addressed variant (8-bit retro ISAs): cells are little-endian
 * int64 at byte offset cell*8 within mem. */
void wubu_tgemm_mem8(uint8_t *mem, uint32_t A, uint32_t B, uint32_t C,
                     int M, int N, int K);

/* Float32 GEMM for ML inference: C += A*B, float32 row-major.
 * Uses AVX2+FMA when available, scalar fallback otherwise. */
void wubu_tgemm_f32(const float *A, const float *B, float *C,
                    int M, int N, int K);

/* MIR-compatible float32 GEMM: handles int64 cell memory layout */
void wubu_tgemm_f32_mir(int64_t *mem, int64_t a, int64_t b, int64_t c,
                        int M, int N, int K);

/* BF16 GEMM: A and B are BF16 (uint16_t), C is FP32 */
void wubu_tgemm_bf16_avx512(const uint16_t *A_bf16, const uint16_t *B_bf16, float *C,
                             int M, int N, int K);

/* Dispatch: mode=0 int64, mode=1 float32 */
void wubu_tgemm_dispatch(int mode, int64_t *mem, int64_t A, int64_t B,
                         int64_t C, int M, int N, int K);

#endif /* WUBU_TGEMM_H */
