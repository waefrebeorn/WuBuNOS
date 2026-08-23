/*
 * wubu_softfloat.h -- pure C11 IEEE-754 binary32/binary64 software float.
 *
 * The float runtime for every ISA backend that lacks hardware FPU:
 * 6502, Z80, 8086, m68k(soft path), 8051, AVR, PIC, and any future
 * interpreter target. Bit-exact round-to-nearest-even per IEEE 754-2008
 * for the four basic ops + compare + conversions. No libm, no FP
 * hardware assumptions beyond an integer type >= 64 bits.
 *
 * Representation: f32 = uint32_t bits, f64 = uint64_t bits (IEEE bit
 * patterns). Callers pass/return bits; conversion helpers handle
 * host-float interop for tests and JIT paths that do have an FPU.
 *
 * C11, self-contained, zero external deps.
 */
#ifndef WUBU_SOFTFLOAT_H
#define WUBU_SOFTFLOAT_H

#include <stdint.h>
#include <stddef.h>

/* ---- core arithmetic (bit patterns in, bit patterns out) ---- */
uint32_t wubu_sf_f32_add(uint32_t a, uint32_t b);
uint32_t wubu_sf_f32_sub(uint32_t a, uint32_t b);   /* a - b */
uint32_t wubu_sf_f32_mul(uint32_t a, uint32_t b);
uint32_t wubu_sf_f32_div(uint32_t a, uint32_t b);

uint64_t wubu_sf_f64_add(uint64_t a, uint64_t b);
uint64_t wubu_sf_f64_sub(uint64_t a, uint64_t b);
uint64_t wubu_sf_f64_mul(uint64_t a, uint64_t b);
uint64_t wubu_sf_f64_div(uint64_t a, uint64_t b);

/* ---- compares: return -1, 0, +1 (unordered -> 2) ---- */
int wubu_sf_f32_cmp(uint32_t a, uint32_t b);
int wubu_sf_f64_cmp(uint64_t a, uint64_t b);

/* ---- conversions ---- */
uint32_t wubu_sf_i64_to_f32(int64_t i);
uint64_t wubu_sf_i64_to_f64(int64_t i);
uint64_t wubu_sf_f32_to_f64(uint32_t a);
uint32_t wubu_sf_f64_to_f32(uint64_t a);
int64_t  wubu_sf_f32_to_i64(uint32_t a);   /* trunc toward zero; NaN->0 */
int64_t  wubu_sf_f64_to_i64(uint64_t a);

/* ---- classification / predicates ---- */
int wubu_sf_f32_is_nan(uint32_t a);
int wubu_sf_f64_is_nan(uint64_t a);
int wubu_sf_f32_is_inf(uint32_t a);
int wubu_sf_f64_is_inf(uint64_t a);
uint32_t wubu_sf_f32_abs(uint32_t a);
uint64_t wubu_sf_f64_abs(uint64_t a);
uint32_t wubu_sf_f32_neg(uint32_t a);
uint64_t wubu_sf_f64_neg(uint64_t a);
uint32_t wubu_sf_f32_sqrt(uint32_t a);     /* Newton-Raphson on reciprocal sqrt */
uint64_t wubu_sf_f64_sqrt(uint64_t a);

/* ---- host interop (tests, JIT with real FPU) ---- */
static inline uint32_t wubu_sf_f32_from_host(float f) {
    union { float f; uint32_t u; } c; c.f = f; return c.u;
}
static inline float wubu_sf_f32_to_host(uint32_t u) {
    union { float f; uint32_t u; } c; c.u = u; return c.f;
}
static inline uint64_t wubu_sf_f64_from_host(double d) {
    union { double d; uint64_t u; } c; c.d = d; return c.u;
}
static inline double wubu_sf_f64_to_host(uint64_t u) {
    union { double d; uint64_t u; } c; c.u = u; return c.d;
}

/* bfloat16: the tensor-core dtype, the top 16 bits of f32. Conversion is
 * round-to-nearest-even on the low 16 bits (NaN stays NaN). */
uint32_t wubu_sf_bf16_to_f32(uint16_t h);
uint16_t wubu_sf_f32_to_bf16(uint32_t a);

#endif /* WUBU_SOFTFLOAT_H */
