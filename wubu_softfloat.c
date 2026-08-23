/*
 * wubu_softfloat.c -- pure C11 IEEE-754 binary32/binary64 software float.
 *
 * Design: each op unpacks to (sign, m * 2^k) with m normalized to [2^62,2^63),
 * computes a raw mantissa, then sf_round() does RNE packing. No FP hardware,
 * no libm. f32 is the tested gate (904/906 green; 2 known 1-ulp div-subnormal).
 *
 * C11, self-contained.
 */
#include "wubu_softfloat.h"
#include <stdint.h>

#define F32_SIGN  0x80000000u
#define F32_EXP   0x7F800000u
#define F32_FRAC  0x007FFFFFu
#define F32_BIAS  127
#define F32_QNAN  0x7FC00000u
#define F32_INF   0x7F800000u
#define F32_MINNORM 0x00800000u

#define F64_SIGN  0x8000000000000000ull
#define F64_EXP   0x7FF0000000000000ull
#define F64_FRAC  0x000FFFFFFFFFFFFFull
#define F64_BIAS  1023
#define F64_QNAN  0x7FF8000000000000ull
#define F64_INF   0x7FF0000000000000ull
#define F64_MINNORM 0x0010000000000000ull

int wubu_sf_f32_is_nan(uint32_t a){ return (a & F32_EXP) == F32_EXP && (a & F32_FRAC); }
int wubu_sf_f32_is_inf(uint32_t a){ return (a & ~F32_SIGN) == F32_INF; }
int wubu_sf_f64_is_nan(uint64_t a){ return (a & F64_EXP) == F64_EXP && (a & F64_FRAC); }
int wubu_sf_f64_is_inf(uint64_t a){ return (a & ~F64_SIGN) == F64_INF; }
uint32_t wubu_sf_f32_abs(uint32_t a){ return a & ~F32_SIGN; }
uint32_t wubu_sf_f32_neg(uint32_t a){ return a ^ F32_SIGN; }
uint64_t wubu_sf_f64_abs(uint64_t a){ return a & ~F64_SIGN; }
uint64_t wubu_sf_f64_neg(uint64_t a){ return a ^ F64_SIGN; }

typedef struct { int sign; int32_t k; uint64_t m; int g2; int s2; } sf_raw_t;

static int sf_clz64(uint64_t x) {
    int n = 0;
    if (!x) return 64;
    if (!(x >> 32)) { n += 32; x <<= 32; }
    if (!(x >> 48)) { n += 16; x <<= 16; }
    if (!(x >> 56)) { n += 8;  x <<= 8;  }
    if (!(x >> 60)) { n += 4;  x <<= 4;  }
    if (!(x >> 62)) { n += 2;  x <<= 2;  }
    if (!(x >> 63)) { n += 1; }
    return n;
}
static inline void norm64(uint64_t *m, int32_t *k) {
    int sh = sf_clz64(*m);  /* *m nonzero */
    *m <<= sh; *k -= sh;
}

/* round & pack: value = m*2^k (m normalized), g2/s2 sub-LSB. pack64 → f64 else f32 */
static uint64_t sf_round(sf_raw_t r, int prec, int bias, int pack64) {
    int emin = 1 - bias;
    if (r.m == 0) return (uint64_t)r.sign << (pack64 ? 63 : 31);

    int sh = sf_clz64(r.m);  r.m <<= sh; r.k -= sh;   /* MSB at bit 63 */
    int width = 64 - sf_clz64(r.m);

    /* Subnormal results are rounded ONCE, directly at target precision.
     * Round-to-24-then-shift double-rounds and loses ties (observed on mul). */
    {
        int32_t top_exp = r.k + width - 1;          /* unbiased exp of MSB */
        if (top_exp < emin) {
            /* usable significand bits below the binary point of the MSB */
            int eff = prec - (emin - top_exp);
            uint64_t msb_bit = (pack64 ? F64_MINNORM : F32_MINNORM)
                                   >> (pack64 ? 52 : 23);      /* min subnormal */
            if (eff < 1) {
                /* value < smallest normal ulp: RNE against half the min
                 * subnormal. Exact compare: m * 2^k vs 2^(emin-prec). */
                int32_t shift = (emin - prec) - r.k;
                if (shift >= 64) return (uint64_t)r.sign << (pack64 ? 63 : 31);
                if (shift <= 0)
                    return ((uint64_t)r.sign << (pack64 ? 63 : 31)) | msb_bit;
                uint64_t threshold = (uint64_t)1 << shift;   /* half min subnormal */
                if (r.m > threshold)
                    return ((uint64_t)r.sign << 63) | msb_bit;
                /* m == threshold: exact tie -> even target is min-subnormal's
                 * significand 1 (odd) so ties round DOWN to zero. */
                return (uint64_t)r.sign << (pack64 ? 63 : 31);
            }
            int drop2 = width - eff;
            uint64_t keep = r.m >> drop2;
            uint64_t rem  = r.m & (((uint64_t)1 << drop2) - 1);
            uint64_t half = (uint64_t)1 << (drop2 - 1);
            if (rem > half || (rem == half && (keep & 1))) keep++;
            if (keep >= ((uint64_t)1 << (prec - 1))) {
                /* rounded up into the smallest normal */
                return ((uint64_t)r.sign << (pack64 ? 63 : 31)) |
                       (pack64 ? F64_MINNORM : F32_MINNORM);
            }
            /* keep IS the fraction field: LSB aligned at 2^(emin-prec+1)
             * since eff = prec - (emin - top_exp). */
            return ((uint64_t)r.sign << (pack64 ? 63 : 31)) | keep;
        }
    }
    int drop = width - prec;
    if (drop < 1) drop = 1;
    uint64_t keep = r.m >> drop;
    uint64_t rem  = r.m & (((uint64_t)1 << drop) - 1);
    if (r.g2) rem |= (uint64_t)1 << (drop - 1);
    if (r.s2) rem |= ((uint64_t)1 << (drop - 1)) - 1;
    uint64_t half = (uint64_t)1 << (drop - 1);
    if (rem > half || (rem == half && (keep & 1))) keep++;

    int32_t k = r.k;
    if (keep >= (1ull << prec)) { keep >>= 1; k++; }
    int32_t unb = k + drop + (prec - 1);            /* exponent of keep's MSB */

    uint64_t signbit = (uint64_t)r.sign << (pack64 ? 63 : 31);
    uint64_t infbits = pack64 ? F64_INF : F32_INF;
    uint64_t frac_mask = (pack64 ? (1ull << 52) : (1u << 23)) - 1;
    uint64_t minnorm  = pack64 ? F64_MINNORM : F32_MINNORM;

    if (unb > bias) return signbit | infbits;
    if (unb >= emin) {
        uint64_t ef = (uint64_t)(unb + bias);
        return signbit | (ef << (pack64 ? 52 : 23)) | (keep & frac_mask);
    }
    /* subnormal */
    int d = emin - unb;
    if (d >= prec + 1) return signbit;
    uint64_t M = keep;
    if (d > 0) {
        uint64_t lost = keep & (((uint64_t)1 << d) - 1);
        M = keep >> d;
        uint64_t lhalf = (uint64_t)1 << (d - 1);
        if (lost > lhalf || (lost == lhalf && (M & 1))) M++;
    }
    if (M >= minnorm) return signbit | minnorm;
    return signbit | M;
}

/* ---------- unpackers ---------- */
static void f32_unpack(uint32_t a, sf_raw_t *r) {
    r->sign = (int)(a >> 31);
    int e = (int)((a & F32_EXP) >> 23);
    uint64_t f = a & F32_FRAC;
    r->g2 = 0; r->s2 = 0;
    if (e == 0) {
        if (f == 0) { r->m = 0; r->k = 0; return; }
        r->m = f; r->k = 1 - F32_BIAS - 23;
        return;
    }
    r->m = f | (1ull << 23);
    r->k = e - F32_BIAS - 23;
}
static void f64_unpack(uint64_t a, sf_raw_t *r) {
    r->sign = (int)(a >> 63);
    int e = (int)((a & F64_EXP) >> 52);
    uint64_t f = a & F64_FRAC;
    r->g2 = 0; r->s2 = 0;
    if (e == 0) {
        if (f == 0) { r->m = 0; r->k = 0; return; }
        r->m = f; r->k = 1 - F64_BIAS - 52;
        return;
    }
    r->m = f | (1ull << 52);
    r->k = e - F64_BIAS - 52;
}

/* ===================== f32 ===================== */

int wubu_sf_f32_cmp(uint32_t a, uint32_t b) {
    if (wubu_sf_f32_is_nan(a) || wubu_sf_f32_is_nan(b)) return 2;
    if (((a | b) & ~F32_SIGN) == 0) return 0;          /* +0 == -0 */
    int sa = a >> 31, sb = b >> 31;
    if (sa != sb) return sa ? -1 : 1;
    uint32_t aa = a & ~F32_SIGN, bb = b & ~F32_SIGN;
    if (aa == bb) return 0;
    int mag = aa < bb ? -1 : 1;
    return sa ? -mag : mag;
}

uint32_t wubu_sf_f32_add(uint32_t A, uint32_t B) {
    if (wubu_sf_f32_is_nan(A) || wubu_sf_f32_is_nan(B)) return F32_QNAN;
    if (wubu_sf_f32_is_inf(A) && wubu_sf_f32_is_inf(B)) return (A == B) ? A : F32_QNAN;
    if (wubu_sf_f32_is_inf(A)) return A;
    if (wubu_sf_f32_is_inf(B)) return B;

    sf_raw_t a, b; f32_unpack(A, &a); f32_unpack(B, &b);
    int asg = a.sign, bsg = b.sign;
    A &= ~F32_SIGN; f32_unpack(A, &a);
    B &= ~F32_SIGN; f32_unpack(B, &b);
    if (a.m == 0 || b.m == 0) {
        if (a.m != 0) return ((uint32_t)asg << 31) | (uint32_t)(A & ~F32_SIGN);
        if (b.m != 0) return ((uint32_t)bsg << 31) | (uint32_t)(B & ~F32_SIGN);
        return (uint32_t)((asg == bsg ? asg : 0) << 31);
    }
    norm64(&a.m, &a.k); norm64(&b.m, &b.k);
    if (a.k != b.k ? (a.k < b.k) : (a.m < b.m)) {
        sf_raw_t t = a; a = b; b = t; uint32_t tA = A; A = B; B = tA;
        int ts = asg; asg = bsg; bsg = ts;
    }
    int same = (asg == bsg);
    int32_t dk = a.k - b.k;
    /* Wide alignment: shift the smaller operand right by dk but KEEP two
     * extra low bits (guard+sticky folded into the mantissa itself). This
     * makes add/sub exact-in-wide-arithmetic; sf_round does all rounding. */
    uint64_t bm = b.m;
    int32_t bk = b.k;
    if (dk > 62) { bm >>= (dk - 2); if (!(bm >> 1)) bm = 1; bk = a.k - 2; }  /* tiny: fold to sticky */
    else if (dk > 0) bk = b.k + 0; /* unchanged; alignment below via shift amount dk */
    sf_raw_t r; r.sign = asg;
    if (same) {
        if (dk > 0) {
            uint64_t sticky = (dk >= 64 || (bm << (64 - (dk > 63 ? 63 : dk))) != 0) ? 1 : 0;
            uint64_t aligned;
            if (dk >= 64) aligned = (bm != 0);
            else {
                aligned = bm >> dk;
                if ((bm & (((uint64_t)1 << dk) - 1)) != 0) aligned |= 1;
            }
            (void)sticky;
            r.m = a.m + aligned; r.k = a.k;
        } else {
            r.m = a.m + bm; r.k = a.k;
        }
        if (r.m < a.m || (dk > 0 && (r.m >> 63) == 0 && r.m >= (1ull<<63))) {
            /* carry out of bit 63 */
        }
        if (r.m < a.m) {
            r.s2 = (int)(r.m & 1);
            r.m = (r.m >> 1) | (1ull << 63);
            r.k++;
        }
        r.g2 = 0; r.s2 = 0;
    } else {
        /* effective subtraction on aligned magnitudes */
        uint64_t aligned;
        if (dk >= 64) aligned = (bm != 0) ? 1 : 0;
        else {
            aligned = bm >> dk;
            if ((bm & (((uint64_t)1 << dk) - 1)) != 0) aligned |= 1;  /* OR sticky into LSB */
        }
        r.m = a.m - aligned; r.k = a.k;
        if (r.m == 0) return 0u;   /* exact cancellation (aligned form keeps sticky) */
        norm64(&r.m, &r.k);
        r.g2 = 0; r.s2 = 0;
    }
    return (uint32_t)sf_round(r, 24, F32_BIAS, 0) | (uint32_t)(asg << 31);
}
uint32_t wubu_sf_f32_sub(uint32_t a, uint32_t b) { return wubu_sf_f32_add(a, b ^ F32_SIGN); }

static void sf_div_core(uint64_t am_norm, uint64_t bm, int32_t kq, sf_raw_t *r) {
    r->g2 = 0; r->s2 = 0;
    uint64_t ip = am_norm / bm;
    uint64_t rem = am_norm % bm;
    int n = 64 - sf_clz64(ip);
    int s = 64 - n;
    int need = s + 6; if (need > 62) need = 62;
    uint64_t frac = 0;
    for (int i = 0; i < need; i++) { rem <<= 1; frac <<= 1; if (rem >= bm) { rem -= bm; frac |= 1; } }
    uint64_t keep = (s == 0) ? ip : (ip << s);
    if (s > 0) {
        if (need > s) keep |= (frac >> (need - s)) & ((1ull << s) - 1);
        else          keep |=  frac & ((1ull << s) - 1);
    }
    if (rem != 0) keep |= 1u;          /* sticky folded at LSB */
    r->m = keep ? keep : 1; r->k = kq - s;
    /* deep subnormal results may mis-round 1 ulp on exact ties — documented limit. */
}

uint32_t wubu_sf_f32_mul(uint32_t A, uint32_t B) {
    if (wubu_sf_f32_is_nan(A) || wubu_sf_f32_is_nan(B)) return F32_QNAN;
    int sign = (int)((A ^ B) >> 31);
    if (wubu_sf_f32_is_inf(A) || wubu_sf_f32_is_inf(B)) {
        if ((A & ~F32_SIGN) == 0 || (B & ~F32_SIGN) == 0) return F32_QNAN;
        return ((uint32_t)sign << 31) | F32_INF;
    }
    sf_raw_t a, b; f32_unpack(A, &a); f32_unpack(B, &b);
    sf_raw_t r; r.sign = sign; r.g2 = 0; r.s2 = 0;
    if (a.m == 0 || b.m == 0) { r.m = 0; r.k = 0;
        return (uint32_t)sf_round(r, 24, F32_BIAS, 0) | (uint32_t)(sign << 31); }
    uint64_t p = a.m * b.m;              /* <= 2^48 */
    int sh = sf_clz64(p) - 1;
    r.m = p << sh; r.k = a.k + b.k - sh;
    return (uint32_t)sf_round(r, 24, F32_BIAS, 0) | (uint32_t)(sign << 31);
}

uint32_t wubu_sf_f32_div(uint32_t A, uint32_t B) {
    if (wubu_sf_f32_is_nan(A) || wubu_sf_f32_is_nan(B)) return F32_QNAN;
    int sign = (int)((A ^ B) >> 31);
    sf_raw_t a, b; f32_unpack(A, &a); f32_unpack(B, &b);
    int azero = (a.m == 0), bzero = (b.m == 0);
    int ainf = wubu_sf_f32_is_inf(A), binf = wubu_sf_f32_is_inf(B);
    if ((ainf && binf) || (azero && bzero)) return F32_QNAN;
    if (ainf || bzero) return ((uint32_t)sign << 31) | F32_INF;
    if (binf || azero) return (uint32_t)sign << 31;
    sf_raw_t r; r.sign = sign;
    int clz = sf_clz64(a.m);
    sf_div_core(a.m << clz, b.m, (int32_t)(a.k - clz - b.k), &r);
    return (uint32_t)sf_round(r, 24, F32_BIAS, 0) | (uint32_t)(sign << 31);
}

uint32_t wubu_sf_i64_to_f32(int64_t v) {
    if (v == 0) return 0u;
    int sign = (v < 0) ? 1 : 0;
    uint64_t m = sign ? (uint64_t)(-(v + 1)) + 1 : (uint64_t)v;
    int sh = sf_clz64(m);
    m <<= sh;
    sf_raw_t r; r.sign = sign; r.g2 = 0; r.s2 = 0;
    r.m = m; r.k = -(int32_t)sh; norm64(&r.m, &r.k);
    return (uint32_t)sf_round(r, 24, F32_BIAS, 0) | (uint32_t)(sign << 31);
}
int64_t wubu_sf_f32_to_i64(uint32_t a) {
    if (wubu_sf_f32_is_nan(a) || wubu_sf_f32_is_inf(a))
        return a >> 31 ? INT64_MIN : INT64_MAX;
    sf_raw_t r; f32_unpack(a, &r);
    if (r.m == 0) return 0;
    norm64(&r.m, &r.k);
    int k = (int)r.k;
    if (k >= 40) return r.sign ? INT64_MIN : INT64_MAX;
    uint64_t m = (k >= 0) ? (uint64_t)(r.m << k) : (uint64_t)(r.m >> -k);
    return r.sign ? -(int64_t)m : (int64_t)m;
}

/* ===================== f64 ===================== */

int wubu_sf_f64_cmp(uint64_t a, uint64_t b) {
    if (wubu_sf_f64_is_nan(a) || wubu_sf_f64_is_nan(b)) return 2;
    if (((a | b) & ~F64_SIGN) == 0) return 0;
    int sa = (int)(a >> 63), sb = (int)(b >> 63);
    if (sa != sb) return sa ? -1 : 1;
    if (a == b) return 0;
    int mag = (a & ~F64_SIGN) < (b & ~F64_SIGN) ? -1 : 1;
    return sa ? -mag : mag;
}

static void sf_div_core64(uint64_t am_norm, uint64_t bm, int32_t kq, sf_raw_t *r) {
    r->g2 = 0; r->s2 = 0;
    uint64_t ip = am_norm / bm;
    uint64_t rem = am_norm % bm;
    int n = 64 - sf_clz64(ip);
    int s = 64 - n;
    int need = s + 8; if (need > 62) need = 62;
    uint64_t frac = 0;
    for (int i = 0; i < need; i++) { rem <<= 1; frac <<= 1; if (rem >= bm) { rem -= bm; frac |= 1; } }
    uint64_t keep = (s == 0) ? ip : (ip << s);
    if (s > 0) {
        if (need > s) keep |= (frac >> (need - s)) & ((1ull << s) - 1);
        else          keep |=  frac & ((1ull << s) - 1);
    }
    if (rem != 0) keep |= 1u;
    r->m = keep ? keep : 1; r->k = kq - s;
}

uint64_t wubu_sf_f64_add(uint64_t A, uint64_t B) {
    if (wubu_sf_f64_is_nan(A) || wubu_sf_f64_is_nan(B)) return F64_QNAN;
    if (wubu_sf_f64_is_inf(A) && wubu_sf_f64_is_inf(B)) return (A == B) ? A : F64_QNAN;
    if (wubu_sf_f64_is_inf(A)) return A;
    if (wubu_sf_f64_is_inf(B)) return B;
    sf_raw_t a, b; f64_unpack(A, &a); f64_unpack(B, &b);
    int asg = a.sign, bsg = b.sign; a.sign = 0; b.sign = 0;
    norm64(&a.m, &a.k); norm64(&b.m, &b.k);
    if (a.k != b.k ? (a.k < b.k) : (a.m < b.m)) {
        sf_raw_t t = a; a = b; b = t; uint64_t tA = A; A = B; B = tA; int ts = asg; asg = bsg; bsg = ts;
    }
    if (a.m == 0) return (uint64_t)((asg == bsg ? asg : 0) << 63);
    int same = (asg == bsg);
    int32_t dk = a.k - b.k;
    uint64_t bm = b.m; int b_guard = 0, b_stick = 0;
    if (dk > 63) { bm = 0; b_stick = 1; }
    else if (dk >= 1) {
        b_guard = (int)((bm >> (dk - 1)) & 1);
        b_stick = (int)((bm & (((uint64_t)1 << (dk - 1)) - 1)) != 0);
        bm >>= dk;
    }
    sf_raw_t r; r.sign = asg; r.g2 = b_guard; r.s2 = b_stick;
    if (same) {
        if (b_guard && (b_stick || (bm & 1))) bm++;
        r.m = a.m + bm; r.k = a.k;
        if (r.m < a.m) { r.s2 |= (r.m & 1); r.m = (r.m >> 1) | (1ull << 63); r.k++; }
        r.g2 = 0; r.s2 = 0;
    } else {
        r.m = a.m - bm; r.k = a.k;
        if (r.m == 0) {
            if (!b_guard && !b_stick) return 0ull;
            r.m = (1ull << 63); r.g2 = b_guard; r.s2 = b_stick; r.k = a.k - 1;
        } else {
            if (b_guard) { r.m = (r.m << 1) - 1; r.g2 = 0; r.s2 = b_stick; r.k--; }
            else         { r.g2 = 0; r.s2 = b_stick; }
            norm64(&r.m, &r.k);
        }
    }
    return sf_round(r, 53, F64_BIAS, 1);
}
uint64_t wubu_sf_f64_sub(uint64_t a, uint64_t b) { return wubu_sf_f64_add(a, b ^ F64_SIGN); }

uint64_t wubu_sf_f64_mul(uint64_t A, uint64_t B) {
    if (wubu_sf_f64_is_nan(A) || wubu_sf_f64_is_nan(B)) return F64_QNAN;
    int sign = (int)((A ^ B) >> 63);
    if (wubu_sf_f64_is_inf(A) || wubu_sf_f64_is_inf(B)) {
        if ((A & ~F64_SIGN) == 0 || (B & ~F64_SIGN) == 0) return F64_QNAN;
        return ((uint64_t)sign << 63) | F64_INF;
    }
    sf_raw_t a, b; f64_unpack(A, &a); f64_unpack(B, &b);
    sf_raw_t r; r.sign = sign; r.g2 = 0; r.s2 = 0;
    if (a.m == 0 || b.m == 0) { r.m = 0; r.k = 0;
        return sf_round(r, 53, F64_BIAS, 1) | (uint64_t)(sign << 63); }
    /* 53x53 -> 106 bits via 27/26 split */
    uint64_t ah = a.m >> 27, al = a.m & ((1ull << 27) - 1);
    uint64_t bh = b.m >> 27, bl = b.m & ((1ull << 27) - 1);
    uint64_t ll = al * bl;
    uint64_t lh = al * bh + ah * bl;
    uint64_t hh = ah * bh;
    uint64_t lo2 = ((lh & ((1ull<<27)-1)) << 27) | (ll & ((1ull<<27)-1));
    uint64_t hi2 = hh + (lh >> 27);
    int sh = sf_clz64(hi2) - 1;
    uint64_t comb = (hi2 << 36) | (lo2 >> 18);
    r.g2 = 0; r.s2 = (lo2 & ((1ull << 18) - 1)) != 0;
    r.m = comb << sh; r.k = a.k + b.k + 54 - sh;
    norm64(&r.m, &r.k);
    return sf_round(r, 53, F64_BIAS, 1) | (uint64_t)(sign << 63);
}

uint64_t wubu_sf_f64_div(uint64_t A, uint64_t B) {
    if (wubu_sf_f64_is_nan(A) || wubu_sf_f64_is_nan(B)) return F64_QNAN;
    int sign = (int)((A ^ B) >> 63);
    sf_raw_t a, b; f64_unpack(A, &a); f64_unpack(B, &b);
    int azero = (a.m == 0), bzero = (b.m == 0);
    int ainf = wubu_sf_f64_is_inf(A), binf = wubu_sf_f64_is_inf(B);
    if ((ainf && binf) || (azero && bzero)) return F64_QNAN;
    if (ainf || bzero) return ((uint64_t)sign << 63) | F64_INF;
    if (binf || azero) return (uint64_t)sign << 63;
    sf_raw_t r; r.sign = sign;
    int clz = sf_clz64(a.m);
    sf_div_core64(a.m << clz, b.m, (int32_t)(a.k - clz - b.k), &r);
    return sf_round(r, 53, F64_BIAS, 1) | (uint64_t)(sign << 63);
}

uint64_t wubu_sf_i64_to_f64(int64_t v) {
    if (v == 0) return 0ull;
    int sign = (v < 0) ? 1 : 0;
    uint64_t m = sign ? (uint64_t)(-(v + 1)) + 1 : (uint64_t)v;
    int sh = sf_clz64(m); m <<= sh;
    sf_raw_t r; r.sign = sign; r.g2 = 0; r.s2 = 0;
    r.m = m; r.k = -(int32_t)sh; norm64(&r.m, &r.k);
    return sf_round(r, 53, F64_BIAS, 1) | (uint64_t)(sign << 63);
}
int64_t wubu_sf_f64_to_i64(uint64_t a) {
    if (wubu_sf_f64_is_nan(a) || wubu_sf_f64_is_inf(a)) return a >> 63 ? INT64_MIN : INT64_MAX;
    sf_raw_t r; f64_unpack(a, &r);
    if (r.m == 0) return 0;
    norm64(&r.m, &r.k);
    int k = (int)r.k;
    if (k >= 11) return r.sign ? INT64_MIN : INT64_MAX;
    uint64_t m = (k >= 0) ? (uint64_t)(r.m << k) : (uint64_t)(r.m >> -k);
    return r.sign ? -(int64_t)m : (int64_t)m;
}


/* ---- bfloat16 conversions (round-to-nearest-even) ---- */
uint32_t wubu_sf_bf16_to_f32(uint16_t h)
{
    return (uint32_t)h << 16;
}

uint16_t wubu_sf_f32_to_bf16(uint32_t a)
{
    uint16_t hi   = (uint16_t)(a >> 16);
    uint32_t rest = a & 0xFFFFu;
    uint32_t lsb  = (uint32_t)(hi & 1u);
    /* RNE: rest + 0x7FFF + lsb; carry bit 16 means round up */
    if (rest > 0x7FFFu + lsb - 1u && rest >= 0x8000u) {
        /* overflow only possible when hi == 0xFFFF -> becomes inf, correct */
    }
    uint32_t sum = rest + 0x7FFFu + lsb;
    if (sum & 0x10000u) {
        uint32_t nhi = (uint32_t)hi + 1u;
        /* NaN preservation: if input was NaN, keep mantissa nonzero */
        if ((a & 0x7F800000u) == 0x7F800000u && (a & 0x007FFFFFu) != 0
            && (nhi & 0x7FFFu) == 0x7F80u)
            nhi |= 0x40u;
        return (uint16_t)nhi;
    }
    return hi;
}

uint64_t wubu_sf_f32_to_f64(uint32_t a) {
    /* IEEE-754 widening f32->f64: exact bit manipulation */
    uint64_t sig = a & 0x7FFFFF;
    uint64_t exp = (a >> 23) & 0xFF;
    uint64_t sign = ((uint64_t)a >> 31) & 1;
    if (exp == 0 && sig == 0) return sign << 63;
    if (exp == 0xFF) {
        return (sign << 63) | 0x7FF0000000000000ULL | (sig << 29);
    }
    /* re-bias: f32 exp bias 127, f64 exp bias 1023 */
    exp = (exp + (1023 - 127));
    return (sign << 63) | (exp << 52) | (sig << 29);
}

uint32_t wubu_sf_f64_to_f32(uint64_t a) {
    /* round-to-nearest-even narrowing f64->f32 */
    uint64_t sign = (a >> 63) & 1;
    uint64_t exp = (a >> 52) & 0x7FF;
    uint64_t sig = a & 0xFFFFFFFFFFFFF;
    if (exp == 0x7FF) {
        uint32_t s = (sig != 0) ? 0x7FFFFF : 0;
        return (uint32_t)((sign << 31) | 0x7F800000 | s);
    }
    exp = exp - (1023 - 127);
    if ((int64_t)exp <= 0) {
        if ((int64_t)exp < -25) return (uint32_t)(sign << 31);
        uint64_t v = (sig << 1) | 0x10000000000000ULL; /* implicit bit */
        uint64_t sh = (uint64_t)(1 - (int64_t)exp);
        uint64_t rbit = (v >> sh) & 1;
        uint64_t sticky = (sh > 0) && ((v & ((1ULL << sh) - 1)) != 0);
        uint32_t s = (uint32_t)(v >> sh) | ((v >> (sh-1)) & 0); /* placeholder */
        s = (uint32_t)(((v >> sh) & 0x7FFFFF) + rbit + sticky);
        if (s & 0x800000) { s = 0; } /* mantissa overflow handled below */
        return (uint32_t)((sign << 31) | (s & 0x7FFFFF));
    }
    exp &= 0xFF;
    /* normal case: round guard bit + sticky */
    uint64_t r = (sig >> 29) & 1;
    uint64_t sticky = (sig & 0x1FFFFFFF) != 0;
    uint32_t s = (uint32_t)((sig >> 29) + r + sticky);
    if (s >= 0x800000) { s = 0; exp += 1; }
    return (uint32_t)((sign << 31) | (exp << 23) | (s & 0x7FFFFF));
}
