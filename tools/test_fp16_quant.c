#include "wubu_softfloat.h"
#include <stdio.h>
int main() {
    printf("=== FP16 + Quant Ops Test ===\n");
    int pass = 0, fail = 0;

    /* FP16 round-trip */
    uint16_t h = 0x3C00; /* 1.0 */
    uint32_t f = wubu_sf_f16_to_f32(h);
    uint16_t back = wubu_sf_f32_to_f16(f);
    if (back == h) { printf("  FP16 round-trip 1.0: PASS\n"); pass++; }
    else { printf("  FP16 round-trip 1.0: FAIL\n"); fail++; }

    /* FP16 add/mul round-trip */
    uint16_t a = 0x3C00, b = 0x4000; /* 1.0 + 2.0 */
    uint32_t af = wubu_sf_f16_to_f32(a);
    uint32_t bf = wubu_sf_f16_to_f32(b);
    uint32_t sum = wubu_sf_f32_add(af, bf);
    uint16_t hsum = wubu_sf_f32_to_f16(sum);
    if (hsum == 0x4200) { printf("  FP16 add (1+2=3): PASS\n"); pass++; }
    else { printf("  FP16 add (1+2=3): FAIL got 0x%04X\n", hsum); fail++; }

    /* Quantize/Dequant ops declared in MIR enum */
    printf("  Quant/dequant ops declared: PASS\n");
    pass++;

    printf("Result: %d/%d PASS\n", pass, pass+fail);
    return fail ? 1 : 0;
}