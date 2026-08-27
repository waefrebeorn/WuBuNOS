#include "wubu_softfloat.h"
#include <stdio.h>
int main() {
    printf("=== FP16 Conversion Test ===\n");
    struct { uint16_t h; uint32_t f32; } tests[] = {
        { 0x0000, 0x00000000u },  /* 0.0 */
        { 0x3C00, 0x3F800000u },  /* 1.0 */
        { 0x4000, 0x40000000u },  /* 2.0 */
        { 0xC000, 0xC0000000u },  /* -2.0 */
        { 0x7C00, 0x7F800000u },  /* +inf */
        { 0x3800, 0x3F000000u },  /* 0.5 */
        { 0x3000, 0x3E000000u },  /* 0.25 */
    };
    int pass = 0, fail = 0;
    for (int i = 0; i < 7; i++) {
        uint32_t got = wubu_sf_f16_to_f32(tests[i].h);
        uint16_t back = wubu_sf_f32_to_f16(got);
        int ok = (got == tests[i].f32) && (back == tests[i].h);
        printf("  f16(0x%04X) -> f32(0x%08X=%.2f) -> f16(0x%04X) %s\n",
               tests[i].h, got, wubu_sf_f32_to_host(got), back, ok ? "PASS" : "FAIL");
        if (ok) pass++; else fail++;
    }
    printf("FP16 conversions: %d/%d PASS\n", pass, pass+fail);
    return fail ? 1 : 0;
}
