/*
 * tools/test_auto_tune.c — Test the auto-tuning framework.
 */
#include "wubu_auto_tune.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    printf("=== Auto-Tune Framework Test ===\n");

    /* Test with a small GEMM */
    tune_tgemm_t best = wubu_tune_tgemm(64, 64, 64);

    /* Verify the result is sensible */
    if (best.gflops <= 0.0) {
        printf("FAIL: no valid configuration measured\n");
        return 1;
    }
    if (best.mc <= 0 || best.nc <= 0 || best.kc <= 0 ||
        best.mr <= 0 || best.nr <= 0) {
        printf("FAIL: invalid tile sizes\n");
        return 1;
    }

    printf("\nBest configuration:\n");
    printf("  mc=%d nc=%d kc=%d mr=%d nr=%d\n",
           best.mc, best.nc, best.kc, best.mr, best.nr);
    printf("  %.2f GFLOPS\n", best.gflops);
    printf("PASS: auto-tune framework works\n");
    return 0;
}
