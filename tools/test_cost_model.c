#include "wubu_cost_model.h"
#include <stdio.h>
int main() {
    printf("=== Cost Model Test ===\n");
    tune_tgemm_t best = wubu_predict_best_config(128, 128, 64);
    printf("Best: mc=%d nc=%d kc=%d mr=%d nr=%d\n",
           best.mc, best.nc, best.kc, best.mr, best.nr);
    return 0;
}
