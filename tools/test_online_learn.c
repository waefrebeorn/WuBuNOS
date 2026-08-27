#include "wubu_online_learn.h"
#include <stdio.h>
int main() {
    printf("=== Online Learning Test ===\n");
    wubu_reset_learner();

    /* Simulate profiling data: kernel runs at different configs */
    tune_tgemm_t cfg1 = {64,64,64,8,4,1.5};
    tune_tgemm_t cfg2 = {128,128,64,8,4,2.3};
    tune_tgemm_t cfg3 = {32,32,32,4,8,0.8};

    wubu_log_profile(64, 64, 64, &cfg1);
    wubu_log_profile(128, 128, 64, &cfg2);
    wubu_log_profile(64, 64, 64, &cfg1);  /* repeat: ring buffer */
    wubu_log_profile(32, 32, 32, &cfg3);

    double best64 = wubu_best_seen_gflops(64, 64, 64);
    printf("Best 64x64x64: %.2f GFLOPS (expected 1.50)\n", best64);

    wubu_learning_summary();

    return (best64 > 1.4 && best64 < 1.6) ? 0 : 1;
}