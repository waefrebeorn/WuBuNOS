/* wubu_online_learn.h — online learning from production profiling */
#ifndef WUBU_ONLINE_LEARN_H
#define WUBU_ONLINE_LEARN_H

#include "wubu_auto_tune.h"

/* Log a profiling data point (shape + config + measured GFLOPS) */
void wubu_log_profile(int M, int N, int K, const tune_tgemm_t *cfg);

/* Get best GFLOPS seen for a given shape */
double wubu_best_seen_gflops(int M, int N, int K);

/* Print learning summary statistics */
void wubu_learning_summary(void);

/* Reset learner state (for testing) */
void wubu_reset_learner(void);

#endif /* WUBU_ONLINE_LEARN_H */
