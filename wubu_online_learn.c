/*
 * wubu_online_learn.c — Online learning from production profiling.
 *
 * Phase 4 AGI compiler component: the compiler improves itself by
 * collecting runtime performance data and updating its cost model.
 *
 * Workflow:
 *   1. Run kernels with various configs
 *   2. Log (shape, config, runtime) tuples to a ring buffer
 *   3. Periodically retrain the cost model from accumulated data
 *   4. Use the updated model to select better configs
 *
 * C11, self-contained.
 */

#include "wubu_cost_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Profiling data point ---- */

typedef struct {
    int M, N, K;          /* problem shape */
    int mc, nc, kc;       /* config that was used */
    int mr, nr;
    double gflops;        /* measured performance */
    uint64_t timestamp;   /* when it was measured */
} profile_point_t;

/* ---- Online learner state ---- */

#define PROFILE_BUF_SIZE 1024
#define RETRAIN_INTERVAL 64   /* retrain after every N new points */

typedef struct {
    profile_point_t buf[PROFILE_BUF_SIZE];
    int count;            /* valid entries in buf */
    int since_retrain;    /* points since last model update */
    double best_gflops;   /* best GFLOPS seen so far */
} online_learner_t;

static online_learner_t learner;

/* ---- Log a profiling point ---- */

void wubu_log_profile(int M, int N, int K, const tune_tgemm_t *cfg)
{
    if (learner.count >= PROFILE_BUF_SIZE) {
        /* Ring buffer: overwrite oldest entry */
        memmove(&learner.buf[0], &learner.buf[1],
                (PROFILE_BUF_SIZE - 1) * sizeof(profile_point_t));
        learner.count = PROFILE_BUF_SIZE - 1;
    }

    profile_point_t *p = &learner.buf[learner.count++];
    p->M = M; p->N = N; p->K = K;
    p->mc = cfg->mc; p->nc = cfg->nc; p->kc = cfg->kc;
    p->mr = cfg->mr; p->nr = cfg->nr;
    p->gflops = cfg->gflops;
    p->timestamp = (uint64_t)time(NULL);

    if (cfg->gflops > learner.best_gflops) {
        learner.best_gflops = cfg->gflops;
    }

    learner.since_retrain++;

    /* Retrain model when enough new data accumulated */
    if (learner.since_retrain >= RETRAIN_INTERVAL) {
        printf("[online-learn] retraining cost model from %d data points...\n",
               learner.count);
        wubu_train_cost_model((const tune_tgemm_t *)learner.buf,
                              learner.count, M, N, K);
        learner.since_retrain = 0;
    }
}

/* ---- Get current best performance for a shape ---- */

double wubu_best_seen_gflops(int M, int N, int K)
{
    double best = 0.0;
    for (int i = 0; i < learner.count; i++) {
        if (learner.buf[i].M == M && learner.buf[i].N == N &&
            learner.buf[i].K == K) {
            if (learner.buf[i].gflops > best)
                best = learner.buf[i].gflops;
        }
    }
    return best;
}

/* ---- Print learning summary ---- */

void wubu_learning_summary(void)
{
    printf("=== Online Learning Summary ===\n");
    printf("Data points collected: %d\n", learner.count);
    printf("Best GFLOPS ever:      %.2f\n", learner.best_gflops);
    printf("Points since retrain:  %d / %d\n",
           learner.since_retrain, RETRAIN_INTERVAL);

    /* Show recent improvement trend */
    if (learner.count >= 10) {
        double recent[10];
        for (int i = 0; i < 10; i++)
            recent[i] = learner.buf[learner.count - 10 + i].gflops;

        double avg = 0.0;
        for (int i = 0; i < 10; i++) avg += recent[i];
        avg /= 10.0;

        printf("Recent 10 avg GFLOPS:  %.2f\n", avg);
    }
}

/* ---- Reset learner state (for testing) ---- */

void wubu_reset_learner(void)
{
    memset(&learner, 0, sizeof(learner));
}
