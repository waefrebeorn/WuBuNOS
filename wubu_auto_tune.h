/* wubu_auto_tune.h — auto-tuning framework public API */
#ifndef WUBU_AUTO_TUNE_H
#define WUBU_AUTO_TUNE_H

#include <stdint.h>

typedef struct {
    int mc;          /* M tile */
    int nc;          /* N tile */
    int kc;          /* K tile */
    int mr;          /* M register block */
    int nr;          /* N register block */
    double gflops;   /* measured performance */
} tune_tgemm_t;

/* Auto-tune T_GEMM for the given dimensions. Returns best config. */
tune_tgemm_t wubu_tune_tgemm(int M, int N, int K);

#endif /* WUBU_AUTO_TUNE_H */
