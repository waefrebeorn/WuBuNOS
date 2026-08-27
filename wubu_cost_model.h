/* wubu_cost_model.h — learned cost model for kernel selection */
#ifndef WUBU_COST_MODEL_H
#define WUBU_COST_MODEL_H

#include "wubu_auto_tune.h"

/* Predict best GEMM config for given dimensions using the learned model */
tune_tgemm_t wubu_predict_best_config(int M, int N, int K);

/* Train the model from benchmark data */
void wubu_train_cost_model(const tune_tgemm_t *data, int n, int M, int N, int K);

#endif /* WUBU_COST_MODEL_H */
