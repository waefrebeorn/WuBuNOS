/*
 * wubu_host_tensor.c - Host-side implementations of AGI tensor ops.
 * Called from JIT code via wubu_tensor_dispatch.
 * C11, self-contained, no external dependencies.
 */

#include "wubu_mir.h"
#include "wubu_softfloat.h"
#include <math.h>

/* Helper: load f32 from mem slot (int64_t cell interpreted as IEEE-754 bits) */
static float load_f32(int64_t *mem, int64_t slot) {
    return *(float*)&mem[slot];
}

/* Helper: store f32 to mem slot */
static void store_f32(int64_t *mem, int64_t slot, float val) {
    mem[slot] = (int64_t)*(uint32_t*)&val;
}

/* Softmax: exp(x_i - max) / sum(exp(x_j - max)) */
static void tensor_softmax(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    if (N <= 0) return;
    float *src = (float*)&mem[a];
    float *d = (float*)&mem[dst];
    float max_val = src[0];
    for (int i = 1; i < N; i++) if (src[i] > max_val) max_val = src[i];
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = expf(src[i] - max_val);
        d[i] = e;
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) d[i] *= inv_sum;
}

/* Elementwise operations */
static void tensor_tanh(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) store_f32(mem, dst + i, tanhf(load_f32(mem, a + i)));
}

static void tensor_sigmoid(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        store_f32(mem, dst + i, 1.0f / (1.0f + expf(-x)));
    }
}

static void tensor_gelu(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        float sqrt2pi = 0.7978845608f;
        float gelu = 0.5f * x * (1.0f + tanhf(sqrt2pi * (x + 0.044715f * x * x * x)));
        store_f32(mem, dst + i, gelu);
    }
}

static void tensor_relu(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        store_f32(mem, dst + i, x > 0.0f ? x : 0.0f);
    }
}

static void tensor_exp(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) store_f32(mem, dst + i, expf(load_f32(mem, a + i)));
}

static void tensor_sqrt(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        store_f32(mem, dst + i, x > 0.0f ? sqrtf(x) : 0.0f);
    }
}

/* Sum reduction */
static void tensor_sum(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += load_f32(mem, a + i);
    store_f32(mem, dst, sum);
}

/* RMSNorm */
static void tensor_rms_norm(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float eps = 1e-6f;
    float sum_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        sum_sq += x * x;
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / N + eps);
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        store_f32(mem, dst + i, x * inv_rms);
    }
}

/* LayerNorm */
static void tensor_layernorm(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float eps = 1e-5f;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += load_f32(mem, a + i);
    float mean = sum / N;
    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i) - mean;
        var += x * x;
    }
    float inv_std = 1.0f / sqrtf(var / N + eps);
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        store_f32(mem, dst + i, (x - mean) * inv_std);
    }
}

/* Embedding lookup */
static void tensor_embedding(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t dim) {
    int32_t *indices = (int32_t*)&mem[a];
    float *weights = (float*)&mem[b];
    float *out = (float*)&mem[dst];
    for (int64_t i = 0; i < dim; i++) {
        int32_t idx = indices[i];
        for (int d = 0; d < 64; d++)
            out[i * 64 + d] = weights[idx * 64 + d];
    }
}

/* Argmax */
static void tensor_argmax(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float *src = (float*)&mem[a];
    float max_val = src[0];
    int max_idx = 0;
    for (int i = 1; i < N; i++) if (src[i] > max_val) { max_val = src[i]; max_idx = i; }
    store_f32(mem, dst, (float)max_idx);
}

/* SwiGLU */
static void tensor_swglu(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    int H = N / 2;
    for (int i = 0; i < H; i++) {
        float x = load_f32(mem, a + 2*i);
        float y = load_f32(mem, a + 2*i + 1);
        store_f32(mem, dst + i, (1.0f / (1.0f + expf(-x))) * y);
    }
}

/* Clamp */
static void tensor_clamp(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float min_val = load_f32(mem, b);
    float max_val = load_f32(mem, b + 1);
    for (int i = 0; i < N; i++) {
        float x = load_f32(mem, a + i);
        if (x < min_val) x = min_val;
        else if (x > max_val) x = max_val;
        store_f32(mem, dst + i, x);
    }
}

/* Dropout (training mode placeholder) */
static void tensor_dropout(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    float p = load_f32(mem, b);
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < N; i++) store_f32(mem, dst + i, load_f32(mem, a + i) * scale);
}

/* Rope - placeholder (rotary position embedding) */
static void tensor_rope(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    /* TODO: implement rotary position embedding */
    for (int i = 0; i < N; i++) store_f32(mem, dst + i, load_f32(mem, a + i));
}

/* Conv2D - placeholder */
static void tensor_conv2d(int64_t *mem, int64_t a, int64_t b, int64_t dst, int64_t N) {
    /* TODO: implement convolution */
    for (int i = 0; i < N; i++) store_f32(mem, dst + i, load_f32(mem, a + i));
}

/* Main dispatcher - called from JIT
 * Signature: void wubu_tensor_dispatch(int64_t *mem, uint32_t op,
 *                                     int64_t a, int64_t b, int64_t dst, int64_t N)
 */
void wubu_tensor_dispatch(int64_t *mem, uint32_t op,
                          int64_t a, int64_t b, int64_t dst, int64_t N) {
    switch (op) {
        case MIR_T_SOFTMAX:    tensor_softmax(mem, a, b, dst, N); break;
        case MIR_T_TANH:       tensor_tanh(mem, a, b, dst, N); break;
        case MIR_T_SIGMOID:    tensor_sigmoid(mem, a, b, dst, N); break;
        case MIR_T_GELU:       tensor_gelu(mem, a, b, dst, N); break;
        case MIR_T_RELU:       tensor_relu(mem, a, b, dst, N); break;
        case MIR_T_EXP:        tensor_exp(mem, a, b, dst, N); break;
        case MIR_T_SQRT:       tensor_sqrt(mem, a, b, dst, N); break;
        case MIR_T_SUM:        tensor_sum(mem, a, b, dst, N); break;
        case MIR_T_RMS_NORM:   tensor_rms_norm(mem, a, b, dst, N); break;
        case MIR_T_LAYERNORM:  tensor_layernorm(mem, a, b, dst, N); break;
        case MIR_T_EMBEDDING:  tensor_embedding(mem, a, b, dst, N); break;
        case MIR_T_ARGMAX:     tensor_argmax(mem, a, b, dst, N); break;
        case MIR_T_SWIGLU:     tensor_swglu(mem, a, b, dst, N); break;
        case MIR_T_CLAMP:      tensor_clamp(mem, a, b, dst, N); break;
        case MIR_T_DROPOUT:    tensor_dropout(mem, a, b, dst, N); break;
        case MIR_T_ROPE:       tensor_rope(mem, a, b, dst, N); break;
        case MIR_T_CONV2D:     tensor_conv2d(mem, a, b, dst, N); break;
        default: break;
    }
}