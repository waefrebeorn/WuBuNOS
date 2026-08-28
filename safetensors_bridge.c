/*
 * safetensors_bridge.c — Bridge SafeTensors → HLIR → MIR pipeline.
 *
 * Loads model weights from .safetensors files into HLIR constant nodes,
 * then lowers to MIR for execution. This closes the model-loading loop
 * entirely within wubunos.
 */
#include "safetensors_bridge.h"
#include "safetensors.h"
#include "wubu_hlir.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* F16 → F32 conversion */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) f32 = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF; f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result; memcpy(&result, &f32, 4); return result;
}

/* BF16 → F32 conversion */
static float bf16_to_f32(uint16_t h) {
    uint32_t f32 = ((uint32_t)h) << 16;
    float result; memcpy(&result, &f32, 4); return result;
}

/* Load SafeTensors weights into HLIR graph as constants. */
int safetensors_to_hlir(safetensors_t *st, hlir_graph_t *g) {
    if (!st || !g) return -1;
    hlir_graph_init(g);

    for (int i = 0; i < st->n_tensors; i++) {
        safetensors_tensor_t *t = &st->tensors[i];

        /* Build HLIR tensor shape */
        hlir_tensor_t shape = hlir_tensor(t->n_dims, t->shape, t->dtype);

        /* Convert data to float32 for HLIR (handle F16/BF16 conversion) */
        int64_t nelems = shape.nelems;
        float *f32_data = (float *)calloc(nelems, sizeof(float));
        if (!f32_data) return -1;

        if (strcmp(t->dtype_str, "F32") == 0) {
            memcpy(f32_data, t->data, nelems * sizeof(float));
        } else if (strcmp(t->dtype_str, "F16") == 0) {
            uint16_t *src = (uint16_t *)t->data;
            for (int64_t j = 0; j < nelems; j++)
                f32_data[j] = f16_to_f32(src[j]);
        } else if (strcmp(t->dtype_str, "BF16") == 0) {
            uint16_t *src = (uint16_t *)t->data;
            for (int64_t j = 0; j < nelems; j++)
                f32_data[j] = bf16_to_f32(src[j]);
        } else if (strcmp(t->dtype_str, "I32") == 0) {
            int32_t *src = (int32_t *)t->data;
            for (int64_t j = 0; j < nelems; j++)
                f32_data[j] = (float)src[j];
        } else if (strcmp(t->dtype_str, "I8") == 0) {
            int8_t *src = (int8_t *)t->data;
            for (int64_t j = 0; j < nelems; j++)
                f32_data[j] = (float)src[j];
        } else {
            /* Unknown dtype: zero-fill */
            free(f32_data);
            continue;
        }

        /* Create HLIR constant node */
        char node_name[256];
        snprintf(node_name, sizeof(node_name), "weight_%s", t->name);
        hlir_constant(g, node_name, &shape, f32_data);
        free(f32_data);
    }

    return 0;
}
