/*
 * tools/test_e2e_mlp.c — End-to-end MLP inference test.
 *
 * Runs a 1-layer MLP: output = relu(input @ weight + bias)
 * through HLIR → MIR → interpreter pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_hlir.h"
#include "wubu_mir.h"

static void ref_mlp(const float *input, const float *W, const float *bias,
                    float *output, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float sum = bias[j];
        for (int k = 0; k < in_dim; k++)
            sum += input[k] * W[k * out_dim + j];
        output[j] = fmaxf(0.0f, sum);
    }
}

int main(void) {
    int in_dim = 4, out_dim = 8;
    int nW = in_dim * out_dim;

    float W[32], bias[8], input[4], ref_out[8];

    srand(42);
    for (int i = 0; i < nW; i++) W[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < out_dim; i++) bias[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < in_dim; i++) input[i] = (float)(rand() % 10) / 10.0f;

    ref_mlp(input, W, bias, ref_out, in_dim, out_dim);

    printf("Reference: ");
    for (int i = 0; i < out_dim; i++) printf("%.4f ", ref_out[i]);
    printf("\n");

    /* Build HLIR graph */
    hlir_graph_t g;
    hlir_graph_init(&g);

    /* Use 2D shapes: [1, in_dim] @ [in_dim, out_dim] = [1, out_dim] */
    int64_t in_dims[2] = {1, in_dim};
    hlir_tensor_t in_shape = hlir_tensor(2, in_dims, 0);
    hlir_node_t *input_node = hlir_placeholder(&g, "input", &in_shape);

    int64_t w_dims[2] = {in_dim, out_dim};
    hlir_tensor_t w_shape = hlir_tensor(2, w_dims, 0);
    hlir_node_t *weight_node = hlir_constant(&g, "W", &w_shape, W);

    int64_t out_dims[2] = {1, out_dim};
    hlir_tensor_t out_shape = hlir_tensor(2, out_dims, 0);

    /* MatMul: input @ W */
    hlir_node_t *mm_inputs[2] = {input_node, weight_node};
    hlir_node_t *matmul = hlir_op(&g, HLIR_MATMUL, "matmul", mm_inputs, 2, &out_shape, NULL, 0);

    /* Bias (broadcast over batch) */
    int64_t b_dims[2] = {1, out_dim};
    hlir_tensor_t b_shape = hlir_tensor(2, b_dims, 0);
    hlir_node_t *bias_node = hlir_constant(&g, "bias", &b_shape, bias);

    /* Add: matmul + bias */
    hlir_node_t *add_inputs[2] = {matmul, bias_node};
    hlir_node_t *add = hlir_op(&g, HLIR_ADD, "add", add_inputs, 2, &out_shape, NULL, 0);

    /* ReLU */
    hlir_node_t *relu = hlir_op(&g, HLIR_RELU, "relu", &add, 1, &out_shape, NULL, 0);
    (void)relu;

    printf("HLIR: %d nodes\n", g.n);

    /* Lower to MIR */
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int rc = hlir_lower_mir(&g, &prog);
    if (rc != 0) { printf("HLIR→MIR failed\n"); return 1; }
    printf("MIR: %u instructions\n", prog.n);

    /* Setup memory: batch=1, in_dim=4 */
    int ncells = prog.total_mem > 0 ? (int)prog.total_mem + 1 : 200;
    int64_t *mem = (int64_t*)calloc(ncells, sizeof(int64_t));

    /* Store input at cell 1 (base of placeholder) */
    for (int i = 0; i < in_dim; i++) {
        union { float f; int32_t i; } u;
        u.f = input[i];
        mem[1 + i] = (int64_t)u.i;
    }

    prog.mem = mem;

    /* Run interpreter */
    int64_t result = wubu_mir_interp(&prog);
    printf("Interpreter VR0: %lld\n", (long long)result);

    /* Read output from memory */
    printf("HLIR out: ");
    int found_out = 0;
    for (int i = 1; i < ncells; i++) {
        union { float f; int32_t i; } u;
        u.i = (int32_t)mem[i];
        /* Check if this cell matches any reference output */
        for (int j = 0; j < out_dim; j++) {
            if (fabsf(u.f - ref_out[j]) < 1e-3f) {
                printf("[cell %d]=%.4f ", i, u.f);
                found_out = 1;
                break;
            }
        }
    }
    if (!found_out) {
        /* Just print cells that look like float outputs */
        for (int i = 1; i < ncells && i < 20; i++) {
            union { float f; int32_t i; } u;
            u.i = (int32_t)mem[i];
            if (u.f > -100.0f && u.f < 100.0f && u.f != 0.0f)
                printf("[%d]=%.4f ", i, u.f);
        }
    }
    printf("\n");

    free(mem);
    wubu_mir_free(&prog);
    hlir_graph_free(&g);
    return 0;
}
