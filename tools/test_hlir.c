/*
 * test_hlir.c -- Test the HLIR (high-level IR) graph builder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_hlir.h"

static int total = 0, pass = 0;
#define CHECK(c, m) do { total++; if (c) pass++; else printf("  FAIL: %s\n", m); } while(0)

static hlir_tensor_t make_shape(int n_dims, const int64_t *dims, int dtype)
{
    return hlir_tensor(n_dims, dims, dtype);
}

static void test_placeholder(void)
{
    printf("-- Test: placeholder --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t dims[] = {4, 8};
    hlir_tensor_t shape = make_shape(2, dims, 0); /* F32 */
    hlir_node_t *x = hlir_placeholder(&g, "input", &shape);
    CHECK(x != NULL, "placeholder created");
    CHECK(x->op == HLIR_PLACEHOLDER, "op is placeholder");
    CHECK(x->output.n_dims == 2, "shape has 2 dims");
    CHECK(x->output.dims[0] == 4, "dim0 = 4");
    CHECK(x->output.dims[1] == 8, "dim1 = 8");
    CHECK(g.n_inputs == 1, "graph has 1 input");
    CHECK(g.n == 1, "graph has 1 node");
    hlir_graph_free(&g);
}

static void test_add_mul(void)
{
    printf("-- Test: add + mul --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t d[] = {8};
    hlir_tensor_t shape = make_shape(1, d, 0);
    hlir_node_t *x = hlir_placeholder(&g, "x", &shape);
    hlir_node_t *y = hlir_placeholder(&g, "y", &shape);
    hlir_node_t *add = hlir_add(&g, x, y);
    hlir_node_t *mul = hlir_mul(&g, add, y);
    CHECK(add != NULL, "add node created");
    CHECK(mul != NULL, "mul node created");
    CHECK(g.n == 4, "graph has 4 nodes");
    CHECK(add->n_inputs == 2, "add has 2 inputs");
    CHECK(mul->n_inputs == 2, "mul has 2 inputs");
    hlir_graph_free(&g);
}

static void test_relu(void)
{
    printf("-- Test: relu --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t d[] = {16};
    hlir_tensor_t shape = make_shape(1, d, 0);
    hlir_node_t *x = hlir_placeholder(&g, "x", &shape);
    hlir_node_t *rel = hlir_relu(&g, x);
    CHECK(rel->op == HLIR_RELU, "op is relu");
    CHECK(rel->output.dims[0] == 16, "shape preserved");
    hlir_graph_free(&g);
}

static void test_constant(void)
{
    printf("-- Test: constant --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t d[] = {3, 4};
    hlir_tensor_t shape = make_shape(2, d, 0);
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    hlir_node_t *c = hlir_constant(&g, "weights", &shape, data);
    CHECK(c != NULL, "constant created");
    CHECK(c->op == HLIR_CONSTANT, "op is constant");
    CHECK(c->data != NULL, "data is stored");
    CHECK(c->output.dims[0] == 3, "dim0 = 3");
    CHECK(c->output.dims[1] == 4, "dim1 = 4");
    hlir_graph_free(&g);
}

static void test_transformer_block(void)
{
    printf("-- Test: transformer block --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t d[] = {4, 8};
    hlir_tensor_t shape = make_shape(2, d, 0);
    hlir_node_t *x = hlir_placeholder(&g, "x", &shape);

    hlir_node_t *norm1 = hlir_rmsnorm(&g, x, 1e-6f);
    hlir_node_t *rotated = hlir_rope(&g, norm1, 32);
    hlir_node_t *act = hlir_gelu(&g, rotated);
    hlir_node_t *out = hlir_residual_add(&g, act, x);
    hlir_set_output(&g, out);

    CHECK(g.n == 5, "5 nodes in transformer block");
    CHECK(g.n_outputs == 1, "1 output");
    CHECK(out->op == HLIR_RESIDUAL_ADD, "output is residual add");

    hlir_node_t *sorted[32];
    int n = hlir_topo_sort(&g, sorted);
    CHECK(n == 5, "topo sort returns 5 nodes");
    CHECK(sorted[0]->op == HLIR_PLACEHOLDER, "placeholder is first");
    CHECK(sorted[4]->op == HLIR_RESIDUAL_ADD, "residual is last");

    hlir_graph_free(&g);
}

static void test_attention(void)
{
    printf("-- Test: attention --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t qd[] = {2, 8, 16};
    hlir_tensor_t qshape = make_shape(3, qd, 0);
    hlir_node_t *q = hlir_placeholder(&g, "q", &qshape);
    hlir_node_t *k = hlir_placeholder(&g, "k", &qshape);
    hlir_node_t *v = hlir_placeholder(&g, "v", &qshape);
    hlir_node_t *att = hlir_attention(&g, q, k, v, 0.125f);
    CHECK(att != NULL, "attention created");
    CHECK(att->n_inputs == 3, "attention has 3 inputs");
    hlir_graph_free(&g);
}

static void test_swiglu_block(void)
{
    printf("-- Test: SwiGLU block --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);
    int64_t d[] = {4, 4096};
    hlir_tensor_t shape = make_shape(2, d, 0);
    hlir_node_t *x = hlir_placeholder(&g, "x", &shape);
    hlir_node_t *sw = hlir_swiglu(&g, x);
    hlir_node_t *proj = hlir_relu(&g, sw);
    hlir_node_t *out = hlir_residual_add(&g, proj, x);
    hlir_set_output(&g, out);
    CHECK(g.n == 4, "swiglu block has 4 nodes");
    hlir_graph_free(&g);
}

int main(void)
{
    printf("=== HLIR TEST ===\n\n");
    test_placeholder();
    test_add_mul();
    test_relu();
    test_constant();
    test_transformer_block();
    test_attention();
    test_swiglu_block();

    printf("\n=== %s ===\n", pass == total ? "PASS" : "FAIL");
    printf("(PASS: %d, TOTAL: %d)\n", pass, total);
    return (pass == total) ? 0 : 1;
}
