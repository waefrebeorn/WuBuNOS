/*
 * test_hlir_lower.c -- Test HLIR → MIR lowering + execution.
 *
 * Builds HLIR graphs, lowers to MIR, runs through the interpreter,
 * and checks results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_hlir.h"
#include "wubu_mir.h"

static int total = 0, pass = 0;
#define CHECK(c, m) do { total++; if (c) pass++; else printf("  FAIL: %s\n", m); } while(0)

/* Run a lowered MIR program through the interpreter */
static int64_t run_hlir_graph(hlir_graph_t *g)
{
    wubu_mir_prog_t prog;
    int rc = hlir_lower_mir(g, &prog);
    if (rc != 0) return -99999;
    int64_t result = wubu_mir_interp(&prog);
    wubu_mir_free(&prog);
    return result;
}

static void test_scalar_add(void)
{
    printf("-- Test: scalar add via HLIR→MIR --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    /* Constants: 20 + 22 = 42 */
    float val_a = 20.0f, val_b = 22.0f;
    hlir_node_t *a = hlir_constant(&g, "a", &shape, &val_a);
    hlir_node_t *b = hlir_constant(&g, "b", &shape, &val_b);
    hlir_node_t *sum = hlir_add(&g, a, b);
    hlir_set_output(&g, sum);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fabsf(fresult - 42.0f) < 0.01f, "20 + 22 = 42");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_scalar_mul(void)
{
    printf("-- Test: scalar mul via HLIR→MIR --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float val_a = 6.0f, val_b = 7.0f;
    hlir_node_t *a = hlir_constant(&g, "a", &shape, &val_a);
    hlir_node_t *b = hlir_constant(&g, "b", &shape, &val_b);
    hlir_node_t *prod = hlir_mul(&g, a, b);
    hlir_set_output(&g, prod);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fabsf(fresult - 42.0f) < 0.01f, "6 * 7 = 42");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_relu_pos(void)
{
    printf("-- Test: relu(5.0) = 5.0 --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float val = 5.0f;
    hlir_node_t *x = hlir_constant(&g, "x", &shape, &val);
    hlir_node_t *r = hlir_relu(&g, x);
    hlir_set_output(&g, r);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fresult > 0.0f, "relu(5) > 0");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_relu_neg(void)
{
    printf("-- Test: relu(-3.0) = 0.0 --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float val = -3.0f;
    hlir_node_t *x = hlir_constant(&g, "x", &shape, &val);
    hlir_node_t *r = hlir_relu(&g, x);
    hlir_set_output(&g, r);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fresult == 0.0f, "relu(-3) = 0");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_residual(void)
{
    printf("-- Test: residual add 10 + 32 = 42 --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float val_a = 10.0f, val_b = 32.0f;
    hlir_node_t *a = hlir_constant(&g, "a", &shape, &val_a);
    hlir_node_t *b = hlir_constant(&g, "b", &shape, &val_b);
    hlir_node_t *res = hlir_residual_add(&g, a, b);
    hlir_set_output(&g, res);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fabsf(fresult - 42.0f) < 0.01f, "residual 10 + 32 = 42");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_chain(void)
{
    printf("-- Test: (2 + 3) * 4 = 20 --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float va = 2.0f, vb = 3.0f, vc = 4.0f;
    hlir_node_t *a = hlir_constant(&g, "a", &shape, &va);
    hlir_node_t *b = hlir_constant(&g, "b", &shape, &vb);
    hlir_node_t *c = hlir_constant(&g, "c", &shape, &vc);
    hlir_node_t *sum = hlir_add(&g, a, b);
    hlir_node_t *prod = hlir_mul(&g, sum, c);
    hlir_set_output(&g, prod);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fabsf(fresult - 20.0f) < 0.01f, "(2+3)*4 = 20");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

static void test_sigmoid(void)
{
    printf("-- Test: sigmoid(0) ≈ 0.5 --\n");
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t d[] = {1};
    hlir_tensor_t shape = hlir_tensor(1, d, 0);

    float val = 0.0f;
    hlir_node_t *x = hlir_constant(&g, "x", &shape, &val);
    hlir_node_t *s = hlir_op(&g, HLIR_SIGMOID, "sigmoid", &x, 1, &shape, NULL, 0);
    hlir_set_output(&g, s);

    int64_t result = run_hlir_graph(&g);
    float fresult = *(float *)&result;
    CHECK(fabsf(fresult - 0.5f) < 0.05f, "sigmoid(0) ≈ 0.5");
    printf("    result = %f\n", fresult);

    hlir_graph_free(&g);
}

int main(void)
{
    printf("=== HLIR→MIR LOWERING TEST ===\n\n");
    test_scalar_add();
    test_scalar_mul();
    test_relu_pos();
    test_relu_neg();
    test_residual();
    test_chain();
    test_sigmoid();

    printf("\n=== %s ===\n", pass == total ? "PASS" : "FAIL");
    printf("(PASS: %d, TOTAL: %d)\n", pass, total);
    return (pass == total) ? 0 : 1;
}
