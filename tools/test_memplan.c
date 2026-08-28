#include <stdio.h>
#include <string.h>
#include "wubu_memplan.h"

/* Test 1: Linear chain — no reuse possible (each node feeds the next) */
static void test_linear_chain(void) {
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t dims[2] = { 256, 256 };
    hlir_tensor_t t = hlir_tensor(2, dims, 0); /* F32 */

    /* Chain: a = const, b = add(a, a), c = add(b, b), d = add(c, c) */
    hlir_node_t *a = hlir_op(&g, HLIR_CONSTANT, "a", NULL, 0, &t, NULL, 0);
    hlir_node_t *inputs_b[2] = { a, a };
    hlir_node_t *b = hlir_op(&g, HLIR_ADD, "b", inputs_b, 2, &t, NULL, 0);
    hlir_node_t *inputs_c[2] = { b, b };
    hlir_node_t *c = hlir_op(&g, HLIR_ADD, "c", inputs_c, 2, &t, NULL, 0);
    hlir_node_t *inputs_d[2] = { c, c };
    hlir_op(&g, HLIR_ADD, "d", inputs_d, 2, &t, NULL, 0);

    memplan_t *plan = memplan_create(&g);
    if (!plan) { printf("FAIL: test_linear_chain — plan creation failed\n"); return; }

    /* In a linear chain a→b→c→d, each node's output feeds only the next.
     * Once b is computed, a's buffer can be reused for c (ping-pong).
     * So only 2 buffers are needed. */
    printf("  linear_chain: %d buffers, peak=%lld, naive=%lld\n",
           plan->n_buffers, (long long)plan->peak_memory, (long long)plan->naive_memory);

    if (plan->n_buffers == 2) {
        printf("PASS: test_linear_chain (ping-pong reuse: 2 buffers)\n");
    } else {
        printf("FAIL: test_linear_chain — expected 2 buffers, got %d\n", plan->n_buffers);
    }

    memplan_free(plan);
    hlir_graph_free(&g);
}

/* Test 2: Parallel branches — reuse possible after branches converge */
static void test_parallel_branches(void) {
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t dims[2] = { 128, 128 };
    hlir_tensor_t t = hlir_tensor(2, dims, 0); /* F32 */

    /* Two independent branches that converge:
     * a = const, b = const, c = add(a, b), d = relu(a), e = add(c, d) */
    hlir_node_t *a = hlir_op(&g, HLIR_CONSTANT, "a", NULL, 0, &t, NULL, 0);
    hlir_node_t *b = hlir_op(&g, HLIR_CONSTANT, "b", NULL, 0, &t, NULL, 0);
    hlir_node_t *inputs_c[2] = { a, b };
    hlir_node_t *c = hlir_op(&g, HLIR_ADD, "c", inputs_c, 2, &t, NULL, 0);
    hlir_node_t *inputs_d[1] = { a };
    hlir_node_t *d = hlir_op(&g, HLIR_RELU, "d", inputs_d, 1, &t, NULL, 0);
    hlir_node_t *inputs_e[2] = { c, d };
    hlir_op(&g, HLIR_ADD, "e", inputs_e, 2, &t, NULL, 0);

    memplan_t *plan = memplan_create(&g);
    if (!plan) { printf("FAIL: test_parallel_branches — plan creation failed\n"); return; }

    printf("  parallel_branches: %d buffers, peak=%lld, naive=%lld\n",
           plan->n_buffers, (long long)plan->peak_memory, (long long)plan->naive_memory);

    /* With reuse, we should need fewer than 5 buffers */
    if (plan->n_buffers < 5) {
        printf("PASS: test_parallel_branches (reuse: %d < 5)\n", plan->n_buffers);
    } else {
        printf("FAIL: test_parallel_branches — expected < 5 buffers, got %d\n", plan->n_buffers);
    }

    memplan_free(plan);
    hlir_graph_free(&g);
}

/* Test 3: Independent ops — full reuse possible */
static void test_independent_ops(void) {
    hlir_graph_t g;
    hlir_graph_init(&g);

    int64_t dims[1] = { 1024 };
    hlir_tensor_t t = hlir_tensor(1, dims, 0); /* F32 */

    /* 4 independent constants — all can share one buffer */
    hlir_op(&g, HLIR_CONSTANT, "a", NULL, 0, &t, NULL, 0);
    hlir_op(&g, HLIR_CONSTANT, "b", NULL, 0, &t, NULL, 0);
    hlir_op(&g, HLIR_CONSTANT, "c", NULL, 0, &t, NULL, 0);
    hlir_op(&g, HLIR_CONSTANT, "d", NULL, 0, &t, NULL, 0);

    memplan_t *plan = memplan_create(&g);
    if (!plan) { printf("FAIL: test_independent_ops — plan creation failed\n"); return; }

    printf("  independent_ops: %d buffers, peak=%lld, naive=%lld\n",
           plan->n_buffers, (long long)plan->peak_memory, (long long)plan->naive_memory);

    /* All 4 are independent (no inputs), so they can share 1 buffer */
    if (plan->n_buffers <= 2) {
        printf("PASS: test_independent_ops (reuse: %d <= 2)\n", plan->n_buffers);
    } else {
        printf("FAIL: test_independent_ops — expected <= 2 buffers, got %d\n", plan->n_buffers);
    }

    memplan_free(plan);
    hlir_graph_free(&g);
}

/* Test 4: Empty graph */
static void test_empty_graph(void) {
    hlir_graph_t g;
    hlir_graph_init(&g);

    memplan_t *plan = memplan_create(&g);
    if (!plan) { printf("FAIL: test_empty_graph — plan creation failed\n"); return; }

    if (plan->n_buffers == 0 && plan->peak_memory == 0) {
        printf("PASS: test_empty_graph\n");
    } else {
        printf("FAIL: test_empty_graph — expected 0 buffers, got %d\n", plan->n_buffers);
    }

    memplan_free(plan);
    hlir_graph_free(&g);
}

int main(void) {
    printf("=== Memory Plan Tests ===\n");
    test_linear_chain();
    test_parallel_branches();
    test_independent_ops();
    test_empty_graph();
    printf("=== Done ===\n");
    return 0;
}
