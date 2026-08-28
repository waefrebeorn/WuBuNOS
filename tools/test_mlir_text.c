#include <stdio.h>
#include <string.h>
#include "mlir_text.h"

/* Test 1: Simple arith ops */
static void test_simple_arith(void) {
    const char *src =
        "%c0 = arith.constant 1.0 : f32\n"
        "%c1 = arith.constant 2.0 : f32\n"
        "%sum = arith.addf(%c0, %c1) : f32\n"
        "%prod = arith.mulf(%sum, %c1) : f32\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  simple_arith: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 3) {
        printf("PASS: test_simple_arith (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_simple_arith — expected >= 3 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 2: Function with module */
static void test_func_module(void) {
    const char *src =
        "module {\n"
        "  func.func @add(%a: f32, %b: f32) -> f32 {\n"
        "    %sum = arith.addf(%a, %b) : f32\n"
        "    return %sum : f32\n"
        "  }\n"
        "}\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  func_module: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 1) {
        printf("PASS: test_func_module (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_func_module — expected >= 1 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 3: Tensor types */
static void test_tensor_types(void) {
    const char *src =
        "%t = arith.constant dense<1.0> : tensor<2x3xf32>\n"
        "%c = arith.addf(%t, %t) : tensor<2x3xf32>\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  tensor_types: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 1) {
        printf("PASS: test_tensor_types (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_tensor_types — expected >= 1 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 4: Empty input */
static void test_empty(void) {
    hlir_graph_t g;
    int rc = mlir_text_load("", &g);
    printf("  empty: rc=%d, n=%d\n", rc, g.n);
    if (rc != 0 || g.n == 0) {
        printf("PASS: test_empty (correctly rejected empty input)\n");
    } else {
        printf("FAIL: test_empty — should reject empty input\n");
    }
    hlir_graph_free(&g);
}

/* Test 5: Math ops */
static void test_math_ops(void) {
    const char *src =
        "%x = arith.constant 2.0 : f32\n"
        "%y = math.exp(%x) : f32\n"
        "%z = math.sqrt(%y) : f32\n"
        "%w = math.tanh(%z) : f32\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  math_ops: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 3) {
        printf("PASS: test_math_ops (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_math_ops — expected >= 3 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

int main(void) {
    printf("=== MLIR Text Parser Tests ===\n");
    test_simple_arith();
    test_func_module();
    test_tensor_types();
    test_empty();
    test_math_ops();
    printf("=== Done ===\n");
    return 0;
}
