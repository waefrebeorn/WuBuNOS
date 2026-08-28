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

/* Forward declarations for tests defined after main() */
static void test_linalg_matmul(void);
static void test_memref_ops(void);
static void test_scf_control_flow(void);
static void test_realistic_mlp(void);

int main(void) {
    printf("=== MLIR Text Parser Tests ===\n");
    test_simple_arith();
    test_func_module();
    test_tensor_types();
    test_empty();
    test_math_ops();
    test_linalg_matmul();
    test_memref_ops();
    test_scf_control_flow();
    test_realistic_mlp();
    printf("=== Done ===\n");
    return 0;
}

/* Test 6: linalg matmul */
static void test_linalg_matmul(void) {
    const char *src =
        "func.func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>) -> tensor<4x4xf32> {\n"
        "  %C = linalg.matmul ins(%A, %B) outs(%C_init) -> tensor<4x4xf32>\n"
        "  return %C : tensor<4x4xf32>\n"
        "}\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  linalg_matmul: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 1) {
        printf("PASS: test_linalg_matmul (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_linalg_matmul — expected >= 1 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 7: memref ops */
static void test_memref_ops(void) {
    const char *src =
        "%buf = memref.alloc() : memref<256xf32>\n"
        "%val = memref.load %buf[%idx] : memref<256xf32>\n"
        "%res = arith.addf(%val, %c1) : f32\n"
        "memref.store %res, %buf[%idx] : memref<256xf32>\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  memref_ops: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 3) {
        printf("PASS: test_memref_ops (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_memref_ops — expected >= 3 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 8: scf control flow */
static void test_scf_control_flow(void) {
    const char *src =
        "func.func @loop(%N: index) -> f32 {\n"
        "  %c0 = arith.constant 0 : index\n"
        "  %sum_init = arith.constant 0.0 : f32\n"
        "  %result = scf.for %i = %c0 to %N step %c0 iter_args(%s = %sum_init) -> f32 {\n"
        "    %new_sum = arith.addf(%s, %one) : f32\n"
        "    scf.yield %new_sum : f32\n"
        "  }\n"
        "  return %result : f32\n"
        "}\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  scf_control_flow: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 2) {
        printf("PASS: test_scf_control_flow (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_scf_control_flow — expected >= 2 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}

/* Test 9: Realistic MLP layer (matmul + bias + relu) */
static void test_realistic_mlp(void) {
    const char *src =
        "func.func @linear(%input: tensor<1x256xf32>, %weight: tensor<256x512xf32>,\n"
        "                  %bias: tensor<512xf32>) -> tensor<1x512xf32> {\n"
        "  %c0 = arith.constant 0.0 : f32\n"
        "  %filled = linalg.fill ins(%c0 : f32) outs(%bias : tensor<512xf32>)\n"
        "  %gemm = linalg.matmul ins(%input, %weight) outs(%filled) -> tensor<1x512xf32>\n"
        "  %biased = arith.addf(%gemm, %bias) : tensor<1x512xf32>\n"
        "  %neg = arith.constant 0.0 : f32\n"
        "  %activated = arith.maximumf(%biased, %neg) : tensor<1x512xf32>\n"
        "  return %activated : tensor<1x512xf32>\n"
        "}\n";

    hlir_graph_t g;
    int rc = mlir_text_load(src, &g);
    printf("  realistic_mlp: rc=%d, n=%d\n", rc, g.n);
    if (rc == 0 && g.n >= 4) {
        printf("PASS: test_realistic_mlp (%d ops)\n", g.n);
    } else {
        printf("FAIL: test_realistic_mlp — expected >= 4 ops, got %d\n", g.n);
    }
    hlir_graph_free(&g);
}
