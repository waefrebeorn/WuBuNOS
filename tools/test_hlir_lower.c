/*
 * test_hlir_lower.c -- Test HLIR -> MIR lowering + execution through interpreter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_hlir.h"
#include "wubu_mir.h"

static float f32_from_u32(uint32_t u) {
    union { float f; uint32_t u; } c; c.u = u; return c.f;
}

static int check(const char *name, int got, int want) {
    if (got != want) {
        printf("  FAIL %-25s got=%d want=%d\n", name, got, want);
        return 0;
    }
    printf("  PASS %-25s\n", name);
    return 1;
}

int main(void) {
    printf("=== HLIR->MIR lowering tests ===\n");
    int pass = 0, fail = 0;

    /* Test 1: relu tensor (4 elements, scalar path = first element) */
    {
        hlir_graph_t g;
        hlir_graph_init(&g);
        int64_t dims[] = {4};
        hlir_tensor_t shape = hlir_tensor(1, dims, 0);
        float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
        hlir_node_t *x = hlir_constant(&g, "x", &shape, data);
        hlir_node_t *y = hlir_relu(&g, x);
        (void)y;
        wubu_mir_prog_t prog;
        if (hlir_lower_mir(&g, &prog) == 0) {
            /* Pre-allocate mem so interpreter uses it without freeing */
            prog.mem = calloc(512, sizeof(int64_t));
            prog.total_mem = 255;
            wubu_mir_interp(&prog);
            if (prog.mem) {
                /* x at addr=1, relu output at addr=5 (4 elems, scalar path=first) */
                float in_val  = f32_from_u32((uint32_t)prog.mem[1]);
                float out_val = f32_from_u32((uint32_t)prog.mem[2]);
                int ok1 = fabsf(in_val - (-1.0f)) < 0.01f;
                int ok2 = fabsf(out_val - 0.0f) < 0.01f;
                if (ok1) { printf("  PASS relu input=-1.0 (got %.4f)\n", in_val); pass++; }
                else { printf("  FAIL relu input got=%.4f want=-1.0\n", in_val); fail++; }
                if (ok2) { printf("  PASS relu output=0.0 (got %.4f)\n", out_val); pass++; }
                else { printf("  FAIL relu output got=%.4f want=0.0\n", out_val); fail++; }
            } else { fail++; printf("  FAIL relu: mem null\n"); }
            free(prog.mem);
            wubu_mir_free(&prog);
        } else { fail++; printf("  FAIL relu: lower failed\n"); }
        hlir_graph_free(&g);
    }

    /* Test 2: add + mul (scalar, 1 element each) */
    {
        hlir_graph_t g;
        hlir_graph_init(&g);
        int64_t dims[] = {1};
        hlir_tensor_t shape = hlir_tensor(1, dims, 0);
        float da[] = {2.0f};
        float db[] = {3.0f};
        hlir_node_t *a = hlir_constant(&g, "a", &shape, da);
        hlir_node_t *b = hlir_constant(&g, "b", &shape, db);
        hlir_add(&g, a, b);
        hlir_mul(&g, a, b);
        wubu_mir_prog_t prog;
        if (hlir_lower_mir(&g, &prog) == 0) {
            prog.mem = calloc(512, sizeof(int64_t));
            prog.total_mem = 255;
            wubu_mir_interp(&prog);
            if (prog.mem) {
                /* a at addr 1, b at addr 5, add at addr 9, mul at addr 13 */
                float a_val   = f32_from_u32((uint32_t)prog.mem[1]);
                float b_val   = f32_from_u32((uint32_t)prog.mem[2]);
                float add_val = f32_from_u32((uint32_t)prog.mem[3]);
                float mul_val = f32_from_u32((uint32_t)prog.mem[4]);
                int ok;
                ok = fabsf(a_val - 2.0f) < 0.01f;
                if (ok) { printf("  PASS a=2.0 (got %.4f)\n", a_val); pass++; }
                else { printf("  FAIL a got=%.4f want=2.0\n", a_val); fail++; }
                ok = fabsf(b_val - 3.0f) < 0.01f;
                if (ok) { printf("  PASS b=3.0 (got %.4f)\n", b_val); pass++; }
                else { printf("  FAIL b got=%.4f want=3.0\n", b_val); fail++; }
                ok = fabsf(add_val - 5.0f) < 0.01f;
                if (ok) { printf("  PASS add 2+3=5 (got %.4f)\n", add_val); pass++; }
                else { printf("  FAIL add got=%.4f want=5.0\n", add_val); fail++; }
                ok = fabsf(mul_val - 6.0f) < 0.01f;
                if (ok) { printf("  PASS mul 2*3=6 (got %.4f)\n", mul_val); pass++; }
                else { printf("  FAIL mul got=%.4f want=6.0\n", mul_val); fail++; }
            } else { fail++; printf("  FAIL add+mul: mem null\n"); }
            free(prog.mem);
            wubu_mir_free(&prog);
        } else { fail++; printf("  FAIL add+mul: lower failed\n"); }
        hlir_graph_free(&g);
    }

    /* Test 3: gelu scalar */
    {
        hlir_graph_t g;
        hlir_graph_init(&g);
        int64_t dims[] = {1};
        hlir_tensor_t shape = hlir_tensor(1, dims, 0);
        float data[] = {0.0f};
        hlir_node_t *x = hlir_constant(&g, "x", &shape, data);
        hlir_gelu(&g, x);
        wubu_mir_prog_t prog;
        if (hlir_lower_mir(&g, &prog) == 0) {
            prog.mem = calloc(512, sizeof(int64_t));
            prog.total_mem = 255;
            wubu_mir_interp(&prog);
            if (prog.mem) {
                /* x at addr 1, gelu output at addr 5 */
                float val = f32_from_u32((uint32_t)prog.mem[2]);
                int ok = fabsf(val - 0.0f) < 0.02f;
                if (ok) { printf("  PASS gelu(0)~0 (got %.4f)\n", val); pass++; }
                else { printf("  FAIL gelu(0) got=%.4f want=~0\n", val); fail++; }
            } else { fail++; printf("  FAIL gelu: mem null\n"); }
            free(prog.mem);
            wubu_mir_free(&prog);
        } else { fail++; printf("  FAIL gelu: lower failed\n"); }
        hlir_graph_free(&g);
    }

    printf("\n=== RESULTS ===\n");
    printf("PASS:  %d\n", pass);
    printf("FAIL:  %d\n", fail);
    printf("TOTAL: %d\n", pass + fail);
    return fail ? 1 : 0;
}
