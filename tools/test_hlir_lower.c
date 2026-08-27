/*
 * test_hlir_lower.c -- Test HLIR -> MIR lowering + execution through interpreter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_hlir.h"
#include "wubu_mir.h"

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

    /* Test 1: relu tensor */
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
            int rc = wubu_mir_interp(&prog);
            printf("  test_relu interp rc=%d mem=%p\n", rc, (void*)prog.mem);
            if (prog.mem) {
                /* relu(-1)=0, relu(0)=0, relu(1)=1, relu(2)=2 */
                pass += check("relu[-1]=0", (int)prog.mem[1], 0);
                pass += check("relu[0]=0", (int)prog.mem[2], 0);
                pass += check("relu[1]=1", (int)prog.mem[3], 1);
                pass += check("relu[2]=2", (int)prog.mem[4], 2);
            } else { fail++; printf("  FAIL relu: mem null\n"); }
            wubu_mir_free(&prog);
        } else { fail++; printf("  FAIL relu: lower failed\n"); }
        hlir_graph_free(&g);
    }

    /* Test 2: add + mul */
    {
        hlir_graph_t g;
        hlir_graph_init(&g);
        int64_t dims[] = {1};
        hlir_tensor_t shape = hlir_tensor(1, dims, 0);
        float da[] = {2.0f};
        float db[] = {3.0f};
        hlir_node_t *a = hlir_constant(&g, "a", &shape, da);
        hlir_node_t *b = hlir_constant(&g, "b", &shape, db);
        hlir_node_t *add = hlir_add(&g, a, b);
        hlir_node_t *mul = hlir_mul(&g, add, b);
        (void)mul;
        wubu_mir_prog_t prog;
        if (hlir_lower_mir(&g, &prog) == 0) {
            int rc = wubu_mir_interp(&prog);
            printf("  test_add+mul interp rc=%d mem=%p\n", rc, (void*)prog.mem);
            if (prog.mem) {
                float add_val = wubu_sf_f32_to_host((uint32_t)prog.mem[3]);
                float mul_val = wubu_sf_f32_to_host((uint32_t)prog.mem[4]);
                int ok1 = fabsf(add_val - 5.0f) < 0.01f;
                int ok2 = fabsf(mul_val - 15.0f) < 0.01f;
                if (ok1) { printf("  PASS add 2+3=5 (got %.4f)\n", add_val); pass++; }
                else { printf("  FAIL add got=%.4f want=5.0\n", add_val); fail++; }
                if (ok2) { printf("  PASS mul (2+3)*3=15 (got %.4f)\n", mul_val); pass++; }
                else { printf("  FAIL mul got=%.4f want=15.0\n", mul_val); fail++; }
            } else { fail++; printf("  FAIL add+mul: mem null\n"); }
            wubu_mir_free(&prog);
        } else { fail++; printf("  FAIL add+mul: lower failed\n"); }
        hlir_graph_free(&g);
    }

    /* Test 3: gelu */
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
            int rc = wubu_mir_interp(&prog);
            printf("  test_gelu interp rc=%d mem=%p\n", rc, (void*)prog.mem);
            if (prog.mem) {
                float val = wubu_sf_f32_to_host((uint32_t)prog.mem[1]);
                int ok = fabsf(val - 0.0f) < 0.02f;
                if (ok) { printf("  PASS gelu(0)~0 (got %.4f)\n", val); pass++; }
                else { printf("  FAIL gelu(0) got=%.4f want=~0\n", val); fail++; }
            } else { fail++; printf("  FAIL gelu: mem null\n"); }
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
