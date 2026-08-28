#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "safetensors_bridge.h"
#include "wubu_mir.h"
#include "wubu_softfloat.h"

static int pass_count = 0, fail_count = 0;

static void check(const char *name, int cond) {
    if (cond) { printf("PASS: %s\n", name); pass_count++; }
    else { printf("FAIL: %s\n", name); fail_count++; }
}

/* Build a test .safetensors file */
static int build_test_file(const char *path) {
    float weight[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float bias[3] = {0.1f, 0.2f, 0.3f};
    char json[1024];
    int jlen = snprintf(json, sizeof(json),
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"bias\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[24,36]}}");
    uint64_t hs = jlen;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint8_t h[8]; for (int i=0;i<8;i++) h[i]=(uint8_t)((hs>>(i*8))&0xFF);
    fwrite(h,1,8,f); fwrite(json,1,jlen,f);
    fwrite(weight,1,24,f); fwrite(bias,1,12,f);
    fclose(f);
    return 0;
}

/* Test 1: Load SafeTensors → HLIR */
static void test_bridge(void) {
    build_test_file("/tmp/test_bridge.safetensors");

    safetensors_t st;
    int rc = safetensors_load("/tmp/test_bridge.safetensors", &st);
    check("safetensors load", rc == 0 && st.n_tensors == 2);

    hlir_graph_t g;
    rc = safetensors_to_hlir(&st, &g);
    check("safetensors_to_hlir", rc == 0);
    check("graph has 2 nodes", g.n == 2);

    /* Check weight node */
    int found_weight = 0;
    for (int i = 0; i < g.n; i++) {
        if (strcmp(g.nodes[i]->name, "weight_weight") == 0) {
            found_weight = 1;
            check("weight dtype F32", g.nodes[i]->output.dtype == 0);
            check("weight nelems 6", g.nodes[i]->output.nelems == 6);
            if (g.nodes[i]->data) {
                float *d = (float *)g.nodes[i]->data;
                check("weight data[0]=1.0", d[0] == 1.0f);
                check("weight data[5]=6.0", d[5] == 6.0f);
            }
        }
    }
    check("found weight node", found_weight);

    hlir_graph_free(&g);
    safetensors_free(&st);
}

/* Test 2: Load SafeTensors → HLIR → MIR */
static void test_bridge_to_mir(void) {
    build_test_file("/tmp/test_bridge.safetensors");

    safetensors_t st;
    safetensors_load("/tmp/test_bridge.safetensors", &st);

    hlir_graph_t g;
    safetensors_to_hlir(&st, &g);

    /* Lower to MIR */
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int rc = hlir_lower_mir(&g, &prog);
    check("hlir_lower_mir", rc == 0);
    check("MIR has instructions", prog.n > 0);
    printf("  MIR: %u instructions\n", prog.n);

    wubu_mir_free(&prog);
    hlir_graph_free(&g);
    safetensors_free(&st);
}

/* Test 3: F16 conversion */
static void test_f16_bridge(void) {
    /* Build F16 safetensors */
    uint16_t f16_data[4];
    /* 1.0 = 0x3C00, 2.0 = 0x4000 in F16 */
    f16_data[0] = 0x3C00; f16_data[1] = 0x4000;
    f16_data[2] = 0x4200; f16_data[3] = 0x4400; /* 3.0, 4.0 */

    const char *json = "{\"w\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    int jlen = strlen(json);
    FILE *f = fopen("/tmp/test_f16_bridge.safetensors", "wb");
    uint8_t h[8]; for (int i=0;i<8;i++) h[i]=(uint8_t)(((uint64_t)jlen>>(i*8))&0xFF);
    fwrite(h,1,8,f); fwrite(json,1,jlen,f);
    fwrite(f16_data,1,8,f); fclose(f);

    safetensors_t st;
    int rc = safetensors_load("/tmp/test_f16_bridge.safetensors", &st);
    check("f16 load", rc == 0 && st.n_tensors == 1);

    hlir_graph_t g;
    rc = safetensors_to_hlir(&st, &g);
    check("f16 to hlir", rc == 0 && g.n == 1);

    if (g.n == 1 && g.nodes[0]->data) {
        float *d = (float *)g.nodes[0]->data;
        check("f16 data[0]=1.0", d[0] == 1.0f);
        check("f16 data[1]=2.0", d[1] == 2.0f);
        check("f16 data[2]=3.0", d[2] == 3.0f);
        check("f16 data[3]=4.0", d[3] == 4.0f);
    }

    hlir_graph_free(&g);
    safetensors_free(&st);
}

int main(void) {
    printf("=== SafeTensors Bridge Tests ===\n");
    test_bridge();
    test_bridge_to_mir();
    test_f16_bridge();
    printf("\n=== %d PASS, %d FAIL ===\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
