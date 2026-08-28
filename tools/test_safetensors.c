#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "safetensors.h"

static int pass_count = 0, fail_count = 0;

/* Build a minimal .safetensors file for testing */
static int build_test_file(const char *path) {
    float weight_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float bias_data[3] = {0.1f, 0.2f, 0.3f};
    size_t weight_size = sizeof(weight_data);
    size_t bias_size = sizeof(bias_data);

    char json[1024];
    int json_len = snprintf(json, sizeof(json),
        "{"
        "\"weight\": {\"dtype\": \"F32\", \"shape\": [2, 3], \"data_offsets\": [%zu, %zu]},"
        "\"bias\": {\"dtype\": \"F32\", \"shape\": [3], \"data_offsets\": [%zu, %zu]}"
        "}",
        0, weight_size, weight_size, weight_size + bias_size);

    uint64_t header_size = (uint64_t)json_len;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint8_t hdr_size[8];
    for (int i = 0; i < 8; i++)
        hdr_size[i] = (uint8_t)((header_size >> (i * 8)) & 0xFF);
    fwrite(hdr_size, 1, 8, f);
    fwrite(json, 1, json_len, f);
    fwrite(weight_data, 1, weight_size, f);
    fwrite(bias_data, 1, bias_size, f);
    fclose(f);
    return 0;
}

static void check(const char *name, int cond) {
    if (cond) { printf("PASS: %s\n", name); pass_count++; }
    else { printf("FAIL: %s\n", name); fail_count++; }
}

static void test_load(void) {
    const char *test_path = "/tmp/test.safetensors";
    if (build_test_file(test_path) != 0) {
        check("build_test_file", 0);
        return;
    }

    safetensors_t st;
    int rc = safetensors_load(test_path, &st);
    check("load returns 0", rc == 0);
    check("2 tensors", st.n_tensors == 2);

    const safetensors_tensor_t *w = safetensors_find(&st, "weight");
    check("find weight", w != NULL);
    if (w) {
        check("weight shape [2,3]", w->n_dims == 2 && w->shape[0] == 2 && w->shape[1] == 3);
        check("weight size 24", w->data_size == 24);
        float *d = (float *)w->data;
        check("weight data", d[0]==1.0f && d[1]==2.0f && d[5]==6.0f);
    }

    const safetensors_tensor_t *b = safetensors_find(&st, "bias");
    check("find bias", b != NULL);
    if (b) {
        check("bias shape [3]", b->n_dims == 1 && b->shape[0] == 3);
        check("bias size 12", b->data_size == 12);
        float *d = (float *)b->data;
        check("bias data", d[0]==0.1f && d[1]==0.2f && d[2]==0.3f);
    }

    check("nonexistent returns NULL", safetensors_find(&st, "nope") == NULL);
    safetensors_free(&st);

    /* Bad file */
    safetensors_t st2;
    check("bad file rejected", safetensors_load("/tmp/nonexistent.safetensors", &st2) != 0);
}

static void test_f16(void) {
    uint8_t f16_data[4] = {0x00, 0x3C, 0x00, 0x40};
    const char *json = "{\"w\": {\"dtype\": \"F16\", \"shape\": [2], \"data_offsets\": [0, 4]}}";
    int json_len = strlen(json);

    FILE *f = fopen("/tmp/test_f16.safetensors", "wb");
    uint8_t hdr[8];
    for (int i = 0; i < 8; i++)
        hdr[i] = (uint8_t)(((uint64_t)json_len >> (i * 8)) & 0xFF);
    fwrite(hdr, 1, 8, f);
    fwrite(json, 1, json_len, f);
    fwrite(f16_data, 1, 4, f);
    fclose(f);

    safetensors_t st;
    int rc = safetensors_load("/tmp/test_f16.safetensors", &st);
    check("f16 load", rc == 0 && st.n_tensors == 1);
    if (rc == 0 && st.n_tensors == 1) {
        const safetensors_tensor_t *t = &st.tensors[0];
        check("f16 dtype", strcmp(t->dtype_str, "F16") == 0);
        check("f16 shape [2]", t->n_dims == 1 && t->shape[0] == 2);
        check("f16 size 4", t->data_size == 4);
        check("f16 dtype_size 2", t->dtype_size == 2);
    }
    safetensors_free(&st);
}

static void test_print(void) {
    const char *test_path = "/tmp/test.safetensors";
    safetensors_t st;
    int rc = safetensors_load(test_path, &st);
    if (rc == 0) {
        safetensors_print(&st);
        check("print works", 1);
    } else {
        check("print works", 0);
    }
    safetensors_free(&st);
}

int main(void) {
    printf("=== SafeTensors Tests ===\n");
    test_load();
    test_f16();
    test_print();
    printf("\n=== %d PASS, %d FAIL ===\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
