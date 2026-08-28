/*
 * tools/test_hf_hub.c — HuggingFace Hub download test.
 *
 * Tests the HF Hub runtime by downloading a small model file
 * and parsing it as SafeTensors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hf_hub.h"

int main(void) {
    printf("=== HF Hub Test ===\n\n");

    /* Test 1: Download a small test file */
    printf("Test 1: Download test file from HF Hub\n");
    size_t sz;
    uint8_t *data = hf_hub_download("hf-internal-testing/tiny-random-llama",
                                     "model.safetensors", &sz);
    if (data) {
        printf("  Downloaded %zu bytes\n", sz);

        /* Parse as SafeTensors */
        safetensors_t st;
        int rc = safetensors_load_from_memory(data, sz, &st);
        if (rc == 0) {
            printf("  Parsed: %d tensors\n", st.n_tensors);
            safetensors_print(&st);
            safetensors_free(&st);
        } else {
            printf("  Parse failed: %s\n", st.error);
        }
        free(data);
    } else {
        printf("  Download failed (network may be unavailable)\n");
    }

    printf("\n=== Done ===\n");
    return 0;
}
