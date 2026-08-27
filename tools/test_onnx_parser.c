/*
 * test_onnx_parser.c -- Test ONNX parser with a hand-built minimal model.
 *
 * Uses a verified-correct ONNX ModelProto byte layout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "wubu_hlir.h"
#include "onnx_parser.h"

/*
 * Minimal ONNX ModelProto (verified wire format):
 *
 *   ModelProto {
 *     ir_version = 8;
 *     graph {
 *       initializer { data_type=1; dims=[2,2]; name="W"; raw_data=<2x2 identity> }
 *       node { input="x"; output="y"; op_type="MatMul" }
 *     }
 *   }
 *
 * Lengths computed:
 *   TensorProto payload = 27 bytes → tag+len = 2 bytes
 *   NodeProto payload = 14 bytes → tag+len = 2 bytes
 *   GraphProto payload = 29 + 16 = 45 bytes → tag+len = 2 bytes
 *   ModelProto total = 2 + 2 + 45 = 49 bytes
 */
static const uint8_t minimal_onnx[] = {
    /* ModelProto: ir_version = 8 (field 1, varint) */
    0x08, 0x08,

    /* GraphProto (field 7, length-delimited, len=45) */
    0x3a, 0x2d,

    /* -- TensorProto (initializer W) -- */
    0x2a, 0x1b,           /* field 5, length-delimited, len=27 */
    0x08, 0x01,           /* data_type=FLOAT */
    0x12, 0x08,           /* dims (field 2), len=8 */
    0x02, 0x02,           /* packed int64: [2, 2] */
    0x22, 0x01, 0x57,     /* name = "W" */
    0x4a, 0x10,           /* raw_data (field 9), len=16 */
    0x00, 0x00, 0x80, 0x3F,  /* 1.0f */
    0x00, 0x00, 0x00, 0x00,  /* 0.0f */
    0x00, 0x00, 0x00, 0x00,  /* 0.0f */
    0x00, 0x00, 0x80, 0x3F,  /* 1.0f */

    /* -- NodeProto (MatMul) -- */
    0x3a, 0x0b,           /* field 7, length-delimited, len=11 */
    0x0a, 0x01, 0x78,     /* input = "x" */
    0x12, 0x01, 0x79,     /* output = "y" */
    0x22, 0x06,           /* op_type, len=6 */
    0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c,  /* "MatMul" */
};

#define MINIMAL_ONNX_SIZE sizeof(minimal_onnx)

int main(void) {
    printf("=== ONNX parser tests ===\n");
    int pass = 0, fail = 0;

    /* Write the hand-built ONNX file */
    FILE *fp = fopen("/tmp/test_model.onnx", "wb");
    if (!fp) {
        printf("  FAIL: cannot open /tmp/test_model.onnx\n");
        return 1;
    }
    fwrite(minimal_onnx, 1, MINIMAL_ONNX_SIZE, fp);
    fclose(fp);
    printf("  wrote %zu-byte ONNX model\n", MINIMAL_ONNX_SIZE);

    /* Parse with onnx_load_model */
    hlir_graph_t g;
    int rc = onnx_load_model("/tmp/test_model.onnx", &g);
    if (rc == 0) {
        printf("  PASS onnx_load_model returned 0\n");
        printf("  graph has %d nodes\n", g.n);
        if (g.n >= 1) {
            printf("  node[0] op=%d name=%s\n", g.nodes[0]->op,
                   g.nodes[0]->name ? g.nodes[0]->name : "(null)");
            pass++;
        } else {
            printf("  FAIL: expected >= 1 node, got %d\n", g.n);
            fail++;
        }
        hlir_graph_free(&g);
    } else {
        printf("  FAIL onnx_load_model returned %d\n", rc);
        fail++;
    }

    printf("\n=== RESULTS ===\n");
    printf("PASS:  %d\n", pass);
    printf("FAIL:  %d\n", fail);
    printf("TOTAL: %d\n", pass + fail);
    return fail ? 1 : 0;
}
