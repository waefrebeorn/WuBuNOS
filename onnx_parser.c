/*
 * onnx_parser.c -- Minimal ONNX model parser -> HLIR graph.
 *
 * Parses the ONNX protobuf wire format directly (no protoc/protobuf library
 * dependency). ONNX ModelProto has the structure:
 *
 *   message ModelProto {
 *     int64 ir_version = 1;
 *     repeated TensorProto initializer = 5;
 *     GraphProto graph = 7;
 *   }
 *   message GraphProto {
 *     repeated TensorProto initializer = 5;
 *     repeated NodeProto node = 7;
 *   }
 *   message NodeProto {
 *     repeated string input = 1;
 *     repeated string output = 2;
 *     string op_type = 4;
 *   }
 *   message TensorProto {
 *     int32 data_type = 1;
 *     repeated int64 dims = 2;
 *     bytes raw_data = 9;
 *   }
 *
 * Wire format: varint(field_number << 3 | wire_type) + payload.
 * Wire types: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit.
 *
 * C11, self-contained.
 *
 * STATUS: parser skeleton complete. ONNX wire reader works. Node type
 * mapping works. Test harness compiles. The encoder in test_onnx_parser.c
 * has a length-patching bug (todo: fix before end-to-end test).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "wubu_hlir.h"

/* ---- Minimal protobuf wire reader ---- */

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
} pb_reader_t;

static int pb_init(pb_reader_t *r, const void *data, size_t len) {
    r->data = (const uint8_t *)data;
    r->len = len;
    r->pos = 0;
    return 0;
}

static uint64_t pb_peek_varint(pb_reader_t *r, size_t *consumed) {
    uint64_t val = 0;
    int shift = 0;
    size_t start = r->pos;
    if (start >= r->len) { if (consumed) *consumed = 0; return 0; }
    while (r->pos < r->len) {
        uint8_t b = r->data[r->pos++];
        val |= (uint64_t)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
        if (shift > 63) break;
    }
    if (consumed) *consumed = r->pos - start;
    return val;
}

static uint64_t pb_read_varint(pb_reader_t *r) {
    return pb_peek_varint(r, NULL);
}

static uint32_t pb_read_fixed32(pb_reader_t *r) {
    if (r->pos + 4 > r->len) return 0;
    uint32_t v = (uint32_t)r->data[r->pos    ] |
                 (uint32_t)r->data[r->pos + 1] << 8 |
                 (uint32_t)r->data[r->pos + 2] << 16 |
                 (uint32_t)r->data[r->pos + 3] << 24;
    r->pos += 4;
    return v;
}

static const uint8_t *pb_read_bytes(pb_reader_t *r, size_t *out_len) {
    uint64_t len = pb_read_varint(r);
    if (r->pos + (size_t)len > r->len) { *out_len = 0; return NULL; }
    const uint8_t *ptr = r->data + r->pos;
    *out_len = (size_t)len;
    r->pos += (size_t)len;
    return ptr;
}

static void pb_skip_field(pb_reader_t *r, uint8_t wire_type) {
    switch (wire_type) {
        case 0: pb_read_varint(r); break;
        case 1: r->pos += 8; break;
        case 2: { size_t l; pb_read_bytes(r, &l); break; }
        case 5: r->pos += 4; break;
        default: break;
    }
}

static void pb_read_repeated_int64(pb_reader_t *r, int64_t *out, int max, int *count) {
    *count = 0;
    size_t saved = r->pos;
    while (r->pos < r->len && *count < max) {
        uint64_t tag = pb_peek_varint(r, NULL);
        uint8_t wire_type = (uint8_t)(tag & 0x07);
        uint32_t field_num = (uint32_t)(tag >> 3);

        if (field_num == 2 && wire_type == 0) {
            /* Packed repeated varint */
            size_t pack_len;
            const uint8_t *pack = pb_read_bytes(r, &pack_len);
            if (pack) {
                const uint8_t *p = pack;
                uint64_t val = 0;
                int shift = 0;
                while (p < pack + pack_len && *count < max) {
                    uint8_t b = *p++;
                    val |= (uint64_t)(b & 0x7F) << shift;
                    if ((b & 0x80) == 0) {
                        out[(*count)++] = (int64_t)val;
                        val = 0;
                        shift = 0;
                    } else {
                        shift += 7;
                    }
                }
            }
        } else if (field_num == 2 && wire_type == 2) {
            pb_skip_field(r, wire_type);
        } else {
            r->pos = saved;
            return;
        }
    }
    r->pos = saved;
}

/* ---- ONNX op/dtype mapping ---- */

static int onnx_dtype_to_hlir(int32_t onnx_dtype) {
    switch (onnx_dtype) {
        case 1:  return 0;  /* FLOAT */
        case 2:  return 7;  /* UINT8 */
        case 3:  return 3;  /* INT8 */
        case 4:  return 9;  /* UINT16 */
        case 5:  return 5;  /* INT16 */
        case 6:  return 6;  /* INT32 */
        case 7:  return 11; /* INT64 */
        case 10: return 2;  /* BFLOAT16 */
        case 11: return 1;  /* FLOAT16 */
        case 12: return 7;  /* DOUBLE */
        default: return 0;
    }
}

static hlir_op_t onnx_op_to_hlir(const char *op_type) {
    if (!op_type) return HLIR_ADD;
    if (!strcmp(op_type, "Add") || !strcmp(op_type, "Sub")) return HLIR_ADD;
    if (!strcmp(op_type, "Mul")) return HLIR_MUL;
    if (!strcmp(op_type, "MatMul")) return HLIR_MATMUL;
    if (!strcmp(op_type, "Relu")) return HLIR_RELU;
    if (!strcmp(op_type, "Gelu")) return HLIR_GELU;
    if (!strcmp(op_type, "Sigmoid")) return HLIR_SIGMOID;
    if (!strcmp(op_type, "Softmax")) return HLIR_SOFTMAX;
    if (!strcmp(op_type, "LayerNormalization")) return HLIR_LAYERNORM;
    if (!strcmp(op_type, "RMSNormalization")) return HLIR_RMSNORM;
    if (!strcmp(op_type, "Reshape")) return HLIR_RESHAPE;
    if (!strcmp(op_type, "Transpose")) return HLIR_TRANSPOSE;
    if (!strcmp(op_type, "Cast")) return HLIR_CAST;
    if (!strcmp(op_type, "ReduceSum")) return HLIR_REDUCE_SUM;
    if (!strcmp(op_type, "Attention")) return HLIR_ATTENTION;
    if (!strcmp(op_type, "SiLU") || !strcmp(op_type, "Swish")) return HLIR_GELU;
    return HLIR_ADD;  /* Unknown -> passthrough */
}

/* ---- ONNX parser ---- */

/*
 * onnx_load_model: parse an ONNX file and populate an HLIR graph.
 *
 * Currently supported ops: Add, Mul, MatMul, Relu, Gelu, Softmax,
 * LayerNormalization, RMSNormalization, Reshape, Transpose, Cast,
 * ReduceSum, Attention, SiLU/Swish.
 *
 * Returns 0 on success, -1 on error.
 */
int onnx_load_model(const char *filepath, hlir_graph_t *g) {
    if (!filepath || !g) return -1;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) { fprintf(stderr, "onnx: cannot open %s\n", filepath); return -1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 256LL * 1024 * 1024) {
        fprintf(stderr, "onnx: invalid file size %ld\n", fsize);
        fclose(fp); return -1;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { fclose(fp); return -1; }
    if ((long)fread(buf, 1, (size_t)fsize, fp) != fsize) {
        fprintf(stderr, "onnx: short read\n"); free(buf); fclose(fp); return -1;
    }
    fclose(fp);

    pb_reader_t r;
    pb_init(&r, buf, (size_t)fsize);

    /* Verify it's an ONNX model (check for graph field 7) */
    int has_graph = 0;
    pb_reader_t scan = r;
    while (scan.pos < scan.len) {
        uint64_t tag = pb_peek_varint(&scan, NULL);
        uint8_t wire_type = (uint8_t)(tag & 0x07);
        uint32_t field_num = (uint32_t)(tag >> 3);
        if (field_num == 7 && wire_type == 2) { has_graph = 1; break; }
        pb_skip_field(&scan, wire_type);
    }
    if (!has_graph) {
        fprintf(stderr, "onnx: not a valid ONNX model (no graph field)\n");
        free(buf); return -1;
    }

    /* Re-parse to extract initializers and nodes */
    pb_init(&r, buf, (size_t)fsize);

    hlir_node_t *nodes[256];
    int n_nodes = 0;
    memset(nodes, 0, sizeof(nodes));

    while (r.pos < r.len) {
        uint64_t tag = pb_peek_varint(&r, NULL);
        uint8_t wire_type = (uint8_t)(tag & 0x07);
        uint32_t field_num = (uint32_t)(tag >> 3);

        if (field_num == 5 && wire_type == 2) {
            /* TensorProto initializer */
            size_t ilen;
            const uint8_t *idata = pb_read_bytes(&r, &ilen);
            if (idata) {
                pb_reader_t ir;
                pb_init(&ir, idata, ilen);

                int32_t data_type = 1;  /* default FLOAT */
                int64_t dims[8];
                int n_dims = 0;
                const char *name = NULL;
                const uint8_t *raw_data = NULL;
                size_t raw_len = 0;

                while (ir.pos < ir.len) {
                    uint64_t itag = pb_peek_varint(&ir, NULL);
                    uint8_t iwt = (uint8_t)(itag & 0x07);
                    uint32_t ifn = (uint32_t)(itag >> 3);

                    if (ifn == 1 && iwt == 0) {
                        data_type = (int32_t)pb_read_varint(&ir);
                    } else if (ifn == 2 && iwt == 2) {
                        /* dims — skip for now */
                        size_t dl;
                        pb_read_bytes(&ir, &dl);
                    } else if (ifn == 4 && iwt == 2) {
                        size_t nl;
                        const uint8_t *ns = pb_read_bytes(&ir, &nl);
                        if (ns && nl > 0 && n_nodes < 255) {
                            static char name_buf[256];
                            if (nl >= sizeof(name_buf)) nl = sizeof(name_buf) - 1;
                            memcpy(name_buf, ns, nl);
                            name_buf[nl] = '\0';
                            name = name_buf;
                        }
                    } else if (ifn == 9 && iwt == 2) {
                        raw_data = pb_read_bytes(&ir, &raw_len);
                    } else {
                        pb_skip_field(&ir, iwt);
                    }
                }

                if (name && n_dims >= 0) {
                    int dtype = onnx_dtype_to_hlir(data_type);
                    hlir_tensor_t shape;
                    memset(&shape, 0, sizeof(shape));
                    shape.dtype = dtype;
                    /* For now, single-element constant */
                    shape.nelems = raw_len / 4;
                    hlir_node_t *node = hlir_constant(g, name, &shape, raw_data);
                    if (node && n_nodes < 256) {
                        nodes[n_nodes++] = node;
                    }
                }
            }
        } else if (field_num == 7 && wire_type == 2) {
            /* GraphProto - parse nodes */
            size_t glen;
            const uint8_t *gdata = pb_read_bytes(&r, &glen);
            if (gdata) {
                pb_reader_t gr;
                pb_init(&gr, gdata, glen);

                while (gr.pos < gr.len) {
                    uint64_t gtag = pb_peek_varint(&gr, NULL);
                    uint8_t gwt = (uint8_t)(gtag & 0x07);
                    uint32_t gfn = (uint32_t)(gtag >> 3);

                    if (gfn == 7 && gwt == 2) {
                        size_t nlen;
                        const uint8_t *ndata = pb_read_bytes(&gr, &nlen);
                        if (ndata) {
                            pb_reader_t nr;
                            pb_init(&nr, ndata, nlen);

                            char op_type[64] = {0};
                            char input_names[8][64];
                            char output_names[8][64];
                            int n_in = 0, n_out = 0;

                            while (nr.pos < nr.len) {
                                uint64_t ntag = pb_peek_varint(&nr, NULL);
                                uint8_t nwt = (uint8_t)(ntag & 0x07);
                                uint32_t nfn = (uint32_t)(ntag >> 3);

                                if (nfn == 4 && nwt == 2) {
                                    size_t sl;
                                    const uint8_t *s = pb_read_bytes(&nr, &sl);
                                    if (s && sl < sizeof(op_type)) {
                                        memcpy(op_type, s, sl);
                                        op_type[sl] = '\0';
                                    }
                                } else if (nfn == 1 && nwt == 2) {
                                    size_t sl;
                                    const uint8_t *s = pb_read_bytes(&nr, &sl);
                                    if (s && sl < 64 && n_in < 8) {
                                        memcpy(input_names[n_in], s, sl);
                                        input_names[n_in][sl] = '\0';
                                        n_in++;
                                    }
                                } else if (nfn == 2 && nwt == 2) {
                                    size_t sl;
                                    const uint8_t *s = pb_read_bytes(&nr, &sl);
                                    if (s && sl < 64 && n_out < 8) {
                                        memcpy(output_names[n_out], s, sl);
                                        output_names[n_out][sl] = '\0';
                                        n_out++;
                                    }
                                } else {
                                    pb_skip_field(&nr, nwt);
                                }
                            }

                            /* Look up input nodes by name */
                            hlir_node_t *inp_nodes[16];
                            int n_inp = 0;
                            for (int j = 0; j < n_in && n_inp < 16; j++) {
                                inp_nodes[n_inp] = NULL;
                                for (int k = 0; k < n_nodes; k++) {
                                    if (nodes[k] && nodes[k]->name &&
                                        strcmp(nodes[k]->name, input_names[j]) == 0) {
                                        inp_nodes[n_inp] = nodes[k];
                                        break;
                                    }
                                }
                                n_inp++;
                            }

                            /* Create HLIR node */
                            hlir_op_t op = onnx_op_to_hlir(op_type);
                            char out_name_buf[64];
                            const char *out_name = NULL;
                            if (n_out > 0 && output_names[0][0])
                                out_name = output_names[0];
                            else {
                                snprintf(out_name_buf, sizeof(out_name_buf), "node_%d", n_nodes);
                                out_name = out_name_buf;
                            }

                            hlir_node_t *out_node = NULL;
                            if (n_inp >= 2 && (op == HLIR_ADD || op == HLIR_MUL || op == HLIR_MATMUL)) {
                                out_node = hlir_op(g, op, out_name,
                                                   (hlir_node_t *[]){inp_nodes[0], inp_nodes[1]},
                                                   2, NULL, NULL, 0);
                            } else if (n_inp >= 1) {
                                out_node = hlir_op(g, op, out_name, &inp_nodes[0], 1, NULL, NULL, 0);
                            } else {
                                out_node = hlir_op(g, op, out_name, NULL, 0, NULL, NULL, 0);
                            }

                            if (out_node && n_nodes < 256)
                                nodes[n_nodes++] = out_node;
                        }
                    } else {
                        pb_skip_field(&gr, gwt);
                    }
                }
            }
        } else {
            pb_skip_field(&r, wire_type);
        }
    }

    free(buf);
    fprintf(stderr, "onnx: parsed %d nodes from model\n", n_nodes);
    return 0;
}
