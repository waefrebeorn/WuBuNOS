/*
 * safetensors.c — SafeTensors format parser.
 *
 * Parses .safetensors files (the standard format for ML model weights).
 * Format:
 *   - 8 bytes: header size (uint64 LE)
 *   - N bytes: JSON header { "name": {"dtype": "F16", "shape": [..], "data_offsets": [BEGIN, END]}, ... }
 *   - Rest: raw tensor data bytes
 *
 * Self-contained: minimal JSON parser (no external deps), supports F32/F16/BF16/I8/I32/I64 dtypes.
 * C11, no third-party libraries.
 */
#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ─── Minimal JSON parser ─── */

typedef struct {
    const char *src;
    int         pos;
    int         len;
} json_parser_t;

static void json_skip_ws(json_parser_t *jp) {
    while (jp->pos < jp->len && isspace((unsigned char)jp->src[jp->pos]))
        jp->pos++;
}

static int json_parse_string(json_parser_t *jp, char *buf, int cap) {
    json_skip_ws(jp);
    if (jp->pos >= jp->len || jp->src[jp->pos] != '"') return -1;
    jp->pos++; /* skip opening quote */
    int i = 0;
    while (jp->pos < jp->len && jp->src[jp->pos] != '"' && i < cap - 1) {
        if (jp->src[jp->pos] == '\\') {
            jp->pos++;
            if (jp->pos >= jp->len) break;
            switch (jp->src[jp->pos]) {
            case '"': buf[i++] = '"'; break;
            case '\\': buf[i++] = '\\'; break;
            case '/': buf[i++] = '/'; break;
            case 'n': buf[i++] = '\n'; break;
            case 't': buf[i++] = '\t'; break;
            default: buf[i++] = jp->src[jp->pos]; break;
            }
        } else {
            buf[i++] = jp->src[jp->pos];
        }
        jp->pos++;
    }
    buf[i] = '\0';
    if (jp->pos < jp->len && jp->src[jp->pos] == '"') jp->pos++;
    return i;
}

static int json_skip_value(json_parser_t *jp);

static int json_skip_array(json_parser_t *jp) {
    if (jp->pos >= jp->len || jp->src[jp->pos] != '[') return -1;
    jp->pos++;
    json_skip_ws(jp);
    if (jp->pos < jp->len && jp->src[jp->pos] == ']') { jp->pos++; return 0; }
    do {
        if (json_skip_value(jp) < 0) return -1;
        json_skip_ws(jp);
        if (jp->pos < jp->len && jp->src[jp->pos] == ',') { jp->pos++; continue; }
        break;
    } while (1);
    if (jp->pos < jp->len && jp->src[jp->pos] == ']') jp->pos++;
    return 0;
}

static int json_skip_object(json_parser_t *jp) {
    if (jp->pos >= jp->len || jp->src[jp->pos] != '{') return -1;
    jp->pos++;
    json_skip_ws(jp);
    if (jp->pos < jp->len && jp->src[jp->pos] == '}') { jp->pos++; return 0; }
    do {
        char key[256];
        if (json_parse_string(jp, key, sizeof(key)) < 0) return -1;
        json_skip_ws(jp);
        if (jp->pos < jp->len && jp->src[jp->pos] == ':') jp->pos++;
        if (json_skip_value(jp) < 0) return -1;
        json_skip_ws(jp);
        if (jp->pos < jp->len && jp->src[jp->pos] == ',') { jp->pos++; continue; }
        break;
    } while (1);
    if (jp->pos < jp->len && jp->src[jp->pos] == '}') jp->pos++;
    return 0;
}

static int json_skip_value(json_parser_t *jp) {
    json_skip_ws(jp);
    if (jp->pos >= jp->len) return -1;
    char c = jp->src[jp->pos];
    if (c == '"') { char buf[256]; return json_parse_string(jp, buf, sizeof(buf)); }
    if (c == '[') return json_skip_array(jp);
    if (c == '{') return json_skip_object(jp);
    /* number, true, false, null */
    while (jp->pos < jp->len && jp->src[jp->pos] != ',' && jp->src[jp->pos] != '}' &&
           jp->src[jp->pos] != ']' && !isspace((unsigned char)jp->src[jp->pos]))
        jp->pos++;
    return 0;
}

/* Parse a string value for a given key (assumes we're at the key) */
static int json_get_string_value(json_parser_t *jp, char *buf, int cap) {
    json_skip_ws(jp);
    if (jp->pos >= jp->len || jp->src[jp->pos] != ':') return -1;
    jp->pos++;
    json_skip_ws(jp);
    return json_parse_string(jp, buf, cap);
}

/* Parse an integer value for a given key */
static int json_get_int_value(json_parser_t *jp, int64_t *val) {
    json_skip_ws(jp);
    if (jp->pos >= jp->len || jp->src[jp->pos] != ':') return -1;
    jp->pos++;
    json_skip_ws(jp);
    *val = 0;
    int neg = 0;
    if (jp->pos < jp->len && jp->src[jp->pos] == '-') { neg = 1; jp->pos++; }
    while (jp->pos < jp->len && isdigit((unsigned char)jp->src[jp->pos])) {
        *val = *val * 10 + (jp->src[jp->pos] - '0');
        jp->pos++;
    }
    if (neg) *val = -*val;
    return 0;
}

/* ─── SafeTensors dtype mapping ─── */

static int safetensors_dtype_size(const char *dtype) {
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "I8") == 0) return 1;
    if (strcmp(dtype, "I16") == 0) return 2;
    if (strcmp(dtype, "I32") == 0) return 4;
    if (strcmp(dtype, "I64") == 0) return 8;
    if (strcmp(dtype, "U8") == 0) return 1;
    return 4; /* default F32 */
}

static int safetensors_dtype_to_hlir(const char *dtype) {
    if (strcmp(dtype, "F32") == 0) return 0; /* HLIR_DTYPE_FLOAT32 */
    if (strcmp(dtype, "F16") == 0) return 1; /* HLIR_DTYPE_FLOAT16 */
    if (strcmp(dtype, "BF16") == 0) return 2; /* HLIR_DTYPE_BFLOAT16 */
    if (strcmp(dtype, "I32") == 0) return 3; /* HLIR_DTYPE_INT32 */
    if (strcmp(dtype, "I8") == 0) return 4;  /* HLIR_DTYPE_INT8 */
    return 0;
}

/* ─── Public API ─── */

int safetensors_load(const char *filepath, safetensors_t *st) {
    memset(st, 0, sizeof(*st));
    strncpy(st->filename, filepath, sizeof(st->filename) - 1);

    /* Read entire file */
    FILE *f = fopen(filepath, "rb");
    if (!f) { snprintf(st->error, sizeof(st->error), "cannot open %s", filepath); return -1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(fsize);
    if (!data) { fclose(f); snprintf(st->error, sizeof(st->error), "malloc failed"); return -1; }
    if ((long)fread(data, 1, fsize, f) != fsize) { free(data); fclose(f); snprintf(st->error, sizeof(st->error), "short read"); return -1; }
    fclose(f);

    /* Parse header size (8 bytes, uint64 LE) */
    if (fsize < 8) { free(data); snprintf(st->error, sizeof(st->error), "file too small"); return -1; }
    uint64_t header_size = 0;
    for (int i = 0; i < 8; i++)
        header_size |= ((uint64_t)data[i]) << (i * 8);

    if (header_size == 0 || header_size > (uint64_t)(fsize - 8)) {
        free(data);
        snprintf(st->error, sizeof(st->error), "invalid header size: %llu", (unsigned long long)header_size);
        return -1;
    }

    /* Parse JSON header */
    char *json = (char *)malloc(header_size + 1);
    if (!json) { free(data); snprintf(st->error, sizeof(st->error), "malloc failed"); return -1; }
    memcpy(json, data + 8, header_size);
    json[header_size] = '\0';

    json_parser_t jp = { .src = json, .pos = 0, .len = (int)header_size };

    json_skip_ws(&jp);
    if (jp.pos >= jp.len || jp.src[jp.pos] != '{') {
        free(data); free(json);
        snprintf(st->error, sizeof(st->error), "expected '{' at start of header");
        return -1;
    }

    /* Parse each tensor entry */
    jp.pos++; /* skip '{' */
    json_skip_ws(&jp);

    while (jp.pos < jp.len && jp.src[jp.pos] != '}') {
        json_skip_ws(&jp);
        if (jp.pos >= jp.len || jp.src[jp.pos] != '"') break;

        char tensor_name[256];
        if (json_parse_string(&jp, tensor_name, sizeof(tensor_name)) < 0) break;

        json_skip_ws(&jp);
        if (jp.pos < jp.len && jp.src[jp.pos] == ':') jp.pos++;
        json_skip_ws(&jp);

        if (jp.pos >= jp.len || jp.src[jp.pos] != '{') { json_skip_value(&jp); goto next; }

        /* Parse tensor info object */
        jp.pos++; /* skip '{' */
        char dtype[32] = "F32";
        int64_t shape[16] = {0};
        int n_dims = 0;
        int64_t data_begin = 0, data_end = 0;

        while (jp.pos < jp.len && jp.src[jp.pos] != '}') {
            json_skip_ws(&jp);
            if (jp.pos >= jp.len || jp.src[jp.pos] != '"') break;

            char key[64];
            if (json_parse_string(&jp, key, sizeof(key)) < 0) break;

            if (strcmp(key, "dtype") == 0) {
                json_get_string_value(&jp, dtype, sizeof(dtype));
            } else if (strcmp(key, "shape") == 0) {
                /* Parse array of ints */
                json_skip_ws(&jp);
                if (jp.pos < jp.len && jp.src[jp.pos] == ':') jp.pos++;
                json_skip_ws(&jp);
                if (jp.pos < jp.len && jp.src[jp.pos] == '[') {
                    jp.pos++;
                    json_skip_ws(&jp);
                    n_dims = 0;
                    while (jp.pos < jp.len && jp.src[jp.pos] != ']' && n_dims < 16) {
                        json_skip_ws(&jp);
                        int64_t val = 0;
                        int neg = 0;
                        if (jp.pos < jp.len && jp.src[jp.pos] == '-') { neg = 1; jp.pos++; }
                        while (jp.pos < jp.len && isdigit((unsigned char)jp.src[jp.pos])) {
                            val = val * 10 + (jp.src[jp.pos] - '0');
                            jp.pos++;
                        }
                        if (neg) val = -val;
                        shape[n_dims++] = val;
                        json_skip_ws(&jp);
                        if (jp.pos < jp.len && jp.src[jp.pos] == ',') jp.pos++;
                    }
                    if (jp.pos < jp.len && jp.src[jp.pos] == ']') jp.pos++;
                }
            } else if (strcmp(key, "data_offsets") == 0) {
                json_skip_ws(&jp);
                if (jp.pos < jp.len && jp.src[jp.pos] == ':') jp.pos++;
                json_skip_ws(&jp);
                if (jp.pos < jp.len && jp.src[jp.pos] == '[') {
                    jp.pos++;
                    json_skip_ws(&jp);
                    /* Parse first int */
                    json_skip_ws(&jp);
                    int64_t val = 0; int neg = 0;
                    if (jp.pos < jp.len && jp.src[jp.pos] == '-') { neg = 1; jp.pos++; }
                    while (jp.pos < jp.len && isdigit((unsigned char)jp.src[jp.pos])) {
                        val = val * 10 + (jp.src[jp.pos] - '0'); jp.pos++;
                    }
                    if (neg) val = -val;
                    data_begin = val;
                    json_skip_ws(&jp);
                    if (jp.pos < jp.len && jp.src[jp.pos] == ',') jp.pos++;
                    json_skip_ws(&jp);
                    /* Parse second int */
                    val = 0; neg = 0;
                    if (jp.pos < jp.len && jp.src[jp.pos] == '-') { neg = 1; jp.pos++; }
                    while (jp.pos < jp.len && isdigit((unsigned char)jp.src[jp.pos])) {
                        val = val * 10 + (jp.src[jp.pos] - '0'); jp.pos++;
                    }
                    if (neg) val = -val;
                    data_end = val;
                    json_skip_ws(&jp);
                    if (jp.pos < jp.len && jp.src[jp.pos] == ']') jp.pos++;
                }
            } else {
                /* Skip unknown key's value */
                json_skip_value(&jp);
            }

            json_skip_ws(&jp);
            if (jp.pos < jp.len && jp.src[jp.pos] == ',') jp.pos++;
        }
        if (jp.pos < jp.len && jp.src[jp.pos] == '}') jp.pos++;

        /* Store tensor info */
        if (st->n_tensors < SAFETENSORS_MAX_TENSORS) {
            safetensors_tensor_t *t = &st->tensors[st->n_tensors];
            strncpy(t->name, tensor_name, sizeof(t->name) - 1);
            strncpy(t->dtype_str, dtype, sizeof(t->dtype_str) - 1);
            t->dtype = safetensors_dtype_to_hlir(dtype);
            t->dtype_size = safetensors_dtype_size(dtype);
            t->n_dims = n_dims;
            for (int i = 0; i < n_dims; i++) t->shape[i] = shape[i];
            t->data_offset = (size_t)data_begin;
            t->data_size = (size_t)(data_end - data_begin);
            t->data = data + 8 + (size_t)header_size + (size_t)data_begin;
            st->n_tensors++;
        }

next:
        json_skip_ws(&jp);
        if (jp.pos < jp.len && jp.src[jp.pos] == ',') jp.pos++;
    }

    st->file_data = data;
    st->file_size = fsize;
    st->header_json = json;
    st->header_size = (size_t)header_size;
    return 0;
}

void safetensors_free(safetensors_t *st) {
    if (st->file_data) free(st->file_data);
    if (st->header_json) free(st->header_json);
    st->file_data = NULL;
    st->header_json = NULL;
    st->n_tensors = 0;
}

const safetensors_tensor_t *safetensors_find(const safetensors_t *st, const char *name) {
    for (int i = 0; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
    return NULL;
}

void safetensors_print(const safetensors_t *st) {
    printf("SafeTensors: %s (%d tensors)\n", st->filename, st->n_tensors);
    for (int i = 0; i < st->n_tensors; i++) {
        const safetensors_tensor_t *t = &st->tensors[i];
        printf("  %s: dtype=%s shape=[", t->name, t->dtype_str);
        for (int d = 0; d < t->n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%lld", (long long)t->shape[d]);
        }
        printf("] offset=%zu size=%zu\n", t->data_offset, t->data_size);
    }
}
