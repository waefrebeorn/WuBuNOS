#include <stdio.h>
#include <string.h>
#include "mlir_parser.h"

/* Build a minimal MLIR bytecode file for testing.
 * Format: magic "MLïR", version, producer, sections.
 * We create a simple file with a string section and IR section. */

static uint8_t test_file[4096];
static size_t test_pos;

static void emit_byte(uint8_t b) { test_file[test_pos++] = b; }
static void emit_bytes(const uint8_t *d, size_t n) { memcpy(test_file + test_pos, d, n); test_pos += n; }

static void emit_varint(uint64_t v) {
    /* PrefixVarInt: count trailing zeros in first byte */
    if (v < (1ULL << 7)) {
        emit_byte((uint8_t)(v << 1) | 1);
    } else if (v < (1ULL << 14)) {
        emit_byte((uint8_t)((v & 0x3F) << 2) | 2);
        emit_byte((uint8_t)((v >> 6) & 0xFF));
    } else if (v < (1ULL << 21)) {
        emit_byte((uint8_t)((v & 0x1F) << 3) | 4);
        emit_byte((uint8_t)((v >> 5) & 0xFF));
        emit_byte((uint8_t)((v >> 13) & 0xFF));
    } else {
        /* Generic fallback */
        uint8_t buf[9];
        int n = 0;
        uint64_t tmp = v;
        do {
            buf[n++] = (uint8_t)(tmp & 0x7F);
            tmp >>= 7;
        } while (tmp);
        /* Encode with prefix */
        int n_add = n - 1;
        uint8_t first = 0;
        for (int i = 0; i < n_add; i++) first |= (1 << (i + 1));
        first |= (buf[0] << (n_add + 1));
        emit_byte(first);
        for (int i = 1; i < n; i++) emit_byte(buf[i]);
    }
}

static void emit_string(const char *s) {
    size_t len = strlen(s);
    emit_varint(len);
    emit_bytes((const uint8_t *)s, len);
}

static void emit_section_header(uint8_t id, uint64_t length) {
    emit_byte(id); /* no alignment */
    emit_varint(length);
}

static size_t build_test_mlir(void) {
    test_pos = 0;

    /* Magic: "MLïR" = 0x52EF4C4D (LE) */
    emit_byte(0x4D); emit_byte(0x4C); emit_byte(0xEF); emit_byte(0x52);

    /* Version */
    emit_varint(1);

    /* Producer */
    emit_string("test");

    /* ── String section (id=1) ── */
    size_t sec_start = test_pos;
    emit_section_header(1, 0); /* placeholder length */
    size_t data_start = test_pos;

    /* 3 strings: "func", "arith.addf", "func.return" */
    emit_varint(3);
    /* String lengths in reverse order */
    emit_varint(10); /* "func.return" */
    emit_varint(10); /* "arith.addf" */
    emit_varint(4);  /* "func" */
    /* String data */
    emit_string("func");
    emit_string("arith.addf");
    emit_string("func.return");

    /* Patch section length */
    size_t sec_end = test_pos;
    size_t sec_len = sec_end - data_start;
    /* Go back and patch the length */
    test_pos = sec_start + 1; /* skip id byte */
    emit_varint(sec_len);
    test_pos = sec_end;

    /* ── IR section (id=3) ── */
    sec_start = test_pos;
    emit_section_header(3, 0);
    data_start = test_pos;

    /* 1 region, 1 block, 2 ops */
    emit_varint(1); /* n_regions */
    emit_varint(1); /* n_blocks */
    emit_varint(4); /* block_enc: (2 ops << 1) | 0 (no args) */

    /* Op 1: arith.addf (dialect=0, op_name=1) */
    emit_varint(0); /* dialect index */
    emit_varint(1); /* op_name index in string table */
    emit_varint(1); /* n_results */
    emit_varint(2); /* result type encoding (no location) */
    emit_varint(2); /* n_operands */
    emit_varint(0); /* use_order 0 */
    emit_varint(1); /* use_order 1 */
    emit_varint(0); /* n_attrs */
    emit_varint(0); /* n_regions */

    /* Op 2: func.return (dialect=0, op_name=2) */
    emit_varint(0);
    emit_varint(2);
    emit_varint(0); /* n_results */
    emit_varint(0); /* n_operands */
    emit_varint(0); /* n_attrs */
    emit_varint(0); /* n_regions */

    sec_end = test_pos;
    sec_len = sec_end - data_start;
    test_pos = sec_start + 1;
    emit_varint(sec_len);
    test_pos = sec_end;

    return test_pos;
}

int main(void) {
    int pass = 0, fail = 0;

    /* Test 1: Build and parse minimal MLIR bytecode */
    size_t fsize = build_test_mlir();
    FILE *f = fopen("/tmp/test.mlir", "wb");
    if (!f) { printf("FAIL: cannot create test file\n"); return 1; }
    fwrite(test_file, 1, fsize, f);
    fclose(f);

    mlir_model_t model;
    int rc = mlir_load_model("/tmp/test.mlir", &model);
    if (rc == 0 && model.parsed_ok) {
        printf("PASS: MLIR bytecode parsed (graph has %d ops)\n", model.graph.n);
        pass++;
    } else {
        printf("FAIL: MLIR parse failed: %s\n", model.error);
        fail++;
    }
    mlir_model_free(&model);

    /* Test 2: Bad magic */
    FILE *f2 = fopen("/tmp/test_bad.mlir", "wb");
    if (f2) {
        uint8_t bad[4] = {0x00, 0x00, 0x00, 0x00};
        fwrite(bad, 1, 4, f2);
        fclose(f2);
    }
    mlir_model_t model2;
    rc = mlir_load_model("/tmp/test_bad.mlir", &model2);
    if (rc != 0) {
        printf("PASS: bad magic correctly rejected (%s)\n", model2.error);
        pass++;
    } else {
        printf("FAIL: bad magic not rejected\n");
        fail++;
    }
    mlir_model_free(&model2);

    /* Test 3: Empty file */
    FILE *f3 = fopen("/tmp/test_empty.mlir", "wb");
    if (f3) { fclose(f3); }
    mlir_model_t model3;
    rc = mlir_load_model("/tmp/test_empty.mlir", &model3);
    if (rc != 0) {
        printf("PASS: empty file correctly rejected (%s)\n", model3.error);
        pass++;
    } else {
        printf("FAIL: empty file not rejected\n");
        fail++;
    }
    mlir_model_free(&model3);

    printf("\n=== MLIR Parser: %d PASS, %d FAIL ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
