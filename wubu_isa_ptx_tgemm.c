/*
 * wubu_isa_ptx_tgemm.c — Data-independent T_GEMM PTX kernel template.
 * ONE cubin works for ALL matrix sizes. Data via device memory pointers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>

typedef struct {
    char *text;
    size_t n;
    size_t cap;
} ptx_buf_t;

static void ptx_cat(ptx_buf_t *b, const char *fmt, ...) {
    /* We need printf-style formatting for some calls, but PTX register
     * names use % which conflicts. Solution: use %% in format strings
     * for literal %, and %s/%d for actual formatting. */
    va_list ap;
    va_start(ap, fmt);
    int len = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (len <= 0) return;
    if (b->n + (size_t)len + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->n + (size_t)len + 1) cap *= 2;
        b->text = realloc(b->text, cap);
        b->cap = cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->text + b->n, (size_t)len + 1, fmt, ap);
    va_end(ap);
    b->n += (size_t)len;
}

/* Append raw string (no printf interpretation) */
static void ptx_raw(ptx_buf_t *b, const char *s) {
    size_t len = strlen(s);
    if (b->n + len + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->n + len + 1) cap *= 2;
        b->text = realloc(b->text, cap);
        b->cap = cap;
    }
    memcpy(b->text + b->n, s, len);
    b->n += len;
}

/* Generate the data-independent T_GEMM PTX kernel. No comments — ptxas doesn't support them. */
static char *emit_tgemm_kernel(int M, int N, int K) {
    ptx_buf_t b = {0};

    ptx_raw(&b, ".version 8.0\n");
    ptx_raw(&b, ".target sm_89\n");
    ptx_raw(&b, ".address_size 64\n\n");

    ptx_raw(&b, ".visible .entry wubu_tgemm(\n");
    ptx_raw(&b, "    .param .u64 A,\n");
    ptx_raw(&b, "    .param .u64 B,\n");
    ptx_raw(&b, "    .param .u64 C,\n");
    ptx_raw(&b, "    .param .s32 M,\n");
    ptx_raw(&b, "    .param .s32 N,\n");
    ptx_raw(&b, "    .param .s32 K\n");
    ptx_raw(&b, ") {\n");

    ptx_raw(&b, "    .reg .u64 %ra<10>;\n");
    ptx_raw(&b, "    .reg .s64 %rs<20>;\n");
    ptx_raw(&b, "    .reg .s32 %ri<10>;\n");
    ptx_raw(&b, "    .reg .pred p0;\n");
    ptx_raw(&b, "    .reg .pred p1;\n");
    ptx_raw(&b, "    .reg .u32 %tid<1>;\n\n");

    /* Load parameters */
    ptx_raw(&b, "    ld.param.u64 %ra0, [A];\n");
    ptx_raw(&b, "    ld.param.u64 %ra1, [B];\n");
    ptx_raw(&b, "    ld.param.u64 %ra2, [C];\n");
    ptx_raw(&b, "    ld.param.s32 %ri0, [M];\n");
    ptx_raw(&b, "    ld.param.s32 %ri1, [N];\n");
    ptx_raw(&b, "    ld.param.s32 %ri2, [K];\n\n");

    /* Get thread ID */
    ptx_raw(&b, "    mov.u32 %tid0, %tid.x;\n");
    ptx_raw(&b, "    cvt.s64.u32 %rs0, %tid0;\n\n");

    /* Total cells = M * N */
    ptx_raw(&b, "    mul.lo.s32 %ri3, %ri0, %ri1;\n");
    ptx_raw(&b, "    cvt.s64.s32 %rs1, %ri3;\n\n");

    /* Grid-stride loop */
    ptx_raw(&b, "    bra cell_test;\n");
    ptx_raw(&b, "cell_body:\n");

    /* Decode cell -> (i, j) */
    ptx_raw(&b, "    cvt.s32.s64 %ri4, %rs0;\n");
    ptx_raw(&b, "    div.s32 %ri5, %ri4, %ri1;\n");
    ptx_raw(&b, "    rem.s32 %ri6, %ri4, %ri1;\n\n");

    /* A row offset = i * K * 8 */
    ptx_raw(&b, "    mul.lo.s32 %ri7, %ri5, %ri2;\n");
    ptx_raw(&b, "    cvt.s64.s32 %rs2, %ri7;\n");
    ptx_raw(&b, "    shl.b64 %rs2, %rs2, 3;\n");
    ptx_raw(&b, "    add.s64 %rs2, %rs2, %ra0;\n\n");

    /* B col offset = j * 8 */
    ptx_raw(&b, "    cvt.s64.s32 %rs3, %ri6;\n");
    ptx_raw(&b, "    shl.b64 %rs3, %rs3, 3;\n");
    ptx_raw(&b, "    add.s64 %rs3, %rs3, %ra1;\n\n");

    /* C offset = (i * N + j) * 8 */
    ptx_raw(&b, "    mul.lo.s32 %ri7, %ri5, %ri1;\n");
    ptx_raw(&b, "    add.s32 %ri7, %ri7, %ri6;\n");
    ptx_raw(&b, "    cvt.s64.s32 %rs4, %ri7;\n");
    ptx_raw(&b, "    shl.b64 %rs4, %rs4, 3;\n");
    ptx_raw(&b, "    add.s64 %rs4, %rs4, %ra2;\n\n");

    /* K-loop */
    ptx_raw(&b, "    mov.s64 %rs5, 0;\n");
    ptx_raw(&b, "    mov.s32 %ri8, 0;\n");
    ptx_raw(&b, "    bra k_test;\n");
    ptx_raw(&b, "k_body:\n");

    /* A[i][k] */
    ptx_raw(&b, "    cvt.s64.s32 %rs6, %ri8;\n");
    ptx_raw(&b, "    shl.b64 %rs6, %rs6, 3;\n");
    ptx_raw(&b, "    add.s64 %rs7, %rs2, %rs6;\n");
    ptx_raw(&b, "    ld.global.s64 %rs7, [%rs7];\n\n");

    /* B[k][j] */
    ptx_raw(&b, "    cvt.s64.s32 %rs8, %ri8;\n");
    ptx_raw(&b, "    cvt.s64.s32 %rs9, %ri1;\n");
    ptx_raw(&b, "    mul.lo.s64 %rs8, %rs8, %rs9;\n");
    ptx_raw(&b, "    shl.b64 %rs8, %rs8, 3;\n");
    ptx_raw(&b, "    add.s64 %rs8, %rs3, %rs8;\n");
    ptx_raw(&b, "    ld.global.s64 %rs8, [%rs8];\n\n");

    /* acc += a * b */
    ptx_raw(&b, "    mul.lo.s64 %rs7, %rs7, %rs8;\n");
    ptx_raw(&b, "    add.s64 %rs5, %rs5, %rs7;\n\n");

    /* k++ */
    ptx_raw(&b, "    add.s32 %ri8, %ri8, 1;\n");
    ptx_raw(&b, "k_test:\n");
    ptx_raw(&b, "    setp.lt.s32 p0, %ri8, %ri2;\n");
    ptx_raw(&b, "    @p0 bra k_body;\n\n");

    /* Store C[i][j] = acc */
    ptx_raw(&b, "    st.global.s64 [%rs4], %rs5;\n\n");

    /* Grid-stride: cell += blockDim.x */
    ptx_raw(&b, "    mov.u32 %tid0, %ntid.x;\n");
    ptx_raw(&b, "    cvt.s64.u32 %rs6, %tid0;\n");
    ptx_raw(&b, "    add.s64 %rs0, %rs0, %rs6;\n");
    ptx_raw(&b, "cell_test:\n");
    ptx_raw(&b, "    setp.lt.s64 p1, %rs0, %rs1;\n");
    ptx_raw(&b, "    @p1 bra cell_body;\n\n");

    ptx_raw(&b, "    ret;\n");
    ptx_raw(&b, "}\n");

    return b.text;
}

/* Compile PTX to cubin (cached) */
static int compile_tgemm_cubin(int M, int N, int K, uint8_t **out_code, size_t *out_size) {
    const char *cubin_path = "/tmp/wubu_tgemm.cubin";
    const char *ptx_path = "/tmp/wubu_tgemm.ptx";

    static int cached = 0;
    if (!cached) {
        char *ptx = emit_tgemm_kernel(M, N, K);
        if (!ptx) return -1;

        FILE *f = fopen(ptx_path, "w");
        if (!f) { free(ptx); return -1; }
        fputs(ptx, f);
        fclose(f);

        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "ptxas -arch=sm_89 -O2 %s -o %s 2>/tmp/ptxas_tgemm.log",
                 ptx_path, cubin_path);
        int rc = system(cmd);
        if (rc != 0) {
            fprintf(stderr, "[ptx_tgemm] ptxas failed (rc=%d)\n", rc);
            free(ptx);
            return -1;
        }
        cached = 1;
        free(ptx);
        fprintf(stderr, "[ptx_tgemm] compiled data-independent kernel (cached)\n");
    }

    FILE *cf = fopen(cubin_path, "rb");
    if (!cf) return -1;
    fseek(cf, 0, SEEK_END);
    long sz = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    uint8_t *cubin = malloc((size_t)sz);
    if (!cubin) { fclose(cf); return -1; }
    fread(cubin, 1, (size_t)sz, cf);
    fclose(cf);

    *out_code = cubin;
    *out_size = (size_t)sz;
    return 0;
}

/* Public API: data-independent T_GEMM on GPU */
int wubu_ptx_tgemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K) {
    uint8_t *cubin;
    size_t cubin_size;
    if (compile_tgemm_cubin(M, N, K, &cubin, &cubin_size) != 0)
        return -1;

    /* Write cubin for host stub */
    FILE *f = fopen("/tmp/wubu_tgemm_kernel.cubin", "wb");
    if (!f) { free(cubin); return -1; }
    fwrite(cubin, 1, cubin_size, f);
    fclose(f);

    /* Write matrices for host stub */
    char path_a[256], path_b[256], path_c[256];
    snprintf(path_a, sizeof(path_a), "/tmp/wubu_tgemm_A_%d_%d.bin", M, K);
    snprintf(path_b, sizeof(path_b), "/tmp/wubu_tgemm_B_%d_%d.bin", K, N);
    snprintf(path_c, sizeof(path_c), "/tmp/wubu_tgemm_C_%d_%d.bin", M, N);

    f = fopen(path_a, "wb");
    fwrite(A, 8, (size_t)(M * K), f);
    fclose(f);

    f = fopen(path_b, "wb");
    fwrite(B, 8, (size_t)(K * N), f);
    fclose(f);

    /* Launch host stub */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "LD_LIBRARY_PATH='/usr/lib/wsl/lib:$LD_LIBRARY_PATH' "
             "/tmp/gpu_tgemm_stub %s %s %s %d %d %d 2>/tmp/gpu_tgemm.log",
             path_a, path_b, path_c, M, N, K);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[ptx_tgemm] host stub failed (rc=%d)\n", rc);
        free(cubin);
        return -1;
    }

    /* Read result */
    f = fopen(path_c, "rb");
    if (!f) { free(cubin); return -1; }
    fread(C, 8, (size_t)(M * N), f);
    fclose(f);

    free(cubin);
    return 0;
}
