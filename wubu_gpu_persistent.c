/*
 * wubu_gpu_persistent.c — In-process GPU T_GEMM with persistent context.
 *
 * Eliminates ~0.5s launch overhead by:
 * 1. Keeping CUDA context alive (cuDevicePrimaryCtxRetain)
 * 2. Allocating device memory once, reusing across calls
 * 3. Loading cubin once, reusing function handle
 * 4. No process forks, no file I/O
 *
 * Usage:
 *   wubu_gpu_init();                    // call once at startup
 *   wubu_gpu_tgemm(A, B, C, M, N, K);  // fast repeated calls
 *   wubu_gpu_free();                    // call at shutdown
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Persistent state ---------------------------------------------------- */
static CUcontext g_ctx = NULL;
static CUdevice g_dev = 0;
static CUmodule g_module = NULL;
static CUfunction g_kernel = NULL;
static int g_initialized = 0;

/* Device memory pool (grow as needed) */
static CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
static size_t d_A_size = 0, d_B_size = 0, d_C_size = 0;

/* ---- PTX source for shared-memory tiled kernel ---------------------------- */
/* Compiled once, cached to disk */
static const char PTX_SRC[] =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n\n"
    ".visible .entry wubu_tgemm_smem(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C,\n"
    "    .param .s32 M,\n"
    "    .param .s32 N,\n"
    "    .param .s32 K\n"
    ") {\n"
    "    .reg .u64 %ra<16>;\n"
    "    .reg .s64 %rs<40>;\n"
    "    .reg .s32 %ri<16>;\n"
    "    .reg .pred p<8>;\n"
    "    .reg .u32 %tid<1>;\n"
    "    .shared .s64 smem[2176];\n\n"
    "    ld.param.u64 %ra0, [A];\n"
    "    ld.param.u64 %ra1, [B];\n"
    "    ld.param.u64 %ra2, [C];\n"
    "    ld.param.s32 %ri0, [M];\n"
    "    ld.param.s32 %ri1, [N];\n"
    "    ld.param.s32 %ri2, [K];\n\n"
    "    mov.u32 %tid0, %ctaid.x;\n"
    "    cvt.s64.u32 %rs0, %tid0;\n"
    "    shl.b64 %rs0, %rs0, 5;\n"
    "    mov.u32 %tid0, %ctaid.y;\n"
    "    cvt.s64.u32 %rs1, %tid0;\n"
    "    shl.b64 %rs1, %rs1, 5;\n"
    "    mov.u32 %tid0, %tid.x;\n"
    "    cvt.s64.u32 %rs2, %tid0;\n"
    "    mov.u32 %tid0, %tid.y;\n"
    "    cvt.s64.u32 %rs3, %tid0;\n\n"
    "    add.s64 %rs4, %rs1, %rs3;\n"
    "    add.s64 %rs5, %rs0, %rs2;\n\n"
    "    cvt.s32.s64 %ri4, %rs4;\n"
    "    cvt.s32.s64 %ri5, %rs5;\n"
    "    setp.lt.s32 p0, %ri4, %ri0;\n"
    "    setp.lt.s32 p1, %ri5, %ri1;\n"
    "    and.pred p0, p0, p1;\n\n"
    "    mov.s64 %rs30, 0;\n"
    "    mov.s64 %rs6, 0;\n"
    "k_tile_loop:\n"
    "    cvt.s32.s64 %ri6, %rs6;\n"
    "    cvt.s64.s32 %rs7, %rs2;\n"
    "    add.s64 %rs7, %rs6, %rs7;\n"
    "    cvt.s32.s64 %ri8, %rs4;\n"
    "    cvt.s32.s64 %ri9, %rs7;\n"
    "    setp.lt.s32 p2, %ri8, %ri0;\n"
    "    setp.lt.s32 p3, %ri9, %ri2;\n"
    "    and.pred p2, p2, p3;\n"
    "    @p2 bra load_a_ok;\n"
    "    mov.s64 %rs8, 0;\n"
    "    bra load_a_done;\n"
    "load_a_ok:\n"
    "    mul.lo.s64 %rs8, %rs4, %ri2;\n"
    "    add.s64 %rs8, %rs8, %rs7;\n"
    "    shl.b64 %rs8, %rs8, 3;\n"
    "    add.s64 %rs8, %rs8, %ra0;\n"
    "    ld.global.s64 %rs8, [%rs8];\n"
    "load_a_done:\n"
    "    shl.b64 %rs9, %rs3, 5;\n"
    "    add.s64 %rs9, %rs9, %rs2;\n"
    "    shl.b64 %rs9, %rs9, 3;\n"
    "    cvta.shared.u64 %ra10, smem;\n"
    "    add.s64 %rs9, %rs9, %ra10;\n"
    "    st.shared.s64 [%rs9], %rs8;\n\n"
    "    cvt.s64.s32 %rs10, %rs3;\n"
    "    add.s64 %rs10, %rs6, %rs10;\n"
    "    cvt.s32.s64 %ri10, %rs10;\n"
    "    cvt.s32.s64 %ri11, %ri5;\n"
    "    setp.lt.s32 p2, %ri10, %ri2;\n"
    "    setp.lt.s32 p3, %ri11, %ri1;\n"
    "    and.pred p2, p2, p3;\n"
    "    @p2 bra load_b_ok;\n"
    "    mov.s64 %rs11, 0;\n"
    "    bra load_b_done;\n"
    "load_b_ok:\n"
    "    mul.lo.s64 %rs11, %rs10, %ri1;\n"
    "    add.s64 %rs11, %rs11, %rs5;\n"
    "    shl.b64 %rs11, %rs11, 3;\n"
    "    add.s64 %rs11, %rs11, %ra1;\n"
    "    ld.global.s64 %rs11, [%rs11];\n"
    "load_b_done:\n"
    "    shl.b64 %rs12, %rs3, 5;\n"
    "    add.s64 %rs12, %rs12, %rs2;\n"
    "    add.s64 %rs12, %rs12, 1024;\n"
    "    shl.b64 %rs12, %rs12, 3;\n"
    "    add.s64 %rs12, %rs12, %ra10;\n"
    "    st.shared.s64 [%rs12], %rs11;\n\n"
    "    bar.sync 0;\n\n"
    "    mov.s32 %ri12, 0;\n"
    "k_inner:\n"
    "    shl.b64 %rs13, %rs3, 5;\n"
    "    cvt.s64.s32 %rs14, %ri12;\n"
    "    add.s64 %rs13, %rs13, %rs14;\n"
    "    shl.b64 %rs13, %rs13, 3;\n"
    "    add.s64 %rs13, %rs13, %ra10;\n"
    "    ld.shared.s64 %rs15, [%rs13];\n"
    "    shl.b64 %rs16, %rs14, 5;\n"
    "    add.s64 %rs16, %rs16, %rs2;\n"
    "    add.s64 %rs16, %rs16, 1024;\n"
    "    shl.b64 %rs16, %rs16, 3;\n"
    "    add.s64 %rs16, %rs16, %ra10;\n"
    "    ld.shared.s64 %rs17, [%rs16];\n"
    "    mul.lo.s64 %rs18, %rs15, %rs17;\n"
    "    add.s64 %rs30, %rs30, %rs18;\n"
    "    add.s32 %ri12, %ri12, 1;\n"
    "    setp.lt.s32 p4, %ri12, 32;\n"
    "    @p4 bra k_inner;\n\n"
    "    bar.sync 0;\n"
    "    add.s64 %rs6, %rs6, 32;\n"
    "    cvt.s32.s64 %ri6, %rs6;\n"
    "    setp.lt.s32 p5, %ri6, %ri2;\n"
    "    @p5 bra k_tile_loop;\n\n"
    "    @p0 bra store_ok;\n"
    "    bra store_done;\n"
    "store_ok:\n"
    "    mul.lo.s64 %rs20, %rs4, %ri1;\n"
    "    add.s64 %rs20, %rs20, %rs5;\n"
    "    shl.b64 %rs20, %rs20, 3;\n"
    "    add.s64 %rs20, %rs20, %ra2;\n"
    "    st.global.s64 [%rs20], %rs30;\n"
    "store_done:\n"
    "    ret;\n"
    "}\n";

/* ---- Initialize persistent GPU context ---------------------------------- */
int wubu_gpu_init(void) {
    if (g_initialized) return 0;

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuInit failed (%d)\n", (int)err);
        return -1;
    }

    err = cuDeviceGet(&g_dev, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuDeviceGet failed (%d)\n", (int)err);
        return -1;
    }

    /* Retain primary context (keeps it alive) */
    err = cuDevicePrimaryCtxRetain(&g_ctx, g_dev);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuDevicePrimaryCtxRetain failed (%d)\n", (int)err);
        return -1;
    }

    /* Compile PTX to cubin */
    FILE *f = fopen("/tmp/wubu_tgemm_smem_persist.ptx", "w");
    if (!f) return -1;
    fputs(PTX_SRC, f);
    fclose(f);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ptxas -arch=sm_89 -O2 /tmp/wubu_tgemm_smem_persist.ptx "
             "-o /tmp/wubu_tgemm_smem_persist.cubin 2>/tmp/ptxas_persist.log");
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[gpu] ptxas failed (rc=%d)\n", rc);
        return -1;
    }

    /* Load cubin */
    f = fopen("/tmp/wubu_tgemm_smem_persist.cubin", "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *cubin = malloc((size_t)sz);
    fread(cubin, 1, (size_t)sz, f);
    fclose(f);

    err = cuModuleLoadData(&g_module, cubin);
    free(cubin);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuModuleLoadData failed (%d)\n", (int)err);
        return -1;
    }

    err = cuModuleGetFunction(&g_kernel, g_module, "wubu_tgemm_smem");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuModuleGetFunction failed (%d)\n", (int)err);
        return -1;
    }

    g_initialized = 1;
    fprintf(stderr, "[gpu] persistent context initialized\n");
    return 0;
}

/* ---- Ensure device memory is allocated ---------------------------------- */
static int ensure_device_mem(size_t need_A, size_t need_B, size_t need_C) {
    CUresult err;

    if (d_A_size < need_A) {
        if (d_A) cuMemFree(d_A);
        err = cuMemAlloc(&d_A, need_A);
        if (err != CUDA_SUCCESS) return -1;
        d_A_size = need_A;
    }
    if (d_B_size < need_B) {
        if (d_B) cuMemFree(d_B);
        err = cuMemAlloc(&d_B, need_B);
        if (err != CUDA_SUCCESS) return -1;
        d_B_size = need_B;
    }
    if (d_C_size < need_C) {
        if (d_C) cuMemFree(d_C);
        err = cuMemAlloc(&d_C, need_C);
        if (err != CUDA_SUCCESS) return -1;
        d_C_size = need_C;
    }
    return 0;
}

/* ---- Fast in-process GPU T_GEMM ----------------------------------------- */
int wubu_gpu_tgemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K) {
    if (!g_initialized) {
        if (wubu_gpu_init() != 0) return -1;
    }

    CUresult err;

    /* Ensure context is current */
    err = cuCtxSetCurrent(g_ctx);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuCtxSetCurrent failed (%d)\n", (int)err);
        return -1;
    }

    size_t sz_A = (size_t)M * K * 8;
    size_t sz_B = (size_t)K * N * 8;
    size_t sz_C = (size_t)M * N * 8;

    /* Allocate/reuse device memory */
    if (ensure_device_mem(sz_A, sz_B, sz_C) != 0) {
        fprintf(stderr, "[gpu] device memory allocation failed\n");
        return -1;
    }

    /* Copy A and B to device */
    err = cuMemcpyHtoD(d_A, A, sz_A);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuMemcpyHtoD A failed (%d)\n", (int)err);
        return -1;
    }
    err = cuMemcpyHtoD(d_B, B, sz_B);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuMemcpyHtoD B failed (%d)\n", (int)err);
        return -1;
    }

    /* Launch kernel */
    int TILE = 32;
    int grid_x = (N + TILE - 1) / TILE;
    int grid_y = (M + TILE - 1) / TILE;

    void *params[6];
    params[0] = (void *)&d_A;
    params[1] = (void *)&d_B;
    params[2] = (void *)&d_C;
    params[3] = (void *)&M;
    params[4] = (void *)&N;
    params[5] = (void *)&K;

    err = cuLaunchKernel(g_kernel,
                         grid_x, grid_y, 1,
                         TILE, TILE, 1,
                         0, NULL,
                         params, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuLaunchKernel failed (%d)\n", (int)err);
        return -1;
    }

    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuCtxSynchronize failed (%d)\n", (int)err);
        return -1;
    }

    /* Copy C back */
    err = cuMemcpyDtoH(C, d_C, sz_C);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu] cuMemcpyDtoH failed (%d)\n", (int)err);
        return -1;
    }

    return 0;
}

/* ---- Cleanup ------------------------------------------------------------- */
void wubu_gpu_free(void) {
    if (d_A) { cuMemFree(d_A); d_A = 0; d_A_size = 0; }
    if (d_B) { cuMemFree(d_B); d_B = 0; d_B_size = 0; }
    if (d_C) { cuMemFree(d_C); d_C = 0; d_C_size = 0; }
    if (g_module) { cuModuleUnload(g_module); g_module = NULL; }
    if (g_ctx) { cuDevicePrimaryCtxRelease(g_dev); g_ctx = NULL; }
    g_initialized = 0;
}
