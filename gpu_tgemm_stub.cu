/*
 * gpu_tgemm_stub.cu — CUDA host stub for data-independent T_GEMM kernel.
 *
 * Usage: gpu_tgemm_stub <A_bin> <B_bin> <C_bin> <M> <N> <K>
 *
 * Reads A and B matrices from binary files, allocates device memory,
 * launches the data-independent T_GEMM kernel, writes C back to file.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o /tmp/gpu_tgemm_stub gpu_tgemm_stub.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        CUresult err = (call);                                         \
        if (err != CUDA_SUCCESS) {                                     \
            const char *err_str;                                       \
            cuGetErrorString(err, &err_str);                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, err_str);                      \
            exit(2);                                                   \
        }                                                              \
    } while (0)

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <A_bin> <B_bin> <C_bin> <M> <N> <K>\n", argv[0]);
        return 1;
    }

    const char *path_a = argv[1];
    const char *path_b = argv[2];
    const char *path_c = argv[3];
    int M = atoi(argv[4]);
    int N = atoi(argv[5]);
    int K = atoi(argv[6]);

    size_t sz_A = (size_t)M * K * 8;
    size_t sz_B = (size_t)K * N * 8;
    size_t sz_C = (size_t)M * N * 8;

    /* Read A and B from files */
    int64_t *h_A = (int64_t *)malloc(sz_A);
    int64_t *h_B = (int64_t *)malloc(sz_B);
    int64_t *h_C = (int64_t *)malloc(sz_C);
    if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE *f = fopen(path_a, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path_a); return 1; }
    fread(h_A, 1, sz_A, f);
    fclose(f);

    f = fopen(path_b, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path_b); return 1; }
    fread(h_B, 1, sz_B, f);
    fclose(f);

    /* Init CUDA */
    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    /* Load cubin */
    f = fopen("/tmp/wubu_tgemm_kernel.cubin", "rb");
    if (!f) { fprintf(stderr, "cannot open cubin\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long cubin_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *cubin_data = (uint8_t *)malloc((size_t)cubin_size);
    fread(cubin_data, 1, (size_t)cubin_size, f);
    fclose(f);

    CUmodule module;
    CUDA_CHECK(cuModuleLoadData(&module, cubin_data));
    free(cubin_data);

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "wubu_tgemm"));

    /* Allocate device memory */
    CUdeviceptr d_A, d_B, d_C;
    CUDA_CHECK(cuMemAlloc(&d_A, sz_A));
    CUDA_CHECK(cuMemAlloc(&d_B, sz_B));
    CUDA_CHECK(cuMemAlloc(&d_C, sz_C));

    /* Copy A and B to device */
    CUDA_CHECK(cuMemcpyHtoD(d_A, h_A, sz_A));
    CUDA_CHECK(cuMemcpyHtoD(d_B, h_B, sz_B));

    /* Launch kernel: grid-stride over M*N cells */
    int total_cells = M * N;
    int block_size = 256;
    int grid_size = (total_cells + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;  /* max grid dim */

    /* Kernel params: A, B, C, M, N, K */
    void *params[7];
    params[0] = (void *)&d_A;
    params[1] = (void *)&d_B;
    params[2] = (void *)&d_C;
    params[3] = (void *)&M;
    params[4] = (void *)&N;
    params[5] = (void *)&K;

    /* For grid-stride, we need blockDim.x as a parameter */
    params[6] = (void *)&block_size;

    CUDA_CHECK(cuLaunchKernel(kernel,
                              grid_size, 1, 1,       /* grid dim */
                              block_size, 1, 1,      /* block dim */
                              0, NULL,               /* shared mem, stream */
                              params, NULL));        /* kernel args, extra */

    CUDA_CHECK(cuCtxSynchronize());

    /* Copy C back */
    CUDA_CHECK(cuMemcpyDtoH(h_C, d_C, sz_C));

    /* Write C to file */
    f = fopen(path_c, "wb");
    if (!f) { fprintf(stderr, "cannot open %s for writing\n", path_c); return 1; }
    fwrite(h_C, 1, sz_C, f);
    fclose(f);

    /* Cleanup */
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
