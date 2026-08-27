/*
 * gpu_tgemm_fast.cu — Shared-memory tiled T_GEMM as a shared library.
 * Compiled to .so, loaded via dlopen for in-process GPU calls.
 *
 * Compile: nvcc -arch=sm_89 -O2 -shared -Xcompiler -fPIC -o /tmp/libgpu_tgemm.so gpu_tgemm_fast.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define TILE 32

__global__ void tgemm_smem_kernel(const int64_t *A, const int64_t *B, int64_t *C,
                                   int M, int N, int K) {
    __shared__ int64_t As[TILE][TILE];
    __shared__ int64_t Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    int64_t acc = 0;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        if (row < M && k0 + tx < K)
            As[ty][tx] = A[row * K + k0 + tx];
        else
            As[ty][tx] = 0;

        if (k0 + ty < K && col < N)
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; t++)
            acc += As[ty][t] * Bs[t][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

/* Persistent state */
static int64_t *d_A = NULL, *d_B = NULL, *d_C = NULL;
static size_t d_A_sz = 0, d_B_sz = 0, d_C_sz = 0;
static int initialized = 0;

extern "C" int gpu_tgemm_init(void) {
    if (initialized) return 0;
    cudaError_t err = cudaFree(0);  /* Initialize CUDA context */
    if (err != cudaSuccess) return -1;
    initialized = 1;
    return 0;
}

extern "C" int gpu_tgemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K) {
    if (!initialized) gpu_tgemm_init();

    size_t sz_A = (size_t)M * K * 8;
    size_t sz_B = (size_t)K * N * 8;
    size_t sz_C = (size_t)M * N * 8;

    /* Reuse or reallocate device memory */
    if (d_A_sz < sz_A) {
        if (d_A) cudaFree(d_A);
        cudaMalloc((void **)&d_A, sz_A);
        d_A_sz = sz_A;
    }
    if (d_B_sz < sz_B) {
        if (d_B) cudaFree(d_B);
        cudaMalloc((void **)&d_B, sz_B);
        d_B_sz = sz_B;
    }
    if (d_C_sz < sz_C) {
        if (d_C) cudaFree(d_C);
        cudaMalloc((void **)&d_C, sz_C);
        d_C_sz = sz_C;
    }

    cudaMemcpy(d_A, A, sz_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sz_B, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    tgemm_smem_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sz_C, cudaMemcpyDeviceToHost);

    return 0;
}

extern "C" void gpu_tgemm_free(void) {
    if (d_A) { cudaFree(d_A); d_A = NULL; d_A_sz = 0; }
    if (d_B) { cudaFree(d_B); d_B = NULL; d_B_sz = 0; }
    if (d_C) { cudaFree(d_C); d_C = NULL; d_C_sz = 0; }
    initialized = 0;
}
