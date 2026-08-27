/*
 * gpu_tgemm_smem.cu — Shared-memory tiled T_GEMM for RTX 4050 (cc8.9).
 * Written in CUDA C — compiled by nvcc directly, no PTX assembly needed.
 *
 * Usage: gpu_tgemm_smem <A_bin> <B_bin> <C_bin> <M> <N> <K>
 * Compile: nvcc -arch=sm_89 -O2 -o /tmp/gpu_tgemm_smem gpu_tgemm_smem.cu
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TILE 32

__global__ void tgemm_smem(const int64_t *A, const int64_t *B, int64_t *C,
                           int M, int N, int K) {
    __shared__ int64_t As[TILE][TILE];
    __shared__ int64_t Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    int64_t acc = 0;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        /* Load A tile */
        if (row < M && k0 + tx < K)
            As[ty][tx] = A[row * K + k0 + tx];
        else
            As[ty][tx] = 0;

        /* Load B tile */
        if (k0 + ty < K && col < N)
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        /* Partial dot product */
        for (int t = 0; t < TILE; t++)
            acc += As[ty][t] * Bs[t][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

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

    int64_t *h_A = (int64_t *)malloc(sz_A);
    int64_t *h_B = (int64_t *)malloc(sz_B);
    int64_t *h_C = (int64_t *)calloc(sz_C, 1);
    if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

    FILE *f = fopen(path_a, "rb");
    if (!f || fread(h_A, 1, sz_A, f) != sz_A) { fprintf(stderr, "read A failed\n"); return 1; }
    fclose(f);

    f = fopen(path_b, "rb");
    if (!f || fread(h_B, 1, sz_B, f) != sz_B) { fprintf(stderr, "read B failed\n"); return 1; }
    fclose(f);

    /* Allocate device memory */
    int64_t *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sz_A);
    cudaMalloc((void **)&d_B, sz_B);
    cudaMalloc((void **)&d_C, sz_C);

    cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_B, cudaMemcpyHostToDevice);

    /* Launch */
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    tgemm_smem<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sz_C, cudaMemcpyDeviceToHost);

    /* Write result */
    f = fopen(path_c, "wb");
    fwrite(h_C, 1, sz_C, f);
    fclose(f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
