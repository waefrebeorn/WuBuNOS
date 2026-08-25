/*
 * gpu_host_stub.cu -- CUDA host-side launcher for wubu_kernel cubin.
 *
 * Usage: gpu_host_stub <cubin_path> <int64_arg>
 *
 * Loads the cubin, gets the wubu_kernel function, allocates device
 * memory for the result, launches the kernel (1 thread, 1 block),
 * copies the result back, prints it to stdout as a decimal int64.
 *
 * This is the companion to wubu_isa_ptx.c. The PTX driver calls
 * ptxas to produce a cubin, then invokes THIS binary to actually
 * run it on the GPU. Keeping the CUDA runtime/host code in a separate
 * .cu file (compiled by nvcc once) means the driver itself stays
 * plain C11 and doesn't need the CUDA toolkit headers.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o /tmp/gpu_host_stub gpu_host_stub.cu
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

/* Hardware probe: query the REAL device so the compiler can generate
 * hardware-aware code. Prints machine-readable key=value lines on stdout.
 * This is how wubu_isa_ptx.c learns SM count, block limits, shared memory,
 * and warp size — no hardcoding anywhere. */
static int probe_device(void)
{
    CUDA_CHECK(cuInit(0));
    CUdevice dev;
    CUDA_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUDA_CHECK(cuCtxCreate(&ctx, 0, dev));

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    int cc_major = 0, cc_minor = 0;
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int sm_count   = 0;
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    int max_thds   = 0;
    cuDeviceGetAttribute(&max_thds, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    int shmem_blk  = 0;
    cuDeviceGetAttribute(&shmem_blk, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    int warp_size  = 0;
    cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    int regs_blk   = 0;
    cuDeviceGetAttribute(&regs_blk, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
    int l2_size    = 0;
    cuDeviceGetAttribute(&l2_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
    size_t total_mem = 0;
    cuDeviceTotalMem(&total_mem, dev);

    printf("name=%s\n", name);
    printf("cc=%d.%d\n", cc_major, cc_minor);
    printf("sm_count=%d\n", sm_count);
    printf("max_threads_per_block=%d\n", max_thds);
    printf("shared_mem_per_block=%d\n", shmem_blk);
    printf("warp_size=%d\n", warp_size);
    printf("registers_per_block=%d\n", regs_blk);
    printf("l2_cache_bytes=%d\n", l2_size);
    printf("total_mem_bytes=%zu\n", total_mem);

    cuCtxDestroy(ctx);
    return 0;
}

int main(int argc, char **argv)
{
    /* Probe mode: dump hardware profile and exit. */
    if (argc == 2 && strcmp(argv[1], "--probe") == 0)
        return probe_device();

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cubin_path> <int64_arg>\n"
                        "       %s --probe\n", argv[0], argv[0]);
        return 1;
    }

    const char *cubin_path = argv[1];
    int64_t arg = strtoll(argv[2], NULL, 10);

    /* Read cubin file */
    FILE *f = fopen(cubin_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", cubin_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long cubin_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *cubin_data = (uint8_t *)malloc((size_t)cubin_size);
    if (!cubin_data) { fclose(f); return 1; }
    fread(cubin_data, 1, (size_t)cubin_size, f);
    fclose(f);

    /* Init CUDA driver */
    CUDA_CHECK(cuInit(0));

    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));

    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    /* Load cubin module */
    CUmodule module;
    CUDA_CHECK(cuModuleLoadData(&module, cubin_data));
    free(cubin_data);

    /* Get kernel function */
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "wubu_kernel"));

    /* Allocate device memory for result (8 bytes = one int64) */
    CUdeviceptr d_result;
    CUDA_CHECK(cuMemAlloc(&d_result, sizeof(int64_t)));

    /* Set up kernel parameters: wubu_kernel(&result, arg)
     * CUDA driver API passes params as an array of void* pointing
     * to each parameter value. */
    int64_t h_result = 0;
    void *params[2];
    params[0] = (void *)&d_result;   /* pointer to device memory */
    params[1] = (void *)&arg;         /* the scalar argument */

    /* Launch: ONE block of max threads. bar.sync semantics require the
     * cooperating T_GEMM threads to share a block, so parallelism is
     * bounded by max_threads_per_block (queried from the device:
     * 1024 on this GPU). Single-block also keeps the redundant scalar
     * execution coherent across the whole launch. */
    int block_x = 1024;
    int grid_x = 1;
    CUDA_CHECK(cuLaunchKernel(kernel,
                              grid_x, 1, 1,       /* grid dim */
                              block_x, 1, 1,      /* block dim */
                              0, NULL,            /* shared mem, stream */
                              params, NULL));     /* kernel args, extra */

    CUDA_CHECK(cuCtxSynchronize());

    /* Copy result back */
    CUDA_CHECK(cuMemcpyDtoH(&h_result, d_result, sizeof(int64_t)));

    /* Print result to stdout (the driver reads this) */
    printf("%lld\n", (long long)h_result);

    /* Cleanup */
    cuMemFree(d_result);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}