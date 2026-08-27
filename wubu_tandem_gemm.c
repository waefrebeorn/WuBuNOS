/*
 * wubu_tandem_gemm.c — Heterogeneous CPU+GPU T_GEMM with automatic load balancing.
 *
 * The AGI compiler's ultimate dispatch: uses ALL hardware simultaneously.
 * Profiles CPU and GPU throughput, computes optimal split, runs both in parallel.
 *
 * Algorithm:
 *   1. Profile: measure CPU GFLOPS and GPU GFLOPS on a small test case
 *   2. Split: assign rows to CPU and GPU proportional to their throughput
 *   3. Execute: launch CPU worker threads + GPU kernel simultaneously
 *   4. Merge: combine partial C matrices (no overlap, just concatenation)
 *
 * Split strategy:
 *   gpu_fraction = gpu_gflops / (cpu_gflops + gpu_gflops)
 *   gpu_rows = round(M * gpu_fraction)
 *   cpu_rows = M - gpu_rows
 *
 * The B matrix is shared (read-only), so both CPU and GPU use the same B.
 * A is split by rows: A_cpu = A[0..cpu_rows], A_gpu = A[gpu_rows..M]
 * C is split similarly: C_cpu = C[0..cpu_rows], C_gpu = C[gpu_rows..M]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

/* ---- External GPU API ---------------------------------------------------- */
extern int wubu_ptx_tgemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K);

/* ---- CPU T_GEMM (AVX-512 + cache blocking) ------------------------------ */
/* Declared in wubu_isa_x86_64.c — the parallel JIT T_GEMM */
extern void wubu_tgemm_parallel(int64_t *mem, int64_t A, int64_t B, int64_t C, int M, int N, int K);

/* ---- CPU T_GEMM using the same kernel as the JIT benchmark -------------- */
/* We use a direct call to the AVX-512 kernel via the benchmark's approach */

/* Forward declaration — we'll implement a simple CPU GEMM here */
static void cpu_tgemm(int64_t *C, int64_t *A, int64_t *B, int M, int N, int K) {
    /* Use OpenMP parallel k-j-i loop (same as the JIT kernel but in C) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            int64_t a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* ---- GPU T_GEMM via data-independent kernel ----------------------------- */
static int gpu_tgemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K) {
    /* Write matrices for GPU host stub */
    FILE *f = fopen("/tmp/tandem_A_gpu.bin", "wb");
    fwrite(A, 8, (size_t)M * K, f);
    fclose(f);
    f = fopen("/tmp/tandem_B_gpu.bin", "wb");
    fwrite(B, 8, (size_t)K * N, f);
    fclose(f);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "LD_LIBRARY_PATH='/usr/lib/wsl/lib:$LD_LIBRARY_PATH' "
             "/tmp/gpu_tgemm_smem /tmp/tandem_A_gpu.bin /tmp/tandem_B_gpu.bin /tmp/tandem_C_gpu.bin %d %d %d 2>/dev/null",
             M, N, K);
    int rc = system(cmd);

    if (rc != 0) return -1;

    /* Read result */
    f = fopen("/tmp/tandem_C_gpu.bin", "rb");
    if (!f) return -1;
    fread(C, 8, (size_t)M * N, f);
    fclose(f);
    return 0;
}

/* ---- Timing helper ------------------------------------------------------- */
static double nowsec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Hardware profiler --------------------------------------------------- */
typedef struct {
    double cpu_gflops;
    double gpu_gflops;
    int n_cpu_cores;
    int gpu_available;
} hw_profile_t;

static hw_profile_t g_profile = {0};
static int g_profiled = 0;

static void profile_hardware(void) {
    if (g_profiled) return;

    fprintf(stderr, "[tandem] profiling hardware...\n");

    /* Profile CPU: time a 256³ GEMM */
    int M = 256, N = 256, K = 256;
    int64_t *A = calloc((size_t)M * K, 8);
    int64_t *B = calloc((size_t)K * N, 8);
    int64_t *C = calloc((size_t)M * N, 8);
    for (int i = 0; i < M*K; i++) A[i] = (i % 17) - 8;
    for (int i = 0; i < K*N; i++) B[i] = (i % 13) - 6;

    /* Warm up */
    cpu_tgemm(C, A, B, M, N, K);

    /* Time it */
    double t0 = nowsec();
    int reps = 3;
    for (int r = 0; r < reps; r++) {
        memset(C, 0, (size_t)M * N * 8);
        cpu_tgemm(C, A, B, M, N, K);
    }
    double t_cpu = (nowsec() - t0) / reps;
    double cpu_gflops = (2.0 * M * N * K) / t_cpu / 1e9;

    /* Profile GPU: measure overhead vs compute separately */
    double gpu_gflops = 0;
    int gpu_avail = 0;

    /* Write matrices for GPU */
    FILE *f = fopen("/tmp/tandem_A.bin", "wb");
    fwrite(A, 8, (size_t)M * K, f);
    fclose(f);
    f = fopen("/tmp/tandem_B.bin", "wb");
    fwrite(B, 8, (size_t)K * N, f);
    fclose(f);

    /* Use shared-memory kernel */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "LD_LIBRARY_PATH='/usr/lib/wsl/lib:$LD_LIBRARY_PATH' "
             "/tmp/gpu_tgemm_smem /tmp/tandem_A.bin /tmp/tandem_B.bin /tmp/tandem_C.bin %d %d %d 2>/dev/null",
             M, N, K);

    /* Warm up */
    int gpu_rc = system(cmd);
    if (gpu_rc == 0) {
        /* Measure small (overhead-dominated) */
        t0 = nowsec();
        system(cmd);
        double t_small = nowsec() - t0;

        /* Measure large (compute-dominated) */
        int M2 = 512, N2 = 512, K2 = 512;
        int64_t *A2 = calloc((size_t)M2 * K2, 8);
        int64_t *B2 = calloc((size_t)K2 * N2, 8);
        for (int i = 0; i < M2*K2; i++) A2[i] = (i % 17) - 8;
        for (int i = 0; i < K2*N2; i++) B2[i] = (i % 13) - 6;

        f = fopen("/tmp/tandem_A.bin", "wb"); fwrite(A2, 8, (size_t)M2*K2, f); fclose(f);
        f = fopen("/tmp/tandem_B.bin", "wb"); fwrite(B2, 8, (size_t)K2*N2, f); fclose(f);

        char cmd2[1024];
        snprintf(cmd2, sizeof(cmd2),
                 "LD_LIBRARY_PATH='/usr/lib/wsl/lib:$LD_LIBRARY_PATH' "
                 "/tmp/gpu_tgemm_smem /tmp/tandem_A.bin /tmp/tandem_B.bin /tmp/tandem_C.bin %d %d %d 2>/dev/null",
                 M2, N2, K2);

        /* Warm up large */
        system(cmd2);
        t0 = nowsec();
        system(cmd2);
        double t_large = nowsec() - t0;

        /* GPU GFLOPS = (2*M2*N2*K2 - 2*M*N*K) / (t_large - t_small) */
        if (t_large > t_small) {
            double gpu_gflops_raw = (2.0 * M2 * N2 * K2 - 2.0 * M * N * K) / (t_large - t_small) / 1e9;
            gpu_gflops = gpu_gflops_raw;
            gpu_avail = 1;
        }

        free(A2);
        free(B2);
    }

    fprintf(stderr, "[tandem] CPU: %.1f GFLOPS (%d cores), GPU: %.1f GFLOPS (%s)\n",
            cpu_gflops, omp_get_max_threads(), gpu_gflops,
            gpu_avail ? "available" : "unavailable");

    g_profile.cpu_gflops = cpu_gflops;
    g_profile.gpu_gflops = gpu_gflops;
    g_profile.n_cpu_cores = omp_get_max_threads();
    g_profile.gpu_available = gpu_avail;
    g_profiled = 1;

    free(A);
    free(B);
    free(C);
}

/* ---- Tandem CPU+GPU GEMM ------------------------------------------------ */
int wubu_tandem_gemm(int64_t *A, int64_t *B, int64_t *C, int M, int N, int K) {
    profile_hardware();

    /* Compute split based on measured hardware and matrix size.
     * GPU has ~0.5s launch overhead, so only use it for large matrices.
     * Strategy: assign rows proportional to GPU/(CPU+GPU) for large matrices. */
    int gpu_rows = 0, cpu_rows = M;

    if (g_profile.gpu_available && g_profile.gpu_gflops > 0 && M >= 512) {
        /* For large matrices, split proportional to throughput */
        double total = g_profile.cpu_gflops + g_profile.gpu_gflops;
        double gpu_frac = g_profile.gpu_gflops / total;
        /* Clamp: GPU gets at least 25%, at most 75% */
        if (gpu_frac < 0.25) gpu_frac = 0.25;
        if (gpu_frac > 0.75) gpu_frac = 0.75;
        gpu_rows = (int)(M * gpu_frac);
        /* Round to multiple of 32 for GPU alignment */
        gpu_rows = (gpu_rows / 32) * 32;
        if (gpu_rows < 32) gpu_rows = 0;
        if (gpu_rows > M - 32) gpu_rows = M - 32;
        cpu_rows = M - gpu_rows;
    }

    fprintf(stderr, "[tandem] M=%d: CPU rows=%d, GPU rows=%d (%.0f%% GPU)\n",
            M, cpu_rows, gpu_rows, 100.0 * gpu_rows / M);

    /* Launch both simultaneously */
    double t0 = nowsec();

    if (gpu_rows > 0) {
        /* Launch GPU on rows [cpu_rows..M] in a thread */
        int gpu_done = 0;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                /* CPU computes rows [0..cpu_rows] */
                int64_t *C_cpu = C;
                int64_t *A_cpu = A;
                cpu_tgemm(C_cpu, A_cpu, B, cpu_rows, N, K);
            }
            #pragma omp section
            {
                /* GPU computes rows [cpu_rows..M] */
                int64_t *C_gpu = C + (int64_t)cpu_rows * N;
                int64_t *A_gpu = A + (int64_t)cpu_rows * K;
                gpu_tgemm(A_gpu, B, C_gpu, gpu_rows, N, K);
                gpu_done = 1;
            }
        }
    } else {
        /* CPU only */
        cpu_tgemm(C, A, B, M, N, K);
    }

    double dt = nowsec() - t0;
    double gflops = (2.0 * M * N * K) / dt / 1e9;
    fprintf(stderr, "[tandem] done in %.4fs (%.1f GFLOPS)\n", dt, gflops);

    return 0;
}

/* ---- Test harness -------------------------------------------------------- */
static void naive_gemm(int64_t *C, int64_t *A, int64_t *B, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int64_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

int main(int argc, char **argv) {
    int M = 256, N = 256, K = 256;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    int64_t *A = calloc((size_t)M*K, 8);
    int64_t *B = calloc((size_t)N*K, 8);
    int64_t *C = calloc((size_t)M*N, 8);
    int64_t *Cref = calloc((size_t)M*N, 8);

    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = (rand() % 17) - 8;
    for (int i = 0; i < K*N; i++) B[i] = (rand() % 13) - 6;

    /* Reference */
    naive_gemm(Cref, A, B, M, N, K);

    /* Tandem */
    int rc = wubu_tandem_gemm(A, B, C, M, N, K);

    /* Verify */
    int bad = 0;
    for (int i = 0; i < M*N; i++)
        if (C[i] != Cref[i]) { bad++; if (bad <= 5) printf("  C[%d]=%lld ref=%lld\n", i, (long long)C[i], (long long)Cref[i]); }

    printf("Tandem %dx%dx%d: %s (%d wrong)\n", M, N, K, bad ? "MISMATCH" : "OK", bad);

    free(A); free(B); free(C); free(Cref);
    return bad ? 1 : 0;
}
