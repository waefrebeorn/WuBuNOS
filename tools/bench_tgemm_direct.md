# T_GEMM Parallel Benchmark Tool

## Purpose
Standalone benchmark to verify T_GEMM kernel correctness and measure OpenMP speedup outside JIT context.

## Files
- `bench_tgemm_direct.c` — standalone 4-row tiled GEMM benchmark with OpenMP parallel wrapper

## Build and Run

```bash
# Build with OpenMP
gcc -O3 -fopenmp -std=c11 -o /tmp/bench_tgemm tools/bench_tgemm_direct.c

# Run
/tmp/bench_tgemm [M] [N] [K] [reps]

# With multiple threads
OMP_NUM_THREADS=4 /tmp/bench_tgemm 256 256 256
```

## Expected Output
```
sz=256x256x256 nt=4 t=88.801ms GOPS=3.0 bad=0 PASS
```

## Verification Criteria
- bad=0: Correctness against naive reference passed
- GOPS: Billions of operations per second
- speedup: Compare NT=1 vs NT=4 to verify parallelization

## Integration Notes
- Uses `mmap(MAP_ANONYMOUS)` for working buffer (shared across OpenMP threads)
- Direct copy of the 4-row tiled kernel from `wubu_tgemm.c`
- Parallel wrapper `tgemm_parallel()` partitions rows across threads