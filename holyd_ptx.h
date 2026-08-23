/*
 * holyd_ptx.h  --  PTX Backend Header for HolyD Compiler
 * 
 * Public API for NVIDIA Tensor Core PTX emission.
 */

#ifndef WUBUNOS_HOLYC_PTX_H
#define WUBUNOS_HOLYC_PTX_H

#include "holyd.h"
#include <stdint.h>
#include <stddef.h>

/* PTX Target Architecture */
typedef enum {
    HD_TARGET_X86_64 = 0,
    HD_TARGET_PTX    = 1,
} HDTargetArch;

/* PTX Generator Context */
typedef struct PTXGen {
    char      *code;
    size_t     code_size;
    size_t     code_cap;
    int        label_count;
    int        reg_count;
    HDTargetArch target_arch;
} PTXGen;

/* Initialize PTX generator */
void ptx_gen_init(PTXGen *gen);

/* Emit PTX string (internal) */
void ptx_emit(PTXGen *gen, const char *fmt, ...);

/* Emit PTX version and target */
void ptx_emit_header(PTXGen *gen);

/* Emit PTX function prologue for matrix multiply kernel */
void ptx_emit_matmul_kernel_prologue(PTXGen *gen, const char *name);

/* Emit PTX epilogue */
void ptx_emit_epilogue(PTXGen *gen);

/* Emit MMA matrix multiply for FP16 Tensor Cores (m16n8k16) */
void ptx_emit_mma_f16_m16n8k16(PTXGen *gen, 
                                const char *d_a, const char *d_b, const char *d_c);

/* Emit FP16 matrix multiply kernel using Tensor Cores */
void ptx_emit_holyd_gpu_matmul(PTXGen *gen);

/* Public API: Compile HolyD source to PTX for GPU execution */
char *hd_compile_ptx(const char *source);

/* PTX runtime execution via CUDA driver API (stub) */
int hd_exec_ptx(const char *ptx_code, void **args, int num_args);

/* HolyD built-in: GpuMatMul(ptr A, ptr B, ptr C, I64 M, I64 N, I64 K) */
int64_t hd_builtin_gpu_matmul(int64_t A, int64_t B, int64_t C, 
                               int64_t M, int64_t N, int64_t K);

#endif /* WUBUNOS_HOLYC_PTX_H */
