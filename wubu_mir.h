/*
 * wubu_mir.h -- the WuBuOS mid-level IR (the hourglass neck).
 *
 * The compiler doctrine (compiler-doctrine.md): the frontend emits ONE
 * IR, and every ISA is a DRIVER that consumes it. This is the neck of
 * the hourglass: AST (any language) -> MIR -> x86-64 / RISC-V / ARM64
 * / PTX drivers. It is how the compiler unlocks ALL hardware: one
 * frontend, N backends, each a driver in the driver space.
 *
 * MIR is minimal 3-address code with VIRTUAL registers (u32 indices,
 * 0 = the return value register). Because operands are virtual, the
 * register-clobber bug family that plagued the direct x86 codegen
 * (the rdi-clobber binops) is IMPOSSIBLE here — each driver does its
 * own register assignment.
 *
 * Comparisons produce 0/1 in a virtual register (no flags registers in
 * the IR), and branches test a virtual register (MIR_JZ jumps when
 * the vr is 0). Labels are symbolic (MIR_LABEL); drivers resolve them.
 *
 * C11, self-contained.
 */
#ifndef WUBU_MIR_H
#define WUBU_MIR_H

#include <stddef.h>
#include <stdint.h>
#include "holyd_types.h"   /* for HD_MAX_IDENT_LEN (function-name buffers) */

/* a virtual register index. 0 is reserved as the "value" register the
 * program returns (MIR_RET reads it). */
typedef uint32_t wubu_vr_t;

typedef enum {
    MIR_CONST = 1,     /* dst = imm */
    MIR_ADD,           /* dst = a + b */
    MIR_SUB,           /* dst = a - b */
    MIR_MUL,           /* dst = a * b */
    MIR_DIV,           /* dst = a / b */
    MIR_MOD,           /* dst = a % b */
    MIR_AND,           /* dst = a & b (bitwise) */
    MIR_OR,            /* dst = a | b (bitwise) */
    MIR_XOR,           /* dst = a ^ b */
    MIR_SHL,           /* dst = a << b */
    MIR_SHR,           /* dst = a >> b */
    MIR_NEG,           /* dst = -a */
    MIR_NOT,           /* dst = ~a */
    MIR_EQ,            /* dst = (a == b) ? 1 : 0 */
    MIR_NE,            /* dst = (a != b) ? 1 : 0 */
    MIR_LT,            /* dst = (a < b) ? 1 : 0  (signed) */
    MIR_LE,            /* dst = (a <= b) ? 1 : 0 (signed) */
    MIR_GT,            /* dst = (a > b) ? 1 : 0  (signed) */
    MIR_GE,            /* dst = (a >= b) ? 1 : 0 (signed) */
    MIR_ULT,           /* dst = (a < b) ? 1 : 0  (unsigned) */
    MIR_ULE,           /* dst = (a <= b) ? 1 : 0 (unsigned) */
    MIR_UGT,           /* dst = (a > b) ? 1 : 0  (unsigned) */
    MIR_UGE,           /* dst = (a >= b) ? 1 : 0 (unsigned) */
    MIR_MOV,           /* dst = src */
    MIR_JMP,           /* pc = label */
    MIR_JZ,            /* if (src == 0) pc = label */
    MIR_JNZ,           /* if (src != 0) pc = label */
    MIR_LABEL,         /* a jump target (no code) */
    MIR_BREAK,         /* pc = enclosing loop's done label */
    MIR_CONTINUE,      /* pc = enclosing loop's top label */
    MIR_RET,           /* return vr 0 */
    MIR_FRET,          /* return float vr 0 (bits reinterpreted as f32, returned low-32) */
    /* Memory model (arrays + pointers live in a flat int64 memory array) */
    MIR_ALLOC,         /* dst = base_addr; imm = n_elements (reserves memory) */
    MIR_LOAD,          /* dst = mem[a] */
    MIR_STORE,         /* mem[a] = b */
    MIR_CALL,          /* call function func_id (args already in v1..vN) */
    /* Soft-float ops: f32 values travel as IEEE-754 bit patterns inside the
     * int64 register file (upper 32 bits zero). Executed via wubu_softfloat. */
    MIR_FADD,          /* dst = f32(a) + f32(b) */
    MIR_FSUB,          /* dst = f32(a) - f32(b) */
    MIR_FMUL,          /* dst = f32(a) * f32(b) */
    MIR_FDIV,          /* dst = f32(a) / f32(b) */
    MIR_FNEG,          /* dst = -f32(a) */
    MIR_ITOF,          /* dst = (f32)a */
    MIR_FTOI,          /* dst = (int)f32(a) */
    MIR_FEQ,           /* dst = (f32(a) == f32(b)) */
    MIR_FNE,           /* dst = (f32(a) != f32(b)) */
    MIR_FLT,           /* dst = (f32(a) <  f32(b)) */
    MIR_FLE,           /* dst = (f32(a) <= f32(b)) */
    /* f64 (double): bits travel in the full int64 register file */
    MIR_DADD,          /* dst = f64(a) + f64(b) */
    MIR_DSUB,          /* dst = f64(a) - f64(b) */
    MIR_DMUL,          /* dst = f64(a) * f64(b) */
    MIR_DDIV,          /* dst = f64(a) / f64(b) */
    MIR_DNEG,          /* dst = -f64(a) */
    MIR_DITOF,         /* dst = (f64)a */
    MIR_DTOI,          /* dst = (int)f64(a) */
    MIR_F32_TO_F64,    /* dst = (f64)f32(a)  (exact) */
    MIR_F64_TO_F32,    /* dst = (f32)f64(a)  (RNE) */
    /* bfloat16: the AGI tensor dtype. bf16 travels as uint16 in low bits. */
    MIR_BF16_TO_F32,   /* dst = widen_bf16(a)   (exact) */
    MIR_F32_TO_BF16,   /* dst = narrow_f32(a)   (RNE) */
    /* FP16 (IEEE 754 half): 5-bit exp, 10-bit mantissa.
     * Travels as uint16 in low bits; sign-extended to f32 on widen. */
    MIR_F16_TO_F32,    /* dst = widen_f16(a)    (exact) */
    MIR_F32_TO_F16,    /* dst = narrow_f32(a)   (RNE) */
    MIR_F16_ADD,       /* dst = f16(a) + f16(b)  (via f32) */
    MIR_F16_MUL,       /* dst = f16(a) * f16(b)  (via f32) */
    MIR_F16_DIV,       /* dst = f16(a) / f16(b)  (via f32) */
    /* ---- Quantization ops (INT8 for inference) ---- */
    /* Quantize f32 to INT8: scale by (127/max_val), clamp [-128,127].
     * a=input(base), b=scale(base, float), dst=output(base), imm=N.
     * Stores int8 as int32 (sign-extended) for retro-ISA compatibility. */
    MIR_QUANTIZE_I8,   /* dst[i] = clamp(f32[i] * scale, -128, 127) */
    /* Dequantize INT8 to f32: multiply by scale.
     * a=input(base), b=scale(base), dst=output(base), imm=N. */
    MIR_DEQUANTIZE_I8, /* dst[i] = int8[i] * scale */
    /* INT8 matrix multiply: C += A*B where A,B,C are int8 matrices.
     * a=A(base), b=B(base), dst=C(base); imm packs M,N,K.
     * Accumulates into int32 to prevent overflow. */
    MIR_T_GEMM_I8,
    /* T_GEMM: tensor matrix multiply C += A*B (int64 elements).
     * vrs: a=A(base), b=B(base), dst=C(base); imm packs M,N,K:
     *   imm = ((uint64_t)M << 22) | ((uint64_t)N << 11) | (uint64_t)K
     * A is M×K row-major, B is K×N row-major, C is M×N row-major,
     * all in the MIR flat mem[] array. Lowered to a register-tiled
     * triple loop (x86-64 native emits AVX/sse2 integer MADD-style
     * block accumulation). */
    MIR_T_GEMM,
    /* ---- AGI tensor ops (inference primitives) ---- */
    /* T_SOFTMAX: softmax over a 1D vector of f32 values.
     * a=input(base), dst=output(base), imm=N (count).
     * Output[i] = exp(input[i] - max) / sum(exp(input[j] - max)) */
    MIR_T_SOFTMAX,
    /* T_LAYERNORM: RMS-style normalization over a 1D f32 vector.
     * a=input(base), dst=output(base), imm=N (count).
     * mean_sq = sum(x^2)/N; output[i] = input[i] / sqrt(mean_sq + eps) */
    MIR_T_LAYERNORM,
    /* T_ATTENTION: scaled dot-product attention.
     * a=Q(base), b=K(base), dst=V(base); imm packs (H << 24)|(D_head << 16)|T|scale.
     * Q,K,V are T×D_head row-major per head, H heads. Output overwrites V. */
    MIR_T_ATTENTION,
    /* T_EMBEDDING: token embedding lookup.
     * a=table(base), b=token_id, dst=output_row(base), imm=dim.
     * Copies dim elements from table[token_id*dim .. (token_id+1)*dim] to output. */
    MIR_T_EMBEDDING,
    /* T_SWIGLU: SwiGLU activation (gate * sigmoid(gate)) * up.
     * a=gate(base), b=up(base), dst=output(base), imm=N.
     * output[i] = gate[i] * sigmoid(gate[i]) * up[i] */
    MIR_T_SWIGLU,
    /* T_RMS_NORM: RMSNorm with optional weight.
     * a=input(base), b=weight(base or 0), dst=output(base), imm=N.
     * rms = sqrt(sum(x^2)/N + eps); output[i] = weight[i] * x[i] / rms */
    MIR_T_RMS_NORM,
    /* T_ROPE: Rotary Position Embedding.
     * a=input(base), dst=output(base), imm packs (dim << 16) | pos | (head << 8).
     * Applies 2D rotation pairs at position pos for head head, dim dimensions. */
    MIR_T_ROPE,
    /* T_CONV2D: 2D convolution (simplified: valid padding, stride 1).
     * a=input(base), b=kernel(base), dst=output(base);
     * imm packs (IC << 24)|(OC << 16)|(KH << 8)|KW.
     * Input: IC×IH×IW, Kernel: OC×IC×KH×KW, Output: OC×OH×OW. */
    MIR_T_CONV2D,
    /* T_DROPOUT: dropout mask (training only; inference = identity).
     * a=input(base), dst=output(base), imm=N; rate in a->imm (0 = inference).
     * output[i] = (rand() > rate) ? input[i] / (1-rate) : 0 */
    MIR_T_DROPOUT,
    /* T_ARGMAX: find index of maximum f32 value.
     * a=input(base), imm=N. Returns index in vr0. */
    MIR_T_ARGMAX,
    /* T_SUM: sum a 1D f32 vector.
     * a=input(base), imm=N. Returns sum (f32 bits) in vr0. */
    MIR_T_SUM,
    /* T_EXP: elementwise exp(x) on f32 vector.
     * a=input(base), dst=output(base), imm=N. */
    MIR_T_EXP,
    /* T_SQRT: elementwise sqrt(x) on f32 vector.
     * a=input(base), dst=output(base), imm=N. */
    MIR_T_SQRT,
    /* T_TANH: elementwise tanh(x) on f32 vector.
     * a=input(base), dst(output(base), imm=N. */
    MIR_T_TANH,
    /* T_SIGMOID: elementwise sigmoid(x) = 1/(1+exp(-x)).
     * a=input(base), dst=output(base), imm=N. */
    MIR_T_SIGMOID,
    /* T_GELU: elementwise GELU(x) = x * Φ(x).
     * a=input(base), dst=output(base), imm=N. */
    MIR_T_GELU,
    /* T_RELU: elementwise ReLU(x) = max(0,x).
     * a=input(base), dst=output(base), imm=N. */
    MIR_T_RELU,
    /* T_CLAMP: elementwise clamp(x, lo, hi).
     * a=input(base), dst=output(base), imm packs lo/hi as f32 bits in upper/lower 32. */
    MIR_T_CLAMP,
    /* T_GEMM_BIAS: fused GEMM + bias addition.
     * Same encoding as T_GEMM plus b=bias_base. */
    MIR_T_GEMM_BIAS,
    /* FUSED_AFFINE: y = x * weight + bias (single instruction).
     * dst=output, a=x, b=weight; imm packs bias VR in upper 32 bits. */
    MIR_FUSED_AFFINE,
    /* T_LAYERNORM_APPLY: fused layer norm application (gamma, beta).
     * dst=output, a=centered, b=params; imm packs gamma/beta indices. */
    MIR_T_LAYERNORM_APPLY,
    /* T_GEMM_F32: float32 tensor matrix multiply C += A*B.
     * Same encoding as T_GEMM but operates on float32 data.
     * Dispatches to wubu_tgemm_f32() for optimized AVX2+FMA execution. */
    MIR_T_GEMM_F32,
} wubu_mir_op_t;

#define MIR_MAX_FUNCTIONS 256
#define MIR_MAX_CALL_ARGS 32
#define MIR_MAX_CALL_DEPTH 256
typedef struct {
    char name[HD_MAX_IDENT_LEN];   /* function name ("" for anonymous) */
    size_t start;                 /* first instruction index */
    size_t end;                   /* one past last instruction */
} wubu_mir_func_t;

typedef struct {
    wubu_mir_op_t op;
    wubu_vr_t dst;
    wubu_vr_t a;
    wubu_vr_t b;
    int64_t imm;
    uint32_t label;              /* for JMP/JZ: the target label id */
    uint32_t func_id;            /* for MIR_CALL: index into prog->funcs */
    char func_name[HD_MAX_IDENT_LEN]; /* for MIR_CALL: external function name ("" if internal) */
} wubu_mir_instr_t;

typedef struct {
    wubu_mir_instr_t *ins;       /* dynamic array */
    size_t n, cap;
    uint32_t n_labels;           /* next label id */
    uint32_t n_args;             /* number of function arguments (v1..n_args) */
    int64_t total_mem;           /* number of int64 cells reserved via MIR_ALLOC */
    int64_t *mem;                /* direct-access memory (for tests/external init) */
    wubu_vr_t next_vr_hi;        /* high-water mark of high-vr address slots (call convention) */
    wubu_mir_func_t funcs[MIR_MAX_FUNCTIONS];
    int n_funcs;
    /* External function name table for JIT resolution.
     * When func_id == 0xFFFF, the JIT looks up the name here to emit
     * a real call to libc instead of xor eax,eax. */
    char ext_func_names[MIR_MAX_FUNCTIONS][HD_MAX_IDENT_LEN];
    uint32_t ext_func_ids[MIR_MAX_FUNCTIONS];  /* func_id for each ext entry */
    int n_ext_funcs;
} wubu_mir_prog_t;

/* O1: init a program (zeroed = empty) */
void wubu_mir_init(wubu_mir_prog_t *p);
void wubu_mir_free(wubu_mir_prog_t *p);

/* O2: append instructions (returns the new vr for dst ops) */
wubu_vr_t wubu_mir_const(wubu_mir_prog_t *p, int64_t imm);
/* Like wubu_mir_const but forces the destination virtual register to `dst`
 * (used for the call convention: arguments must live in v1..vN). */
wubu_vr_t wubu_mir_const_to(wubu_mir_prog_t *p, wubu_vr_t dst, int64_t imm);
wubu_vr_t wubu_mir_binop(wubu_mir_prog_t *p, wubu_mir_op_t op,
                         wubu_vr_t a, wubu_vr_t b);
/* Return a float vr (bits reinterpreted as f32 on float-less ISAs). */
void     wubu_mir_fret(wubu_mir_prog_t *p, wubu_vr_t a);
wubu_vr_t wubu_mir_unop(wubu_mir_prog_t *p, wubu_mir_op_t op, wubu_vr_t a);
wubu_vr_t wubu_mir_mov(wubu_mir_prog_t *p, wubu_vr_t a);
/* mov INTO a pre-chosen dst (phi-merge: both arms write the same vr) */
wubu_vr_t wubu_mir_mov_to(wubu_mir_prog_t *p, wubu_vr_t dst, wubu_vr_t a);
uint32_t wubu_mir_new_label(wubu_mir_prog_t *p);
void wubu_mir_jmp(wubu_mir_prog_t *p, uint32_t label);
void wubu_mir_jz(wubu_mir_prog_t *p, wubu_vr_t cond, uint32_t label);
void wubu_mir_jnz(wubu_mir_prog_t *p, wubu_vr_t cond, uint32_t label);
void wubu_mir_place_label(wubu_mir_prog_t *p, uint32_t label);
/* reserve n_elements int64 cells in the program's memory; returns base addr */
wubu_vr_t wubu_mir_alloc(wubu_mir_prog_t *p, int64_t n_elements);
/* load/store through an address held in a vr */
wubu_vr_t wubu_mir_load(wubu_mir_prog_t *p, wubu_vr_t addr);
void wubu_mir_store(wubu_mir_prog_t *p, wubu_vr_t addr, wubu_vr_t val);
void wubu_mir_ret(wubu_mir_prog_t *p, wubu_vr_t v);
/* Emit MIR_T_GEMM: mem[dst] += A[i*N+k]*B[k*N+j] accumulation.
 * a=Abase, b=Bbase, dst=Cbase, M/N/K are the matrix shapes. */
void wubu_mir_tgemm(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b,
                   wubu_vr_t dst, int M, int N, int K);
/* Emit MIR_T_GEMM_F32: float32 matrix multiply C += A*B.
 * Same encoding as T_GEMM but operates on float32 data. */
void wubu_mir_tgemm_f32(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b,
                        wubu_vr_t dst, int M, int N, int K);
/* AGI tensor ops */
void wubu_mir_tsoftmax(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_trms_norm(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tlayernorm(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tattention(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tembedding(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tswiglu(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_trope(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tconv2d(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tdropout(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_targmax(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tsum(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_texp(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tsqrt(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_ttanh(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tsigmoid(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tgelu(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_trelu(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
void wubu_mir_tclamp(wubu_mir_prog_t *p, wubu_vr_t a, wubu_vr_t b, wubu_vr_t dst, int64_t imm);
/* Emit MIR_CALL to prog->funcs[func_id] (args must be in v1..vN already).
 * For external calls (func_id == 0xFFFF), name is used by the JIT to
 * resolve the actual libc function. */
void wubu_mir_call_ext(wubu_mir_prog_t *p, uint32_t func_id, const char *name);
void wubu_mir_call(wubu_mir_prog_t *p, uint32_t func_id);

/* Set the number of function arguments (v1..n_args get pre-assigned to arg regs) */
void wubu_mir_set_n_args(wubu_mir_prog_t *p, uint32_t n_args);

/* O3: dump the program (the hourglass neck, visible) */
void wubu_mir_dump(const wubu_mir_prog_t *p);

/* O4: interpret a MIR program directly (the portable execution oracle).
 * Faithfully runs the canonical MIR every driver consumes, so the
 * differential gauntlet can verify ALL backends agree on a host whose
 * native CPU is not the target (e.g. x86-64 running an ARM64 target).
 * Returns the vr0 value at MIR_RET. */
int64_t wubu_mir_interp(const wubu_mir_prog_t *p);

/* O5: convert MIR to SSA form (enables SCCP + GVN) */
int wubu_mir_to_ssa(wubu_mir_prog_t *p);

#endif /* WUBU_MIR_H */
