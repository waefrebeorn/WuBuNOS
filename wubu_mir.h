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
    /* T_GEMM: tensor matrix multiply C += A*B (int64 elements).
     * vrs: a=A(base), b=B(base), dst=C(base); imm packs M,N,K:
     *   imm = ((uint64_t)M << 22) | ((uint64_t)N << 11) | (uint64_t)K
     * A is M×K row-major, B is K×N row-major, C is M×N row-major,
     * all in the MIR flat mem[] array. Lowered to a register-tiled
     * triple loop (x86-64 native emits AVX/sse2 integer MADD-style
     * block accumulation). */
    MIR_T_GEMM
} wubu_mir_op_t;

#define MIR_MAX_FUNCTIONS 256
#define MIR_MAX_CALL_ARGS 8
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
} wubu_mir_instr_t;

typedef struct {
    wubu_mir_instr_t *ins;       /* dynamic array */
    size_t n, cap;
    uint32_t n_labels;           /* next label id */
    uint32_t n_args;             /* number of function arguments (v1..n_args) */
    int64_t total_mem;           /* number of int64 cells reserved via MIR_ALLOC */
    wubu_vr_t next_vr_hi;        /* high-water mark of high-vr address slots (call convention) */
    wubu_mir_func_t funcs[MIR_MAX_FUNCTIONS];
    int n_funcs;
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
/* Emit MIR_CALL to prog->funcs[func_id] (args must be in v1..vN already). */
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

#endif /* WUBU_MIR_H */
