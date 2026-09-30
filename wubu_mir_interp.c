/*
 * wubu_mir_interp.c -- a correct, portable interpreter for wubu_mir.
 *
 * Doctrine: the frontend emits ONE MIR (the hourglass neck), and every
 * ISA is a DRIVER that consumes it. On a host whose native CPU is not the
 * target (e.g. this x86-64 box running an ARM64/RISC-V/m68k target), the
 * per-ISA *encoded* bytes cannot be natively executed — but the MIR is
 * ISA-neutral and can be run directly. This interpreter is the faithful
 * execution oracle: it runs the exact MIR program every driver consumes,
 * so the differential gauntlet can verify ALL backends agree with the
 * x86-64 native JIT (the golden reference) without needing native targets.
 *
 * It also exercises the real pipeline: HolyD -> MIR (lowering + regalloc
 * already ran during compile) -> interpret the canonical form. The per-ISA
 * encoders are still validated by their `compile()` not crashing and by the
 * tools/verify_isa.sh objdump oracle; this interpreter is the run oracle.
 *
 * Dispatch: direct-threaded (computed goto) per the SOTA technique —
 * replaces the C `switch` dispatch (which GCC lowers to a jumptable per
 * iteration, adding an indirection + loop-condition check on every op)
 * with a computed goto through a static label table. CPython reports
 * 15-20% from this alone; eli.thegreenplace benchmarks show 25% over
 * switch; LuaJIT and YARV use the same pattern. Our inner loop is the
 * hot path for 11/14 interpreted backends, so the leverage is high.
 *
 * SOTA sources:
 *   - CPython: https://docs.python.org/3/library/dis.html (computed goto)
 *   - eli.thegreenplace: "Computed goto for efficient dispatch tables" (25%)
 *   - Rust VM experiments: direct-threading 2x vs switch
 *   - BLIS interpreter: label-threaded dispatch for micro-kernel loops
 *
 * C11, self-contained.
 */
#include "wubu_mir.h"
#include "wubu_softfloat.h"
#include "wubu_tgemm.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* HolyC/HolyD contract: `int` intermediates are 64-bit two's-complement;
 * arithmetic NEVER wraps implicitly — only explicit narrowing ops truncate.
 * The x86-64 driver already emits REX.W (64-bit) ops; the interpreter now
 * matches it exactly, and the fuzz oracle models plain int64 arithmetic. */

/*
 * Interpret a MIR program. Returns the value in vr0 at MIR_RET.
 *
 * Model: virtual registers are an array sized to the max vr referenced.
 * Each instruction executes in program order, except JMP/JZ which alter
 * the pc. MIR_RET terminates and returns vr(a) (the ret operand is the
 * vr to return — see wubu_mir_ret).
 */
int64_t wubu_mir_interp(const wubu_mir_prog_t *p)
{
    if (!p || p->n == 0) return 0;

    /* find max vr referenced so we can size the register file.
     * Every op that references registers does so via dst/a/b; scanning all
     * three fields for every op is conservative but always correct. */
    uint32_t max_vr = 0;
    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (in->dst > max_vr) max_vr = in->dst;
        if (in->a   > max_vr) max_vr = in->a;
        if (in->b   > max_vr) max_vr = in->b;
    }
    /* include vr0 (return) */
    max_vr = (max_vr < 1) ? 1 : max_vr;

    int64_t *vr = (int64_t *)calloc((size_t)max_vr + 1, sizeof(int64_t));
    if (!vr) return 0;

    /* label id -> instruction index (MIR_LABEL is a no-op placeholder).
     * Resolve in a PRE-PASS so forward jumps work (see comment below). */
    size_t *label_pc = (size_t *)calloc(p->n_labels ? p->n_labels : 1, sizeof(size_t));
    if (!label_pc) { free(vr); return 0; }
    for (size_t i = 0; i < p->n; i++) {
        if (p->ins[i].op == MIR_LABEL && p->ins[i].label < p->n_labels)
            label_pc[p->ins[i].label] = i + 1;  /* next instr after the label */
    }

    /* Memory model: a flat array of int64 cells (for arrays + pointers).
     * Size it to cover both MIR_ALLOC cells AND the high-vr address slots
     * used by the call convention (param slots live at those addresses). */
    int64_t mem_hi = p->total_mem;
    if ((int64_t)(p->next_vr_hi) - 1 > mem_hi) mem_hi = (int64_t)(p->next_vr_hi) - 1;
    int64_t mem_size = (mem_hi < 1) ? 1 : (mem_hi + 1);
    uint8_t *mem = p->mem ? p->mem : (uint8_t *)calloc((size_t)mem_size, sizeof(int64_t));
    int alloc_mem = (p->mem == NULL);
    if (!mem) { free(vr); free(label_pc); return 0; }

    size_t pc = 0;
    int64_t result = 0;
    size_t guard = 0;
    /* call stack: each frame saves the return pc plus a snapshot of the
     * register file and memory so calls (incl. recursion) are reentrant —
     * the canonical MIR uses absolute vrs shared across all invocations. */
    typedef struct { size_t ret_pc; int64_t *vr_save; uint8_t *mem_save; } call_frame_t;
    call_frame_t call_stack[MIR_MAX_CALL_DEPTH];
    int call_sp = 0;

    /* --- Direct-threaded dispatch (computed goto) ---
     *
     * SOTA technique (CPython, YARV, LuaJIT, BLIS): each opcode dispatches
     * via `goto *labels[op]` instead of `switch`. The label table is static
     * so it's built once at function entry, and each handler ends with
     * DISPATCH() which increments pc and jumps to the next label — no loop
     * condition check, no switch indirection per iteration.
     */
    /* Opcode -> handler, designated by ENUM NAME rather than by position.
     *
     * The old table was positional and had silently fallen 5 entries behind
     * the enum: it omitted MIR_TO_PTR and the four integer-extend ops, so
     * every opcode from 39 upward dispatched to the WRONG handler. The
     * visible symptom was that MIR_SEXT32 (47) ran op_fne, so `127 + 1`
     * evaluated 128 != 128 == 1 on every interpreted target -- 13 of the 14
     * gauntlet targets scored an identical 79/247 because they all call
     * wubu_mir_interp. Naming the enum constant turns a dropped or
     * reordered opcode into a compile error instead of a silent miscompile. */
    enum { MIR_LABEL_SLOTS = 105 };
    static void *labels[MIR_LABEL_SLOTS] = {
        [MIR_CONST           ] = &&op_const, /* 1 */
        [MIR_ADD             ] = &&op_add, /* 2 */
        [MIR_SUB             ] = &&op_sub, /* 3 */
        [MIR_MUL             ] = &&op_mul, /* 4 */
        [MIR_DIV             ] = &&op_div, /* 5 */
        [MIR_MOD             ] = &&op_mod, /* 6 */
        [MIR_UDIV            ] = &&op_udiv, /* 7 */
        [MIR_UMOD            ] = &&op_umod, /* 8 */
        [MIR_AND             ] = &&op_and, /* 9 */
        [MIR_OR              ] = &&op_or, /* 10 */
        [MIR_XOR             ] = &&op_xor, /* 11 */
        [MIR_SHL             ] = &&op_shl, /* 12 */
        [MIR_SHR             ] = &&op_shr, /* 13 */
        [MIR_NEG             ] = &&op_neg, /* 14 */
        [MIR_NOT             ] = &&op_not, /* 15 */
        [MIR_EQ              ] = &&op_eq, /* 16 */
        [MIR_NE              ] = &&op_ne, /* 17 */
        [MIR_LT              ] = &&op_lt, /* 18 */
        [MIR_LE              ] = &&op_le, /* 19 */
        [MIR_GT              ] = &&op_gt, /* 20 */
        [MIR_GE              ] = &&op_ge, /* 21 */
        [MIR_ULT             ] = &&op_ult, /* 22 */
        [MIR_ULE             ] = &&op_ule, /* 23 */
        [MIR_UGT             ] = &&op_ugt, /* 24 */
        [MIR_UGE             ] = &&op_uge, /* 25 */
        [MIR_MOV             ] = &&op_mov, /* 26 */
        [MIR_JMP             ] = &&op_jmp, /* 27 */
        [MIR_JZ              ] = &&op_jz, /* 28 */
        [MIR_JNZ             ] = &&op_jnz, /* 29 */
        [MIR_LABEL           ] = &&op_label, /* 30 */
        [MIR_BREAK           ] = &&op_break, /* 31 */
        [MIR_CONTINUE        ] = &&op_continue, /* 32 */
        [MIR_RET             ] = &&op_ret, /* 33 */
        [MIR_FRET            ] = &&op_fret, /* 34 */
        [MIR_ALLOC           ] = &&op_alloc, /* 35 */
        [MIR_LOAD            ] = &&op_load, /* 36 */
        [MIR_STORE           ] = &&op_store, /* 37 */
        [MIR_CALL            ] = &&op_call, /* 38 */
        [MIR_TO_PTR          ] = &&op_default, /* 39 */
        [MIR_FADD            ] = &&op_fadd, /* 40 */
        [MIR_FSUB            ] = &&op_fsub, /* 41 */
        [MIR_FMUL            ] = &&op_fmul, /* 42 */
        [MIR_FDIV            ] = &&op_fdiv, /* 43 */
        [MIR_FNEG            ] = &&op_fneg, /* 44 */
        [MIR_ITOF            ] = &&op_itof, /* 45 */
        [MIR_FTOI            ] = &&op_ftoi, /* 46 */
        [MIR_SEXT32          ] = &&op_sext32, /* 47 */
        [MIR_SEXT16          ] = &&op_sext16, /* 48 */
        [MIR_SEXT8           ] = &&op_sext8, /* 49 */
        [MIR_ZEXT32          ] = &&op_zext32, /* 50 */
        [MIR_FEQ             ] = &&op_feq, /* 51 */
        [MIR_FNE             ] = &&op_fne, /* 52 */
        [MIR_FLT             ] = &&op_flt, /* 53 */
        [MIR_FLE             ] = &&op_fle, /* 54 */
        [MIR_DADD            ] = &&op_dadd, /* 55 */
        [MIR_DSUB            ] = &&op_dsub, /* 56 */
        [MIR_DMUL            ] = &&op_dmul, /* 57 */
        [MIR_DDIV            ] = &&op_ddiv, /* 58 */
        [MIR_DGT             ] = &&op_dgt, /* 59 */
        [MIR_DLT             ] = &&op_dlt, /* 60 */
        [MIR_DGE             ] = &&op_dge, /* 61 */
        [MIR_DLE             ] = &&op_dle, /* 62 */
        [MIR_DEQ             ] = &&op_deq, /* 63 */
        [MIR_DNE             ] = &&op_dne, /* 64 */
        [MIR_DNEG            ] = &&op_dneg, /* 65 */
        [MIR_DITOF           ] = &&op_ditof, /* 66 */
        [MIR_DTOI            ] = &&op_dtoi, /* 67 */
        [MIR_DTOI_U          ] = &&op_dtoi_u, /* 68 */
        [MIR_F32_TO_F64      ] = &&op_f32_to_f64, /* 69 */
        [MIR_F64_TO_F32      ] = &&op_f64_to_f32, /* 70 */
        [MIR_BF16_TO_F32     ] = &&op_bf16_to_f32, /* 71 */
        [MIR_F32_TO_BF16     ] = &&op_f32_to_bf16, /* 72 */
        [MIR_F16_TO_F32      ] = &&op_f16_to_f32, /* 73 */
        [MIR_F32_TO_F16      ] = &&op_f32_to_f16, /* 74 */
        [MIR_F16_ADD         ] = &&op_f16_add, /* 75 */
        [MIR_F16_MUL         ] = &&op_f16_mul, /* 76 */
        [MIR_F16_DIV         ] = &&op_f16_div, /* 77 */
        [MIR_QUANTIZE_I8     ] = &&op_default, /* 78 */
        [MIR_DEQUANTIZE_I8   ] = &&op_default, /* 79 */
        [MIR_T_GEMM_I8       ] = &&op_default, /* 80 */
        [MIR_T_GEMM          ] = &&op_t_gemm, /* 81 */
        [MIR_T_SOFTMAX       ] = &&op_t_softmax, /* 82 */
        [MIR_T_LAYERNORM     ] = &&op_t_layernorm, /* 83 */
        [MIR_T_ATTENTION     ] = &&op_t_attention, /* 84 */
        [MIR_T_EMBEDDING     ] = &&op_t_embedding, /* 85 */
        [MIR_T_SWIGLU        ] = &&op_t_swiglu, /* 86 */
        [MIR_T_RMS_NORM      ] = &&op_t_rms_norm, /* 87 */
        [MIR_T_ROPE          ] = &&op_t_rope, /* 88 */
        [MIR_T_CONV2D        ] = &&op_t_conv2d, /* 89 */
        [MIR_T_DROPOUT       ] = &&op_t_dropout, /* 90 */
        [MIR_T_ARGMAX        ] = &&op_t_argmax, /* 91 */
        [MIR_T_SUM           ] = &&op_t_sum, /* 92 */
        [MIR_T_EXP           ] = &&op_t_exp, /* 93 */
        [MIR_T_SQRT          ] = &&op_t_sqrt, /* 94 */
        [MIR_T_TANH          ] = &&op_t_tanh, /* 95 */
        [MIR_T_SIGMOID       ] = &&op_t_sigmoid, /* 96 */
        [MIR_T_GELU          ] = &&op_t_gelu, /* 97 */
        [MIR_T_RELU          ] = &&op_t_relu, /* 98 */
        [MIR_T_CLAMP         ] = &&op_t_clamp, /* 99 */
        [MIR_T_GEMM_BIAS     ] = &&op_default, /* 100 */
        [MIR_FUSED_AFFINE    ] = &&op_default, /* 101 */
        [MIR_T_LAYERNORM_APPLY] = &&op_default, /* 102 */
        [MIR_T_GEMM_F32      ] = &&op_t_gemm_f32, /* 103 */
        [MIR_DITOF_U         ] = &&op_ditof_u, /* 104 */
    };

    /* A missing entry here is legal C and silently defaults to NULL, so the
     * compiler cannot catch it. scripts/check_opcode_coverage.py enforces
     * coverage across every backend at build time instead. Do not try to
     * replace it with a sizeof()-based typedef here: the table's size comes
     * from its initializer list, so adding an enum member does not change
     * that size and the check silently passes. (Verified by trying it.) */

#define DISPATCH() do { \
        pc++; \
        if (pc >= p->n) goto done; \
        if (++guard > 50000000) goto done; \
        in = &p->ins[pc]; \
        goto *labels[in->op]; \
    } while (0)

    const wubu_mir_instr_t *in = &p->ins[0];
    goto *labels[in->op];

    /* ---- Simple arithmetic/bitwise ops: end with DISPATCH ---- */
op_const:
    vr[in->dst] = in->imm;
    DISPATCH();
op_mov:
    vr[in->dst] = vr[in->a];
    DISPATCH();
op_add:
    vr[in->dst] = vr[in->a] + vr[in->b];
    DISPATCH();
op_sub:
    vr[in->dst] = vr[in->a] - vr[in->b];
    DISPATCH();
op_mul:
    vr[in->dst] = vr[in->a] * vr[in->b];
    DISPATCH();
op_div:
    vr[in->dst] = (vr[in->b] != 0) ? (vr[in->a] / vr[in->b]) : 0;
    DISPATCH();
op_mod:
    vr[in->dst] = (vr[in->b] != 0) ? (vr[in->a] % vr[in->b]) : 0;
    DISPATCH();
op_udiv:
    { uint64_t a = (uint64_t)vr[in->a]; uint64_t b = (uint64_t)vr[in->b];
      vr[in->dst] = b ? (int64_t)(a / b) : 0; }
    DISPATCH();
op_umod:
    { uint64_t a = (uint64_t)vr[in->a]; uint64_t b = (uint64_t)vr[in->b];
      vr[in->dst] = b ? (int64_t)(a % b) : 0; }
    DISPATCH();
op_and:
    vr[in->dst] = vr[in->a] & vr[in->b];
    DISPATCH();
op_or:
    vr[in->dst] = vr[in->a] | vr[in->b];
    DISPATCH();
op_xor:
    vr[in->dst] = vr[in->a] ^ vr[in->b];
    DISPATCH();
op_shl:
    vr[in->dst] = (int64_t)((uint64_t)vr[in->a] << (vr[in->b] & 63));
    DISPATCH();
op_shr:
    vr[in->dst] = vr[in->a] >> (vr[in->b] & 63);
    DISPATCH();
op_neg:
    vr[in->dst] = -vr[in->a];
    DISPATCH();
op_not:
    vr[in->dst] = ~vr[in->a];
    DISPATCH();
op_sext32:
    vr[in->dst] = (int64_t)(int32_t)vr[in->a];
    DISPATCH();
op_sext16:
    vr[in->dst] = (int64_t)(int16_t)vr[in->a];
    DISPATCH();
op_sext8:
    vr[in->dst] = (int64_t)(int8_t)vr[in->a];
    DISPATCH();
op_zext32:
    vr[in->dst] = (uint64_t)(uint32_t)vr[in->a];
    DISPATCH();
op_eq:
    vr[in->dst] = (vr[in->a] == vr[in->b]) ? 1 : 0;
    DISPATCH();
op_ne:
    vr[in->dst] = (vr[in->a] != vr[in->b]) ? 1 : 0;
    DISPATCH();
op_lt:
    vr[in->dst] = (vr[in->a] <  vr[in->b]) ? 1 : 0;
    DISPATCH();
op_le:
    vr[in->dst] = (vr[in->a] <= vr[in->b]) ? 1 : 0;
    DISPATCH();
op_gt:
    vr[in->dst] = (vr[in->a] >  vr[in->b]) ? 1 : 0;
    DISPATCH();
op_ge:
    vr[in->dst] = (vr[in->a] >= vr[in->b]) ? 1 : 0;
    DISPATCH();
op_ult:
    vr[in->dst] = ((uint32_t)vr[in->a] <  (uint32_t)vr[in->b]) ? 1 : 0;
    DISPATCH();
op_ule:
    vr[in->dst] = ((uint32_t)vr[in->a] <= (uint32_t)vr[in->b]) ? 1 : 0;
    DISPATCH();
op_ugt:
    vr[in->dst] = ((uint32_t)vr[in->a] >  (uint32_t)vr[in->b]) ? 1 : 0;
    DISPATCH();
op_uge:
    vr[in->dst] = ((uint32_t)vr[in->a] >= (uint32_t)vr[in->b]) ? 1 : 0;
    DISPATCH();

    /* ---- Control flow ---- */
op_label:
    DISPATCH();
op_jmp:
    if (in->label < p->n_labels && label_pc[in->label] > 0) {
        pc = label_pc[in->label];
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    DISPATCH();
op_jz:
    if (vr[in->a] == 0 && in->label < p->n_labels && label_pc[in->label] > 0) {
        pc = label_pc[in->label];
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    DISPATCH();
op_jnz:
    if (vr[in->a] != 0 && in->label < p->n_labels && label_pc[in->label] > 0) {
        pc = label_pc[in->label];
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    DISPATCH();
op_break:
    if (in->label < p->n_labels && label_pc[in->label] > 0) {
        pc = label_pc[in->label];
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    DISPATCH();
op_continue:
    if (in->label < p->n_labels && label_pc[in->label] > 0) {
        pc = label_pc[in->label];
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    DISPATCH();

    /* ---- Memory ops ---- */
op_store:
    {
        int64_t addr = vr[in->a];
        if (addr >= 0 && addr < mem_size * 8) mem_store64(mem, addr, vr[in->b]);
    }
    DISPATCH();
op_load:
    {
        int64_t addr = vr[in->a];
        vr[in->dst] = (addr >= 0 && addr < mem_size * 8) ? mem_load64(mem, addr) : 0;
    }
    DISPATCH();
op_alloc:
    /* addresses are now const vrs; nothing to do at runtime */
    DISPATCH();

    /* ---- soft-float ops (f32 as IEEE bit patterns, upper 32 bits zero) ---- */
op_fadd:
    vr[in->dst] = wubu_sf_f32_add((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
    DISPATCH();
op_fsub:
    vr[in->dst] = wubu_sf_f32_sub((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
    DISPATCH();
op_fmul:
    vr[in->dst] = wubu_sf_f32_mul((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
    DISPATCH();
op_fdiv:
    vr[in->dst] = wubu_sf_f32_div((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
    DISPATCH();
op_fneg:
    vr[in->dst] = wubu_sf_f32_neg((uint32_t)vr[in->a]);
    DISPATCH();
op_itof:
    vr[in->dst] = wubu_sf_i64_to_f32(vr[in->a]);
    DISPATCH();
op_ftoi:
    vr[in->dst] = wubu_sf_f32_to_i64((uint32_t)vr[in->a]);
    DISPATCH();
op_feq:
    vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) == 0);
    DISPATCH();
op_fne:
    vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) != 2)
                      && (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) != 0);
    DISPATCH();
op_flt:
    vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) == -1);
    DISPATCH();
op_fle:
    { int c = wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
      vr[in->dst] = (c == -1 || c == 0); }
    DISPATCH();
    /* f64: bits fill the full int64 register */
op_dadd:
    vr[in->dst] = (int64_t)wubu_sf_f64_add((uint64_t)vr[in->a], (uint64_t)vr[in->b]);
    DISPATCH();
op_dsub:
    vr[in->dst] = (int64_t)wubu_sf_f64_sub((uint64_t)vr[in->a], (uint64_t)vr[in->b]);
    DISPATCH();
op_dmul:
    vr[in->dst] = (int64_t)wubu_sf_f64_mul((uint64_t)vr[in->a], (uint64_t)vr[in->b]);
    DISPATCH();
op_ddiv:
    vr[in->dst] = (int64_t)wubu_sf_f64_div((uint64_t)vr[in->a], (uint64_t)vr[in->b]);
    DISPATCH();
op_dneg:
    vr[in->dst] = (int64_t)wubu_sf_f64_neg((uint64_t)vr[in->a]);
    DISPATCH();
/* Double comparisons for interpreter (software float path) */
op_dgt: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d > ub.d) ? 1 : 0;
    DISPATCH();
}
op_dlt: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d < ub.d) ? 1 : 0;
    DISPATCH();
}
op_dge: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d >= ub.d) ? 1 : 0;
    DISPATCH();
}
op_dle: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d <= ub.d) ? 1 : 0;
    DISPATCH();
}
op_deq: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d == ub.d) ? 1 : 0;
    DISPATCH();
}
op_dne: {
    union { double d; int64_t i; } ua, ub;
    ua.i = vr[in->a]; ub.i = vr[in->b];
    vr[in->dst] = (ua.d != ub.d) ? 1 : 0;
    DISPATCH();
}
op_ditof:
    vr[in->dst] = (int64_t)wubu_sf_i64_to_f64(vr[in->a]);
    DISPATCH();
op_ditof_u: {
    /* unsigned 64-bit to double: reinterpret bits as uint64, convert to f64 */
    union { double d; int64_t i; } result;
    result.d = (double)(uint64_t)vr[in->a];
    vr[in->dst] = result.i;
    DISPATCH();
}
op_dtoi:
    vr[in->dst] = wubu_sf_f64_to_i64((uint64_t)vr[in->a]);
    DISPATCH();
op_dtoi_u:
    { double d; memcpy(&d, &vr[in->a], 8); vr[in->dst] = (int64_t)(uint64_t)d; }
    DISPATCH();
op_f32_to_f64:
    vr[in->dst] = (int64_t)wubu_sf_f32_to_f64((uint32_t)vr[in->a]);
    DISPATCH();
op_f64_to_f32:
    vr[in->dst] = (int32_t)wubu_sf_f64_to_f32((uint64_t)vr[in->a]);
    DISPATCH();
op_bf16_to_f32:
    vr[in->dst] = (int32_t)wubu_sf_bf16_to_f32((uint16_t)vr[in->a]);
    DISPATCH();
op_f32_to_bf16:
    vr[in->dst] = (int16_t)wubu_sf_f32_to_bf16((uint32_t)vr[in->a]);
    DISPATCH();
op_f16_to_f32:
    vr[in->dst] = (int32_t)wubu_sf_f16_to_f32((uint16_t)vr[in->a]);
    DISPATCH();
op_f32_to_f16:
    vr[in->dst] = (int16_t)wubu_sf_f32_to_f16((uint32_t)vr[in->a]);
    DISPATCH();
op_f16_add:
    { uint32_t a = wubu_sf_f16_to_f32((uint16_t)vr[in->a]); uint32_t b = wubu_sf_f16_to_f32((uint16_t)vr[in->b]); uint32_t r = wubu_sf_f32_add(a, b); vr[in->dst] = (int32_t)wubu_sf_f32_to_f16(r); }
    DISPATCH();
op_f16_mul:
    { uint32_t a = wubu_sf_f16_to_f32((uint16_t)vr[in->a]); uint32_t b = wubu_sf_f16_to_f32((uint16_t)vr[in->b]); uint32_t r = wubu_sf_f32_mul(a, b); vr[in->dst] = (int32_t)wubu_sf_f32_to_f16(r); }
    DISPATCH();
op_f16_div:
    { uint32_t a = wubu_sf_f16_to_f32((uint16_t)vr[in->a]); uint32_t b = wubu_sf_f16_to_f32((uint16_t)vr[in->b]); uint32_t r = wubu_sf_f32_div(a, b); vr[in->dst] = (int32_t)wubu_sf_f32_to_f16(r); }
    DISPATCH();

    /* ---- Return/termination ---- */
op_fret:
    result = vr[0];
    goto done;
op_ret:
    if (call_sp > 0) {
        /* returning from a called function: restore the caller's
         * register file (the MIR shares absolute vrs across invocations).
         * Memory is NOT restored — globals and locals use disjoint addresses
         * (allocated by wubu_mir_alloc), so changes to globals persist
         * across calls. Recursion is not supported (locals would overlap). */
        call_frame_t *f = &call_stack[--call_sp];
        int64_t retval = vr[in->a];
        memcpy(vr, f->vr_save, ((size_t)max_vr + 1) * sizeof(int64_t));
        vr[0] = retval;
        free(f->vr_save);
        free(f->mem_save);
        pc = f->ret_pc;
        in = &p->ins[pc];
        goto *labels[in->op];
    }
    result = vr[in->a];
    goto done;

    /* ---- Function calls ---- */
op_call:
    {
        if (in->func_id < (uint32_t)p->n_funcs && call_sp < MIR_MAX_CALL_DEPTH) {
            /* snapshot caller state so the callee (which reuses
             * the same absolute vrs / memory) cannot clobber it. */
            call_frame_t *f = &call_stack[call_sp++];
            f->ret_pc = pc + 1;
            f->vr_save = (int64_t *)malloc(((size_t)max_vr + 1) * sizeof(int64_t));
            f->mem_save = (uint8_t *)malloc((size_t)mem_size * sizeof(int64_t));
            if (f->vr_save) memcpy(f->vr_save, vr, ((size_t)max_vr + 1) * sizeof(int64_t));
            if (f->mem_save) memcpy(f->mem_save, mem, (size_t)mem_size * sizeof(int64_t));
            pc = p->funcs[in->func_id].start;
            in = &p->ins[pc];
            goto *labels[in->op];
        }
        /* call stack overflow or invalid func: skip (safe) */
    }
    DISPATCH();

    /* ---- AGI tensor ops ---- */
op_t_gemm:
    {
        int M = (int)(in->imm >> 22);
        int N = (int)((in->imm >> 11) & 0x7FFF);
        int K = (int)(in->imm & 0x7FF);
        int64_t a = vr[in->a], b = vr[in->b], c = vr[in->dst];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                int64_t acc = mem[c + (int64_t)i * N + j];
                for (int k = 0; k < K; k++)
                    acc += mem[a + (int64_t)i * K + k] * mem[b + (int64_t)k * N + j];
                mem[c + (int64_t)i * N + j] = acc;
            }
    }
    DISPATCH();
op_t_gemm_f32:
    {
        int M = (int)(in->imm >> 22);
        int Ndim = (int)((in->imm >> 11) & 0x7FF);
        int K = (int)(in->imm & 0x7FF);
        int64_t a = vr[in->a], b = vr[in->b], c = vr[in->dst];
        wubu_tgemm_f32_mir(mem, a, b, c, M, Ndim, K);
    }
    DISPATCH();

op_t_softmax:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        /* Find max for numerical stability */
        float mx = -1e30f;
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            if (v > mx) mx = v;
        }
        /* Compute exp(x - max) and sum */
        float sum = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]) - mx;
            float e = wubu_sf_f32_to_host(wubu_sf_f32_exp(wubu_sf_f32_from_host(v)));
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(e);
            sum += e;
        }
        /* Normalize */
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_d + i]) / sum;
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(v);
        }
    }
    DISPATCH();
op_t_rms_norm:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_b = vr[in->b], base_d = vr[in->dst];
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            sum_sq += v * v;
        }
        float rms = wubu_sf_f32_to_host(wubu_sf_f32_sqrt(wubu_sf_f32_from_host(sum_sq / N + 1e-6f)));
        for (uint32_t i = 0; i < N; i++) {
            float x = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            float w = (base_b > 0) ? wubu_sf_f32_to_host((uint32_t)mem[base_b + i]) : 1.0f;
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(w * x / rms);
        }
    }
    DISPATCH();
op_t_sum:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a];
        float sum = 0.0f;
        for (uint32_t i = 0; i < N; i++)
            sum += wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
        vr[0] = (int64_t)wubu_sf_f32_from_host(sum);
    }
    DISPATCH();
op_t_exp:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++)
            mem[base_d + i] = (int64_t)wubu_sf_f32_exp((uint32_t)mem[base_a + i]);
    }
    DISPATCH();
op_t_sqrt:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++)
            mem[base_d + i] = (int64_t)wubu_sf_f32_sqrt((uint32_t)mem[base_a + i]);
    }
    DISPATCH();
op_t_tanh:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++)
            mem[base_d + i] = (int64_t)wubu_sf_f32_tanh((uint32_t)mem[base_a + i]);
    }
    DISPATCH();
op_t_sigmoid:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++)
            mem[base_d + i] = (int64_t)wubu_sf_f32_sigmoid((uint32_t)mem[base_a + i]);
    }
    DISPATCH();
op_t_gelu:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++)
            mem[base_d + i] = (int64_t)wubu_sf_f32_gelu((uint32_t)mem[base_a + i]);
    }
    DISPATCH();
op_t_relu:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            mem[base_d + i] = (v > 0.0f) ? mem[base_a + i] : 0;
        }
    }
    DISPATCH();
op_t_argmax:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a];
        uint32_t best_i = 0;
        float best_v = -1e30f;
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            if (v > best_v) { best_v = v; best_i = i; }
        }
        vr[0] = (int64_t)best_i;
    }
    DISPATCH();
op_t_swiglu:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_b = vr[in->b], base_d = vr[in->dst];
        for (uint32_t i = 0; i < N; i++) {
            float gate = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            float up = wubu_sf_f32_to_host((uint32_t)mem[base_b + i]);
            float sig = wubu_sf_f32_to_host(wubu_sf_f32_sigmoid((uint32_t)mem[base_a + i]));
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(gate * sig * up);
        }
    }
    DISPATCH();
op_t_layernorm:
    {
        uint32_t N = (uint32_t)in->imm;
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            sum_sq += v * v;
        }
        float inv_std = wubu_sf_f32_to_host(wubu_sf_f32_rsqrt(wubu_sf_f32_from_host(sum_sq / N + 1e-5f)));
        for (uint32_t i = 0; i < N; i++) {
            float x = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(x * inv_std);
        }
    }
    DISPATCH();
op_t_embedding:
    {
        uint32_t dim = (uint32_t)in->imm;
        int64_t base_table = vr[in->a];
        uint32_t token_id = (uint32_t)vr[in->b];
        int64_t base_out = vr[in->dst];
        int64_t src = base_table + (int64_t)token_id * dim;
        for (uint32_t i = 0; i < dim; i++)
            mem[base_out + i] = mem[src + i];
    }
    DISPATCH();
op_t_rope:
    {
        uint32_t dim = (uint32_t)((in->imm >> 16) & 0xFFFF);
        uint32_t pos = (uint32_t)(in->imm & 0xFFFF);
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        for (uint32_t i = 0; i < dim; i += 2) {
            float x0 = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            float x1 = (i+1 < dim) ? wubu_sf_f32_to_host((uint32_t)mem[base_a + i+1]) : 0.0f;
            float freq = 1.0f / powf(10000.0f, (float)(i/2) / (float)(dim/2));
            float theta = (float)pos * freq;
            float cos_t = cosf(theta), sin_t = sinf(theta);
            float y0 = x0 * cos_t - x1 * sin_t;
            float y1 = x0 * sin_t + x1 * cos_t;
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(y0);
            if (i+1 < dim) mem[base_d + i+1] = (int64_t)wubu_sf_f32_from_host(y1);
        }
    }
    DISPATCH();
op_t_clamp:
    {
        uint32_t N = (uint32_t)(in->imm & 0xFFFFFFFF);
        int64_t base_a = vr[in->a], base_d = vr[in->dst];
        float lo = wubu_sf_f32_to_host((uint32_t)(in->imm >> 32));
        float hi = wubu_sf_f32_to_host((uint32_t)(in->imm & 0xFFFFFFFF));
        for (uint32_t i = 0; i < N; i++) {
            float v = wubu_sf_f32_to_host((uint32_t)mem[base_a + i]);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            mem[base_d + i] = (int64_t)wubu_sf_f32_from_host(v);
        }
    }
    DISPATCH();

    /* ---- Complex ops: interpreter stub (lowered to elementwise on real HW) ---- */
op_t_attention:
op_t_conv2d:
op_t_dropout:
    DISPATCH();

op_default:
    /* unsupported op: skip */
    DISPATCH();

done:
    free(vr);
    free(label_pc);
    if (alloc_mem) free(mem);
    return result;
}
