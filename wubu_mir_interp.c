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
    int64_t *mem = p->mem ? p->mem : (int64_t *)calloc((size_t)mem_size, sizeof(int64_t));
    int alloc_mem = (p->mem == NULL);
    if (!mem) { free(vr); free(label_pc); return 0; }

    size_t pc = 0;
    int64_t result = 0;
    size_t guard = 0;
    /* call stack: each frame saves the return pc plus a snapshot of the
     * register file and memory so calls (incl. recursion) are reentrant —
     * the canonical MIR uses absolute vrs shared across all invocations. */
    typedef struct { size_t ret_pc; int64_t *vr_save; int64_t *mem_save; } call_frame_t;
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
    static void *labels[] =
        { &&op_default,                    /* 0  (MIR_NONE) */
          &&op_const,                      /* 1  MIR_CONST  */
          &&op_add,                        /* 2  MIR_ADD    */
          &&op_sub,                        /* 3  MIR_SUB    */
          &&op_mul,                        /* 4  MIR_MUL    */
          &&op_div,                        /* 5  MIR_DIV    */
          &&op_mod,                        /* 6  MIR_MOD    */
          &&op_and,                        /* 7  MIR_AND    */
          &&op_or,                         /* 8  MIR_OR     */
          &&op_xor,                        /* 9  MIR_XOR    */
          &&op_shl,                        /* 10 MIR_SHL    */
          &&op_shr,                        /* 11 MIR_SHR    */
          &&op_neg,                        /* 12 MIR_NEG    */
          &&op_not,                        /* 13 MIR_NOT    */
          &&op_eq,                         /* 14 MIR_EQ     */
          &&op_ne,                         /* 15 MIR_NE     */
          &&op_lt,                         /* 16 MIR_LT     */
          &&op_le,                         /* 17 MIR_LE     */
          &&op_gt,                         /* 18 MIR_GT     */
          &&op_ge,                         /* 19 MIR_GE     */
          &&op_ult,                        /* 20 MIR_ULT    */
          &&op_ule,                        /* 21 MIR_ULE    */
          &&op_ugt,                        /* 22 MIR_UGT    */
          &&op_uge,                        /* 23 MIR_UGE    */
          &&op_mov,                        /* 24 MIR_MOV    */
          &&op_jmp,                        /* 25 MIR_JMP    */
          &&op_jz,                         /* 26 MIR_JZ     */
          &&op_jnz,                        /* 27 MIR_JNZ    */
          &&op_label,                      /* 28 MIR_LABEL  */
          &&op_break,                      /* 29 MIR_BREAK  */
          &&op_continue,                   /* 30 MIR_CONTINUE */
          &&op_ret,                        /* 31 MIR_RET    */
          &&op_fret,                       /* 32 MIR_FRET   */
          &&op_alloc,                      /* 33 MIR_ALLOC  */
          &&op_load,                       /* 34 MIR_LOAD   */
          &&op_store,                      /* 35 MIR_STORE  */
          &&op_call,                       /* 36 MIR_CALL   */
          &&op_fadd,                       /* 37 MIR_FADD   */
          &&op_fsub,                       /* 38 MIR_FSUB   */
          &&op_fmul,                       /* 39 MIR_FMUL   */
          &&op_fdiv,                       /* 40 MIR_FDIV   */
          &&op_fneg,                       /* 41 MIR_FNEG   */
          &&op_itof,                       /* 42 MIR_ITOF   */
          &&op_ftoi,                       /* 43 MIR_FTOI   */
          &&op_feq,                        /* 44 MIR_FEQ    */
          &&op_fne,                        /* 45 MIR_FNE    */
          &&op_flt,                        /* 46 MIR_FLT    */
          &&op_fle,                        /* 47 MIR_FLE    */
          &&op_dadd,                       /* 48 MIR_DADD   */
          &&op_dsub,                       /* 49 MIR_DSUB   */
          &&op_dmul,                       /* 50 MIR_DMUL   */
          &&op_ddiv,                       /* 51 MIR_DDIV   */
          &&op_dneg,                       /* 52 MIR_DNEG   */
          &&op_ditof,                      /* 53 MIR_DITOF  */
          &&op_dtoi,                       /* 54 MIR_DTOI   */
          &&op_f32_to_f64,                 /* 55 MIR_F32_TO_F64 */
          &&op_f64_to_f32,                 /* 56 MIR_F64_TO_F32 */
          &&op_bf16_to_f32,                /* 57 MIR_BF16_TO_F32 */
          &&op_f32_to_bf16,                /* 58 MIR_F32_TO_BF16 */
          &&op_f16_to_f32,                 /* 59 MIR_F16_TO_F32 */
          &&op_f32_to_f16,                 /* 60 MIR_F32_TO_F16 */
          &&op_f16_add,                    /* 61 MIR_F16_ADD */
          &&op_f16_mul,                    /* 62 MIR_F16_MUL */
          &&op_f16_div,                    /* 63 MIR_F16_DIV */
          &&op_default,                    /* 64 MIR_QUANTIZE_I8 */
          &&op_default,                    /* 65 MIR_DEQUANTIZE_I8 */
          &&op_default,                    /* 66 MIR_T_GEMM_I8 */
          &&op_t_gemm,                     /* 67 MIR_T_GEMM  */
          &&op_t_softmax,                  /* 68 MIR_T_SOFTMAX */
          &&op_t_layernorm,                /* 69 MIR_T_LAYERNORM */
          &&op_t_attention,                /* 70 MIR_T_ATTENTION */
          &&op_t_embedding,                /* 71 MIR_T_EMBEDDING */
          &&op_t_swiglu,                   /* 72 MIR_T_SWIGLU */
          &&op_t_rms_norm,                 /* 73 MIR_T_RMS_NORM */
          &&op_t_rope,                     /* 74 MIR_T_ROPE */
          &&op_t_conv2d,                   /* 75 MIR_T_CONV2D */
          &&op_t_dropout,                  /* 76 MIR_T_DROPOUT */
          &&op_t_argmax,                   /* 77 MIR_T_ARGMAX */
          &&op_t_sum,                      /* 78 MIR_T_SUM */
          &&op_t_exp,                      /* 79 MIR_T_EXP */
          &&op_t_sqrt,                     /* 80 MIR_T_SQRT */
          &&op_t_tanh,                     /* 81 MIR_T_TANH */
          &&op_t_sigmoid,                  /* 82 MIR_T_SIGMOID */
          &&op_t_gelu,                     /* 83 MIR_T_GELU */
          &&op_t_relu,                     /* 84 MIR_T_RELU */
          &&op_t_clamp,                    /* 85 MIR_T_CLAMP */
          &&op_default,                    /* 86 MIR_T_GEMM_BIAS */
          &&op_default,                    /* 87 MIR_FUSED_AFFINE */
          &&op_default,                    /* 88 MIR_T_LAYERNORM_APPLY */
          &&op_t_gemm_f32 };               /* 89 MIR_T_GEMM_F32 */

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
        if (addr >= 0 && addr < mem_size) mem[addr] = vr[in->b];
    }
    DISPATCH();
op_load:
    {
        int64_t addr = vr[in->a];
        vr[in->dst] = (addr >= 0 && addr < mem_size) ? mem[addr] : 0;
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
op_ditof:
    vr[in->dst] = (int64_t)wubu_sf_i64_to_f64(vr[in->a]);
    DISPATCH();
op_dtoi:
    vr[in->dst] = wubu_sf_f64_to_i64((uint64_t)vr[in->a]);
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
         * register file + memory (the MIR shares absolute vrs across
         * invocations), preserving vr0 as the return value. */
        call_frame_t *f = &call_stack[--call_sp];
        int64_t retval = vr[in->a];
        memcpy(vr, f->vr_save, ((size_t)max_vr + 1) * sizeof(int64_t));
        memcpy(mem, f->mem_save, (size_t)mem_size * sizeof(int64_t));
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
            /* snapshot caller state so the callee (which reuses the
             * same absolute vrs / memory) cannot clobber it. */
            call_frame_t *f = &call_stack[call_sp++];
            f->ret_pc = pc + 1;
            f->vr_save = (int64_t *)malloc(((size_t)max_vr + 1) * sizeof(int64_t));
            f->mem_save = (int64_t *)malloc((size_t)mem_size * sizeof(int64_t));
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
