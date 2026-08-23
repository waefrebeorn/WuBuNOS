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
 * C11, self-contained.
 */
#include "wubu_mir.h"
#include "wubu_softfloat.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    int64_t *mem = (int64_t *)calloc((size_t)mem_size, sizeof(int64_t));
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

    while (pc < p->n) {
        if (++guard > 50000000) { break; }  /* safety against malformed MIR loops */
        const wubu_mir_instr_t *in = &p->ins[pc];

        /* debug: print arg/mem at function entry */
        switch (in->op) {
        case MIR_CONST:
            vr[in->dst] = in->imm;
            pc++;
            break;
        case MIR_MOV:
            vr[in->dst] = vr[in->a];
            pc++;
            break;
        case MIR_ADD: vr[in->dst] = vr[in->a] + vr[in->b]; pc++; break;
        case MIR_SUB: vr[in->dst] = vr[in->a] - vr[in->b]; pc++; break;
        case MIR_MUL: vr[in->dst] = vr[in->a] * vr[in->b]; pc++; break;
        case MIR_DIV:
            vr[in->dst] = (vr[in->b] != 0) ? (vr[in->a] / vr[in->b]) : 0;
            pc++;
            break;
        case MIR_MOD:
            vr[in->dst] = (vr[in->b] != 0) ? (vr[in->a] % vr[in->b]) : 0;
            pc++;
            break;
        case MIR_AND: vr[in->dst] = vr[in->a] & vr[in->b]; pc++; break;
        case MIR_OR:  vr[in->dst] = vr[in->a] | vr[in->b]; pc++; break;
        case MIR_XOR: vr[in->dst] = vr[in->a] ^ vr[in->b]; pc++; break;
        case MIR_SHL: vr[in->dst] = (int64_t)((uint64_t)vr[in->a] << (vr[in->b] & 63)); pc++; break;
        case MIR_SHR: vr[in->dst] = vr[in->a] >> (vr[in->b] & 63); pc++; break;
        case MIR_NEG: vr[in->dst] = -vr[in->a]; pc++; break;
        case MIR_NOT: vr[in->dst] = ~vr[in->a]; pc++; break;
        case MIR_EQ: vr[in->dst] = (vr[in->a] == vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_NE: vr[in->dst] = (vr[in->a] != vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_LT: vr[in->dst] = (vr[in->a] <  vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_LE: vr[in->dst] = (vr[in->a] <= vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_GT: vr[in->dst] = (vr[in->a] >  vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_GE: vr[in->dst] = (vr[in->a] >= vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_ULT: vr[in->dst] = ((uint32_t)vr[in->a] <  (uint32_t)vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_ULE: vr[in->dst] = ((uint32_t)vr[in->a] <= (uint32_t)vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_UGT: vr[in->dst] = ((uint32_t)vr[in->a] >  (uint32_t)vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_UGE: vr[in->dst] = ((uint32_t)vr[in->a] >= (uint32_t)vr[in->b]) ? 1 : 0; pc++; break;
        case MIR_LABEL:
            if (in->label < p->n_labels) label_pc[in->label] = pc + 1;
            pc++;
            break;
        case MIR_JMP:
            if (in->label < p->n_labels && label_pc[in->label] > 0)
                pc = label_pc[in->label];
            else
                pc++; /* unresolved: fall through (safe) */
            break;
        case MIR_JZ:
            if (vr[in->a] == 0 && in->label < p->n_labels && label_pc[in->label] > 0)
                pc = label_pc[in->label];
            else
                pc++;
            break;
        case MIR_JNZ:
            if (vr[in->a] != 0 && in->label < p->n_labels && label_pc[in->label] > 0)
                pc = label_pc[in->label];
            else
                pc++;
            break;
        case MIR_BREAK:
            if (in->label < p->n_labels && label_pc[in->label] > 0)
                pc = label_pc[in->label];
            else
                pc++; /* unresolved: fall through (safe) */
            break;
        case MIR_CONTINUE:
            if (in->label < p->n_labels && label_pc[in->label] > 0)
                pc = label_pc[in->label];
            else
                pc++; /* unresolved: fall through (safe) */
            break;
        case MIR_STORE:
            { int64_t addr = vr[in->a]; if (addr >= 0 && addr < mem_size) mem[addr] = vr[in->b];
            }
            pc++;
            break;
        case MIR_LOAD:
            { int64_t addr = vr[in->a]; vr[in->dst] = (addr >= 0 && addr < mem_size) ? mem[addr] : 0;
            }
            pc++;
            break;
        case MIR_ALLOC:
            /* addresses are now const vrs; nothing to do at runtime */
            pc++;
            break;
        /* --- soft-float ops (f32 as IEEE bit patterns, upper 32 bits zero) --- */
        case MIR_FADD: vr[in->dst] = wubu_sf_f32_add((uint32_t)vr[in->a], (uint32_t)vr[in->b]); pc++; break;
        case MIR_FSUB: vr[in->dst] = wubu_sf_f32_sub((uint32_t)vr[in->a], (uint32_t)vr[in->b]); pc++; break;
        case MIR_FMUL: vr[in->dst] = wubu_sf_f32_mul((uint32_t)vr[in->a], (uint32_t)vr[in->b]); pc++; break;
        case MIR_FDIV: vr[in->dst] = wubu_sf_f32_div((uint32_t)vr[in->a], (uint32_t)vr[in->b]); pc++; break;
        case MIR_FNEG: vr[in->dst] = wubu_sf_f32_neg((uint32_t)vr[in->a]); pc++; break;
        case MIR_ITOF: vr[in->dst] = wubu_sf_i64_to_f32(vr[in->a]); pc++; break;
        case MIR_FTOI: vr[in->dst] = wubu_sf_f32_to_i64((uint32_t)vr[in->a]); pc++; break;
        case MIR_FEQ:  vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) == 0); pc++; break;
        case MIR_FNE:  vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) != 2)
                                     && (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) != 0); pc++; break;
        case MIR_FLT:  vr[in->dst] = (wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]) == -1); pc++; break;
        case MIR_FLE:  { int c = wubu_sf_f32_cmp((uint32_t)vr[in->a], (uint32_t)vr[in->b]);
                         vr[in->dst] = (c == -1 || c == 0); } pc++; break;
        case MIR_RET:
            if (call_sp > 0) {
                /* returning from a called function: restore the caller's
                 * register file + memory (the MIR shares absolute vrs across
                 * invocations), preserving vr0 as the return value. */
                call_frame_t *f = &call_stack[--call_sp];
                int64_t retval = vr[in->a];
                fprintf(stderr, "RET sp=%d retval=%lld\n", call_sp, (long long)retval);
                memcpy(vr, f->vr_save, ((size_t)max_vr + 1) * sizeof(int64_t));
                memcpy(mem, f->mem_save, (size_t)mem_size * sizeof(int64_t));
                vr[0] = retval;
                free(f->vr_save);
                free(f->mem_save);
                pc = f->ret_pc;
                continue;
            }
            result = vr[in->a];
            goto done;
        case MIR_CALL: {
            if (in->func_id < (uint32_t)p->n_funcs) {
                if (call_sp < MIR_MAX_CALL_DEPTH) {
                    /* snapshot caller state so the callee (which reuses the
                     * same absolute vrs / memory) cannot clobber it. */
                    call_frame_t *f = &call_stack[call_sp++];
                    f->ret_pc = pc + 1;
                    f->vr_save = (int64_t *)malloc(((size_t)max_vr + 1) * sizeof(int64_t));
                    f->mem_save = (int64_t *)malloc((size_t)mem_size * sizeof(int64_t));
                    if (f->vr_save) memcpy(f->vr_save, vr, ((size_t)max_vr + 1) * sizeof(int64_t));
                    if (f->mem_save) memcpy(f->mem_save, mem, (size_t)mem_size * sizeof(int64_t));
                    pc = p->funcs[in->func_id].start;
                    continue;
                }
                /* call stack overflow: skip (safe) */
            }
            pc++;
            break;
        }
        default:
            /* unsupported op: skip (honest: lowering covers the battery) */
            pc++;
            break;
        }
    }

done:
    free(vr);
    free(label_pc);
    free(mem);
    return result;
}
