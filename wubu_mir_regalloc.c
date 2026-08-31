/*
 * wubu_mir_regalloc.c -- linear scan register allocator for MIR programs.
 *
 * Algorithm (linear scan on SSA):
 *   1. Scan instructions forward to compute live ranges [first_def, last_use]
 *      for each virtual register.
 *   2. Sort intervals by start position (first_def).
 *   3. Walk intervals linearly:
 *      - Expire intervals whose last_use < current start (free their regs).
 *      - If a free physical register exists, assign it.
 *      - Otherwise, spill the interval whose last_use is furthest in the
 *        future (the "farthest" heuristic). If the current interval has a
 *        later end than the victim, spill the current one instead.
 *   4. Pre-assign v0 -> physical reg 0 (return), v1..n_args -> reg 1..n_args.
 *
 * Key properties:
 *   - MIR is SSA: each vr is defined exactly once, so intervals are simple.
 *   - v0 is the return register (always physical reg 0).
 *   - Argument vrs (v1..n_args) are pre-assigned to physical regs 1..n_args.
 *   - If n_phys_regs is too small, some vrs spill (reg=-1, stack=offset).
 *
 * C11, self-contained.
 */

#include "wubu_mir_regalloc.h"
#include <stdlib.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Helpers: classify MIR opcodes                                      */
/* ------------------------------------------------------------------ */

static int op_has_dst(wubu_mir_op_t op)
{
    switch (op) {
    case MIR_CONST:
    case MIR_ADD: case MIR_SUB: case MIR_MUL: case MIR_DIV: case MIR_MOD:
    case MIR_AND: case MIR_OR: case MIR_XOR:
    case MIR_SHL: case MIR_SHR:
    case MIR_NEG: case MIR_NOT:
    case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE: case MIR_GT: case MIR_GE:
    case MIR_ULT: case MIR_ULE: case MIR_UGT: case MIR_UGE:
    case MIR_MOV:
    case MIR_LOAD:    /* dst = mem[addr] */
    case MIR_ALLOC:   /* dst = base address of a fresh cell */
    case MIR_FADD: case MIR_FSUB: case MIR_FMUL: case MIR_FDIV:
    case MIR_FNEG:
    case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE:
    case MIR_ITOF: case MIR_FTOI:
    case MIR_DADD: case MIR_DSUB: case MIR_DMUL: case MIR_DDIV:
    case MIR_DNEG: case MIR_DITOF: case MIR_DTOI:
    case MIR_F32_TO_F64: case MIR_F64_TO_F32:
    case MIR_BF16_TO_F32: case MIR_F32_TO_BF16:
        return 1;
    default:
        return 0;
    }
}

static int op_num_srcs(wubu_mir_op_t op)
{
    switch (op) {
    case MIR_CONST:
    case MIR_LABEL:
    case MIR_JMP:
    case MIR_ALLOC:
        return 0;
    case MIR_NEG: case MIR_NOT:
    case MIR_MOV:
    case MIR_LOAD:    /* single source: the address vr */
    case MIR_JZ:
    case MIR_RET:
        return 1;
    default:
        return 2; /* binops and MIR_STORE (addr, val) */
    }
}

/* ------------------------------------------------------------------ */
/* Interval (live range) for a single virtual register               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t vr;
    int32_t  start;   /* first_def position */
    int32_t  end;     /* last_use position */
} interval_t;

static int interval_cmp(const void *a, const void *b)
{
    const interval_t *ia = (const interval_t *)a;
    const interval_t *ib = (const interval_t *)b;
    if (ia->start < ib->start) return -1;
    if (ia->start > ib->start) return 1;
    if (ia->end < ib->end) return -1;
    if (ia->end > ib->end) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Allocator                                                           */
/* ------------------------------------------------------------------ */

wubu_reg_assign_t *wubu_mir_alloc_regs(const wubu_mir_prog_t *p,
                                        int n_phys_regs,
                                        size_t *out_count)
{
    if (!p || n_phys_regs <= 0) {
        if (out_count) *out_count = 0;
        return NULL;
    }

    size_t n_ins = p->n;

    /* ---- Step 0: find max vr in the program ---- */
    uint32_t max_vr = 0;
    for (size_t i = 0; i < n_ins; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (op_has_dst(in->op) && in->dst > max_vr) max_vr = in->dst;
        int ns = op_num_srcs(in->op);
        if (ns >= 1 && in->a > max_vr) max_vr = in->a;
        if (ns >= 2 && in->b > max_vr) max_vr = in->b;
    }
    /* Always include v0 in the result array (return register) */
    if (max_vr < 1) max_vr = 0;

    size_t n_vr = (size_t)max_vr + 1;

    /* ---- Step 1: compute live ranges [first_def, last_use] ---- */
    int32_t *first_def = (int32_t *)malloc(n_vr * sizeof(int32_t));
    int32_t *last_use  = (int32_t *)malloc(n_vr * sizeof(int32_t));
    if (!first_def || !last_use) {
        free(first_def);
        free(last_use);
        if (out_count) *out_count = 0;
        return NULL;
    }

    for (size_t v = 0; v < n_vr; v++) {
        first_def[v] = -1;
        last_use[v]  = -1;
    }

    for (size_t i = 0; i < n_ins; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (op_has_dst(in->op)) {
            if (first_def[in->dst] < 0)
                first_def[in->dst] = (int32_t)i;
            last_use[in->dst] = (int32_t)i;
        }
        /* MIR_T_GEMM reads dst (C base address) AND writes to it —
         * dst must stay live through the T_GEMM instruction. */
        if (in->op == MIR_T_GEMM) {
            if (first_def[in->dst] < 0)
                first_def[in->dst] = (int32_t)i;
            last_use[in->dst] = (int32_t)i;
        }
        int ns = op_num_srcs(in->op);
        if (ns >= 1) last_use[in->a] = (int32_t)i;
        if (ns >= 2) last_use[in->b] = (int32_t)i;
    }

    /* ---- Step 1b: extend live ranges across loop back-edges ---- */
    /* For each label, find its position. For each JMP/JZ/JNZ that jumps
     * backward to a label, extend the last_use of all VRs used between
     * the label and the branch to the branch position. This ensures
     * loop-carried VRs stay live throughout the loop body. */
    /* Build label -> position map */
    int32_t *label_pos = (int32_t *)malloc((p->n_labels > 0 ? p->n_labels : 1) * sizeof(int32_t));
    if (label_pos) {
        for (uint32_t l = 0; l < p->n_labels; l++) label_pos[l] = -1;
        for (size_t i = 0; i < n_ins; i++) {
            if (p->ins[i].op == MIR_LABEL && p->ins[i].label < p->n_labels)
                label_pos[p->ins[i].label] = (int32_t)i;
        }
        /* For each backward branch, extend live ranges */
        for (size_t i = 0; i < n_ins; i++) {
            const wubu_mir_instr_t *in = &p->ins[i];
            if ((in->op == MIR_JMP || in->op == MIR_JZ || in->op == MIR_JNZ) &&
                in->label < p->n_labels) {
                int32_t target = label_pos[in->label];
                if (target >= 0 && target < (int32_t)i) {
                    /* Backward branch: extend all VRs in [target, i] to i */
                    for (size_t j = (size_t)target; j <= i; j++) {
                        const wubu_mir_instr_t *jn = &p->ins[j];
                        if (op_has_dst(jn->op) && jn->dst < n_vr)
                            if (last_use[jn->dst] < (int32_t)i) last_use[jn->dst] = (int32_t)i;
                        int ns2 = op_num_srcs(jn->op);
                        if (ns2 >= 1 && jn->a < n_vr)
                            if (last_use[jn->a] < (int32_t)i) last_use[jn->a] = (int32_t)i;
                        if (ns2 >= 2 && jn->b < n_vr)
                            if (last_use[jn->b] < (int32_t)i) last_use[jn->b] = (int32_t)i;
                    }
                }
            }
        }
        free(label_pos);
    }

    /* ---- Step 1c: extend live ranges across function calls ---- */
    /* For each CALL, any VR that is used both before and after the call
     * must survive the call (be in a callee-saved register or spilled).
     * Extend last_use of such VRs to the instruction after the call.
     * This prevents the allocator from assigning caller-saved registers
     * to VRs that hold global variable addresses across function calls. */
    for (size_t i = 0; i < n_ins; i++) {
        if (p->ins[i].op != MIR_CALL) continue;
        for (size_t v = 0; v < n_vr; v++) {
            if (first_def[v] < 0 || first_def[v] >= (int32_t)i) continue;
            if (last_use[v] > (int32_t)i) {
                last_use[v] = (int32_t)(i + 1) > last_use[v] ? (int32_t)(i + 1) : last_use[v];
            }
        }
    }

    /* ---- Step 1d: extend live ranges across function boundaries ---- */
    /* The MIR layout emits function bodies BEFORE the caller code (the entry
     * point calls main, then main's body is emitted after all other function
     * bodies). This means a VR written by the caller (e.g. v1 = arg) has its
     * first_def INSIDE main's body (high instruction index) but its last_use
     * INSIDE the callee's body (low instruction index). The linear scan would
     * see last_use < first_def and treat the VR as dead.
     *
     * Fix: for each function body, find VRs that are READ inside the body
     * and WRITTEN outside the body. Extend their live range to span from the
     * earliest write to the latest read. This ensures args (v1..vN) stay live
     * from the caller's setup through the callee's body. */
    for (int f = 0; f < p->n_funcs; f++) {
        uint32_t fstart = p->funcs[f].start;
        uint32_t fend = p->funcs[f].end;
        if (fend <= fstart) continue;
        for (size_t v = 0; v < n_vr; v++) {
            if (first_def[v] < 0) continue;
            /* Is this VR read inside the function body? */
            int32_t read_inside = -1;
            for (uint32_t i = fstart; i < fend; i++) {
                const wubu_mir_instr_t *in = &p->ins[i];
                int ns = op_num_srcs(in->op);
                if (ns >= 1 && in->a == (wubu_vr_t)v) { read_inside = (int32_t)i; break; }
                if (ns >= 2 && in->b == (wubu_vr_t)v) { read_inside = (int32_t)i; break; }
                if (in->op == MIR_STORE && in->b == (wubu_vr_t)v) { read_inside = (int32_t)i; break; }
            }
            if (read_inside < 0) continue;
            /* Is this VR written OUTSIDE the function body? */
            if (first_def[v] < (int32_t)fstart || first_def[v] >= (int32_t)fend) {
                /* Extend live range to cover both the write and the read */
                if (first_def[v] < read_inside) {
                    /* Normal case: write before read — extend last_use */
                    if (last_use[v] < read_inside) last_use[v] = read_inside;
                } else {
                    /* Reverse case: read before write (callee body before caller body)
                     * Extend first_def to before the read, and last_use to after the write */
                    first_def[v] = (int32_t)fstart;  /* live from start of callee */
                    if (last_use[v] < first_def[v]) last_use[v] = first_def[v];
                    /* Also extend to cover the actual write position */
                    if (last_use[v] < (int32_t)first_def[v] + 1) last_use[v] = (int32_t)first_def[v] + 1;
                }
            }
        }
    }

    /* ---- Step 2: build sorted interval list ---- */
    int n_intervals = 0;
    for (size_t v = 0; v < n_vr; v++)
        if (first_def[v] >= 0) n_intervals++;

    interval_t *intervals = (interval_t *)malloc(
        (n_intervals > 0 ? n_intervals : 1) * sizeof(interval_t));
    if (!intervals) {
        free(first_def);
        free(last_use);
        if (out_count) *out_count = 0;
        return NULL;
    }

    int idx = 0;
    for (size_t v = 0; v < n_vr; v++) {
        if (first_def[v] >= 0) {
            intervals[idx].vr    = (uint32_t)v;
            intervals[idx].start = first_def[v];
            intervals[idx].end   = (last_use[v] >= 0) ? last_use[v] : first_def[v];
            idx++;
        }
    }

    qsort(intervals, n_intervals, sizeof(interval_t), interval_cmp);

    /* ---- Step 3: allocate and initialize result ---- */
    wubu_reg_assign_t *assign = (wubu_reg_assign_t *)calloc(n_vr, sizeof(wubu_reg_assign_t));
    if (!assign) {
        free(intervals);
        free(first_def);
        free(last_use);
        if (out_count) *out_count = 0;
        return NULL;
    }
    for (size_t v = 0; v < n_vr; v++) assign[v].reg = -1;
    for (size_t v = 0; v < n_vr; v++) {
        assign[v].reg   = -1;
        assign[v].stack = 0;
        assign[v].spill_after = 0;
        assign[v].split_until = 0;
    }

    /* ---- Step 4: pre-assign argument registers ----
     * v0 is the implicit return register and is ALWAYS physical reg 0.
     * Argument vrs v1..n_args are pre-assigned to physical regs 1..n_args
     * (capped at 6 arg regs, plus v0 makes 7). */
    uint32_t n_args = p->n_args;
    if (n_args > 6) n_args = 6;

    if (getenv("WUBU_DEBUG_REGS")) {
        fprintf(stderr, "[DEBUG_REGS] n_vr=%u n_intervals=%d n_phys_regs=%d n_args=%u\n",
                n_vr, n_intervals, n_phys_regs, n_args);
        for (int i = 0; i < n_intervals && i < 15; i++) {
            fprintf(stderr, "[DEBUG_REGS] interval[%d] vr=%u start=%d end=%d\n",
                    i, intervals[i].vr, intervals[i].start, intervals[i].end);
        }
    }
    /* v0 -> reg 0 always (return register). Args -> regs 1..n_args. */
    if (n_vr > 0) {
        assign[0].reg = 0;
        assign[0].stack = 0;
    }
    for (uint32_t a = 1; a <= n_args && a < n_vr; a++) {
        int32_t phys = (int32_t)a;  /* v1 -> reg 1, v2 -> reg 2, ... */
        if (phys < n_phys_regs) {
            assign[a].reg = phys;
        }
    }

    /* ---- Step 5: linear scan ---- */
    /* reg_vr[r] = vr currently in physical register r, or -1 */
    int32_t *reg_vr = (int32_t *)malloc(n_phys_regs * sizeof(int32_t));
    /* active[]: list of vrs currently holding a physical register */
    uint32_t *active = (uint32_t *)malloc((n_intervals + 1) * sizeof(uint32_t));
    int32_t next_spill_slot = 0;
    int active_count = 0;

    if (!reg_vr || !active) {
        free(reg_vr);
        free(active);
        free(assign);
        free(intervals);
        free(first_def);
        free(last_use);
        if (out_count) *out_count = 0;
        return NULL;
    }

    for (int r = 0; r < n_phys_regs; r++)
        reg_vr[r] = -1;

    /* Reserve physical registers for pre-assigned vrs (v0, args).
     * v0 is the implicit return register (always physical reg 0).
     * arg vrs v1..n_args are pre-assigned to regs 1..n_args.
     * These registers are occupied but NOT seeded into the active set —
     * if a pre-assigned VR doesn't appear in any MIR instruction its
     * first_def/last_use would be -1, causing the expire logic to
     * immediately drop it and free the register for other VRs.
     * Instead, just mark the physical registers as occupied. */
    reg_vr[0] = (int32_t)0;  /* v0 always in physical reg 0 */
    for (uint32_t a = 1; a <= n_args && a < n_vr; a++) {
        int32_t phys = (int32_t)a;  /* v1 -> reg 1, v2 -> reg 2, ... */
        if (phys >= 0 && phys < n_phys_regs) {
            reg_vr[phys] = (int32_t)a;  /* mark as occupied */
        }
    }

    for (int i = 0; i < n_intervals; i++) {
        uint32_t vr  = intervals[i].vr;
        int32_t  pos = intervals[i].start;

        /* Skip pre-assigned vrs (v0 return register and args already placed above) */
        if (vr == 0 || (vr >= 1 && vr <= n_args)) continue;

        /* Expire active intervals whose last_use < pos */
        int write = 0;
        for (int j = 0; j < active_count; j++) {
            uint32_t av = active[j];
            int32_t aend = (last_use[av] >= 0) ? last_use[av] : first_def[av];
            if (aend < pos) {
                int32_t ar = assign[av].reg;
                if (ar >= 0 && ar < n_phys_regs)
                    reg_vr[ar] = -1;
                continue;  /* drop from active */
            }
            active[write++] = av;
        }
        active_count = write;

        /* Find a free physical register */
        int32_t chosen = -1;
        if (getenv("WUBU_DEBUG_REGS")) {
            fprintf(stderr, "[DEBUG_REGS] allocating vr=%u pos=%d active_count=%d reg_vr=", vr, pos, active_count);
            for (int r = 0; r < n_phys_regs; r++) fprintf(stderr, "%d ", reg_vr[r]);
            fprintf(stderr, "\n");
        }
        for (int r = 0; r < n_phys_regs; r++) {
            if (reg_vr[r] < 0) {
                chosen = r;
                break;
            }
        }

        if (chosen >= 0) {
            /* Assign register */
            assign[vr].reg   = chosen;
            assign[vr].stack = 0;
            reg_vr[chosen]   = (int32_t)vr;
            active[active_count++] = vr;
        } else {
            /* Pool exhausted — find the active interval with the furthest end */
            int32_t victim_vr = -1;
            int32_t victim_end = -1;
            int victim_idx = -1;
            for (int j = 0; j < active_count; j++) {
                uint32_t av = active[j];
                int32_t ae = (last_use[av] >= 0) ? last_use[av] : first_def[av];
                if (victim_vr < 0 || ae > victim_end) {
                    victim_end = ae;
                    victim_vr = (int32_t)av;
                    victim_idx = j;
                }
            }

            int32_t cur_end = intervals[i].end;

            if (0 && victim_vr >= 0 && victim_end > cur_end) {
                /* ---- Interval SPLITTING (LLVM Greedy-style) ----
                 * The current interval is shorter than the victim's remaining
                 * lifetime. Split the victim at the current position: its
                 * register copy now covers only [victim_first_def, pos), past
                 * `pos` the value is reloaded from a spill slot that lives
                 * until the victim's true end. The current interval takes the
                 * freed register for [pos, cur_end]. This avoids a memory
                 * access for the current value across its whole definition. */
                int32_t vreg = assign[victim_vr].reg;
                int32_t v_first_def = (first_def[victim_vr] >= 0) ? first_def[victim_vr] : 0;
                /* victim keeps its stack slot for the tail fragment */
                int32_t tail_slot = -(next_spill_slot + 1) * 8;
                next_spill_slot++;
                assign[victim_vr].stack = tail_slot;
                assign[victim_vr].spill_after = pos;
                assign[victim_vr].split_until = victim_end + 1;

                assign[vr].reg   = vreg;
                assign[vr].stack = tail_slot;   /* tail reload slot is its own */
                assign[vr].spill_after = cur_end + 1;
                assign[vr].split_until = 0;
                reg_vr[vreg] = (int32_t)vr;
                active[victim_idx] = vr;
            } else {
                /* Spill current vr (shorter than all actives) */
                assign[vr].reg   = -1;
                assign[vr].stack = -(next_spill_slot + 1) * 8;
                assign[vr].spill_after = 0;
                assign[vr].split_until = 0;
                next_spill_slot++;
            }
        }
    }

    free(active);
    free(reg_vr);
    free(intervals);
    free(first_def);
    free(last_use);

    if (out_count) *out_count = n_vr;
    if (getenv("WUBU_DEBUG_REGS")) {
        for (size_t v = 0; v < n_vr && v < 15; v++) {
            fprintf(stderr, "[DEBUG_REGS] assign[%zu] reg=%d stack=%d spill_after=%u\n",
                    v, assign[v].reg, assign[v].stack, assign[v].spill_after);
        }
    }
    return assign;
}

void wubu_mir_free_alloc(wubu_reg_assign_t *alloc)
{
    free(alloc);
}