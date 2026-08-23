/*
 * wubu_mir_regalloc.h -- linear scan register allocator for MIR programs.
 *
 * Allocates physical registers to virtual registers using the classic
 * linear-scan algorithm on SSA-form intervals. Because MIR is SSA (each
 * vr defined exactly once), live ranges are simple [first_def, last_use]
 * intervals — no complex liveness analysis needed.
 *
 * Pre-assigns:
 *   v0       -> physical reg 0 (return register)
 *   v1..n_args -> physical reg 0..min(n_args-1, n_phys_regs-1) (arg registers)
 *
 * Spill convention: reg = -1 means spilled (always reloaded from stack). If
* reg >= 0 with spill_after > 0, the in-register copy is valid only for
* instruction indices < spill_after; past that the value is reloaded from
* stack. split_until bounds the fragment when an interval is split more
* than once.
 *
 * C11, self-contained.
 */
#ifndef WUBU_MIR_REGALLOC_H
#define WUBU_MIR_REGALLOC_H

#include "wubu_mir.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int32_t reg;      /* physical register index, or -1 if spilled to stack */
    int32_t stack;    /* stack offset (bytes) if spilled, else 0 */
    /* ---- interval splitting (LLVM Greedy-style) ----
     * If reg >= 0 and spill_after > 0: hold `reg` only for instruction
     * indices < spill_after, then the value must be re-loaded from
     * `stack` (a fresh spill slot) past that point. split_until marks
     * where the original live range is expected to resume if it's split
     * multiple times (== end+1 when unsplit). */
    int32_t spill_after;  /* instruction index after which reg no longer holds value */
    int32_t split_until;  /* instruction index to which the split-off fragment lives */
} wubu_reg_assign_t;

/* Allocate registers for a MIR program.
 * n_phys_regs: number of available physical registers (e.g. 14 for x86-64, 8 for m68k)
 * out_count: receives the size of the returned array (indexed by vr, 0..max_vr-1)
 * Returns dynamically allocated assignment array (caller frees), indexed by vr. */
wubu_reg_assign_t *wubu_mir_alloc_regs(const wubu_mir_prog_t *p, int n_phys_regs, size_t *out_count);

/* Free a previously allocated assignment array. */
void wubu_mir_free_alloc(wubu_reg_assign_t *alloc);

#endif /* WUBU_MIR_REGALLOC_H */