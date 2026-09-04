/*
 * wubu_mir_opt.c -- MIR optimizer passes (the optimizer compiler core).
 *
 * Five classical passes over the hourglass-neck IR:
 *   1. FOLD    -- constant folding (compile-time eval of binops on constants)
 *   2. STRENGTH -- strength reduction (mul/div -> shift, *1/+0/-0/*0 elim)
 *   3. DCE     -- dead code elimination (remove unused result vrs)
 *   4. LICM    -- loop-invariant code motion (hoist pure computations)
 *   5. UNROLL  -- loop unrolling (small constant trip counts)
 *
 * Each pass is a bit in mir_opt_flags_t; wubu_mir_optimize runs the
 * requested passes in canonical order regardless of flag order.
 *
 * C11, self-contained.
 */
#include "wubu_mir_opt.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* MIR integer semantics are 32-bit (the canonical HolyD `int` width the
 * x86-64 JIT golden and the portable interpreter agree on). Folding must
 * wrap to 32 bits to match runtime execution, or it will disagree with the
 * golden reference. */
#define WRAP32(x) ((int64_t)(int32_t)(x))

/* The frontend allocates virtual registers starting at 1<<16 (and emits
 * fresh vrs as instr_index+1 as well), so vr numbers can be far larger than
 * any small fixed table. Compute the TRUE max vr so optimizer passes size
 * their tracking arrays correctly and never index out of bounds. */
static uint32_t mir_max_vr(const wubu_mir_prog_t *p)
{
    uint32_t m = 0;
    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (in->dst > m) m = in->dst;
        if (in->a > m)   m = in->a;
        if (in->b > m)   m = in->b;
    }
    return m + 1;
}

/* ---- Pass 1: Constant Folding ---- */
static void fold_pass(wubu_mir_prog_t *p)
{
    /* Build a map: vr -> constant value (if known). vrs are NOT bounded by
     * the instruction count (frontend uses a 1<<16 base), so size by max_vr. */
    uint32_t nvr = mir_max_vr(p);
    int64_t *const_val = (int64_t *)calloc(nvr ? nvr : 1, sizeof(int64_t));
    int *is_const = (int *)calloc(nvr ? nvr : 1, sizeof(int));
    if (!const_val || !is_const) { free(const_val); free(is_const); return; }

    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_CONST && in->dst < nvr) {
            const_val[in->dst] = in->imm;
            is_const[in->dst] = 1;
        } else if (in->op != MIR_LABEL && in->op != MIR_RET &&
                   in->op != MIR_JMP && in->op != MIR_JZ &&
                   in->dst < nvr) {
            is_const[in->dst] = 0;
        }
    }

    /* Second pass: fold binops where both operands are constants */
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL || in->op == MIR_JMP ||
            in->op == MIR_JZ || in->op == MIR_RET ||
            in->op == MIR_CONST)
            continue;

        /* Unary ops: check operand 'a' */
        /* NOTE: MIR_NEG is NOT folded here because it may be applied to
         * float/double bits (f64 stored as int64_t raw bits). Integer
         * negation of f64 bits is NOT the same as float negation (which
         * only flips the sign bit). The JIT handles MIR_NEG correctly
         * for both int and float. */
        if (in->op == MIR_NOT || in->op == MIR_MOV) {
            if (in->a < nvr && is_const[in->a]) {
                int64_t result;
                switch (in->op) {
                case MIR_NOT: result = ~const_val[in->a]; break;
                case MIR_MOV: result = const_val[in->a]; break;
                default: result = const_val[in->a]; break;
                }
                in->op = MIR_CONST;
                in->imm = result;
                in->a = 0; in->b = 0;
                if (in->dst < nvr) {
                    const_val[in->dst] = result;
                    is_const[in->dst] = 1;
                }
            }
            continue;
        }

        /* Binary ops: check both operands (wrap to 32-bit MIR semantics) */
        if (in->a < nvr && in->b < nvr &&
            is_const[in->a] && is_const[in->b]) {
            int64_t a = const_val[in->a];
            int64_t b = const_val[in->b];
            int64_t result = 0;
            switch (in->op) {
            case MIR_ADD: result = WRAP32(a + b); break;
            case MIR_SUB: result = WRAP32(a - b); break;
            case MIR_MUL: result = WRAP32(a * b); break;
            case MIR_DIV: result = b != 0 ? WRAP32(a / b) : 0; break;
            case MIR_MOD: result = b != 0 ? WRAP32(a % b) : 0; break;
            case MIR_AND: result = WRAP32(a & b); break;
            case MIR_OR:  result = WRAP32(a | b); break;
            case MIR_XOR: result = WRAP32(a ^ b); break;
            case MIR_SHL: result = WRAP32((int64_t)((uint32_t)a << (b & 31))); break;
            case MIR_SHR: result = WRAP32((int64_t)((int32_t)a >> (b & 31))); break;
            case MIR_EQ:  result = (a == b) ? 1 : 0; break;
            case MIR_NE:  result = (a != b) ? 1 : 0; break;
            case MIR_LT:  result = (a < b) ? 1 : 0; break;
            case MIR_LE:  result = (a <= b) ? 1 : 0; break;
            case MIR_GT:  result = (a > b) ? 1 : 0; break;
            case MIR_GE:  result = (a >= b) ? 1 : 0; break;
            case MIR_ULT: result = ((uint32_t)a < (uint32_t)b) ? 1 : 0; break;
            case MIR_ULE: result = ((uint32_t)a <= (uint32_t)b) ? 1 : 0; break;
            case MIR_UGT: result = ((uint32_t)a > (uint32_t)b) ? 1 : 0; break;
            case MIR_UGE: result = ((uint32_t)a >= (uint32_t)b) ? 1 : 0; break;
            default: continue;
            }
            in->op = MIR_CONST;
            in->imm = result;
            in->a = 0; in->b = 0;
            if (in->dst < nvr) {
                const_val[in->dst] = result;
                is_const[in->dst] = 1;
            }
        }
    }
    free(const_val);
    free(is_const);
}

/* ---- Pass 2: Strength Reduction ---- */
/*
 * Build a vr -> constant value table (vr numbers are NOT instruction
 * indices in this IR: wubu_mir_const/binop/load assign dst = instr_index+1,
 * but wubu_mir_mov_to may reuse a vr for a later instr, and the frontend
 * uses very high vr bases like (1<<16) for generated temporaries — so we
 * must map by scanning for CONST definitions, never by indexing ins[vr-1]).
 */
static void strength_pass(wubu_mir_prog_t *p)
{
    uint32_t nvr = mir_max_vr(p);
    int64_t *cval = (int64_t *)calloc(nvr ? nvr : 1, sizeof(int64_t));
    int *is_c = (int *)calloc(nvr ? nvr : 1, sizeof(int));
    int *def_count = (int *)calloc(nvr ? nvr : 1, sizeof(int));
    if (!cval || !is_c || !def_count) { free(cval); free(is_c); free(def_count); return; }

    /* Count definitions per vr. A vr defined more than once (e.g. a loop
     * variable with phi-merge) is NOT a single constant. */
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *d = &p->ins[i];
        if (d->op == MIR_LABEL || d->op == MIR_JMP || d->op == MIR_JZ ||
            d->op == MIR_JNZ || d->op == MIR_BREAK || d->op == MIR_CONTINUE ||
            d->op == MIR_RET)
            continue;
        if (d->dst < nvr) def_count[d->dst]++;
    }

    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *d = &p->ins[i];
        if (d->op == MIR_CONST && d->dst < nvr && def_count[d->dst] <= 1) {
            cval[d->dst] = d->imm; is_c[d->dst] = 1;
        }
    }
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL || in->op == MIR_JMP ||
            in->op == MIR_JZ || in->op == MIR_RET ||
            in->op == MIR_CONST)
            continue;

        int64_t bv = (in->b < nvr && is_c[in->b]) ? cval[in->b] : INT64_MIN;

        /* x + 0 -> x ; x - 0 -> x ; x | 0 -> x ; x ^ 0 -> x */
        if ((in->op == MIR_ADD || in->op == MIR_SUB ||
             in->op == MIR_OR  || in->op == MIR_XOR) && bv == 0) {
            in->op = MIR_MOV; in->b = 0;
        }
        /* x * 0 -> 0 ; x & 0 -> 0 */
        else if ((in->op == MIR_MUL || in->op == MIR_AND) && bv == 0) {
            in->op = MIR_CONST; in->imm = 0; in->a = 0; in->b = 0;
        }
        /* x * 1 -> x ; x / 1 -> x ; x % 1 -> 0 ; x << 0 -> x ; x >> 0 -> x */
        else if (in->op == MIR_MUL && bv == 1) { in->op = MIR_MOV; in->b = 0; }
        else if ((in->op == MIR_DIV || in->op == MIR_SHR || in->op == MIR_SHL) && bv == 0) {
            in->op = MIR_MOV; in->b = 0;
        }
        else if (in->op == MIR_MOD && bv == 1) { in->op = MIR_CONST; in->imm = 0; in->a = 0; in->b = 0; }
        /* x * -1 -> -x (NEG) */
        else if (in->op == MIR_MUL && bv == -1) {
            in->op = MIR_NEG; in->b = 0;
        }
        /* NOTE: MUL->SHL and DIV->SHR strength reduction intentionally disabled.
         * MIR_SHL/MIR_SHR use VR indices for the shift amount (vr[b]), NOT
         * immediate values. The peephole cannot insert a CONST instruction
         * to materialize k, so setting in->b = k makes vr[b] read garbage
         * (e.g. vr[1] = argument value, not shift amount). Keep MUL/DIV. */
    }
}

/* ---- Pass 3: Dead Code Elimination ---- */
/*
 * Mark all vrs that are "used" (read by some instruction, or the RET value).
 * Then remove instructions whose dst is never used by converting them to
 * CONST 0 (harmless value-producing ops only).
 *
 * CRITICAL: MIR_ALLOC / MIR_LOAD / MIR_STORE carry memory side effects and
 * must NEVER be deleted, even if their result vr is unused. Dropping an
 * ALLOC would shrink the flat memory array and corrupt every later address;
 * dropping a STORE would silently lose a write; dropping a LOAD would change
 * the result of a later aliased load. So we skip them in both the "used"
 * marking (they have no value dst anyway) and the deletion loop.
 */
static void dce_pass(wubu_mir_prog_t *p)
{
    uint32_t nvr = mir_max_vr(p);
    int *used = (int *)calloc(nvr ? nvr : 1, sizeof(int));
    if (!used) return;

    /* Mark uses */
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL) continue;
        if (in->op == MIR_RET) {
            if (in->a < nvr) used[in->a] = 1;
            continue;
        }
        if (in->op == MIR_JZ || in->op == MIR_JNZ) {
            if (in->a < nvr) used[in->a] = 1;
            continue;
        }
        if (in->op == MIR_CONST || in->op == MIR_JMP ||
            in->op == MIR_LABEL) continue;
        /* STORE mem[a] = b : reads BOTH a (addr) and b (value) as inputs.
         * These must be treated as live so DCE never deletes the
         * computation feeding the stored value (e.g. loop counter i=i+1). */
        if (in->op == MIR_STORE) {
            if (in->a < nvr) used[in->a] = 1;
            if (in->b < nvr) used[in->b] = 1;
            continue;
        }
        /* MIR_ALLOC has no inputs (it is a base-address source); it must
         * survive (handled by the deletion guard) but marks nothing.
         * MIR_LOAD reads its address operand `a` — that address computation
         * must be kept alive, or DCE deletes the instruction feeding it. */
        if (in->op == MIR_ALLOC) continue;
        if (in->op == MIR_LOAD) {
            if (in->a < nvr) used[in->a] = 1;
            continue;
        }
        /* MIR_CALL: the call reads v1..v_nargs as argument VRs. Without this,
         * DCE would treat MIR_MOV v1, arg as dead and remove it, leaving
         * the callee to read garbage from v1. Conservatively mark the first
         * 32 VRs live — the call ABI in this compiler passes args in v1..vN
         * (N ≤ MIR_MAX_CALL_ARGS). Marking more than needed is safe; marking
         * fewer would be wrong. */
        if (in->op == MIR_CALL) {
            for (uint32_t v = 1; v <= 32 && v < nvr; v++) used[v] = 1;
            continue;
        }
        /* All other ops read a (and possibly b) */
        if (in->a < nvr) used[in->a] = 1;
        if (in->b < nvr) used[in->b] = 1;
    }

    /* Remove dead value-producing instructions: zero out their dst.
     * Never touch ALLOC / LOAD / STORE (memory side effects). */
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL || in->op == MIR_JMP ||
            in->op == MIR_JZ || in->op == MIR_JNZ ||
            in->op == MIR_BREAK || in->op == MIR_CONTINUE ||
            in->op == MIR_RET ||
            in->op == MIR_CONST || in->op == MIR_ALLOC ||
            in->op == MIR_LOAD || in->op == MIR_STORE)
            continue;
        if (in->dst < nvr && !used[in->dst]) {
            in->op = MIR_CONST;
            in->imm = 0;
            in->a = 0;
            in->b = 0;
        }
    }
    free(used);
}

/* ---- Combined fold+dce to fixpoint ---- */
static void fold_dce_pass(wubu_mir_prog_t *p)
{
    /* Iterate fold + DCE until no more changes */
    int changed = 1;
    int iterations = 0;
    while (changed && iterations < 10) {
        changed = 0;
        iterations++;

        /* Track constants (dynamic vr sizing; wrap to 32-bit) */
        uint32_t nvr = mir_max_vr(p);
        int64_t *const_val = (int64_t *)calloc(nvr ? nvr : 1, sizeof(int64_t));
        int *is_const = (int *)calloc(nvr ? nvr : 1, sizeof(int));
        int *def_count = (int *)calloc(nvr ? nvr : 1, sizeof(int));
        if (!const_val || !is_const || !def_count) {
            free(const_val); free(is_const); free(def_count); return;
        }

        /* Count definitions per vr. A vr defined more than once (e.g. a
         * phi-merge written from two branches) is NOT a single constant and
         * must never be folded, or we pick one (wrong) arm's value. */
        for (size_t i = 0; i < p->n; i++) {
            wubu_mir_instr_t *in = &p->ins[i];
            if (in->op == MIR_LABEL || in->op == MIR_JMP || in->op == MIR_RET ||
                in->op == MIR_JZ || in->op == MIR_JNZ || in->op == MIR_BREAK ||
                in->op == MIR_CONTINUE)
                continue;
            if (in->dst < nvr) def_count[in->dst]++;
        }

        for (size_t i = 0; i < p->n; i++) {
            wubu_mir_instr_t *in = &p->ins[i];
            if (in->op == MIR_CONST && in->dst < nvr) {
                const_val[in->dst] = in->imm;
                is_const[in->dst] = 1;
            } else if (in->op != MIR_LABEL && in->op != MIR_RET &&
                       in->op != MIR_JMP && in->op != MIR_JZ &&
                       in->op != MIR_JNZ && in->op != MIR_BREAK &&
                       in->op != MIR_CONTINUE && in->dst < nvr) {
                /* A multi-defined vr (phi) can never be treated as a constant. */
                is_const[in->dst] = 0;
            }
        }

        /* Fold */
        for (size_t i = 0; i < p->n; i++) {
            wubu_mir_instr_t *in = &p->ins[i];
            if (in->op == MIR_LABEL || in->op == MIR_JMP ||
                in->op == MIR_JZ || in->op == MIR_JNZ || in->op == MIR_BREAK ||
                in->op == MIR_CONTINUE || in->op == MIR_RET ||
                in->op == MIR_CONST)
                continue;

            /* Never fold an instruction whose destination vr is multiply
             * defined (phi merge): doing so would overwrite the other arm. */
            if (in->dst < nvr && def_count[in->dst] > 1) continue;

            /* Skip MIR_NEG for float constants — integer negation of f64 bits
             * is not the same as float negation. The JIT handles it correctly. */
            if (in->op == MIR_NOT || in->op == MIR_MOV) {
                if (in->a < nvr && is_const[in->a] && def_count[in->a] <= 1) {
                    int64_t result;
                    switch (in->op) {
                    case MIR_NOT: result = ~const_val[in->a]; break;
                    case MIR_MOV: result = const_val[in->a]; break;
                    default: result = const_val[in->a]; break;
                    }
                    in->op = MIR_CONST;
                    in->imm = result;
                    in->a = 0; in->b = 0;
                    const_val[in->dst] = result;
                    is_const[in->dst] = 1;
                    changed = 1;
                }
                continue;
            }

            if (in->a < nvr && in->b < nvr &&
                is_const[in->a] && is_const[in->b] &&
                def_count[in->a] <= 1 && def_count[in->b] <= 1) {
                int64_t a = const_val[in->a];
                int64_t b = const_val[in->b];
                int64_t result = 0;
                switch (in->op) {
                case MIR_ADD: result = WRAP32(a + b); break;
                case MIR_SUB: result = WRAP32(a - b); break;
                case MIR_MUL: result = WRAP32(a * b); break;
                case MIR_DIV: result = b != 0 ? WRAP32(a / b) : 0; break;
                case MIR_MOD: result = b != 0 ? WRAP32(a % b) : 0; break;
                case MIR_AND: result = WRAP32(a & b); break;
                case MIR_OR:  result = WRAP32(a | b); break;
                case MIR_XOR: result = WRAP32(a ^ b); break;
                case MIR_SHL: result = WRAP32((int64_t)((uint32_t)a << (b & 31))); break;
                case MIR_SHR: result = WRAP32((int64_t)((int32_t)a >> (b & 31))); break;
                case MIR_EQ:  result = (a == b) ? 1 : 0; break;
                case MIR_NE:  result = (a != b) ? 1 : 0; break;
                case MIR_LT:  result = (a < b) ? 1 : 0; break;
                case MIR_LE:  result = (a <= b) ? 1 : 0; break;
                case MIR_GT:  result = (a > b) ? 1 : 0; break;
                case MIR_GE:  result = (a >= b) ? 1 : 0; break;
                case MIR_ULT: result = ((uint32_t)a < (uint32_t)b) ? 1 : 0; break;
                case MIR_ULE: result = ((uint32_t)a <= (uint32_t)b) ? 1 : 0; break;
                case MIR_UGT: result = ((uint32_t)a > (uint32_t)b) ? 1 : 0; break;
                case MIR_UGE: result = ((uint32_t)a >= (uint32_t)b) ? 1 : 0; break;
                default: continue;
                }
                in->op = MIR_CONST;
                in->imm = result;
                in->a = 0; in->b = 0;
                const_val[in->dst] = result;
                is_const[in->dst] = 1;
                changed = 1;
            }
        }
        free(const_val);
        free(is_const);
        free(def_count);
    }
}

/* ---- Pass 4: Loop-Invariant Code Motion ---- */
static void licm_pass(wubu_mir_prog_t *p)
{
    /* SSA-form MIR makes LICM complex (no phi nodes for loop-variant detection).
     * Placeholder: requires SSA reconstruction for proper analysis. */
    (void)p;
}

/* ---- Pass 5: Loop Unrolling ---- */
static void unroll_pass(wubu_mir_prog_t *p)
{
    /* SSA-form MIR makes loop unrolling complex (no phi nodes).
     * Placeholder: requires loop analysis with SSA reconstruction. */
    (void)p;
}

/* ---- Pass 6: Instruction Combining ----
 * Algebraic identities for SSA-form MIR:
 *   x - x = 0,  x ^ x = 0   (self-sub/xor → zero)
 *   x & x = x,  x | x = x   (self-and/or → identity)
 *   x + x = x * 2           (self-add → double, strength handles)
 * These are safe: when both operands are the same vr, the result
 * depends only on that vr's value, so we can replace the instruction.
 */
static void combine_pass(wubu_mir_prog_t *p)
{
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL) continue;
        if (in->a != in->b) continue; /* only when both operands are same vr */

        switch (in->op) {
        case MIR_SUB:
        case MIR_XOR:
            /* x - x = 0, x ^ x = 0 */
            in->op = MIR_CONST;
            in->imm = 0;
            in->a = 0;
            in->b = 0;
            break;
        case MIR_AND:
        case MIR_OR:
            /* x & x = x, x | x = x → MOV dst = a */
            in->op = MIR_MOV;
            in->a = in->a;
            in->b = 0;
            break;
        case MIR_ADD:
            /* x + x = x * 2 — would need new CONST, skip for now */
            break;
        case MIR_MUL:
            /* x * x = x^2 — no simple reduction */
            break;
        case MIR_DIV:
            /* x / x = 1 (if x != 0) — unsafe for x=0, skip */
            break;
        default:
            break;
        }
    }
}

/* ---- Pass 7: Common Subexpression Elimination (CSE) ----
 * For each binary operation, check if an identical operation was
 * already computed earlier. If so, replace with MOV dst = earlier_dst.
 * This eliminates redundant computation across all backends.
 *
 * Example:  ADD(v3, v1, v2)  ...  ADD(v4, v1, v2)
 *       →  ADD(v3, v1, v2)  ...  MOV(v4, v3)
 *
 * Only works within the same basic block (no labels between).
 */
static void cse_pass(wubu_mir_prog_t *p)
{
    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL) continue;
        if (in->op != MIR_ADD && in->op != MIR_SUB && in->op != MIR_MUL &&
            in->op != MIR_DIV && in->op != MIR_MOD && in->op != MIR_AND &&
            in->op != MIR_OR && in->op != MIR_XOR && in->op != MIR_SHL &&
            in->op != MIR_SHR) continue;
        if (in->a == in->b) continue; /* handled by combine_pass */

        /* Look backwards for identical operation */
        for (size_t j = i; j > 0; j--) {
            size_t k = j - 1;
            const wubu_mir_instr_t *prev = &p->ins[k];
            if (prev->op == MIR_LABEL) break; /* stop at block boundary */
            if (prev->op != in->op) continue;
            if (prev->a == in->a && prev->b == in->b) {
                /* Found identical computation — replace with MOV */
                in->op = MIR_MOV;
                in->a = prev->dst;
                in->b = 0;
                break;
            }
            /* Also check commutative ops with swapped operands */
            if ((in->op == MIR_ADD || in->op == MIR_MUL ||
                 in->op == MIR_AND || in->op == MIR_OR || in->op == MIR_XOR) &&
                prev->a == in->b && prev->b == in->a) {
                in->op = MIR_MOV;
                in->a = prev->dst;
                in->b = 0;
                break;
            }
        }
    }
}

/* ---- Main optimizer entry point ---- */
void wubu_mir_optimize(wubu_mir_prog_t *p, mir_opt_flags_t flags)
{
    if (flags & MIR_OPT_SSA)     wubu_mir_to_ssa(p);
    if (flags & MIR_OPT_SCCP)    wubu_mir_sccp(p);
    if (flags & MIR_OPT_FOLD)    fold_dce_pass(p);
    if (flags & MIR_OPT_STRENGTH) strength_pass(p);
    if (flags & MIR_OPT_DCE)     dce_pass(p);
    if (flags & MIR_OPT_LICM)    licm_pass(p);
    if (flags & MIR_OPT_UNROLL)  unroll_pass(p);
    if (flags & MIR_OPT_COMBINE) combine_pass(p);
    if (flags & MIR_OPT_CSE)     cse_pass(p);
    if (flags & MIR_OPT_GVN)     wubu_mir_gvn(p);
    if (flags & MIR_OPT_FUSE)    wubu_mir_fuse(p);
}
