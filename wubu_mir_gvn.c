/*
 * wubu_mir_gvn.c -- Global Value Numbering (GVN) for WuBu MIR.
 *
 * Eliminates redundant computations by assigning value numbers to
 * expressions and replacing duplicate expressions with the first result.
 *
 * C11, self-contained.
 */

#include "wubu_mir.h"
#include <stdlib.h>
#include <string.h>

/* ---- Value numbering ---- */

/* Hash table entry for an expression */
typedef struct {
    uint64_t hash;
    uint32_t dst;   /* VR where result is stored */
    uint8_t  op;    /* MIR op */
    uint32_t a, b;  /* operands */
} gvn_expr_t;

/* Hash table */
typedef struct {
    gvn_expr_t *entries;
    size_t n, cap;
} gvn_table_t;

static uint64_t gvn_hash(uint8_t op, uint32_t a, uint32_t b) {
    uint64_t h = (uint64_t)op * 0x9E3779B9u;
    h ^= ((uint64_t)a * 0x517CC1B7u);
    h ^= ((uint64_t)b * 0x9E3779B9u) << 1;
    return h;
}

static void gvn_init(gvn_table_t *t) {
    memset(t, 0, sizeof(*t));
    t->cap = 64;
    t->entries = calloc(t->cap, sizeof(gvn_expr_t));
}

static void gvn_free(gvn_table_t *t) {
    free(t->entries);
    memset(t, 0, sizeof(*t));
}

/* Find or insert expression. Returns 1 if new, 0 if already exists. */
static int gvn_find_or_insert(gvn_table_t *t, uint8_t op, uint32_t a,
                              uint32_t b, uint32_t *out_dst) {
    uint64_t h = gvn_hash(op, a, b);
    size_t idx = h % t->cap;

    for (size_t i = 0; i < t->cap; i++) {
        size_t slot = (idx + i) % t->cap;
        if (t->entries[slot].hash == 0 && t->entries[slot].dst == 0) {
            /* Empty slot: insert */
            if (t->n >= t->cap * 2 / 3) {
                /* Resize */
                size_t old_cap = t->cap;
                gvn_expr_t *old = t->entries;
                t->cap *= 2;
                t->entries = calloc(t->cap, sizeof(gvn_expr_t));
                for (size_t j = 0; j < old_cap; j++) {
                    if (old[j].hash != 0 || old[j].dst != 0) {
                        size_t nidx = old[j].hash % t->cap;
                        for (size_t k = 0; k < t->cap; k++) {
                            size_t s = (nidx + k) % t->cap;
                            if (t->entries[s].hash == 0 && t->entries[s].dst == 0) {
                                t->entries[s] = old[j];
                                break;
                            }
                        }
                    }
                }
                free(old);
                idx = h % t->cap;
                slot = idx;
            }
            t->entries[slot].hash = h;
            t->entries[slot].op = op;
            t->entries[slot].a = a;
            t->entries[slot].b = b;
            t->entries[slot].dst = *out_dst;
            t->n++;
            return 1;
        }
        if (t->entries[slot].hash == h &&
            t->entries[slot].op == op &&
            t->entries[slot].a == a &&
            t->entries[slot].b == b) {
            /* Found: replace with existing dst */
            *out_dst = t->entries[slot].dst;
            return 0;
        }
    }
    return 1;
}

/* ---- GVN pass ---- */

int wubu_mir_gvn(wubu_mir_prog_t *p) {
    if (!p || p->n == 0) return 0;

    gvn_table_t t;
    gvn_init(&t);

    int elim = 0;

    for (size_t i = 0; i < p->n; i++) {
        wubu_mir_instr_t *ins = &p->ins[i];
        uint8_t op = ins->op;

        /* Only GVN pure ops: binops, unary ops, conversions */
        if (op == MIR_CONST || op == MIR_MOV) continue;
        if (op == MIR_ADD || op == MIR_SUB || op == MIR_MUL ||
            op == MIR_DIV || op == MIR_MOD || op == MIR_AND ||
            op == MIR_OR || op == MIR_XOR || op == MIR_SHL ||
            op == MIR_SHR || op == MIR_NEG || op == MIR_NOT) {
            /* Check if we've seen this expression */
            uint32_t existing_dst = ins->dst;
            int is_new = gvn_find_or_insert(&t, op, ins->a, ins->b, &existing_dst);
            if (!is_new && existing_dst != ins->dst) {
                /* Replace with MOV from existing result */
                ins->op = MIR_MOV;
                ins->a = existing_dst;
                ins->b = 0;
                elim++;
            }
        }
    }

    gvn_free(&t);
    return elim;
}
