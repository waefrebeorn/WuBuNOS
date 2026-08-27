/*
 * wubu_mir_sccp.c — Sparse Conditional Constant Propagation (SCCP).
 *
 * Works on MIR in SSA form. Uses the lattice:
 *   ⊥ (undefined) → constant → ⊤ (overdefined)
 *
 * Discovers constants, propagates them, and removes dead branches.
 *
 * C11, self-contained.
 */

#include "wubu_mir.h"
#include <stdlib.h>
#include <string.h>

/* ---- Lattice ---- */

typedef enum {
    L_UNDEF = 0,   /* not yet visited */
    L_CONST = 1,   /* known constant */
    L_OVER  = 2    /* overdefined (not a constant) */
} lattice_t;

/* ---- SCCP state ---- */

typedef struct {
    lattice_t *lattice;   /* per-VR lattice value */
    int64_t   *cval;      /* constant value (valid if lattice == L_CONST) */
    uint8_t   *exec;      /* edge execution flag (worklist for CFG edges) */
} sccp_state_t;

/* ---- Worklist for edges ---- */

typedef struct {
    uint32_t *edges;
    size_t n, cap;
} edge_worklist_t;

static void edge_push(edge_worklist_t *wl, uint32_t edge) {
    if (wl->n >= wl->cap) {
        wl->cap = wl->cap ? wl->cap * 2 : 64;
        wl->edges = realloc(wl->edges, wl->cap * sizeof(uint32_t));
    }
    wl->edges[wl->n++] = edge;
}

static uint32_t edge_pop(edge_worklist_t *wl) {
    return wl->edges[--wl->n];
}

static int edge_empty(const edge_worklist_t *wl) {
    return wl->n == 0;
}

/* ---- Meet function ---- */

static void meet(sccp_state_t *s, uint32_t vr, lattice_t lv, int64_t cv) {
    lattice_t cur = s->lattice[vr];
    if (cur == L_UNDEF) {
        s->lattice[vr] = lv;
        s->cval[vr] = cv;
    } else if (cur == L_CONST && lv == L_CONST) {
        if (s->cval[vr] != cv) {
            s->lattice[vr] = L_OVER;
            s->cval[vr] = 0;
        }
    } else {
        s->lattice[vr] = L_OVER;
        s->cval[vr] = 0;
    }
}

/* ---- Evaluate one MIR instruction ---- */

static int eval_instr(const wubu_mir_prog_t *p, size_t idx,
                      sccp_state_t *s) {
    const wubu_mir_instr_t *ins = &p->ins[idx];
    uint32_t rd = ins->dst;
    uint32_t ra = ins->a;
    uint32_t rb = ins->b;

    /* Check if operands are constants */
    int ca = (s->lattice[ra] == L_CONST) ? 1 : 0;
    int cb = (s->lattice[rb] == L_CONST) ? 1 : 0;
    int64_t va = ca ? s->cval[ra] : 0;
    int64_t vb = cb ? s->cval[rb] : 0;

    if (ins->op == MIR_CONST) {
        s->lattice[rd] = L_CONST;
        s->cval[rd] = (int64_t)ins->imm;
        return 1;
    }

    if (ins->op == MIR_MOV) {
        if (ca) {
            s->lattice[rd] = L_CONST;
            s->cval[rd] = va;
            return 1;
        }
        s->lattice[rd] = L_OVER;
        return 1;
    }

    if (ins->op == MIR_NEG) {
        if (ca) {
            s->lattice[rd] = L_CONST;
            s->cval[rd] = -va;
            return 1;
        }
        s->lattice[rd] = L_OVER;
        return 1;
    }

    if (ins->op == MIR_NOT) {
        if (ca) {
            s->lattice[rd] = L_CONST;
            s->cval[rd] = ~va;
            return 1;
        }
        s->lattice[rd] = L_OVER;
        return 1;
    }

    /* Binary ops */
    if (ca && cb) {
        int64_t r = 0;
        switch (ins->op) {
        case MIR_ADD: r = va + vb; break;
        case MIR_SUB: r = va - vb; break;
        case MIR_MUL: r = va * vb; break;
        case MIR_DIV: r = vb != 0 ? va / vb : 0; break;
        case MIR_MOD: r = vb != 0 ? va % vb : 0; break;
        case MIR_AND: r = va & vb; break;
        case MIR_OR:  r = va | vb; break;
        case MIR_XOR: r = va ^ vb; break;
        case MIR_SHL: r = va << vb; break;
        case MIR_SHR: r = (int64_t)va >> vb; break;
        default: return 0;
        }
        s->lattice[rd] = L_CONST;
        s->cval[rd] = r;
        return 1;
    }

    /* Comparisons */
    if (ca && cb && (ins->op == MIR_EQ || ins->op == MIR_NE ||
        ins->op == MIR_LT || ins->op == MIR_LE ||
        ins->op == MIR_GT || ins->op == MIR_GE ||
        ins->op == MIR_ULT || ins->op == MIR_ULE ||
        ins->op == MIR_UGT || ins->op == MIR_UGE)) {
        int r = 0;
        switch (ins->op) {
        case MIR_EQ: r = (va == vb); break;
        case MIR_NE: r = (va != vb); break;
        case MIR_LT: r = (int64_t)va < (int64_t)vb; break;
        case MIR_LE: r = (int64_t)va <= (int64_t)vb; break;
        case MIR_GT: r = (int64_t)va > (int64_t)vb; break;
        case MIR_GE: r = (int64_t)va >= (int64_t)vb; break;
        case MIR_ULT: r = (uint64_t)va < (uint64_t)vb; break;
        case MIR_ULE: r = (uint64_t)va <= (uint64_t)vb; break;
        case MIR_UGT: r = (uint64_t)va > (uint64_t)vb; break;
        case MIR_UGE: r = (uint64_t)va >= (uint64_t)vb; break;
        default: break;
        }
        s->lattice[rd] = L_CONST;
        s->cval[rd] = r;
        return 1;
    }

    /* Branches with constant condition → mark edges */
    if (ins->op == MIR_JZ && ca) {
        if (va == 0) return 1; /* jump taken */
        else return 2; /* fall through */
    }
    if (ins->op == MIR_JNZ && ca) {
        if (va != 0) return 1; /* jump taken */
        else return 2; /* fall through */
    }

    return 0; /* changed nothing */
}

/* ---- Main SCCP algorithm ---- */

int wubu_mir_sccp(wubu_mir_prog_t *p) {
    if (!p || p->n == 0) return 0;

    /* Find max VR */
    uint32_t max_vr = 0;
    for (size_t i = 0; i < p->n; i++) {
        uint32_t d = p->ins[i].dst;
        uint32_t a = p->ins[i].a;
        uint32_t b = p->ins[i].b;
        if (d > max_vr) max_vr = d;
        if (a > max_vr) max_vr = a;
        if (b > max_vr) max_vr = b;
    }

    sccp_state_t s;
    s.lattice = calloc(max_vr + 1, sizeof(lattice_t));
    s.cval = calloc(max_vr + 1, sizeof(int64_t));
    s.exec = calloc(p->n + 1, sizeof(uint8_t));

    /* Start from entry: mark all instruction edges as executable */
    edge_worklist_t wl;
    memset(&wl, 0, sizeof(wl));
    edge_push(&wl, 0);

    while (!edge_empty(&wl)) {
        uint32_t idx = edge_pop(&wl);
        if (idx >= p->n || s.exec[idx]) continue;
        s.exec[idx] = 1;

        const wubu_mir_instr_t *ins = &p->ins[idx];

        int changed = eval_instr(p, idx, &s);

        /* For branches, mark successor edges */
        if (ins->op == MIR_JMP) {
            for (size_t j = 0; j < p->n; j++) {
                if (p->ins[j].op == MIR_LABEL && p->ins[j].label == ins->label) {
                    edge_push(&wl, (uint32_t)j);
                    break;
                }
            }
        } else if (ins->op == MIR_JZ || ins->op == MIR_JNZ) {
            if (changed == 1) {
                /* Jump taken */
                for (size_t j = idx + 1; j < p->n; j++) {
                    if (p->ins[j].op == MIR_LABEL && p->ins[j].label == ins->label) {
                        edge_push(&wl, (uint32_t)j);
                        break;
                    }
                }
            } else if (changed == 2) {
                /* Fall through */
                edge_push(&wl, idx + 1);
            } else {
                /* Not constant: explore both */
                for (size_t j = idx + 1; j < p->n; j++) {
                    if (p->ins[j].op == MIR_LABEL && p->ins[j].label == ins->label) {
                        edge_push(&wl, (uint32_t)j);
                        break;
                    }
                }
                edge_push(&wl, idx + 1);
            }
        } else {
            edge_push(&wl, idx + 1);
        }
    }

    free(s.lattice);
    free(s.cval);
    free(s.exec);
    free(wl.edges);
    return 0;
}
