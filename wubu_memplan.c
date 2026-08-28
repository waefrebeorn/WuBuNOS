/*
 * wubu_memplan.c — Memory planning optimization pass for HLIR graphs.
 *
 * Analyzes tensor lifetimes in the HLIR graph and reuses buffers
 * for tensors that don't overlap in lifetime. This reduces peak
 * memory usage during inference — critical for small-GPU deployment.
 *
 * Algorithm:
 * 1. Topological sort of the HLIR graph
 * 2. For each node, compute "birth" (first use) and "death" (last use) step
 * 3. Greedy buffer allocation: reuse a buffer when its current tenant dies
 * 4. Emit a memory plan: each node gets a buffer_id; shared buffers mean reuse
 *
 * Self-contained: operates on hlir_graph_t, produces a memory plan.
 * C11, no third-party libraries.
 */
#include "wubu_memplan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Buffer tracker ─── */

typedef struct {
    int      buffer_id;
    uint32_t tenant_node;   /* which node currently owns this buffer, or -1 if free*/
    int      death_step;    /* when this buffer becomes free */
} buffer_slot_t;

/* ─── Lifetime analysis ─── */

static void compute_lifetimes(hlir_graph_t *g, int *birth, int *death) {
    /* Initialize */
    for (int i = 0; i < g->n; i++) {
        birth[i] = g->n;  /* born late (will be minimized) */
        death[i] = -1;    /* dies early (will be maximized) */
    }

    /* Walk nodes in topological order.
     * A node is "born" at its position in the topo order.
     * A node "dies" at the last step where it's used as an input. */
    /* HLIR graph is already in topo order from hlir_topo_sort */
    for (int step = 0; step < g->n; step++) {
        hlir_node_t *n = g->nodes[step];
        if (step < birth[step]) birth[step] = step;

        /* This node's inputs extend their death to this step */
        for (int j = 0; j < n->n_inputs; j++) {
            hlir_node_t *in = n->inputs[j];
            /* Find the input node's index */
            for (int k = 0; k < g->n; k++) {
                if (g->nodes[k] == in) {
                    if (step > death[k]) death[k] = step;
                    break;
                }
            }
        }
    }

    /* Output nodes die at the end */
    for (int i = 0; i < g->n_outputs; i++) {
        for (int k = 0; k < g->n; k++) {
            if (g->nodes[k] == g->outputs[i]) {
                death[k] = g->n;
                break;
            }
        }
    }
}

/* ─── Size estimation ─── */

static int64_t tensor_size(const hlir_tensor_t *t) {
    if (t->n_dims == 0) return 8; /* scalar */
    int64_t nelems = 1;
    for (int i = 0; i < t->n_dims; i++) nelems *= t->dims[i];
    int elem_size = 4; /* default F32 */
    switch (t->dtype) {
        case 0: elem_size = 4; break; /* F32 */
        case 1: elem_size = 2; break; /* F16 */
        case 2: elem_size = 2; break; /* BF16 */
        case 3: elem_size = 4; break; /* I32 */
        case 4: elem_size = 1; break; /* I8 */
        case 5: elem_size = 1; break; /* I4 */
        default: elem_size = 4; break;
    }
    return nelems * elem_size;
}

/* ─── Memory plan ─── */

memplan_t *memplan_create(hlir_graph_t *g) {
    memplan_t *plan = (memplan_t *)calloc(1, sizeof(memplan_t));
    if (!plan) return NULL;
    plan->n_nodes = g->n;
    plan->node_buf = (int *)calloc(g->n, sizeof(int));
    plan->node_size = (int64_t *)calloc(g->n, sizeof(int64_t));
    plan->n_buffers = 0;
    plan->buf_cap = 64;
    plan->buffers = (int64_t *)calloc(plan->buf_cap, sizeof(int64_t));
    if (!plan->node_buf || !plan->node_size || !plan->buffers) {
        free(plan->node_buf); free(plan->node_size); free(plan->buffers); free(plan);
        return NULL;
    }

    /* Compute lifetimes */
    int *birth = (int *)malloc(g->n * sizeof(int));
    int *death = (int *)malloc(g->n * sizeof(int));
    if (!birth || !death) { free(birth); free(death); memplan_free(plan); return NULL; }
    compute_lifetimes(g, birth, death);

    /* Greedy buffer allocation */
    buffer_slot_t *slots = (buffer_slot_t *)calloc(g->n, sizeof(buffer_slot_t));
    if (!slots) { free(birth); free(death); memplan_free(plan); return NULL; }
    int n_slots = 0;

    for (int step = 0; step < g->n; step++) {
        hlir_node_t *n = g->nodes[step];
        int64_t size = tensor_size(&n->output);
        plan->node_size[step] = size;

        /* Skip nodes that don't allocate (PLACEHOLDER, etc.) */
        if (size <= 0) {
            plan->node_buf[step] = -1;
            continue;
        }

        /* Try to find a free buffer slot */
        int best = -1;
        for (int s = 0; s < n_slots; s++) {
            if (slots[s].death_step < step) {
                /* This slot is free; reuse it */
                if (best < 0 || slots[s].buffer_id < slots[best].buffer_id)
                    best = s;
            }
        }

        if (best >= 0) {
            /* Reuse existing buffer */
            plan->node_buf[step] = slots[best].buffer_id;
            slots[best].tenant_node = step;
            slots[best].death_step = death[step];
            /* Update buffer size if this tenant is larger */
            if (size > plan->buffers[slots[best].buffer_id])
                plan->buffers[slots[best].buffer_id] = size;
        } else {
            /* Allocate new buffer */
            int buf_id = n_slots;
            if (buf_id >= plan->buf_cap) {
                plan->buf_cap *= 2;
                plan->buffers = (int64_t *)realloc(plan->buffers, plan->buf_cap * sizeof(int64_t));
            }
            plan->buffers[buf_id] = size;
            plan->node_buf[step] = buf_id;
            slots[n_slots].buffer_id = buf_id;
            slots[n_slots].tenant_node = step;
            slots[n_slots].death_step = death[step];
            n_slots++;
            if (n_slots > plan->n_buffers) plan->n_buffers = n_slots;
        }
    }

    /* Compute peak memory */
    int64_t peak = 0;
    for (int s = 0; s < n_slots; s++) {
        if (slots[s].death_step >= 0)
            peak += plan->buffers[slots[s].buffer_id];
    }
    plan->peak_memory = peak;

    /* Compute naive memory (no reuse) */
    int64_t naive = 0;
    for (int i = 0; i < g->n; i++) {
        if (plan->node_size[i] > 0) naive += plan->node_size[i];
    }
    plan->naive_memory = naive;

    free(slots);
    free(birth);
    free(death);
    return plan;
}

void memplan_free(memplan_t *plan) {
    if (!plan) return;
    free(plan->node_buf);
    free(plan->node_size);
    free(plan->buffers);
    free(plan);
}

void memplan_print(const memplan_t *plan) {
    printf("Memory Plan: %d buffers, peak=%lld bytes, naive=%lld bytes, savings=%.1f%%\n",
           plan->n_buffers,
           (long long)plan->peak_memory,
           (long long)plan->naive_memory,
           plan->naive_memory > 0 ?
           100.0 * (1.0 - (double)plan->peak_memory / plan->naive_memory) : 0.0);
    for (int i = 0; i < plan->n_buffers; i++) {
        printf("  buffer[%d]: %lld bytes\n", i, (long long)plan->buffers[i]);
    }
}
