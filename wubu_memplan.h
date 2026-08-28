#ifndef WUBU_MEMPLAN_H
#define WUBU_MEMPLAN_H

#include "wubu_hlir.h"
#include <stdint.h>

/* Memory plan: maps each HLIR node to a buffer, with reuse */
typedef struct {
    int      n_nodes;
    int     *node_buf;     /* buffer_id for each node, -1 = no alloc */
    int64_t *node_size;    /* size in bytes for each node's output */

    int      n_buffers;    /* total unique buffers */
    int      buf_cap;      /* buffer array capacity */
    int64_t *buffers;      /* size of each buffer */

    int64_t  peak_memory;  /* peak memory with reuse */
    int64_t  naive_memory; /* memory without reuse (sum of all) */
} memplan_t;

/* Create a memory plan for an HLIR graph */
memplan_t *memplan_create(hlir_graph_t *g);

/* Free a memory plan */
void memplan_free(memplan_t *plan);

/* Print plan summary */
void memplan_print(const memplan_t *plan);

#endif /* WUBU_MEMPLAN_H */
