#ifndef SAFETENSORS_BRIDGE_H
#define SAFETENSORS_BRIDGE_H

#include "safetensors.h"
#include "wubu_hlir.h"

/* Load SafeTensors weights into an HLIR graph as constant nodes.
 * Each tensor becomes an HLIR_CONSTANT with f32 data.
 * Returns 0 on success, -1 on error. */
int safetensors_to_hlir(safetensors_t *st, hlir_graph_t *g);

#endif /* SAFETENSORS_BRIDGE_H */
