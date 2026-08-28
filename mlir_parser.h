#ifndef MLIR_PARSER_H
#define MLIR_PARSER_H

#include "wubu_hlir.h"
#include <stdint.h>

/* MLIR bytecode magic: 0x52EF4C4D (little-endian "MLïR") */
#define MLIR_MAGIC 0x52EF4C4D

/* MLIR bytecode section IDs */
#define MLIR_SECTION_STRING 1
#define MLIR_SECTION_DIALECT 2
#define MLIR_SECTION_IR 3
#define MLIR_SECTION_TYPE 4
#define MLIR_SECTION_ATTR 5

/* mlir_model: parsed MLIR bytecode → HLIR graph */
typedef struct {
    hlir_graph_t graph;
    int parsed_ok;
    char error[256];
} mlir_model_t;

/* Parse MLIR bytecode file → HLIR graph.
 * Returns 0 on success, -1 on error (model.error contains reason). */
int mlir_load_model(const char *filepath, mlir_model_t *model);

/* Free resources */
void mlir_model_free(mlir_model_t *model);

#endif /* MLIR_PARSER_H */
