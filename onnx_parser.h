/*
 * onnx_parser.h -- ONNX model loader for WuBuOS HLIR compiler pipeline.
 */
#ifndef ONNX_PARSER_H
#define ONNX_PARSER_H

#include <stdint.h>
#include "wubu_hlir.h"

/* Load an ONNX model file into an HLIR computation graph.
 * Returns 0 on success, -1 on error. */
int onnx_load_model(const char *filepath, hlir_graph_t *g);

#endif /* ONNX_PARSER_H */
