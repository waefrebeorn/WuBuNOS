#ifndef MLIR_TEXT_H
#define MLIR_TEXT_H

#include "wubu_hlir.h"
#include <stdio.h>
#include <stdlib.h>

/* Parse MLIR text format source string into an HLIR graph.
 * Returns 0 on success, -1 on error. */
int mlir_text_load(const char *source, hlir_graph_t *g);

/* Parse MLIR text format file into an HLIR graph.
 * Returns 0 on success, -1 on error. */
static inline int mlir_text_load_file(const char *filepath, hlir_graph_t *g) {
    FILE *f = fopen(filepath, "r");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(fsize + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, fsize, f);
    buf[fsize] = '\0';
    fclose(f);
    int rc = mlir_text_load(buf, g);
    free(buf);
    return rc;
}

#endif /* MLIR_TEXT_H */
