#ifndef WUBU_ISA_WASM_H
#define WUBU_ISA_WASM_H

#include "wubu_isa_driver.h"
#include "wubu_mir.h"

extern const wubu_isa_driver_t wubu_isa_wasm;

/* Direct compile API (no driver registry needed) */
int wubu_isa_wasm_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size);

#endif /* WUBU_ISA_WASM_H */
