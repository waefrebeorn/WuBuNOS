/*
 * holyd_codegen.c  --  WuBuNOS HolyD Code Generator (Facade)
 * Modular codegen: emit, expr, stmt, api submodules.
 * This file is now a thin facade - real implementation in submodules.
 */

#include "holyd_codegen.h"
#include "holyd_codegen_internal.h"

/* This file intentionally left minimal - all implementation moved to:
 *   holyd_codegen_emit.c    - x86-64 emission helpers, instruction patterns
 *   holyd_codegen_expr.c    - Expression code generation
 *   holyd_codegen_stmt.c    - Statement code generation, hd_gen_init
 *   holyd_codegen_api.c     - Public API: hd_compile, hd_eval, hd_compile_func
 *
 * Internal declarations in holyd_codegen_internal.h
 */

const char *holyd_codegen_version(void) {
    return "HolyD Codegen v1.0 (modular)";
}