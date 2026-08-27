#!/bin/bash -e
# Build test_mir_selfhost with all dependencies

CC=gcc
CFLAGS="-O2 -std=c11 -I. -include wubu_gnu_compat.h"

SOURCES="
  tools/test_mir_only.c
  wubu_mir.c wubu_mir_opt.c wubu_mir_lower.c wubu_mir_regalloc.c
  wubu_mir_interp.c x86_peephole.c wubu_softfloat.c wubu_tgemm.c wubu_host_tensor.c
  wubu_mir_ssa.c wubu_mir_sccp.c wubu_mir_gvn.c wubu_mir_fuse.c
  holyd_mir_eval.c holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c
  holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c
  holyd_codegen_api.c holyd_runtime.c wubu_preproc.c
  jit_stub.c
  -lm -o tools/test_mir_only
"

$CC $CFLAGS $SOURCES