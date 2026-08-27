/*
 * holyd_codegen.h  --  HolyD Code Generator
 * Emits x86-64 machine code from AST. Self-contained, C11.
 */
#ifndef WUBUNOS_HOLYC_CODEGEN_H
#define WUBUNOS_HOLYC_CODEGEN_H

#include "holyd_types.h"

/* JIT Memory Management */
void *jit_alloc_exec(size_t size);
void *jit_lock_exec(void *ptr, size_t size);
void jit_free_exec(void *ptr, size_t size);

/* -- Executable Output -------------------------------------------- */
int hd_write_elf(const char *filename,
                 const uint8_t *code, size_t code_size,
                 const uint8_t *data, size_t data_size,
                 const size_t *patch_offsets,
                 const size_t *patch_globals,
                 size_t n_patches);
int hd_write_pe(const char *filename,
                const uint8_t *code, size_t code_size,
                const uint8_t *data, size_t data_size,
                const size_t *patch_offsets,
                const size_t *patch_globals,
                size_t n_patches);
int hd_write_bin(const char *filename,
                 const uint8_t *code, size_t code_size,
                 int bootable);

/* Code Generator API */
void hd_gen_init(HDGen *gen);
int hd_gen_node(HDGen *gen, const HDASTNode *node);
int hd_gen_function(HDGen *gen, const HDASTNode *func);
int gen_expr(HDGen *gen, const HDASTNode *node);
int gen_stmt(HDGen *gen, const HDASTNode *node);
void emit_prologue(HDGen *gen);
void emit_epilogue(HDGen *gen);

/* Get generated machine code */
const uint8_t *hd_gen_code(const HDGen *gen, size_t *out_size);

/* HolyD personality runtime (host effects: Print / FpWriteFile). */
int64_t wubu_print(const char *s);
int64_t wubu_fp_write_file(const char *name, const char *contents);

/* Register the HolyD runtime functions as extern C symbols so the JIT
 * resolves `Print` / `FpWriteFile` to real host addresses. */
void hd_register_holyd_runtime(HDGen *gen);

/* High-level compile/execute */
void *hd_compile(const char *source, size_t *out_size);
int64_t hd_eval(const char *source);

#endif /* WUBUNOS_HOLYC_CODEGEN_H */