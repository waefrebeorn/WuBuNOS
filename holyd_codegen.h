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

/* Program emitter (self-hosting rung 3): emit module code + function bodies
 * as ONE position-independent image. `funcs[i].code` is a function body (may
 * carry its own global disp32 patches); `call_offs[j]` is a `call rel32`
 * position INSIDE module_code whose displacement targets funcs[call_fn[j]].
 * The emitter lays module + funcs contiguously as .text and fixes every
 * RIP-relative global disp32 (in module and in each function) against the
 * .data RVA, and rewrites each call rel32 to be image-relative. */
typedef struct {
    const uint8_t *code;
    size_t code_size;
    const size_t *global_offs;   /* disp32 positions within this body */
    const size_t *global_goff;   /* data-section offset each points at */
    size_t n_global_patches;
} HDProgramFunc;

int hd_write_pe_program(const char *filename,
                        const uint8_t *module_code, size_t module_size,
                        const size_t *module_global_offs, const size_t *module_global_goff,
                        size_t n_module_globals,
                        const size_t *call_offs, const size_t *call_fn, size_t n_calls,
                        const HDProgramFunc *funcs, size_t n_funcs,
                        const uint8_t *data, size_t data_size);
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
void *hd_compile_func(const char *source, const char *func_name);
int64_t hd_eval(const char *source);

#endif /* WUBUNOS_HOLYC_CODEGEN_H */