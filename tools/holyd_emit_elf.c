/*
 * holyd_emit_elf.c -- thin HolyD -> ELF emitter (no wuburuntime broker).
 *
 * Replicates the `-emit-elf` path of holyd.c but links only the frontend +
 * old codegen + elf writer, so a freestanding source can be compiled straight
 * to a loadable ELF object without dragging in the wuburuntime broker.  This
 * is the compiler side of the "full circle" bootable-image pipeline:
 *
 *   holyd_emit_elf <src.hc> <out.elf>
 *
 * The emitted ELF is a single PT_LOAD image of the compiled main() body plus
 * its .data, with RIP-relative globals patched -- the same bytes the wubuos
 * loader (tokenizer_test / test_backend_consistency) already runs.
 */
#include "holyd_ast.h"
#include "holyd_lexer.h"
#include "holyd_parser.h"
#include "holyd_codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = 0;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: holyd_emit_elf <src.hc> <out.elf>\n");
        return 2;
    }
    char *src = read_file(argv[1]);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    HDLexer lex;
    hd_lex_init(&lex, src);
    if (lex.has_error) { fprintf(stderr, "lex: %s\n", lex.error); free(src); return 1; }

    HDParser parse;
    hd_parse_init(&parse, &lex);
    HDASTNode *ast = hd_parse_compilation_unit(&parse);
    if (parse.has_error || !ast) { fprintf(stderr, "parse error\n"); hd_ast_free(ast); free(src); return 1; }

    HDGen gen;
    hd_gen_init(&gen);
    emit_prologue(&gen);
    if (ast->kind == HD_AST_BLOCK) {
        for (int i = 0; i < ast->n_stmts; i++) {
            HDASTNode *stmt = ast->stmts[i];
            if (stmt->kind == HD_AST_IF || stmt->kind == HD_AST_WHILE ||
                stmt->kind == HD_AST_FOR || stmt->kind == HD_AST_DO_WHILE ||
                stmt->kind == HD_AST_VAR_DECL || stmt->kind == HD_AST_FUNC_DECL)
                gen_stmt(&gen, stmt);
            else
                gen_expr(&gen, stmt);
        }
    }
    emit_epilogue(&gen);
    hd_ast_free(ast);
    free(src);

    if (gen.has_error) { fprintf(stderr, "codegen: %s\n", gen.error); free(gen.code); free(gen.data); return 1; }

    int main_idx = -1;
    for (int f = 0; f < gen.n_functions; f++)
        if (gen.functions[f].name && strcmp(gen.functions[f].name, "main") == 0) { main_idx = f; break; }
    if (main_idx < 0) { fprintf(stderr, "no main()\n"); free(gen.code); free(gen.data); return 1; }

    HDFunction mainfunc = gen.functions[main_idx];
    if (!mainfunc.func_ptr || mainfunc.code_size == 0) { fprintf(stderr, "main() has no code\n"); free(gen.code); free(gen.data); return 1; }

    size_t func_size = mainfunc.code_size;
    uint8_t *func_code = malloc(func_size);
    if (!func_code) { free(gen.code); free(gen.data); return 1; }
    memcpy(func_code, mainfunc.func_ptr, func_size);

    /* patch RIP-relative globals */
    for (int i = 0; i < mainfunc.n_global_patches; i++) {
        size_t pp = mainfunc.global_patches[i].code_patch_pos;
        size_t go = mainfunc.global_patches[i].global_offset;
        if (pp + 4 <= func_size) {
            int32_t disp32 = (int32_t)(func_size + go - pp - 4);
            memcpy(func_code + pp, &disp32, 4);
        }
    }

    size_t patch_offsets[1] = {0}, patch_globals[1] = {0};
    if (hd_write_elf(argv[2], func_code, func_size, gen.data, gen.data_size,
                     patch_offsets, patch_globals, 0) != 0) {
        fprintf(stderr, "failed to write %s\n", argv[2]);
        free(func_code); free(gen.code); free(gen.data);
        return 1;
    }
    printf("holyd_emit_elf: %s (%zu bytes code, %zu bytes data)\n",
           argv[2], func_size, gen.data_size);
    free(func_code); free(gen.code); free(gen.data);
    return 0;
}
