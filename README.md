# WuBuNOS — the WuBuOS Compiler

**The compiler that runs ON the kernel.** WuBuNOS is the from-scratch C11
toolchain inside WuBuOS: a HolyC JIT with 11 ISA backends, a mid-level IR
with optimizer passes, and a self-hosting battery that proves every C11
construct the kernel needs.

## What it is

- **HolyC frontend** — lexer → parser → AST (the TempleOS language, reborn in C11)
- **MIR** — mid-level IR: virtual registers, label-relative jumps, ISA-neutral
- **11 ISA backends** — x86-64, ARM64, RISC-V, MIPS, M68000, AVR, 8051, 8086, Z80, 6502, PTX
- **Optimizer** — CSE, register allocation, peephole, MIR optimization passes
- **Self-hosting battery** — 155 C11 constructs verified against gcc (crash-isolated)
- **Brainfuck** — yes, real 8-command → x86-64 JIT (the meme is load-bearing)

## Project topology

```
wubuwizard = THE BRAIN   (inference engine, model loading, KV-cache, training)
WuBuNOS     = THE COMPILER (this repo — targets every ISA)
WuBuOS      = THE BODY    (kernel, GUI, Styx/9P namespace — links both)
```

WuBuOS has both wubuwizard and WuBuNOS as submodules:
- `wubuos/src/compiler/` → this repo (WuBuNOS)
- `wubuos/src/brain/` → wubuwizard

## Build

WuBuOS builds WuBuNOS as part of its test suite:

```bash
cd wubuos && make test_holyc   # 84/84 HolyC compiler tests
cd wubuos && make test_drivers # 11 ISA drivers differential-tested vs gcc
```

Standalone (WuBuNOS repo directly):

```bash
gcc -O0 -g -Isrc/compiler -Isrc/jit \
  src/jit/jit.c src/jit/jit_encode.c src/jit/wubu_x86.c \
  src/compiler/holyc_lexer.c src/compiler/holyc_parse.c \
  src/compiler/holyc_parse_ast.c src/compiler/holyc_codegen.c \
  src/compiler/holyc_codegen_emit.c src/compiler/holyc_codegen_expr.c \
  src/compiler/holyc_codegen_stmt.c src/compiler/holyc_codegen_api.c \
  src/compiler/wubu_preproc.c src/compiler/holyc_runtime.c \
  src/compiler/holyc_test.c -o holyc_test -ldl
./holyc_test
```

## License

WaefreBeorn Umbrella License v3.0
