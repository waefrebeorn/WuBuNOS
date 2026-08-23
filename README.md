# WuBuNOS — The Compiler

> **The compiler that runs ON the kernel.**

WuBuNOS is the from-scratch C11 toolchain inside WuBuOS: a HolyD compiler
frontend, a mid-level IR with optimizer passes, 14 ISA backends, and a
self-hosting battery that proves every C11 construct the kernel needs.

**Stats:** 50 C files, 20 header files, ~25,208 lines of code.

---

## HolyD — the founding myth

> *"can we make all expansions to HolyC 'HolyD' like a big joke that I swang my
> big dick to make ai make this since other humans didn't and were puny"*
> — WaefreBeorn, quoting the comedian whose line stuck culturally

**HolyD** is the language name. It is deliberately a joke with teeth: the
other humans were puny and didn't build the self-hosting C11 toolchain that
runs on the AGI kernel, so it got built anyway — with swagger. Every
identifier carries the `HD_` prefix (`HD_AST_*`, `hd_build_mir`, …) and the
frontend lives in `holyd_*.c`. Treat the name as gospel in this repo: if you
see `HolyC` or `HC_`, that's a leftover and should be renamed to `HolyD`/`HD_`.

---

## Architecture Overview

WuBuNOS follows the classic retargetable compiler design — one frontend,
one IR, many backends:

```
HolyD Source
     │
     ▼
┌─────────────────────────────────────────┐
│  FRONTEND  (holyd_*.c/h)                │
│  Lexer → Parser → AST → MIR Emitter     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  MID-LEVEL IR  (wubu_mir*.c/h)          │
│  3-address code, virtual registers,     │
│  symbolic labels, SSA form              │
│                                         │
│  Optimizer Passes:                      │
│    • Constant Folding                   │
│    • Strength Reduction                 │
│    • Dead Code Elimination (DCE)        │
│    • Common Subexpression Elim. (CSE)   │
│    • Loop-Invariant Code Motion (LICM)  │
│    • Loop Unrolling                     │
│    • Instruction Combining              │
│                                         │
│  Register Allocation:                   │
│    • Linear-scan on SSA intervals       │
│    • Configurable physical reg count    │
│    • Spill-to-stack fallback            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  BACKENDS  (wubu_isa_*.c)               │
│  11 ISA drivers consuming the same MIR  │
│                                         │
│  Native JITs:        Interpreters:      │
│  • x86-64            • RISC-V (RV64I)   │
│  • ARM64 (AArch64)   • MIPS             │
│  • PTX (NVIDIA GPU)  • M68000           │
│                      • 8086             │
│                      • 6502             │
│                      • Z80              │
│                      • 8051             │
│                      • AVR              │
└─────────────────────────────────────────┘
```

---

## The Compilation Pipeline

### 1. Preprocessing (`wubu_preproc.c/h`)
Tokenizes and expands `#define` macros, handles `#include` resolution,
and strips comments before the lexer sees the text.

### 2. Lexing (`holyd_lexer.c/h`)
Converts preprocessed source text into a token stream: keywords
(`if`, `while`, `class`, `switch`), identifiers, literals (integer,
string, float), operators, and punctuation. The HolyD dialect includes
TempleOS-era keywords like `Switch`, `Case`, `Throw`, `Catch`, `try`,
`if`, `else`, `while`, `for`, `do`, `return`, `break`, `continue`,
`extern`, `import`, `static`, `public`, `define`, `class`, `union`,
`enum`, `sizeof`, `true`, `false`, `NULL`.

### 3. Parsing (`holyd_parse.c`, `holyd_parse_ast.c`, `holyd_ast.h`)
Recursive-descent parser that builds an Abstract Syntax Tree (AST).
Handles the full HolyD grammar: declarations, expressions (with proper
operator precedence), statements, class definitions, function bodies,
and preprocessor directives embedded in the parse.

### 4. AST → MIR Lowering (`holyd_codegen.c/h`)
The AST is lowered to **MIR** (Mid-level IR) — the hourglass neck of
the compiler. MIR is:

- **3-address code:** each instruction has at most 3 operands.
- **Virtual registers:** operands are `uint32_t` indices, not physical
  registers. This makes the register-clobber bug family impossible —
  each backend does its own register assignment.
- **SSA form:** each virtual register is defined exactly once, making
  live-range analysis trivial (linear scan works).
- **Symbolic labels:** jumps reference labels by name; each backend
  resolves them to concrete offsets.

MIR instruction set includes: `MIR_CONST`, `MIR_ADD/SUB/MUL/DIV/MOD`,
bitwise ops (`AND/OR/XOR/SHL/SHR`), `MIR_NEG/NOT`, comparisons
(`EQ/NE/LT/LE/GT/GE` producing 0/1), `MIR_MOV`, `MIR_JMP`, `MIR_JZ`,
`MIR_LABEL`, `MIR_RET`.

### 5. Optimization (`wubu_mir_opt.c/h`, `wubu_mir_regalloc.c/h`, `x86_peephole.c`)

Seven classical MIR optimization passes run in canonical order:

| Pass | Flag | What it does |
|------|------|-------------|
| Constant Folding | `MIR_OPT_FOLD` | Evaluates binops on constants at compile time |
| Strength Reduction | `MIR_OPT_STRENGTH` | Replaces `mul`/`div` with shifts; eliminates `*1`, `+0`, `*0` |
| Dead Code Elimination | `MIR_OPT_DCE` | Removes instructions whose result vr is never used |
| Loop-Invariant Code Motion | `MIR_OPT_LICM` | Hoists pure computations out of loops |
| Loop Unrolling | `MIR_OPT_UNROLL` | Unrolls loops with small constant trip counts |
| Instruction Combining | `MIR_OPT_COMBINE` | Applies algebraic identities (e.g., `x+0 → x`) |
| Common Subexpression Elimination | `MIR_OPT_CSE` | Reuses results of identical computations |

**Register allocation** uses the classic **linear-scan** algorithm on
SSA-form intervals. Since MIR is SSA, each virtual register has a simple
`[first_def, last_use]` live range — no complex liveness analysis needed.
The allocator is parameterized by physical register count (e.g., 14 for
x86-64, 8 for M68k) and spills to the stack when registers are exhausted.

**Peephole optimization** (`x86_peephole.c`) runs after code emission on
x86-64, collapsing redundant `mov` pairs and simplifying addressing modes.

### 6. Code Emission (ISA Drivers)

Each backend implements the `wubu_isa_driver_t` vtable:

```c
typedef struct wubu_isa_driver {
    const char *name;          // "x86-64", "riscv", "6502"
    const char *family;        // "native", "gpu", "portable"
    wubu_isa_exec_t exec;      // NATIVE (JIT) or INTERPRETED

    int  (*compile)(const wubu_mir_prog_t *p,
                    uint8_t **out_code, size_t *out_size);
    int64_t (*run)(const uint8_t *code, size_t size, int64_t arg);
    void (*describe)(void);
} wubu_isa_driver_t;
```

- **compile:** lowers MIR to machine code bytes for that ISA.
- **run:** executes the compiled bytes natively (JIT) or via bundled
  interpreter.
- **describe:** reports ISA family and execution model.

**Oracle doctrine:** every driver's encodings are verified byte-for-byte
against GNU binutils `objdump` before shipping. The differential battery
(`make test_drivers`) runs every driver against gcc on 33+ expressions —
no guessed opcodes.

---

## The ISA Driver Space

| Driver | File | Execution | Architecture |
|--------|------|-----------|-------------|
| x86-64 | `wubu_isa_x86_64.c` | **Native JIT** | AMD64 / Intel 64 |
| ARM64 | `wubu_isa_arm64.c` | **Native JIT** | AArch64 (ARMv8+) |
| PTX | `wubu_isa_ptx.c`, `holyd_ptx.c` | **Native JIT** | NVIDIA GPU (PTX/SASS) |
| RISC-V | `wubu_isa_riscv.c` | Interpreter | RV64I (2010 ISA) |
| MIPS | `wubu_isa_mips.c` | Interpreter | MIPS (Berkeley RISC lineage) |
| M68000 | `wubu_isa_m68k.c` | Interpreter | Motorola 68,000 (1979) |
| 8086 | `wubu_isa_8086.c` | Interpreter | Intel 8086 (1978, x86 root) |
| 6502 | `wubu_isa_6502.c` | Interpreter | MOS 6502 (1975) |
| Z80 | `wubu_isa_z80.c` | Interpreter | Zilog Z80 (1976) |
| 8051 | `wubu_isa_8051.c` | Interpreter | Intel 8051 (1978, MCU) |
| AVR | `wubu_isa_avr.c` | Interpreter | Atmel AVR (Arduino Uno) |

The three **native JITs** (x86-64, ARM64, PTX) emit machine code into
executable memory and call it directly. The eight **interpreters** bundle
a full ISA simulator that executes the emitted bytecode. Both paths are
driven by the same MIR, so correctness is guaranteed by construction.

---

## Self-Hosting Battery

The test suite (`holyd_test.c`, `test_isa_driver.c`) is a **self-hosting
battery** — it compiles and executes every C11 construct WuBuOS needs,
then cross-checks results:

- **84 HolyD compiler tests** (`make test_holyd`): lexer, parser, AST,
  codegen, and end-to-end compilation of every language construct.
- **11 ISA driver differential tests** (`make test_drivers`): each
  driver executes 33+ expressions and is verified against gcc output.
- **Brainfuck** (`brainfuck.c`): a complete 8-command brainfuck → x86-64
  JIT compiler, proving the pipeline works end-to-end on a real (if
  esoteric) language. The meme is load-bearing.

---

## Project Topology

```
wubuwizard = THE BRAIN    (inference engine — model loading, KV-cache, training)
WuBuNOS     = THE COMPILER (this repo — targets every ISA)
WuBuOS      = THE BODY     (kernel, GUI, Styx/9P namespace — links both)
```

WuBuOS has both wubuNOS and wubuwizard as submodules:
- `wubuos/src/compiler/` → this repo (WuBuNOS)
- `wubuos/src/brain/` → wubuwizard

WuBuNOS compiles HolyD programs that run **ring-0 on the WuBuOS kernel**.

---

## Build Instructions

### Standalone (this repo directly)

```bash
cd wubunos
make all
```

This builds the full compiler test binary and runs the self-hosting battery.

### As part of WuBuOS

```bash
cd wubuos
make test_holyd    # 84/84 HolyD compiler tests
make test_drivers  # 11 ISA drivers differential-tested vs gcc
```

### Manual compilation (no Makefile)

```bash
gcc -O0 -g -I. \
  holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c \
  holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c \
  holyd_codegen_api.c holyd_runtime.c holyd_test.c \
  wubu_preproc.c \
  wubu_mir.c wubu_mir_opt.c wubu_mir_regalloc.c wubu_mir_lower.c \
  wubu_isa_driver.c \
  wubu_isa_x86_64.c wubu_isa_arm64.c wubu_isa_mips.c \
  wubu_isa_m68k.c wubu_isa_riscv.c wubu_isa_8086.c \
  wubu_isa_6502.c wubu_isa_z80.c wubu_isa_8051.c wubu_isa_avr.c \
  wubu_isa_ptx.c holyd_ptx.c \
  x86_peephole.c \
  brainfuck.c test_isa_driver.c \
  -o wubunos_test -ldl -lm
./wubunos_test
```

---

## File Map

| Path | Purpose |
|------|---------|
| `holyd_lexer.c/h` | HolyD tokenizer |
| `holyd_parse.c`, `holyd_parse_ast.c`, `holyd_parser.h` | Recursive-descent parser + AST builder |
| `holyd_ast.h` | AST node definitions |
| `holyd_codegen.c/h`, `holyd_codegen_emit.c`, `holyd_codegen_expr.c`, `holyd_codegen_stmt.c`, `holyd_codegen_api.c` | AST → MIR lowering |
| `holyd_types.h` | Core type definitions |
| `holyd_runtime.c` | Runtime support (memory, syscalls) |
| `holyd_test.c` | Self-hosting test battery |
| `holyd.h` | Master include for the HolyD frontend |
| `wubu_preproc.c/h` | C preprocessor (macros, includes) |
| `wubu_mir.c/h` | Mid-level IR: construction, printing, utilities |
| `wubu_mir_opt.c/h` | Optimizer passes (fold, strength, DCE, CSE, LICM, unroll) |
| `wubu_mir_regalloc.c/h` | Linear-scan register allocator |
| `wubu_mir_lower.c/h` | MIR → driver-ready lowering |
| `wubu_isa_driver.c/h` | ISA driver vtable + registry |
| `wubu_isa_x86_64.c` | x86-64 native JIT backend |
| `wubu_isa_arm64.c` | ARM64 native JIT backend |
| `wubu_isa_mips.c` | MIPS interpreter backend |
| `wubu_isa_m68k.c` | Motorola 68k interpreter backend |
| `wubu_isa_riscv.c` | RISC-V interpreter backend |
| `wubu_isa_8086.c` | Intel 8086 interpreter backend |
| `wubu_isa_6502.c` | MOS 6502 interpreter backend |
| `wubu_isa_z80.c` | Zilog Z80 interpreter backend |
| `wubu_isa_8051.c` | Intel 8051 interpreter backend |
| `wubu_isa_avr.c` | Atmel AVR interpreter backend |
| `wubu_isa_ptx.c/h`, `holyd_ptx.c/h` | NVIDIA PTX GPU backend |
| `x86_peephole.c` | x86-64 peephole optimizer |
| `brainfuck.c` | Brainfuck → x86-64 JIT (end-to-end proof) |
| `test_isa_driver.c` | Differential ISA driver tests |

---

## License

WaefreBeorn Umbrella License v3.0

---

## Repository organization (READ FIRST — user-corrected 2026-08-16)

Three real, separate repos. **Do not confuse them:**

| Repo | Local path | GitHub | Role |
|------|-----------|--------|------|
| WuBuOS | `/home/wubu/wubuos` | `waefrebeorn/WuBuOS` | THE BODY — kernel, GUI, firmware, containers |
| WuBuNOS | `/home/wubu/wubunos` | `waefrebeorn/WuBuNOS` | THE COMPILER — this repo |
| wubuwizard | `/home/wubu/wubuwizard` | `waefrebeorn/wubuwizard` | THE BRAIN — AGI engine |

This compiler repo is reached by the OS repo via a **symlink**:
`/home/wubu/wubuos/src/compiler → /home/wubu/wubunos`. The OS `Makefile`
uses `$(COMP) = src/compiler`, so `make holyd` in `wubuos` compiles sources
from here directly. There is **no submodule** for the compiler (the `.gitmodules`
entry was stale and is removed).

`wubuos/src/brain` IS a real **gitlink submodule** → wubuwizard.
Two different linkage mechanisms, on purpose: compiler = symlink (single
checkout), brain = submodule (version-pinned).

**Edit compiler sources HERE (`wubunos`), never in `wubuos/src/compiler/`.**

## Build

Built entirely from the OS repo (compiler has no standalone Makefile):

```bash
cd /home/wubu/wubuos
make holyd                          # build the compiler driver
make gauntlet_runner                # build the universal test gauntlet
make test_gauntlet                  # run 4,450 tests across 14 ISAs
make test_holyd                     # run holyd's self-tests
```

The `test_gauntlet_runner.c`, `isa-test/`, `peephole_superopt/`, and
`dev/compiler_diff.c` live here now (moved Aug 2026 from the OS repo's
`tools/` dir, which referenced them via `tools/...` before the symlink
repoint). See `~/vault/WALKWAY.md` for the 8-wave improvement plan.