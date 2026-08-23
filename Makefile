# WuBuNOS — The Compiler Makefile
#
# Standalone build for interpreter-based tests.
# For the full build (x86-64/ARM64 native JIT, HolyC tests, gauntlet),
# build through the OS repo which wires up the JIT runtime:
#   cd /home/wubu/wubunos/../wubuos && make test_isa_driver test_holyc test_gauntlet
#
# Targets:
#   all              — build test_isa_driver + test_mir_opt
#   test_isa_driver  — ISA driver differential test (10 interpreter drivers)
#   test_mir_opt     — MIR optimizer test
#   gauntlet         — build gauntlet_runner (requires OS repo JIT runtime)
#   clean            — remove binaries

CC      = gcc
OS_ROOT ?= /home/wubu/wubuos/src
CFLAGS  = -O0 -g -std=c11 -D_POSIX_C_SOURCE=200809L -DWUBU_HOSTED \
          -include wubu_gnu_compat.h -I. -I$(OS_ROOT)/jit -I$(OS_ROOT)/runtime
LDFLAGS = -lm

# ---- Source groups ----

# MIR mid-level IR + optimizer
MIR     = wubu_mir.c wubu_mir_opt.c wubu_mir_lower.c wubu_mir_regalloc.c \
          wubu_mir_interp.c x86_peephole.c wubu_softfloat.c wubu_tgemm.c

# ISA drivers (10 interpreter-based + 2 JIT stubs)
ISA     = wubu_isa_driver.c wubu_isa_jit_stubs.c \
          wubu_isa_mips.c wubu_isa_m68k.c wubu_isa_riscv.c \
          wubu_isa_8086.c wubu_isa_6502.c wubu_isa_z80.c \
          wubu_isa_8051.c wubu_isa_avr.c wubu_isa_pic.c \
          wubu_isa_amdgpu.c wubu_isa_ptx.c wubu_elf64_cubin.c

# Interpreters (compiler repo)
INTERP  = wubu_m68k_interp.c wubu_z80_interp.c wubu_8051_interp.c \
          wubu_avr_interp.c wubu_pic_interp.c

# OS runtime interpreters (used by 8086, 6502, riscv, mips drivers)
OS_INTERP = $(OS_ROOT)/runtime/wubu_6502_interp.c $(OS_ROOT)/runtime/wubu_riscv_interp.c \
            $(OS_ROOT)/runtime/wubu_mips_interp.c $(OS_ROOT)/runtime/wubu_dos_emu.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_mem.c $(OS_ROOT)/runtime/wubu_dos_emu_regs.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_alu.c $(OS_ROOT)/runtime/wubu_dos_emu_int.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_decode.c wubu_softfloat.c

# Frontend (for gauntlet which needs hc_eval_mir)
FRONT   = holyc_lexer.c holyc_parse.c holyc_parse_ast.c holyc_codegen.c \
          holyc_codegen_emit.c holyc_codegen_expr.c holyc_codegen_stmt.c \
          holyc_codegen_api.c holyc_runtime.c holyc_mir_eval.c wubu_preproc.c

# JIT runtime (for gauntlet — provides jit_alloc_exec, jit_lock_exec, etc.)
JIT_SRC = $(wildcard $(OS_ROOT)/jit/jit.c $(OS_ROOT)/jit/jit_encode.c \
                $(OS_ROOT)/jit/wubu_x86.c $(OS_ROOT)/jit/wubu_disasm.c \
                $(OS_ROOT)/jit/wubu_arm64.c $(OS_ROOT)/jit/x86_regalloc.c \
                $(OS_ROOT)/jit/jit_codegen_x86.c $(OS_ROOT)/jit/wubu_rv64.c \
                $(OS_ROOT)/jit/wubu_wasm.c $(OS_ROOT)/jit/jit_codegen_rv64.c \
                $(OS_ROOT)/jit/jit_codegen_wasm.c \
                $(wildcard $(OS_ROOT)/jit/jit_minic*.c $(OS_ROOT)/jit/jit_codegen_arm64.c \
                          $(OS_ROOT)/jit/jit_branch_profile.c $(OS_ROOT)/jit/jit_minic_cg.c))

# Gauntlet test framework
GAUN    = test_gauntlet/wubu_test_gauntlet.c test_gauntlet_runner.c \
          test_gauntlet/suites/gauntlet_comprehensive.c \
          $(wildcard test_gauntlet/suites/gauntlet_gcc_*.c) \
          $(wildcard test_gauntlet/suites/gauntlet_fujitsu.c) \
          $(wildcard test_gauntlet/suites/gauntlet_extern_*.c) \
          $(wildcard test_gauntlet/suites/gauntlet_compcert.c) \
          $(wildcard test_gauntlet/suites/gauntlet_c_testsuite.c) \
          $(wildcard test_gauntlet/suites/gauntlet_llvm.c) \
          $(wildcard test_gauntlet/suites/gauntlet_lacc.c) \
          $(wildcard test_gauntlet/suites/gauntlet_tinycc.c) \
          $(wildcard test_gauntlet/suites/gauntlet_chibicc.c) \
          $(wildcard test_gauntlet/suites/gauntlet_writing_*.c) \
          $(wildcard test_gauntlet/suites/gauntlet_slimcc.c)

# ---- Binaries ----

all: test_isa_driver test_mir_opt

# ISA driver differential test (10 interpreter-based drivers)
test_isa_driver: $(MIR) $(ISA) $(INTERP) $(OS_INTERP) test_isa_driver.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@

# MIR optimizer test
test_mir_opt: $(MIR) $(ISA) $(INTERP) $(OS_INTERP) isa-test/test_mir_opt.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@

# Universal test gauntlet (needs full JIT runtime — see OS Makefile for canonical build)
gauntlet: $(FRONT) $(MIR) $(ISA) $(INTERP) $(OS_INTERP) $(JIT_SRC) $(GAUN)
	$(CC) $(CFLAGS) -I. -Itest_gauntlet $^ $(LDFLAGS) -o gauntlet_runner

# ---- Run targets ----

test: test_isa_driver test_softfloat test_peephole test_elf_cubin test_mir_float test_fuzz_diff test_tgemm
	@echo "=== ISA Driver Test ==="
	./test_isa_driver

# Soft-float runtime test (pure C11 IEEE-754 software float)
test_softfloat: wubu_softfloat.c wubu_softfloat.h tools/test_softfloat.c
	$(CC) $(CFLAGS) -I. -o $@ tools/test_softfloat.c wubu_softfloat.c -lm
	./$@

# ELF64 cubin container writer
test_elf_cubin: wubu_elf64_cubin.c wubu_elf64_cubin.h tools/test_elf_cubin.c
	$(CC) $(CFLAGS) -I. -o $@ tools/test_elf_cubin.c wubu_elf64_cubin.c
	./$@

# x86 peephole self-tests (driven by the superoptimizer discovery loop)
test_peephole: x86_peephole.c x86_peephole.h tools/test_x86_peephole.c
	$(CC) $(CFLAGS) -I. -o $@ tools/test_x86_peephole.c x86_peephole.c
	./$@

# MIR-level float ops through the soft-float runtime
test_mir_float: wubu_mir.c wubu_mir_interp.c wubu_mir_opt.c wubu_mir_lower.c \
                wubu_mir_regalloc.c x86_peephole.c wubu_softfloat.c tools/test_mir_float.c
	$(CC) $(CFLAGS) -I. -o $@ tools/test_mir_float.c wubu_mir.c wubu_mir_interp.c \
		wubu_mir_opt.c wubu_mir_lower.c wubu_mir_regalloc.c x86_peephole.c wubu_softfloat.c -lm
	./$@

# Differential fuzz oracle: random-MIR cross-check across backends
# (x86-64 JIT, 8086, interpreter) — finds correctness bugs via differential
# comparison, no per-case oracle needed.
test_fuzz_diff: tools/test_fuzz_diff.c $(MIR) $(ISA) $(INTERP) $(OS_INTERP)
	$(CC) $(CFLAGS) -I. -o $@ $^
	./$@ 3000

# T_GEMM tensor-op: correctness vs naive C + interpreter/interp/JIT timing
# Lean build: real x86-64 JIT driver + all interp drivers + jit_stub for exec alloc.
# NOTE: filter out wubu_isa_jit_stubs.c (which provides a stub wubu_isa_x86_64)
# so the real JIT in wubu_isa_x86_64.c links instead.
test_tgemm: tools/test_tgemm.c $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) \
            $(INTERP) $(OS_INTERP) wubu_isa_x86_64.c jit_stub.c \
            jit_stub_arm64.c
	$(CC) $(CFLAGS) -I. -I$(OS_ROOT) -o $@ $^
	./$@ 24 24 24

clean:
	rm -f test_isa_driver test_mir_opt gauntlet_runner
	rm -f *.o

.PHONY: all test clean gauntlet
