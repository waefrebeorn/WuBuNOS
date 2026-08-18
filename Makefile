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
          wubu_mir_interp.c x86_peephole.c

# ISA drivers (10 interpreter-based + 2 JIT stubs)
ISA     = wubu_isa_driver.c wubu_isa_jit_stubs.c \
          wubu_isa_mips.c wubu_isa_m68k.c wubu_isa_riscv.c \
          wubu_isa_8086.c wubu_isa_6502.c wubu_isa_z80.c \
          wubu_isa_8051.c wubu_isa_avr.c wubu_isa_pic.c \
          wubu_isa_amdgpu.c wubu_isa_ptx.c

# Interpreters (compiler repo)
INTERP  = wubu_m68k_interp.c wubu_z80_interp.c wubu_8051_interp.c \
          wubu_avr_interp.c wubu_pic_interp.c

# OS runtime interpreters (used by 8086, 6502, riscv, mips drivers)
OS_INTERP = $(OS_ROOT)/runtime/wubu_6502_interp.c $(OS_ROOT)/runtime/wubu_riscv_interp.c \
            $(OS_ROOT)/runtime/wubu_mips_interp.c $(OS_ROOT)/runtime/wubu_dos_emu.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_mem.c $(OS_ROOT)/runtime/wubu_dos_emu_regs.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_alu.c $(OS_ROOT)/runtime/wubu_dos_emu_int.c \
            $(OS_ROOT)/runtime/wubu_dos_emu_decode.c

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
          test_gauntlet/suites/gauntlet_comprehensive.c

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

test: test_isa_driver
	@echo "=== ISA Driver Test ==="
	./test_isa_driver

clean:
	rm -f test_isa_driver test_mir_opt gauntlet_runner
	rm -f *.o

.PHONY: all test clean gauntlet
