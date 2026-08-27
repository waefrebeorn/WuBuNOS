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
          wubu_mir_interp.c x86_peephole.c wubu_softfloat.c wubu_tgemm.c \
          wubu_host_tensor.c wubu_mir_ssa.c wubu_mir_sccp.c wubu_mir_gvn.c wubu_mir_fuse.c wubu_auto_tune.c

# ISA drivers (10 interpreter-based + 2 JIT stubs + 1 WASM emitter)
ISA     = wubu_isa_driver.c wubu_isa_jit_stubs.c \
          wubu_isa_mips.c wubu_isa_m68k.c wubu_isa_riscv.c \
          wubu_isa_8086.c wubu_isa_6502.c wubu_isa_z80.c \
          wubu_isa_8051.c wubu_isa_avr.c wubu_isa_pic.c \
          wubu_isa_amdgpu.c wubu_isa_ptx.c wubu_elf64_cubin.c \
          jit/wubu_isa_wasm.c

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
FRONT   = holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c \
          holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c \
          holyd_codegen_api.c holyd_runtime.c holyd_mir_eval.c wubu_preproc.c

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
test_isa_driver: $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) test_isa_driver.c jit_stub.c jit_stubs_arm64.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@

# MIR optimizer test
test_mir_opt: $(MIR) $(ISA) $(INTERP) $(OS_INTERP) isa-test/test_mir_opt.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@

# Universal test gauntlet (needs full JIT runtime — see OS Makefile for canonical build)
gauntlet: $(FRONT) $(MIR) $(ISA) $(INTERP) $(OS_INTERP) $(JIT_SRC) $(GAUN)
	$(CC) $(CFLAGS) -I. -Itest_gauntlet $^ $(LDFLAGS) -o gauntlet_runner

# ---- Run targets ----

test: test_isa_driver test_softfloat test_peephole test_elf_cubin test_mir_float test_fuzz_diff test_tgemm test_gap_audit test_selfhost_battery
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
test_fuzz_diff: tools/test_fuzz_diff.c $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) jit_stub.c jit_stubs_arm64.c
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
# Gap audit: adversarial 9P frame fuzzer (2000 inputs across 20 categories)
test_gap_audit: isa-test/gap_audit.c $(MIR)
	$(CC) $(CFLAGS) -I. -o $@ isa-test/gap_audit.c $(MIR)
	./$@

# Self-hosting battery: every C11 construct the compiler must support
# Uses x86-64 JIT + all available interpreter backends.
# Links ALL ISA drivers to satisfy wubu_isa_driver.c's registry,
# including the vulkan stub and jit_stub.c for arm64.
test_selfhost_battery: isa-test/selfhost_battery.c $(FRONT) $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) jit_stub.c jit_stubs_arm64.c
	$(CC) $(CFLAGS) -I. -I$(OS_ROOT) $^ -o $@ -lm
	./$@


clean:
	rm -f test_isa_driver test_mir_opt gauntlet_runner
	rm -f *.o

.PHONY: all test clean gauntlet
test_crash_items: tools/test_crash_items.c
	$(CC) $(CFLAGS) -I. $^ wubu_mir.c wubu_mir_opt.c wubu_mir_lower.c wubu_mir_regalloc.c wubu_mir_interp.c x86_peephole.c wubu_softfloat.c wubu_tgemm.c wubu_host_tensor.c wubu_mir_ssa.c wubu_mir_sccp.c wubu_mir_gvn.c wubu_mir_fuse.c wubu_isa_driver.c wubu_isa_x86_64.c wubu_isa_ptx.c wubu_isa_amdgpu.c wubu_isa_mips.c wubu_isa_m68k.c wubu_isa_riscv.c wubu_isa_8086.c wubu_isa_6502.c wubu_isa_z80.c wubu_isa_8051.c wubu_isa_avr.c wubu_isa_pic.c wubu_elf64_cubin.c wubu_isa_vulkan.c wubu_isa_spirv.c wubu_m68k_interp.c wubu_z80_interp.c wubu_8051_interp.c wubu_avr_interp.c wubu_pic_interp.c holyd_mir_eval.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c holyd_codegen_api.c holyd_mir_eval.c wubu_preproc.c -lm -o $@
	./$@


# Auto-tuning test
test_auto_tune: tools/test_auto_tune.c wubu_auto_tune.c wubu_auto_tune.h wubu_tgemm.c wubu_tgemm.h
	$(CC) $(CFLAGS) -I. $< wubu_auto_tune.c wubu_tgemm.c -lm -o $@
	./$@

# Learned cost model test
test_cost_model: tools/test_cost_model.c wubu_cost_model.c wubu_cost_model.h wubu_auto_tune.c wubu_auto_tune.h wubu_tgemm.c wubu_tgemm.h
	$(CC) $(CFLAGS) -I. $< wubu_cost_model.c wubu_auto_tune.c wubu_tgemm.c -lm -o $@
	./$@

# Genetic kernel evolution test (Phase 4: self-improving compiler)
test_genetic: tools/test_genetic.c wubu_genetic_kernels.c wubu_genetic_kernels.h wubu_genome.h wubu_tgemm.c wubu_tgemm.h
	$(CC) $(CFLAGS) -I. $< wubu_genetic_kernels.c wubu_tgemm.c -lm -o $@
	./$@

# Online learning test (Phase 4: self-improving compiler)
test_online_learn: tools/test_online_learn.c wubu_online_learn.c wubu_online_learn.h wubu_cost_model.c wubu_cost_model.h wubu_auto_tune.c wubu_tgemm.c
	$(CC) $(CFLAGS) -I. $< wubu_online_learn.c wubu_cost_model.c wubu_auto_tune.c wubu_tgemm.c -lm -o $@
	./$@

# FP16 + quantization test (precision format expansion)
test_fp16: tools/test_fp16.c wubu_softfloat.c wubu_softfloat.h
	$(CC) $(CFLAGS) -I. $< wubu_softfloat.c -lm -o $@
	./$@

# HLIR test
test_hlir: tools/test_hlir.c wubu_hlir.c wubu_hlir.h wubu_mir.c wubu_mir.h
	$(CC) $(CFLAGS) -I. $< wubu_hlir.c wubu_mir.c -lm -o $@
	./$@

# HLIR→MIR lowering test
test_hlir_lower: tools/test_hlir_lower.c wubu_hlir.c wubu_hlir.h wubu_mir.c wubu_mir_interp.c wubu_mir.h wubu_softfloat.c
	$(CC) $(CFLAGS) -I. $< wubu_hlir.c wubu_mir.c wubu_mir_interp.c wubu_softfloat.c -lm -o $@
	./$@

# ONNX parser test
test_onnx_parser: tools/test_onnx_parser.c onnx_parser.c onnx_parser.h wubu_hlir.c wubu_hlir.h wubu_mir.c wubu_mir_interp.c wubu_softfloat.c
	$(CC) $(CFLAGS) -I. $< onnx_parser.c wubu_hlir.c wubu_mir.c wubu_mir_interp.c wubu_softfloat.c -lm -o $@
	./$@

# WASM backend test
test_wasm_backend: tools/test_wasm_backend.c jit/wubu_isa_wasm.c jit/wubu_isa_wasm.h wubu_isa_driver.h wubu_mir.h wubu_mir.c
	$(CC) $(CFLAGS) -I. -Ijit $< jit/wubu_isa_wasm.c wubu_mir.c -lm -o $@
	./$@

# MIR pipeline self-hosting battery
# Uses hd_eval_mir() (MIR path) instead of hd_eval() (legacy direct codegen)
# Links ALL ISA drivers to satisfy wubu_isa_driver.c's registry.
test_mir_battery: tools/test_mir_battery.c $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) holyd_mir_eval.c holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c holyd_codegen_api.c holyd_runtime.c wubu_preproc.c jit_stub.c jit_stubs_arm64.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@
	./$@

# Debug: dump MIR for failing battery cases
test_mir_dump: tools/test_mir_dump.c $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) holyd_mir_eval.c holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c holyd_codegen_api.c holyd_runtime.c wubu_preproc.c jit_stub.c jit_stubs_arm64.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@
	./$@

test_mir_debug: tools/test_mir_debug.c $(MIR) $(filter-out wubu_isa_jit_stubs.c,$(ISA)) wubu_isa_x86_64.c wubu_isa_vulkan.c wubu_isa_spirv.c $(INTERP) $(OS_INTERP) holyd_mir_eval.c holyd_lexer.c holyd_parse.c holyd_parse_ast.c holyd_codegen.c holyd_codegen_emit.c holyd_codegen_expr.c holyd_codegen_stmt.c holyd_codegen_api.c holyd_runtime.c wubu_preproc.c jit_stub.c jit_stubs_arm64.c
	$(CC) $(CFLAGS) -I. $^ $(LDFLAGS) -o $@
	./$@
