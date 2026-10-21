# PORTFOLIO - REALITY CHECK (GOALS-1000 stack: wubunos)

Per-goal capability evidence. Each entry names file:line in Makefile and the
owning test target that prints ALL PASSED.

## n831  STACK/P1  wubunos/Makefile
**Capability:** header-dependency tracking with -MMD -MP plus -include .d.
**Evidence:** Makefile:18 adds -MMD -MP to CFLAGS; Makefile:32 adds
-include $(wildcard *.d tools/*.d); Makefile:150 defines test_header_deps, which
compiles tools/test_header_deps.c (it #includes tools/test_header_deps.h) and
greps the emitted test_header_deps.d to prove the header is tracked.
**Test:** run: make test_header_deps  -> prints ALL PASSED

## n834  STACK/P1  wubunos/Makefile
**Capability:** zero-warning gate -- CFLAGS carries -Wall -Wextra and a locked
baseline (tools/warning_baseline.txt) pins the existing warning set so the
compiler build cannot regress.
**Evidence:** Makefile:19 adds -Wall -Wextra to CFLAGS; Makefile:358 defines
test_zero_warnings (rebuilds the full compiler via `make -B test_isa_driver`,
greps warning: lines, and diffs against tools/warning_baseline.txt);
tools/warning_baseline.txt locks the 125-warning baseline.
**Test:** make test_zero_warnings prints ALL PASSED.

## n844  STACK/P2  wubunos/Makefile
**Capability:** compile-flag parity gate -- every compiler recipe is classified as
SAME (canonical CFLAGS) or DEVIATES (GAUNTLET_CFLAGS / explicit codegen override),
locked in a manifest; the gate fails on new or undocumented drift.
**Evidence:** Makefile defines test_compile_flag_parity; tools/check_compile_flags.py
classifies every recipe flag source; tools/compile_flag_parity.txt locks the
49-recipe manifest (8 documented deviations).
**Test:** make test_compile_flag_parity prints ALL PASSED.
