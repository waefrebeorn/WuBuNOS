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
