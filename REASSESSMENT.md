# WuBu Reassessment & Replan — 2026-10-12

> Honest status against the 1000-goal list, replacing the stale
> GAP_ANALYSIS.md. Measured this session, not assumed.

## Where things ACTUALLY stand (measured)

### Compiler (wubunos)
- **Core gauntlet: `pass=3458 fail=0 error=0`** (247 cases). 100%.
- **Self-host battery: 155/155** — the compiler compiles itself.
- **Full `make test`: 195/195.**
- **Many-arg functions work**: verified 8/9/12/16-arg calls return correct
  results via `test_single`. The flat-VR model + allocator spill already
  handle arbitrary arg counts. (GAP_ANALYSIS's "9+ args broken / no stack
  frame" is STALE — there IS an rbp frame, and >8 args spill correctly.)
- **NaN fix landed this session**: f32 FLT/FLE were NaN-incorrect (bare
  setb/setbe); now guarded with setnp. fuzz_diff 0 mismatches.
- **Full depot: 4588 tests / 14 ISAs.** The `--core` suite is 100%; the
  full depot includes ~1547 upstream GCC/LLVM/LACC extractions that use
  unsupported features (VLA, `__attribute__`, GCC extensions) — these are
  "EROR (parse)" not correctness bugs. The real frontier is raising
  full-depot coverage, not fixing core correctness.

### 1000-goal list
- **21/1000 marked done.** Concentrated in NAMESPACE (6), UX (9), AGENT (5),
  RUNTIME (1). **Zero in COMPILER, BRAIN, MATH, WORLDSIM.**
- Honest read: I shipped a lot of small, correct, well-tested meta-layer and
  UX modules and narrated progress, instead of driving the hard high-value
  domains. The easy breadth is done; the hard depth is untouched.

## What "done" really means here

The 1000 list is a roadmap, not a checklist to tick by shipping thin
modules. Real progress = the hard domains moving with verified evidence:
- COMPILER: full-depoit gauntlet % climbing, more upstream features parsing.
- BRAIN: a real training run that improves a measured metric.
- MATH: Lean theorems proving (not stubs).
- WORLDSIM: a physics sim matching an analytic reference.

## Replan — the real high-value work, in priority order

1. **COMPILER full-depot coverage** (biggest lever, measurable). The core is
   100%; push the full 4588-test depot. Attack the largest EROR clusters:
   char/string types (byte-addressable), unions (35 cases), full pointers,
   VLA. Each is a real architectural feature with a measurable gauntlet delta.
   - Byte-addressable memory (chars/strings) — enables the most cases.
   - Union layout (max-size, shared offset 0) — 35 cases, contained.
   - Parse support for common GCC/LLVM extraction features (reduce EROR).
2. **BRAIN: a real measured training run.** Not more modules — run
   wubu_train on a corpus and show loss decreasing on a held-out metric.
3. **MATH: prove one real theorem in Lean** (Poincaré metric), not a stub.
4. **WORLDSIM: one physics primitive matching its analytic reference.**

## Discipline going forward
- Every claim backed by a real command's output (gauntlet number, test
  count, loss curve) — never narrative.
- Prefer depth in a hard domain over breadth of small modules.
- Update this doc with measured numbers each session.
