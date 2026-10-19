# Struct by-value returns in the MIR interpreter — status

Measured 2026-10-19 with `test_mir_battery` (source → parser → MIR →
interpreter, the oracle path; `hd_eval_mir(src, NULL)`).

## Battery totals

```
PASS 106   WRONG 14   CRASH 0   TOTAL 120
```

`sizeof` is now fully green (goal 128 closed: array typedef 40, padded struct
16, pointer 8). The remaining 14 wrong cases cluster into two families.

## Family A — struct returns by value (7 cases)

```
struct return           expect 7
struct return 2m        expect 49
struct return member    expect 42
struct return member b  expect 7
struct return chain     expect 99
struct return global    expect 55
```

All of them return a struct from a function and then read a member. The battery
prints only the expected value, not the actual one, so the first debugging step
is a probe that prints both.

Existing notes say the intended convention is "callee copies the struct into a
buffer and returns the buffer address in v0, caller reads members through it"
(also: DOT on a call result must handle FUNC_CALL as the left operand). Either
that convention is not implemented in `holyd_mir_eval.c`, or the call site drops
the returned address and reads the callee's frame instead.

## Family B — float literal/arithmetic (rest)

```
float add   expect 1082130432   (1.5f + 2.5f as a bit pattern)
```

Float values are GPR-carried in this backend with `xmm0` as inline scratch, and
`docs/FLOAT-RETURN-124.md` already records that returning a float in `xmm0` has
no calling convention at all. `float add` failing is consistent with that gap,
not with a separate arithmetic bug.

## Why this is one cluster, not seven goals

Six of the seven Family A cases exercise the same ABI path; fixing the return
convention should flip them together. That is why the registry's
`holyd_mir_eval.c` cluster (7 open goals) is the right next target rather than
closing individual entries.

## Next step

Write a probe that prints actual vs expected for `struct return`, then read the
`MIR_RET`/`MIR_CALL` paths in `holyd_mir_eval.c`. Do not touch `MIR_FRET` until
the struct convention is settled — the two interact through v0.

## Actual values (measured, not assumed)

The harness now prints the computed value on failure (`g_last_result`). That turns
two vague families into two precise defects:

```
WRONG struct return            expect 7,        got 0
WRONG struct return 2m         expect 49,       got 0
WRONG struct return member     expect 42,       got 0
WRONG struct return member b   expect 7,        got 0
WRONG struct return chain      expect 99,       got 0
WRONG struct in struct return  expect 77,       got 0
WRONG struct return global     expect 55,       got 0

WRONG float add                expect 1082130432, got 4616189618054758400
```

**Struct returns evaluate to exactly 0** — the returned buffer address never
reaches the caller (or v0 is overwritten by the epilogue). This is one defect for
seven cases, as suspected.

**Float arithmetic is integer arithmetic.** `1082130432` is `0x40800000` = `4.0f`;
`4616189618054758400` is `0x4008000000000000` = `3.0` as a double. So `1.5 + 2.5`
was computed as `1 + 2`, and the result was then reinterpreted as a double
bit-pattern. Float literals are being parsed/truncated to integers in this path —
a lexer/constant-pool issue, not an ABI one, and separate from the `xmm0`
return gap in `docs/FLOAT-RETURN-124.md`.

## Root cause (identified in the interpreter)

`wubu_mir_interp.c`, `op_ret:` restores the caller's register file and then:

```c
memcpy(cmem + f->frame_off, csave, f->frame_cells * sizeof(int64_t));
```

i.e. **every cell at or above `globals_end` is rolled back to its
call-time contents**. The struct-return buffer is allocated at compile time by
`wubu_mir_alloc(g->prog, size * 8)` — a static address in the MIR's memory, and
for nested frames it lands *above* `globals_end`.

Sequence for `struct A f(){ struct A s; s.x=7; return s; } t=f(); t.x;`:

1. callee copies `s` into the static buffer and puts the buffer address in v0;
2. `op_ret` reads the address into `retval`, restores `vr`, and **restores the
   frame memory — wiping the buffer that was just written**;
3. the caller reads member `x` through the address and gets 0.

That is exactly the observed value: every struct-return probe returns 0, not a
garbage number. The return convention is implemented; the call-frame restore
destroys its result.

## Fix options (next pass)

1. **Exclude the buffer from the restore** — needs the buffer range recorded in
   the program (`struct_ret_base`/`struct_ret_bytes`) and the memcpy split into
   the ranges below and above it.
2. **Allocate buffers below `globals_end`** so they are file-scope persistent,
   which the restore already leaves alone. Simplest and matches the intent of
   "the buffer outlives the callee's frame".
3. **Copy the buffer out at return** — needs the size at `op_ret`, which the
   instruction does not carry.

Option 2 is the smallest change and the one the comment in `HD_AST_RETURN`
already implies.

## Why `sizeof(s)` is 16 for two ints (design limit, not a maths bug)

Struct layout here is **cell-granular**: `mir_struct_t.total_size` counts int64
cells, and every member is rounded up to a whole cell (`hd_parse.c` sets
`type->size` per member). So `struct S{int a;int b;}` occupies 2 cells = 16
bytes, and `sizeof(s)` correctly reports the storage it actually has.

Measured: reverting only the `sizeof(typedef-name)` change leaves both
`sizeof expr struct` and `sizeof in expr` failing identically, so they are
pre-existing and unrelated to that fix (PASS 114 vs 115 with it).

Consequences:
- `sizeof(struct{char;int;double}) == 16` passes because 3 cells is also the
  padded C size — the two models agree by coincidence there.
- `sizeof(int[10]) == 40` passes because arrays scale cells to bytes explicitly.
- Two `int`s disagree because C packs them into 8 bytes while this model gives
  each member a full 8-byte cell.

Making these byte-exact is a **struct-layout change** (member offsets in bytes,
alignment/padding, then cells = ceil(bytes/8)), not a `sizeof` tweak. It also
determines whether `struct in struct return` can copy nested members correctly,
since the arena copy loop is driven by `total_size`. Until then both probes stay
OPEN with this note as the reason.
