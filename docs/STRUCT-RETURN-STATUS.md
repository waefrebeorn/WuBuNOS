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
