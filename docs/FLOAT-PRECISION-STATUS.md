# Float precision — `float` is not single precision (2026-10-19)

## Measurement

Probe `float add` in `test_mir_battery`:

```
float a=1.5; float b=2.5; a+b;     expect 1082130432
```

Everything along the path was checked separately:

| step | result |
|---|---|
| `1.5;` literal | `0x3FF8000000000000` = 1.5 double — correct |
| `float a=1.5; a;` | `0x3FF8000000000000` — correct |
| two vars, read each | both correct |
| `[dadd]` operands at the opcode | `1.5000` and `2.5000` — correct |
| `wubu_sf_f64_add(1.5, 2.5)` in isolation | `0x4010000000000000` = 4.0 — correct |
| interpreter return (`[done] result`) | `0x4010000000000000` = **4.0** |

The battery's expected value is `1082130432` = `0x40800000`, which is **4.0 as
a float32**. The interpreter produces **4.0 as a float64**, and that is exactly
what it should do given the type system it has.

## Root cause

`grep -c HD_TYPE_F32 holyd_types.h` → **0**. There is no single-precision type in
the AST type system, so:

- `float` in source is bound as `HD_TYPE_F64`;
- every float literal is `HD_TYPE_F64`;
- `MIR_DADD` (f64) is chosen, never `MIR_FADD` (f32);
- the result is a double, bit-for-bit correct for a double.

`MIR_FADD` exists in `wubu_mir.h` and the ISA backends have f32 support, but the
front end cannot express an f32 value, so that opcode is unreachable from
source.

## What this means for the probe

`float add` cannot pass without adding single-precision to the type system. The
test is not wrong — it encodes C semantics where `float` arithmetic is
single-precision — but neither is the arithmetic path. The gap is upstream:
**no `HD_TYPE_F32`**.

## Work required (not done)

1. Add `HD_TYPE_F32` to the type-kind enum with `size = 4`.
2. Map the `float` keyword to it in the parser's type keywords (today it falls
   through to F64).
3. Emit `MIR_FADD/FSUB/FMUL/FDIV` for f32 operands, and convert (f32→f64)
   on widening, with `MIR_F2D`/`MIR_D2F`-style casts where the backend has them.
4. Make `sizeof(float) == 4` and keep the storage cell aligned to 8 bytes with
   the value truncated to f32 on store.

Until then the probe stays WRONG with this note as the reason, and no change to
the ADD path is warranted — the ADD path is correct.
