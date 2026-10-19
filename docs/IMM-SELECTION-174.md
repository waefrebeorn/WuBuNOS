# Goal 174 — immediate selection: the criterion as written is not encodable

Registry goal 174 (module `wubu_isa_x86_64.c`) says:

> make test_isa_driver green: 0x1234 uses imm32, 0x123456789ABCDEF0 uses movabs

## What the encoder does today

`wubu_isa_x86_64.c`, `MIR_CONST` (around line 733):

```c
/* mov reg, imm — on x86-64 REX.W + B8+rd is ALWAYS movabs
 * (imm64); there is no mov r64, imm32 form. So emit imm64. */
```

That comment is correct. For a 64-bit destination there is **no** `mov r64, imm32`
encoding: `REX.W + B8+rd` takes a full imm64. So "0x1234 uses imm32" cannot be
met by shortening the B8 form — that form does not exist.

A shorter encoding *does* exist for small values: `REX.W + C7 /0 id`
(`mov r/m64, imm32`, sign-extended), 7 bytes instead of 10. It is valid only
when the value fits in a signed 32-bit range, i.e. `-2^31 <= v <= 2^31-1`.
So the achievable version of the goal is:

- `0x1234` → `48 C7 C0+rd 34 12 00 00` (7 bytes)
- `0x123456789ABCDEF0` → `48 B8+rd …8 bytes…` (10 bytes, movabs)

## Why it was not implemented this pass

`x86_peephole.c` already contains `shrink_movabs`, and it is **commented out** at
its call site with the reason:

```c
// n = shrink_movabs(code, n);   /* REMOVED: changes instr sizes, corrupts rel32 offsets */
```

A post-pass that rewrites instruction lengths invalidates every rel32 that
follows it. Re-introducing the shrink at the encoder (where the size is chosen
before any branch offset is computed) avoids that, but it changes emitted
lengths for every constant in the program — so it must be verified against the
full `test_isa_driver` (195 checks) and the JIT/backpatch path, not against a
handful of bytes. That verification is the work, and it was not started here
rather than half-done.

## Corrected criterion

> 0x1234 is emitted as `C7 /0 id` (7 bytes) and 0x123456789ABCDEF0 as `B8+rd
> imm64` (10 bytes); a value outside int32 range but inside uint32 (e.g.
> 0x80000000) stays movabs, because C7 sign-extends and would change the value.

The `0x80000000` case is the one a naive "fits in 32 bits" test gets wrong, and
it belongs in the criterion so the trap is written down.
