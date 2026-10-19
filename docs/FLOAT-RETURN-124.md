# Goal 124 — float/double returns in xmm0: not implemented, and not a test gap

Registry goal 124 wants `double f(void)` and `float g(void)` to return in `xmm0`
on the JIT.

## What the code actually does

`wubu_isa_x86_64.c` handles `MIR_RET` and `MIR_FRET` in **one shared case**
(line ~1653). The return value is moved through the **general-purpose**
register discipline (`emit_mov_reg`, `emit_store_rbp`, `emit_load_rbp` on the
return slot `vr 0`). Nothing in that path touches an xmm register.

Float arithmetic elsewhere is also GPR-carried: the encoder moves between
`rax` and `xmm0` inline (`66 48 0F 6E C0` / `66 48 0F 7E C0`), does the SSE op
in `xmm0`, then moves the bits back to `rax`. In other words the JIT is
**soft-float shaped**: f64 values live as bit patterns in integer registers, and
`xmm0` is scratch inside an expression, not a return convention.

## Consequence

Two float-returning functions cannot interoperate with SysV-compiled code, and
`MIR_FRET` carries no meaning distinct from `MIR_RET`. The criterion cannot be
satisfied by adding a test: the return convention itself does not exist.

Meeting it means:

1. Give `MIR_FRET` its own emission: load the value into `xmm0` (`movq`/`movsd`)
   and mark the function's return class in `funcs[]`.
2. Teach every call site (`MIR_CALL`) to expect an xmm0 result for such
   functions, and to spill it into a VR rather than assuming rax.
3. Keep the GPR path for internal calls so the existing soft-float programs keep
   working, or migrate them wholesale.

That is a calling-convention change across the encoder and every caller, so it
is not a candidate for a quick pass. Goal 124 stays open with this reason, next
to the other ISA goals that need a real design decision rather than a test.

## Relationship to 175

Goal 175 (SSE2 f64 arithmetic encoding bit-exact vs softfloat) is a *different*
claim and can be verified without touching returns: the arithmetic already runs
in xmm registers. 124 is about the ABI boundary only.
