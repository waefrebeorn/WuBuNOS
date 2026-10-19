# Goal 122 — memory-operand clobber: the obvious invariant is wrong

Registry goal 122 wants `p[i] + p[j]` and `*(p+i) * q` to survive without the
SIB index or base being clobbered by the binop.

## What was tried

A single-instruction invariant over the decoded stream: for every instruction
with a memory operand, the destination register (`modrm.reg`) must differ from
the base, index and SIB-base registers of that operand.

On an 18-value spilling program this reported **1 hit out of 21 memory
operands** — and the hit was a legitimate store-to-address (`mov [rax], rax`,
storing a value to the address it already holds). That instruction is correct:
x86 computes the effective address before the write, so a register may be both
the address and the value.

So the invariant is too strict and the failure was a false positive. The test was
removed rather than loosened until it passed; `test_isa_driver` is back to
200/200 with the frame and REX checks intact.

## What a real check needs

Clobbering is a **sequence** property, not an instruction property: the bug is
"compute address into R, then overwrite R, then use it as base/index". That
requires walking instructions and tracking, per register, whether it is live as
an address between definition and use — a small dataflow pass over the decoded
stream, not a per-instruction predicate.

The decoder already reports `modrm_reg/rm`, SIB and displacements, so the data
pass has what it needs. Goal 122 stays open until that pass exists.
