# Goal 118 — 16-byte stack alignment: measured, oracle identified, tracker missing

## Measured

Across 18 compiled programs (3–20 live values) the prologue is always
`48 81 EC imm32` (`sub rsp, imm`) with **`imm ≡ 8 (mod 16)`** — 18 of 18, e.g.
`nv=20 sub rsp=168`. That follows from `frame = n_spilled*8 + 16 + 64`: whenever
the spill count is even, the frame is 8 mod 16.

This is **not** by itself a violation. Entry to a function is at `rsp ≡ 8`
(after the caller's `call` pushed the return address), so a further 8 keeps the
same parity and the encoder compensates at call sites:
`wubu_isa_x86_64.c:1372` emits `sub rsp,8` before `push K`, with a comment
explaining that `K` must land at `[rsp+8]` once the call pushes its return
address.

Conclusion: asserting `frame % 16 == 0` in the prologue tests the wrong thing —
it is what produced a false failure (18/18 "misaligned") and was reverted.

## Correct oracle

Walk the decoded stream and track an rsp delta relative to function entry
(`entry ≡ 8 mod 16`):

- `push r64` → delta −= 8
- `pop r64`  → delta += 8
- `48 81 EC imm32` (`sub rsp, imm`) → delta −= imm
- `48 81 C4 imm32` / `48 83 C4 ib` (`add rsp, …`) → delta += imm
- `48 89 EC` (`mov rsp, rbp`) → delta unknown from here; stop tracking
- `E8 rel32` (`call`) → assert `(8 + delta) % 16 == 0`, which is the SysV
  requirement (`rsp ≡ 0 mod 16` immediately before the call)

The driver already links the encoder and the decoder, so this lives in
`test_isa_driver.c` next to the REX and frame checks.

## Missing piece

`x86_decode` reports `imm_bytes` but not the immediate **value**. To feed the
tracker it needs `int64_t imm` populated for `imm_bytes` 1/4/8 (and the sign
convention: `sub rsp` immediates are unsigned here, `add`/`rel32` signed).

## Also still true

A program with no `MIR_CALL` emits no `E8`, so the tracker must **prove it saw
at least one call site** before it can claim anything — otherwise it is the same
vacuous pass as the 3-local frame test.
