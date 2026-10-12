# WuBuNOS Compiler — Research Fix List (Kevin-Bacon converged)

Every gap below is an AGI-substrate gap: the compiler is what the AGI *thinks in*.
Each fix cites the proven approach from a real compiler (chibicc, TCC, claudes-c-compiler,
CMU 15-411, System V ABI) and maps it to our exact code site.

Legend: 🔴 = high impact / blocks many tests · 🟡 = medium · ⚪ = architectural (bigger)

---

## GAP 1 — 🔴 Recursion: two calls in one expression clobber the first result
**Symptom:** `fib(n-1)+fib(n-2)` → 0; `fib(n-1)+0` → 1 (correct). Two recursive
(or any two) calls in one binop: the first call's return value is destroyed by the
second call. Sequential `a=fib(2); b=fib(2)` also fails. Single-call recursion works.

**Root cause (traced):** frontend emits `a=gen_expr(left); b=gen_expr(right); ADD(a,b)`.
Both `a` and `b` are VRs. The return value lives in VR0 = physical reg 0 = **r10**
(caller-saved, and NOT in the save/restore list `caller_saved[]={r11,r12,r13,r14,r15,r8,r9,rdx}`).
The callee-saved preference I added (rv→r12) did not fix it — the interval/cross-call
detection isn't landing rv1 in a preserved home reliably.

**Proven fix (chibicc codegen.c `gen_expr` ND_ADD/ND_FUNCALL):**
chibicc never keeps two live values in registers across a sub-call. Pattern:
```
gen_expr(rhs); push();        // first operand result -> STACK
gen_expr(lhs); pop("%rdi");   // reload into a scratch reg
add %rdi, %rax
```
and for calls:
```
for each arg: gen_expr(arg); push();
pop args into argreg64[i];    // AFTER all args evaluated
call
```
**Our mapping:** in `holyd_mir_eval.c` binop (line ~2737) and CALL arg setup (~4620),
when an operand contains a MIR_CALL, force the *other* operand's VR to a **stack
(spill) home** before emitting the second call — i.e. materialize `a` to its stack
slot (`mov [rbp-X], a_home`) right after the first call, so the second call cannot
touch it. Simplest robust version: always spill a call-result VR to its stack slot
immediately (set `assign[rv].reg = -1`), making every call result memory-resident.
Correctness first; the regalloc can re-optimize later.

**MEASURED IMPACT (0cd0145 spill + 6678d05 epilogue):** the regalloc spill
(VRs whose live range crosses a MIR_CALL → stack slot) dropped the x86-64 gauntlet
FAIL count from **329 → 142** (a 57% reduction) — validating GAP 1+2 as the
highest-leverage fix. The epilogue fix (called functions unwind their frame) fixes
fib(2)=1. Both gated 195/195 + selfhost 155/155 + fuzz_diff 0.

**STATUS:** fib(2) fixed. Remaining: fib(10) and `x=5;y=fib(1)` still fail — a
residual operand-preservation-across-calls bug (a local/call-result stored before a
second call is clobbered by it). Next probe: whether the second call's frame or the
8-reg save/restore overwrites the first operand's spill slot / local store. **Refined root cause
(traced via MIR dump):** the MIR is correct (`store mem[g]=v46; call fib; mov v65544=v0;
store mem[h]=v59`). The bug is in x86 codegen of **two sequential calls to the SAME
function**: the internal-call register save/restore (`wubu_isa_x86_64.c` ~1594,
`caller_saved[]={r11,r12,r13,r14,r15,r8,r9,rdx}` pushed before, popped after) interacts
badly with the recursive callee's own frame. Symptom: the LAST global store before the
second recursive call is dropped (`h=fib(n-2)` leaves h unchanged; `g=fib(n-1)` works).
Single-call recursion and two-calls-to-DIFFERENT-functions both work. Next probe: the
8-reg push/pop balance vs the callee prologue/epilogue, and whether r10 (VR0 return reg,
excluded from the save list) survives the second same-function call. NOT yet fixed.
**Verify when fixed:** fib(2)=1, fib(10)=55, `a=fib(2);b=fib(2)`→11, ackermann.

---

## GAP 2 — 🔴 Callee-saved vs caller-saved discipline in the x86 regalloc
**Symptom:** same as GAP 1 root; any VR live across a call may land in r10/r11/r8/r9/rdx
(caller-saved) and be clobbered.

**Proven fix (claudes-c-compiler arm/codegen README + CMU 15-411 lab 3):**
Two-pool linear scan: **callee-saved regs (r12-r15) for values whose live range spans a
call; caller-saved regs only for call-free ranges.** The callee saves/restores callee-saved
regs it uses in prologue/epilogue (we already `push rbx`; must also push any r12-r15 used).
**Our mapping:** `wubu_mir_regalloc.c` Step 5 free-reg search — I added
`range_crosses_call()` + `x86_callee_saved[]={2,3,4,5}` preference, but it must be paired
with GAP 1's spill (belt-and-suspenders) until the interval math is proven. Also the
internal-call save/restore (`wubu_isa_x86_64.c` ~1594) currently saves r12-r15 around
*every* call — if we trust callee-saved homes we can drop r12-r15 from that list to save
work, but ONLY after GAP 1 is solid. **Do not remove the save/restore yet.**

---

## GAP 3 — 🔴 Float/double return from internal calls uses wrong register
**Symptom:** `double r=fabs(-3.0); (int)r` → 0 (external works, internal broken).
Internal float-returning functions put the result in rax, but SysV says XMM0.

**Proven fix (System V ABI + chibicc `load`/return typing):**
Scalar float/double return in **XMM0**, int/pointer in **RAX**. chibicc's `gen_expr`
leaves float results in xmm0 and only moves to rax for int context.
**Our mapping:** `holyd_mir_eval.c` frontend already tracks float-returning builtins
(line ~568 `fret[]`). Internal `MIR_RET`/`MIR_FRET` (x86 ~1625) must, for a float-typed
function, `movq xmm0, <val>` and the caller must read xmm0→ its float VR (DADD/DCMP path),
not rax. The CALL capture `mov_to(rv, 0)` must branch on the callee's return type:
float → capture from xmm0, int → from rax/r10. Tag each function's return type in
`g->prog->funcs[fid]` (we have `fret[]`; persist it).

---

## GAP 4 — 🔴 Byte-addressable memory / 4-byte int layout
**Symptom:** `access_through_char_pointer.c` fails at `byte_ptr+16`; `sizeof(int[2][2])`
mismatches; 2D char array from string vs explicit-init disagree on row stride.
**Root cause:** the cell model allocates **1 cell = 8 bytes per element**, but C says
int=4 bytes, char=1 byte, little-endian. `char[8]` = 1 cell (byte-packed, OK) but
`int[4]` = 4 cells (32 bytes) vs C 16 bytes. Type-punning `(char*)&int` then reading
byte 4..7 reads the *next* cell, not the int's upper bytes.

**Proven fix (C memory model, CMU 15-122; chibicc `load`/`store` by type):**
Real compilers lay out `int` at 4 bytes, `char` at 1, little-endian, and `load`/`store`
emit width-appropriate moves (`movl` for int, `movb` for char). sizeof uses true byte sizes.
**Our mapping (architectural, ⚪):** either (a) make the allocator byte-accurate
(int→4 bytes, pack multiple per cell) — large change touching `wubu_mir_alloc`, all
load/store, `hd_type_size`, sizeof; or (b) keep the cell model but make **type-punning
reads** byte-exact within a cell (a char* into an int cell reads the int's 4 bytes then
the cell's padding as 0). Option (b) is far less invasive and fixes the specific tests.
The 2D-char row-stride conflict (GAP 5) is the same root.

---

## GAP 5 — 🟡 2D char array: string-init vs explicit-init row stride conflict
**Symptom:** `char n[2][6]={"Hi","Yo"}` needs rows byte-packed (6 bytes);
`char n[2][6]={{72,0},{87,0}}` needs cell-strided rows (8 bytes). Same type, two layouts.
**Proven fix:** a single truthful layout — rows are `array_size` **bytes** (6), and
element access within a row is byte-strided; the allocator reserves `rows*row_bytes`.
Once GAP 4 (byte-accurate sizing) lands, this resolves automatically. Until then,
revert attempts that fix one case and break the other (done twice already — keep reverted).

---

## GAP 6 — 🟡 char*[] multi-level indexing (`s[0][0]`, `*s[i]`)
**Symptom:** `char *s[]={"yes"}; s[0][0]` reads full cell / crashes.
**Root cause:** the walk-to-base-IDENT in `mir_index_stride`/char_elem stops at the root
var (`s`: ARRAY(PTR(I8))) and misses the inner `char*` → doesn't set byte stride/byte-load.
**Proven fix (chibicc type-driven load):** resolve the element type by walking the FULL
type chain (ARRAY→PTR→I8), not just the root var's element. `s[0]` is char*, so the outer
`[0]` strides 1 and byte-loads.
**Our mapping:** in `mir_index_stride` and INDEX char_elem (`holyd_mir_eval.c` ~858, ~3886),
for an INDEX/DEREF base, descend `type->base` repeatedly until a non-ARRAY/non-PTR kind,
and use THAT for stride + char_elem. (I attempted pieces; needs the full-chain walk, done
carefully without breaking the explicit-2D-char case.)

---

## GAP 7 — 🟡 Stack args beyond 6 (9+ arg functions)
**Proven fix (SysV ABI):** args 7+ go on the stack right-to-left; `[rsp+0]`=arg7.
**Our mapping:** MIR_CALL x86 currently only sets rdi/rsi/rdx/rcx/r8/r9. Add stack-arg
push for n_args>6, and the callee prologue reads them above rbp. Blocks gcc_torture
9-arg tests and recursion-with-many-args.

---

## GAP 8 — ⚪ Stale-binary false measurements (process, not code)
**Symptom:** gauntlet reported identical FAIL=329 before and after a real fix because
`make gauntlet_runner` said "up to date" (dependency tracking misses `holyd_mir_eval.c`).
**Fix:** force-rebuild (`rm -f gauntlet_runner && make gauntlet_runner`) before any
measurement; add `holyd_mir_eval.c holyd_parse.c wubu_mir_regalloc.c wubu_isa_x86_64.c`
to the gauntlet_runner prerequisites in the Makefile so incremental builds are correct.
Also: run `--target x86-64` (NOT positional `x86-64`, which is ignored → runs all 14 ISAs).

---

## Prioritized order (highest leverage first)
1. **GAP 1 + GAP 2** (recursion / call-result preservation) — chibicc push/pop discipline
   + spill call results. Unblocks fib, ackermann, mutual recursion, and most of the
   231 `exp=0 got=1` first-assertion failures that stem from clobbered call results.
2. **GAP 3** (float internal-call XMM0) — unblocks fabs/sqrt/copysign internal returns.
3. **GAP 4 + GAP 5** (byte-accurate layout) — the big one; unblocks the whole
   char/type-punning chapter_16 cluster. Start with option (b) (in-cell byte punning).
4. **GAP 6** (char*[] indexing) — full type-chain walk.
5. **GAP 7** (stack args).
6. **GAP 8** (Makefile deps) — do alongside any of the above so measurements are trustworthy.

Every fix ships behind: `make test` 195/195 + selfhost 155/155 + fuzz_diff 0 mismatches,
then a force-rebuilt `gauntlet_runner --target x86-64` to confirm the FAIL count drops.
