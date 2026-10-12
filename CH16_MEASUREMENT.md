# Chapter 16 (chars/strings) full-depot measurement — 2026-10-12

Measured with a batch harness running each valid test through test_single
(expected=0). Baseline: **11/40 pass** (8 errored, 21 failed).

## What passes (11)
Basic char literals, char arithmetic, return_char_constant, simple char
expressions, byte store+load roundtrips on arrays, char pointer reads
where the pointee is a freshly-allocated array.

## The dominant blocker: scalar address-of (`&x`) cell layout
`access_through_char_pointer.c` and friends fail because taking the
address of a SCALAR local returns the wrong cell:

    int x = 100; char *p = (char*)&x;
    p[0] == 100   (correct)
    p[1] == 33    (WRONG — should be 0; upper bytes of int 100)

`p[1]` is a CONSTANT that scales with the number of locals (33 with 1
local, 35 with 3) — it is the variable/VR count metadata, not x's value.
So `&x` for a scalar points at a metadata cell, and byte 0 coincidentally
holds the stored value while bytes 1+ hold allocator bookkeeping.

Arrays are fine: `int a[2]; a[0]=100; a[1]=0; ((char*)a)[1] == 0` passes
(fresh zeroed allocation). The bug is specific to scalars whose value
lives in a VR and whose address is taken — the spill cell isn't where
`mir_find_var_addr` points, or isn't zeroed.

## Byte-addressable loads themselves are CORRECT
`p[0]` reads the right byte every time; char pointers into arrays work;
byte store+load roundtrips work. The gap is purely the scalar `&x`
address resolution + cell zeroing, NOT the byte-load width (that work is
done and verified: d2d9aa6).

## Next fix target
Make `&scalar` allocate/zero a dedicated cell and spill the scalar's
current value there (full 8-byte store), so byte-punning through
`(char*)&x` reads the scalar's actual bytes. Unlocks
access_through_char_pointer.c + ~10 sibling tests.

## Other failing clusters (lower priority)
- escape_sequences / control_characters / string_special_characters:
  string literal escape handling (\n, \t, \0 in the middle).
- return_char_constant (got=99): char constant return value.
- static_initializers, type_specifiers, adjacent_strings: parse/EROR.
