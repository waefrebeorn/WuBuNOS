# Chapter 16 (chars/strings) full-depot measurement — 2026-10-12

Measured with a batch harness running each valid test through test_single,
parsing each test's actual `return` expression (char constants, int
literals). **Baseline: 16/33 pass** (3 parse-errors, 14 failed, 7 skipped
as complex-return). The earlier "11/40" was wrong: the harness assumed
expected=0, but char/string tests return char values (e.g. `return 'c'`==99).

## Byte-addressable loads are CORRECT (verified, committed d2d9aa6)
`(char*)&x` byte-punning works: `int x=0x41424344; char *p=(char*)&x;
p[0]==0x44, p[1]==0x43`. `int* &x` deref works. char-array member loads
work. The scalar-address-of "33" readings seen mid-session were an artifact
of a broken experimental edit, NOT a real bug — at clean HEAD scalar `&x`
byte access is correct.

## ROOT CAUSE of the string-pointer cluster: string-literal-to-pointer addr
`char *s = "hi"; s[0]` returns 16 (a constant, regardless of string
content/length) instead of 'h'=104. The string literal IS allocated
correctly (`char s[]="hi"` array init works; `(char*)&int` byte-punning
works), but assigning the literal's address to a `char *` POINTER stores
16, not the byte address. `(int*)s` reads 16; `s` is always <=1000 (a
VR/cell index, not a `total_mem*8` byte offset).

`mir_gen_expr(STRING_LIT)` returns `wubu_mir_alloc(...)` (a correct address
VR), but the pointer-var VAR_DECL init path stores 16 into `s`. Pre-existing
bug (present at the session's first commit 128dbe2 — NOT a regression from
the byte-addressable work). This is THE blocker for the ~8 string-pointer
tests: simple.c, empty_string, pointer_operations, strings_in_function_calls,
addr_of_string, array_of_strings, cast_string_pointer, standard_library_calls.

## Other real failure clusters
- **access_through_char_pointer got=3**: the double `-0.0` byte layout
  (byte 7 should be -128 / 0x80). Float bit-punning through char*.
- **negative_zero got=7**: float special-value handling.
- **got=1 cluster** (terminating_null_bytes, string_special_characters,
  partial_initialize_via_string, literals_and_compound_initializers):
  string-literal-to-array init with embedded NUL + escapes + static arrays
  + external strcmp. Multi-feature; depends on the string-addr fix above.
- **3 parse-errors** (type_specifiers, adjacent_strings x2): adjacent
  string literal concatenation `"a" "b"` -> `"ab"`. Attempted in the
  parser; destabilized ALL string parsing (the lexer leaves a phantom
  STRING token after a literal's trailing quote) and was reverted. Needs
  the lexer's string-trailing-quote quirk fixed first.

## FIXED this session (ec57817, 44455df)
- **addr-16 alloc bug**: wubu_mir_alloc started the first allocation at
  byte 16, colliding with the x86 JIT frame. Fixed: base >= 32. ALL
  first-allocation memory now stable.
- **byte-packed string literals**: strings were cell-packed (stride 8) but
  char* reads byte-stride 1. Now byte-packed. char *s="hi"; s[0]==104,
  strlen==5, s[2]==108 all correct.
- **adjacent string concat**: "a" "b" -> "ab" now parses (was EROR).

## Remaining chapter_16 failures (16/33 passing)
All need ONE of these advanced features:
- **Pointer-to-array types** `T (*p)[N] = &a; p[i][j]`: addr_of_string,
  char_constant_operations (got=7 at array_ptr[0][9]), transfer_by_eightbyte.
  2D arrays work; ptr-to-array with &a does not.
- **External libc calls with output** (puts/strcmp): the calls go through
  dlsym but tests check stdout/side effects, not return values.
- **Escape sequences in compound initializers** + static arrays with NUL:
  terminating_null_bytes, string_special_characters, partial_initialize.
- **negative_zero got=7**: float special-value (double -0.0) handling.

## Next fix target
Pointer-to-array types `T (*p)[N]` — the largest remaining cluster
(3+ tests). `is_ptr_array` tracking already exists; need &a to produce a
ptr-to-array and p[i][j] to decay/index correctly.

### FIXED this session (43ae5b2, 799f6b4)
Pointer-to-array types `int (*p)[N]` now work. Root cause: the `(*p)`
declarator path mis-parsed `int (*p)[3]` as ARRAY(PTR(int),3) (array-of-
pointers) instead of PTR(ARRAY(int,3)). Fixed in the parser (build the
correct type) + DEREF decay (*p returns the sub-array address). All forms
verified: (*p)[1]==20, p[0][1]==20, int *q=*p; q[1]==20, (*p)[0]==10.
Gate 195/195, selfhost 155/155, fuzz_diff 0 mismatches.

The chapter_16 batch stays 16/33 because the remaining failures need
OTHER features (external puts/strcmp stdout verification, static arrays
with NUL, escape sequences in compound initializers, double -0.0 layout),
not ptr-to-array (now correct).
