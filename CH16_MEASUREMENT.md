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

## Next fix target
Fix the pointer-var VAR_DECL init to store the string literal's actual
byte address (not 16). Unlocks the ~8 string-pointer tests — the single
biggest lever in chapter_16.
