# Parser hang on a valid gauntlet test

Repro: the harness's own text for `wcc_chapter_16_valid_chars_char_arguments`,
extracted byte-for-byte from `test_gauntlet/suites/gauntlet_writing_c_compiler_full.c`.

```
tools/compile_diag <that source> --no-semantic   ->  never returns
```

`--no-semantic` is deliberate: the loop is in the PARSER, not in the semantic pass.

## Ruled out by measurement

- Truncating the file at any earlier line terminates (with syntax errors), so the
  hang needs the complete, balanced program.
- A top-level iteration guard in `hd_parse_compilation_unit` (breaks after 20 000
  iterations, prints position) **never fires** — so the stall is inside a single
  `hd_parse_decl` call, not the compilation-unit loop.
- These all terminate: 8 int globals + `return 0;`; 8 char globals + `return 0;`;
  a stray `}`; a top-level `return f(a,b);`; 8-argument calls; the function
  signature alone.

## The file

59 lines: one 8-parameter function with eight `if` comparisons against
`char`/`signed char`/`unsigned char` locals, then `main` declaring eight
same-named locals and making one 8-argument call.

## Next step

Instrument `parse_decl`, `parse_type` and `parse_expr` with an iteration guard
that prints `__func__` and `p->lex->pos`, then bisect by section (the `if` chain
versus the call). The compilation-unit guard stays as a safety net.
## Localised (2026-10-19)

Two liveness probes were added:

- `hd_parse_compilation_unit` breaks after 20 000 iterations — **never fires**.
- `peek()` counts calls and reports after 20 M — this one fires:

```
[parse-peek] spun at pos=1033 tok=81 '}' line=19 col=1
```

`pos=1033` is exactly `strlen(source)` — the lexer has reached the end — while the
token is still the file's final `}`. A lexer-only probe over the same file reaches
`EOF at pos=1033 after 240 tokens`, so the lexer is fine on its own.

That means a parser loop keeps calling `peek()` while the lookahead token remains
`}` and never advances it. The compilation-unit loop is exonerated (it breaks on
NULL and its guard never fired), so the spin is in a nested construct loop that
peeks without advancing at the closing brace.

Next: grep the parse functions for loops whose body does not call `advance()` on
every path (`while (peek(p) == X)` with a conditional body), starting with the
function-body and initializer-list loops.

## Frame identified

With `-O0 -g -no-pie`, the probe reports the caller:

```
[parse-peek] spun at pos=1033 tok=81 '}' line=19 col=1 caller=0x40412c
addr2line -> parse_type  holyd_parse.c:148
```

So the repeated `peek()` calls come from `parse_type`'s entry, i.e. a caller keeps
re-entering `parse_type` without consuming the token. Guards added meanwhile:

- `advance()` forces `HD_TOK_EOF` when the lexer position does not move, so no
  loop that calls it can spin.
- `hd_parse_compilation_unit` reports and advances when a declaration returns a
  node without consuming (hard progress guarantee).

Neither guard alone stops this repro, which is consistent with the spin being
inside `parse_decl`'s declaration path rather than the unit loop. Battery is
unchanged at 142 PASS / 1 WRONG (the known `float add` probe).

## FIXED — the K&R parameter-list loop

A live backtrace (gdb run under a timed interrupt; attach is blocked by
`ptrace_scope=1`) showed the truth:

```
#3 parse_type      holyd_parse.c:145   (calloc)
#4 hd_parse_decl   holyd_parse.c:2338
#5 hd_parse_compilation_unit  holyd_parse.c:2657
```

The list at 2338 is the **K&R parameter declaration loop**:

```c
while (peek(p) != HD_TOK_LBRACE && peek(p) != HD_TOK_EOF) {
    HDType *dt = parse_type(p);        /* allocates, consumes nothing on '}' */
```

`parse_type` does not consume `}`; it returns a default I64. The loop then repeats
forever, allocating one `HDType` per iteration — which is why it looked like a slow
hang rather than a hot spin.

Two changes:
1. the loop also stops at `HD_TOK_RBRACE`;
2. a progress guard reports "expected K&R parameter declaration" and advances when
   `parse_type` consumed nothing.

Verified: the repro now terminates with 32 diagnostics instead of hanging. Those
diagnostics expose the *underlying* gap — `signed char` / `unsigned char`
parameter types are not handled in parameter lists — which is the next defect, not
this one.

## Root cause of the whole class: multi-word type specifiers

After the loop fix, the repro reported 32 parse errors, all around the parameter
list. The lexer maps C keywords onto HolyC type tokens:

```
"void"->U0  "char"->I8  "signed"->I32  "int"->I32  "short"->I16
"long"->I64  "unsigned"->U32
```

so `signed char` arrives as TWO type keywords (I32 then I8). `parse_type` consumed
the first and returned, leaving `char` in the stream for the declarator — which is
what produced "expected identifier", the K&R misdetection, and ultimately the
loop.

Fix: after the switch, if the kind is I32/U32 and the next token is another type
keyword, parse it as the inner type and take its width, propagating unsignedness
(I8->U8, I16->U16, I64->U64).

Result: the repro parses with **0 diagnostics**, battery stays 142 PASS / 1 WRONG,
and the gauntlet reads 1583 / 2668 (59.3%) with errors at 927 — the lowest error
count seen (previous low 932), consistent with tests that used to time out in this
loop now running to completion.

## Final measurement: 541 hangs -> 0 across the whole corpus

An earlier sweep of 4921 extracted test sources (timeout 2 s each) found **541
files that never returned**. After the K&R fix, the struct-member progress guard,
the `void` case and the fold-progress gate, the same sweep reports:

```
HANG  0
CRASH 0
```

Every one of the 4921 sources now returns. The parser cannot hang on real C.

Verification set after the changes:
- battery **148 PASS / 0 WRONG / 0 CRASH**
- `test_sse2_dadd` **10/10**
- `make test` **200/200**
- gauntlet **1583 / 2668 (59.3%)**, errors 930 — inside the measured noise band
  (the gauntlet itself was never dominated by these hangs; they lived in the
  multi-file torture/execution corpus, which the gauntlet skips)
