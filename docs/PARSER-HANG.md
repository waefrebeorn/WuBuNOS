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
