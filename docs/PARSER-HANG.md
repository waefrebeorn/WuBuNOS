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