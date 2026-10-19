# What the parser rejects, counted

Corpus: 4,921 real test sources. Each file's **first** diagnostic is classified
(the first error is what stops progress; later ones are usually fallout).

## Ranked first errors (whole corpus)

| count | class |
|---|---|
| 1733 | expected identifier |
| 832 | expected token N, got N |
| 50 | expected expression |
| 16 | expected K&R parameter declaration |
| 9 | expected typedef name |
| 7 | expected member declaration |

The K&R and member classes are residue from newly added progress guards: they now
report and continue instead of hanging, which is the correct behaviour for input
that really is malformed at that point.

## Inside `expected identifier`

`parse_error_tok()` now appends the offending token, so the class is actionable.
Over the first 800 files:

| token | files |
|---|---|
| `{` | 177 |
| `)` | 29 |
| `1` | 12 |
| `r` | 10 |
| `#` | 9 |
| `i` / `f` / digits | ~40 |

`{` dominates: the parser reaches a point where a declarator name is required and
finds the body opening brace. That is the declarator/definition boundary, not a
missing identifier — the likely cause is a function definition whose return type
was consumed by a path that then expects a name (abstract declarators, function
pointers returning structs, or macro-shaped declarations).

`#` is a separate, simpler class: preprocessor directives that reach the parser.

## Method note

Count the *first* diagnostic per file, and make the message carry the token.
Without the token the top class is unusable: "expected identifier" in 1,733 files
says nothing about what to fix.

## Correction: the corpus is not all translation units

`tools/compile_diag` was parsing **raw** text, so any line beginning with `#`
produced a phantom diagnostic. It now runs `wubu_preprocess()` first, exactly as
the compiler does.

Re-measured with preprocessing:

| count | class |
|---|---|
| 1276 | expected identifier (saw `{`) |
| 437 | expected token N, got N |
| 186 | expected identifier (saw `r`) |
| 129 | expected expression |
| 31 | expected typedef name |

The `{` class is **not a parser bug**. Sampling the files shows many are statement
*fragments*, not translation units:

```
c_testsuite_064711_00184.c:
{
   char a;
   short short b;
   ...
```

A file that opens with `{` cannot be parsed as a declaration, and reporting
"expected identifier" is the correct response.

So this corpus cannot be used as a parser-correctness metric beyond hang/crash
detection, where it is excellent: 4,921 sources, 0 hangs, 0 crashes. Ranking
diagnostics here mostly ranks the corpus's own shape.

## Gauntlet failures classified with HD_DIAG=1

`test_gauntlet_runner.c` now prints `diag <test>: <messages>` when
`hd_build_mir` fails, so the failing tests — not the fragment corpus — can be
classified. One run at 1583/2668 (59.3%) produced 54 distinct classes:

| count | class | example |
|---|---|---|
| 2053 | expected token N, got N | extern_gcc_050237_930621_1 |
| 273 | expected expression | extern_gcc_050242_931004_9 |
| 132 | args: use of undeclared identifier | llvm_065176_printf_chk_1 |
| 78 | expected identifier | llvm_065120_bitfld_1 |
| 76 | expected K&R parameter declaration | llvm_065134_20051113_1 |
| 60 | expected member declaration | gcc_torture_001054_20050826_1 |
| 27 | expected typedef name | llvm_065170_string_opt_17 |

Two lessons from doing it:

1. The numbers were unactionable. `expected token 85, got 80` hid the fact that
   the missing token was `;` and the found token was `{`. The message now carries
   the offending text: `expected token 85, got 80 ('{')`.
2. A single HD_DIAG run writes a ~2 GB log (every diagnostic for every failing
   test). Redirect it to a scratch path and delete it after classification.

Note also that several samples from the first classification (including
extern_gcc_050237_930621_1) now parse with **0 errors**, so fix-then-reclassify
is the correct loop rather than trusting one snapshot.
