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
