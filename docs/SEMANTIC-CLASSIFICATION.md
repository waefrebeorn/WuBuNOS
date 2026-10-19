# Classifying semantic rejections — the method that works (and one that didn't)

## What went wrong

The gauntlet runs tests in parallel worker processes. With `HD_DIAG=1` each worker
prints `TEST <name>` before building and `diag: <messages>` when the build fails,
both to stderr. Pairing a `TEST` line with the next `diag:` line looked correct
and produced a clean-looking list:

```
gcc_compile_004788_pr78527 | line 7: hit_nonpic_addr: use of undeclared identifier
```

but that source (`gauntlet_gcc_compile.c:38`) contains no `hit_nonpic_addr`
anywhere. The lines came from different tests that happened to interleave. **Any
classification built by pairing interleaved stderr is invalid.**

## What is valid

1. **Serialise or label.** Either run the harness with one worker, or put the test
   name in the same line as the message (`diag <name>: ...`). The worker that
   builds a test must own the name; today `run_single_test_nofree(source, ...)`
   receives only the source, so the name has to be threaded in.
2. **Or classify out-of-band.** `tools/compile_diag.c` takes a file and prints
   parse/semantic diagnostics with `line/col`, with no concurrency at all. Feed it
   the exact source text the harness compiles — extracted from the suite table by
   name, byte-for-byte, not via my earlier `unicode_escape` pass, which produced
   texts that disagreed with the harness (one "hang" and several false rejects
   were extraction artefacts).

## What the AST position work bought

`parse_error_at` + `n->line` now report the real line of the offending
identifier:

```
line 3: nope: use of undeclared identifier
```

Before that, every semantic message in a file named the file's last line, which
made even serialised output useless for locating the construct.

## Status

The semantic pass is still gated off (`HD_SEMANTIC=1`): 1546/2668 (57.9%) versus
1581/2668 (59.3%) with it disabled. Two rule fixes have landed since that
measurement — labels counted as declarations, and `__`-prefixed names
(`__VA_ARGS__`) ignored — but they were motivated by the invalid pairing above,
so they must be re-validated with a correctly attributed run before being
claimed as improvements.
