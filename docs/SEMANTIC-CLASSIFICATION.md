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

## The gauntlet is noisy — quantify before claiming parity

Two runs of the SAME binary (semantic pass wired, only the diagnostic format
changed between them) produced:

```
run A:  pass 1581  fail 155  error 932   (59.3%)
run B:  pass 1561  fail 151  error 956   (58.5%)
```

A 20-pass spread with identical accept/reject logic. The harness runs 6 workers,
and under load some tests trip the watchdog or exceed their time budget, moving
between PASS/ERROR. So a ±20-pass band (~±0.75%) exists and **single-run
comparisons cannot resolve a difference smaller than that**.

Consequences:
- The "semantic pass at parity" claim (1581 with it on vs 1581 off) is *within
  the noise band* and should be restated as "no regression larger than the
  measurement noise".
- Goal 125's criterion ("make test_gauntlet_v3 error count stable") cannot be
  attested from these runs: 932 vs 956 is exactly the instability it names.

To make any future claim here, run the same binary at least three times on an
otherwise idle machine and compare medians, not single runs.
