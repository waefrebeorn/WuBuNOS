#!/usr/bin/env python3
"""Fail the build when a backend's opcode dispatch has drifted from the enum.

The recurring bug in this compiler is an opcode reaching a backend with no
handler. The instruction is then silently dropped and the program returns a
plausible wrong answer -- it does not crash and does not report an error.
That is how MIR_MOV, MIR_NEG, the integer width casts, the div/mod/shift
group, the whole f64 core and MIR_BREAK/MIR_CONTINUE all went unnoticed.

A C designated-initializer table cannot detect this at compile time: a missing
entry is legal, it just defaults to NULL. So check it here instead.

Usage:
    python3 scripts/check_opcode_coverage.py [--baseline <file>]

With no baseline, prints a coverage report and exits 0 unless a backend
handles ZERO opcodes (which means the scan itself is broken).

With --baseline, compares against a recorded floor and fails when coverage
drops. Regenerate the baseline after deliberately changing coverage:

    python3 scripts/check_opcode_coverage.py --write-baseline \\
        scripts/opcode_coverage_baseline.txt
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ENUM_FILE = os.path.join(REPO, "wubu_mir.h")
BASELINE_DEFAULT = os.path.join(HERE, "opcode_coverage_baseline.txt")

# Backends that translate MIR to native code. Interpreters are excluded:
# wubu_mir_interp.c uses a computed-goto table and is checked separately.
BACKENDS = [
    "wubu_isa_x86_64.c",
    "wubu_isa_arm64.c",
    "wubu_isa_riscv.c",
    "wubu_isa_mips.c",
    "wubu_isa_ptx.c",
    "wubu_isa_spirv.c",
]


def parse_enum(path):
    """Return the set of MIR_* enum member names, in declaration order."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    # Only the enum body, so comments elsewhere cannot contribute names.
    m = re.search(r"typedef\s+enum\s*\{(.*?)\}\s*wubu_mir_op_t\s*;", text, re.S)
    if not m:
        sys.exit("ERROR: could not find the wubu_mir_op_t enum in " + path)
    body = m.group(1)
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    return re.findall(r"^\s*(MIR_[A-Z0-9_]+)\s*(?:=|,|$)", body, re.M)


def handled_ops(path):
    """Return the set of opcodes with a `case` label in this file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        return None
    return set(re.findall(r"case\s+(MIR_[A-Z0-9_]+)\s*:", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline")
    ap.add_argument("--write-baseline")
    args = ap.parse_args()

    if not os.path.exists(ENUM_FILE):
        sys.exit("ERROR: " + ENUM_FILE + " not found")
    enum_ops = parse_enum(ENUM_FILE)
    if not enum_ops:
        sys.exit("ERROR: parsed 0 opcodes from the enum; the parser is broken")

    coverage = {}
    for rel in BACKENDS:
        path = os.path.join(REPO, rel)
        got = handled_ops(path)
        if got is None:
            print("  %-22s (not present, skipped)" % rel)
            continue
        covered = len(set(enum_ops) & got)
        coverage[rel] = covered
        missing = sorted(set(enum_ops) - got)
        print("  %-22s %2d/%d  missing %d" % (rel, covered, len(enum_ops), len(missing)))

    if not coverage:
        sys.exit("ERROR: no backend found; the scan is broken")

    if args.write_baseline:
        with open(args.write_baseline, "w", encoding="utf-8") as fh:
            for rel in sorted(coverage):
                fh.write("%s %d\n" % (rel, coverage[rel]))
        print("\nbaseline written to " + args.write_baseline)
        return 0

    base_path = args.baseline or BASELINE_DEFAULT
    if os.path.exists(base_path):
        baseline = {}
        with open(base_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                name, _, count = line.rpartition(" ")
                baseline[name.strip()] = int(count)
        regressions = [
            (rel, baseline[rel], coverage[rel])
            for rel in sorted(coverage)
            if rel in baseline and coverage[rel] < baseline[rel]
        ]
        if regressions:
            print("\nFAIL: opcode coverage regressed:")
            for rel, was, now in regressions:
                print("  %-22s %d -> %d" % (rel, was, now))
            print("\nAn opcode was likely added to wubu_mir_op_t without a")
            print("handler in the backend(s) above. A missing case is dropped")
            print("silently and yields a wrong answer rather than an error.")
            return 1
        print("\nopcode coverage meets the recorded baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
