#!/bin/sh
# gauntlet_noise.sh -- run the gauntlet N times and report the distribution.
#
# Single-run comparisons are meaningless: two runs of the same binary have been
# observed at 1581 and 1561 valid passes (a 20-pass band), because six workers
# compete and tests move between PASS and ERROR under load. Any "did this change
# help?" question needs medians from repeated idle runs.
#
# Usage: tools/gauntlet_noise.sh [runs] [binary]
set -u

RUNS=${1:-3}
BIN=${2:-./gauntlet_v3}
OUT=$(mktemp)

i=1
while [ "$i" -le "$RUNS" ]; do
    LOG=$(mktemp)
    echo "run $i/$RUNS ..." >&2
    "$BIN" >"$LOG" 2>&1
    grep -a 'valid_pass_rate' "$LOG" | tail -1 >>"$OUT"
    rm -f "$LOG"
    i=$((i + 1))
    sleep 5
done

echo "--- raw ---"
cat "$OUT"
echo "--- summary ---"
python3 - "$OUT" <<'PY'
import json, sys, statistics
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    rows.append(d)
if not rows:
    print("no runs recorded")
    raise SystemExit(1)
passes = [r["pass"] for r in rows]
errors = [r["error"] for r in rows]
print(f"passes : min {min(passes)}  median {statistics.median(passes):.0f}  max {max(passes)}")
print(f"errors : min {min(errors)}  median {statistics.median(errors):.0f}  max {max(errors)}")
print(f"band   : {max(passes) - min(passes)} passes")
print("A difference between configurations is only real if it exceeds this band.")
PY
rm -f "$OUT"
