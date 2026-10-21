#!/usr/bin/env python3
"""n844: compile-flag parity audit for wubunos/Makefile.

Every recipe that invokes $(CC) is classified by its flag source:
  SAME       -> uses the canonical $(CFLAGS)
  DEVIATES   -> uses $(GAUNTLET_CFLAGS) or any explicit codegen override
               (-O/-std/-march/-mavx2/-mfma/-funroll-loops/-frename-registers/
                -Wall/-Wextra/-fopenmp/-ffast-math/-m32/-m64 ... ) that is not
               in CFLAGS.

The emitted manifest is stable (sorted) and is diffed by the
`make test_compile_flag_parity` gate against the committed
tools/compile_flag_parity.txt; a mismatch means a new / unaccounted flag
deviation was introduced.
"""
import re, sys

s = open('Makefile', encoding='utf-8').read()

# Join backslash-continuations so multi-line recipes become whole lines
# (the leading tab of the recipe is preserved).
logical = re.sub(r'\\\n[ \t]*', ' ', s)

# Canonical CFLAGS codegen token set (for the header comment only).
cm = re.search(r'^CFLAGS[ \t]*=[ \t]*(.*?)$', logical, re.M)
cflags = cm.group(1).split('#')[0].strip() if cm else ''
CODEGEN = ('-O', '-std=', '-march=', '-mavx', '-mfma', '-funroll', '-frename',
           '-fstrict', '-fwrapv', '-ffast-math', '-Wall', '-Wextra', '-W',
           '-m32', '-m64', '-fopenmp')
canonical = sorted(t for t in cflags.split() if t.startswith(CODEGEN))

rows = []
cur = None
for raw in logical.split('\n'):
    if raw.startswith('\t') and '$(CC)' in raw:
        toks = raw.split()
        try:
            i = toks.index('$(CC)')
        except ValueError:
            i = -1
        flags = toks[i + 1:]
        fvar = None
        explicit = []
        for tk in flags:
            if tk == '$(CFLAGS)':
                fvar = 'CFLAGS'
            elif tk == '$(GAUNTLET_CFLAGS)':
                fvar = 'GAUNTLET_CFLAGS'
            elif (tk.startswith('-I') or tk.startswith('-D')
                  or tk in ('-o', '$@', '$^', '$<', '-lm', '-pthread')
                  or tk.endswith('.c') or tk.endswith('.o')
                  or tk.startswith('$(')):
                continue
            else:
                if tk.startswith(CODEGEN):
                    explicit.append(tk)
        if fvar == 'GAUNTLET_CFLAGS':
            status, reason = 'DEVIATES', 'uses GAUNTLET_CFLAGS (arch-scrubbed, Makefile ~L27)'
        elif explicit:
            status, reason = 'DEVIATES', 'explicit overrides: ' + ','.join(explicit)
        else:
            status, reason = 'SAME', 'uses canonical $(CFLAGS)'
        rows.append((cur or '?', fvar or 'NONE', status, reason))
    elif not raw.startswith('\t') and not raw.startswith('#') and ':' in raw \
            and '=' not in raw.split(':', 1)[0]:
        m = re.match(r'^([A-Za-z0-9_./+-]+)\s*:', raw)
        if m:
            cur = m.group(1)

rows.sort()
print('# n844 compile-flag parity audit (wubunos/Makefile)')
print('# canonical CFLAGS codegen tokens: ' + ' '.join(canonical))
print('# target | flag-source | status | reason')
for t, f, st, rs in rows:
    print('%s\t%s\t%s\t%s' % (t, f, st, rs))
sys.exit(0)
