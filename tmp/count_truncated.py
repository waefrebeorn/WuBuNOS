import re, os

suites_dir = 'test_gauntlet/suites'
total = 0
truncated = 0
no_main = 0
brace_mismatch = 0

for f in sorted(os.listdir(suites_dir)):
    if not f.endswith('.c'):
        continue
    with open(os.path.join(suites_dir, f)) as fh:
        content = fh.read()
    
    # Find test entries
    entries = re.findall(r'\{\s*"([^"]+)"\s*,\s*"(.*?)"\s*,\s*(\d+)', content, re.DOTALL)
    
    file_total = 0
    file_bad = 0
    for name, source, expected in entries:
        total += 1
        file_total += 1
        bad = False
        if source.strip().startswith('{'):
            truncated += 1
            bad = True
        elif 'main' not in source:
            no_main += 1
            bad = True
        elif source.count('{') != source.count('}'):
            brace_mismatch += 1
            bad = True
        if bad:
            file_bad += 1
    
    if file_bad > 0:
        print(f"{f}: {file_total} tests, {file_bad} potentially invalid")

print(f"\nTotal tests: {total}")
print(f"Truncated (starts with {{): {truncated}")
print(f"No main: {no_main}")
print(f"Brace mismatch: {brace_mismatch}")
print(f"Potentially valid: {total - truncated - no_main - brace_mismatch}")
