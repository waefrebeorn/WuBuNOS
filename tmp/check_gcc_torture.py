import re, os

# Count invalid tests that are actually in the gauntlet
# The gauntlet runner compiles each test and checks the result
# ERROR means compilation failed

# Let's check a sample of ERROR tests to see if they're truncated
suites_dir = 'test_gauntlet/suites'

# Focus on gcc_torture which has the most ERROR tests
with open(os.path.join(suites_dir, 'gauntlet_gcc_torture.c')) as f:
    content = f.read()

entries = re.findall(r'\{\s*"([^"]+)"\s*,\s*"(.*?)"\s*,\s*(\d+)', content, re.DOTALL)

truncated_count = 0
valid_count = 0
truncated_examples = []

for name, source, expected in entries:
    is_truncated = False
    if source.strip().startswith('{'):
        is_truncated = True
    elif 'main' not in source:
        is_truncated = True
    elif source.count('{') != source.count('}'):
        is_truncated = True
    
    if is_truncated:
        truncated_count += 1
        if len(truncated_examples) < 5:
            truncated_examples.append((name, source[:150]))
    else:
        valid_count += 1

print(f"gcc_torture: {len(entries)} total")
print(f"  Truncated/invalid: {truncated_count}")
print(f"  Potentially valid: {valid_count}")
print(f"\nExamples of truncated tests:")
for name, src in truncated_examples:
    print(f"  {name}:")
    print(f"    {src}...")
