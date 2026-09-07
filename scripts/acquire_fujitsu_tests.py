#!/usr/bin/env python3
"""
acquire_fujitsu_tests.py — Convert Fujitsu Compiler Test Suite into WuBuNOS gauntlet format.

For each Fujitsu C test:
1. Concatenate all .c files (sorted) into one source
2. Store the reference output as expected_output
3. Generate a gauntlet suite file with output-comparison tests

The gauntlet runner must support output comparison mode.
We encode this by adding a special marker to the expected field:
  - expected = -999999 means "use output comparison"
  - The actual expected output is stored in a separate array

Usage: python3 acquire_fujitsu_tests.py <fujitsu_repo_root> <output_dir>
"""
import os
import sys
import json

def escape_c_string(s):
    """Escape a string for C string literal."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 acquire_fujitsu_tests.py <fujitsu_repo_root> <output_dir>")
        sys.exit(1)

    repo_root = sys.argv[1]
    output_dir = sys.argv[2]
    c_dir = os.path.join(repo_root, "C")

    if not os.path.isdir(c_dir):
        print(f"Error: {c_dir} not found")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect all tests
    tests = []
    test_dirs = sorted([d for d in os.listdir(c_dir)
                       if os.path.isdir(os.path.join(c_dir, d)) and d[0].isdigit()])

    for td in test_dirs:
        full = os.path.join(c_dir, td)
        for test_name in sorted(os.listdir(full)):
            test_path = os.path.join(full, test_name)
            if not os.path.isdir(test_path):
                continue

            # Find all .c files
            c_files = sorted([f for f in os.listdir(test_path) if f.endswith('.c')])
            if not c_files:
                continue

            # Find reference output
            ref_files = [f for f in os.listdir(test_path) if f.endswith('.reference_output')]
            if not ref_files:
                continue

            # Read reference output
            ref_path = os.path.join(test_path, ref_files[0])
            with open(ref_path, 'r') as f:
                ref_output = f.read().strip()

            if not ref_output:
                continue

            # Concatenate all .c files into one source
            sources = []
            for cf in c_files:
                cf_path = os.path.join(test_path, cf)
                with open(cf_path, 'r') as f:
                    sources.append(f.read())

            combined = "\n".join(sources)

            # Skip if too large (>64KB)
            if len(combined) > 65536:
                continue

            tests.append({
                'name': f"fujitsu_{test_name}",
                'source': combined,
                'expected_output': ref_output,
            })

    print(f"Collected {len(tests)} tests from Fujitsu C/ suite")

    # Write as JSON for easy processing
    json_path = os.path.join(output_dir, "fujitsu_tests.json")
    with open(json_path, 'w') as f:
        json.dump(tests, f, indent=2)
    print(f"Written JSON to {json_path}")

    # Also write as a C gauntlet file (for return-value tests, we use 0)
    # The output comparison is handled by a separate runner
    c_path = os.path.join(output_dir, "gauntlet_fujitsu_proper.c")
    with open(c_path, 'w') as f:
        f.write("/* Auto-generated gauntlet suite: fujitsu (proper) */\n")
        f.write(f"/* Source: Fujitsu Compiler Test Suite */\n")
        f.write(f"/* Tests: {len(tests)} */\n")
        f.write("/* Each test is a multi-file program concatenated into one source */\n")
        f.write("/* Uses output comparison (expected_output field) */\n\n")
        f.write('#include "wubu_test_gauntlet.h"\n\n')
        f.write('const test_entry_t gauntlet_fujitsu_proper_tests[] = {\n')

        for i, t in enumerate(tests):
            src = escape_c_string(t['source'])
            # Use 0 as expected return; actual comparison is via stdout
            f.write(f'    {{"{t["name"]}", "{src}", 0, TEST_CAT_INTEGER, 0, 100}}')
            if i < len(tests) - 1:
                f.write(',')
            f.write('\n')

        f.write('};\n\n')
        f.write(f'const uint32_t gauntlet_fujitsu_proper_test_count = {len(tests)};\n')

    print(f"Written C gauntlet to {c_path}")

    # Write expected outputs as a separate C array
    out_path = os.path.join(output_dir, "fujitsu_expected_outputs.c")
    with open(out_path, 'w') as f:
        f.write("/* Expected outputs for Fujitsu tests */\n")
        f.write(f"/* {len(tests)} tests */\n\n")
        f.write('#include <stddef.h>\n\n')
        f.write('typedef struct {\n')
        f.write('    const char *test_name;\n')
        f.write('    const char *expected_output;\n')
        f.write('} fujitsu_expected_t;\n\n')
        f.write('const fujitsu_expected_t fujitsu_expected[] = {\n')

        for i, t in enumerate(tests):
            out = escape_c_string(t['expected_output'])
            f.write(f'    {{"{t["name"]}", "{out}"}}')
            if i < len(tests) - 1:
                f.write(',')
            f.write('\n')

        f.write('};\n\n')
        f.write(f'const size_t fujitsu_expected_count = {len(tests)};\n')

    print(f"Written expected outputs to {out_path}")

if __name__ == '__main__':
    main()
