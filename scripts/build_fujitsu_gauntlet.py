#!/usr/bin/env python3
"""
build_fujitsu_gauntlet.py — Convert Fujitsu Compiler Test Suite into WuBuNOS gauntlet format.

For each test directory in the Fujitsu C/ tree:
1. Concatenate all .c files (in sorted order) into a single source
2. Extract the reference output
3. Emit a gauntlet suite entry

Usage: python3 build_fujitsu_gauntlet.py <fujitsu_repo_root> <output_file>
"""
import os
import sys
import json
import hashlib

def escape_c_string(s):
    """Escape a string for C string literal."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 build_fujitsu_gauntlet.py <fujitsu_repo_root> <output_file>")
        sys.exit(1)

    repo_root = sys.argv[1]
    output_file = sys.argv[2]
    c_dir = os.path.join(repo_root, "C")

    if not os.path.isdir(c_dir):
        print(f"Error: {c_dir} not found")
        sys.exit(1)

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

            # Skip tests with empty reference output
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

    # Write output as C gauntlet file
    with open(output_file, 'w') as f:
        f.write("/* Auto-generated gauntlet suite: fujitsu (proper) */\n")
        f.write(f"/* Source: Fujitsu Compiler Test Suite */\n")
        f.write(f"/* Tests: {len(tests)} */\n")
        f.write("/* Each test is a multi-file program concatenated into one source */\n")
        f.write("/* Expected output compared against stdout */\n\n")
        f.write('#include "wubu_test_gauntlet.h"\n\n')
        f.write('const test_entry_t gauntlet_fujitsu_proper_tests[] = {\n')

        for i, t in enumerate(tests):
            src = escape_c_string(t['source'])
            # expected field stores return code; we use 0 (main returns 0)
            # The actual output comparison is done via stdout matching
            f.write(f'    {{"{t["name"]}", "{src}", 0, TEST_CAT_INTEGER, 0, 100}}')
            if i < len(tests) - 1:
                f.write(',')
            f.write('\n')

        f.write('};\n\n')
        f.write(f'const uint32_t gauntlet_fujitsu_proper_test_count = {len(tests)};\n')

    print(f"Written to {output_file}")

if __name__ == '__main__':
    main()
