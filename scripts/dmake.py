#!/usr/bin/env python3
"""
dmake.py — DMake build system for WuBuNOS HolyD gauntlet.

Manages test suite acquisition, generation, building, and running.

Usage:
  python3 dmake.py acquire       — clone/update upstream test repos
  python3 dmake.py generate      — generate gauntlet C files from upstream
  python3 dmake.py build         — build gauntlet binaries
  python3 dmake.py run           — run all gauntlet suites
  python3 dmake.py run SUITE     — run a specific suite
  python3 dmake.py status        — show depot status
  python3 dmake.py clean         — remove generated files
  python3 dmake.py clean-all     — remove everything including upstream clones
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

# ---- Configuration ----

REPO_ROOT = Path("/home/wubu/wubunos")
DEPOT_DIR = REPO_ROOT / "test_gauntlet" / "depot"
SUITES_DIR = REPO_ROOT / "test_gauntlet" / "suites"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Upstream sources: name -> (url, type, description)
UPSTREAM_SOURCES = {
    "fujitsu": {
        "url": "https://github.com/fujitsu/compiler-test-suite.git",
        "type": "git",
        "description": "Fujitsu Compiler Test Suite (C/C++/Fortran)",
        "tests_dir": "C",
        "test_format": "multisource",  # multi-file programs with reference output
    },
    "c-testsuite": {
        "url": "https://github.com/c-testsuite/c-testsuite.git",
        "type": "git",
        "description": "Public database of C compiler test cases",
        "tests_dir": "tests/single-exec",
        "test_format": "singleexec",  # single-file with .expected
    },
    "writing-c-compiler": {
        "url": "https://github.com/nlsandler/writing-a-c-compiler-tests.git",
        "type": "git",
        "description": "Tests from Writing a C Compiler book",
        "tests_dir": "tests",
        "test_format": "chapter_based",  # organized by chapter, valid/invalid
    },
    "incremental": {
        "url": "https://github.com/AMLeng/incremental_c_compiler_tests.git",
        "type": "git",
        "description": "Incremental C Compiler Tests (12 stages)",
        "tests_dir": ".",
        "test_format": "incremental",  # stage_N/valid/*.c, each file is a standalone program
    },
}

# Gauntlet suites that we generate from upstream + our own
GAUNTLETS = {
    "comprehensive": {"source": "builtin", "format": "return_value"},
    "integer": {"source": "builtin", "format": "return_value"},
    "control": {"source": "builtin", "format": "return_value"},
    "bitwise": {"source": "builtin", "format": "return_value"},
    "comparison": {"source": "builtin", "format": "return_value"},
    "memory": {"source": "builtin", "format": "return_value"},
    "stress": {"source": "builtin", "format": "return_value"},
    "float": {"source": "builtin", "format": "return_value"},
    "string": {"source": "builtin", "format": "return_value"},
    "gcc_torture": {"source": "generated", "format": "return_value"},
    "extern_gcc": {"source": "generated", "format": "return_value"},
    "c_testsuite": {"source": "generated", "format": "return_value"},
    "llvm": {"source": "generated", "format": "return_value"},
    "lacc": {"source": "generated", "format": "return_value"},
    "compcert": {"source": "generated", "format": "return_value"},
    "chibicc": {"source": "generated", "format": "return_value"},
    "tinycc": {"source": "generated", "format": "return_value"},
    "slimcc": {"source": "generated", "format": "return_value"},
    "gcc_compile": {"source": "generated", "format": "return_value"},
    "gcc_dg": {"source": "generated", "format": "return_value"},
    "writing_c_compiler": {"source": "generated", "format": "return_value"},
    "fujitsu_proper": {"source": "depot", "format": "output_comparison"},
    "c_testsuite_single": {"source": "depot", "format": "output_comparison"},
}

def run(cmd, cwd=None, check=True):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd or str(REPO_ROOT))
    if check and result.returncode != 0:
        print(f"  ERROR: {cmd}")
        print(f"  {result.stderr[:500]}")
    return result

def acquire():
    """Clone or update upstream test repos."""
    DEPOT_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, info in UPSTREAM_SOURCES.items():
        upstream_path = DEPOT_DIR / name / "upstream"
        
        if upstream_path.exists():
            print(f"  Updating {name}...")
            result = run(f"cd {upstream_path} && git pull --ff-only 2>&1 | tail -3")
            print(f"    {result.stdout.strip()}")
        else:
            print(f"  Cloning {name}...")
            upstream_path.parent.mkdir(parents=True, exist_ok=True)
            result = run(f"git clone --depth 1 {info['url']} {upstream_path} 2>&1 | tail -3")
            print(f"    {result.stdout.strip()}")
        
        # Survey what we got
        if upstream_path.exists():
            c_files = list(upstream_path.rglob("*.c"))
            ref_files = list(upstream_path.rglob("*.reference_output"))
            print(f"    {len(c_files)} .c files, {len(ref_files)} reference outputs")

def generate():
    """Generate gauntlet C files from upstream sources."""
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate Fujitsu single-source tests
    generate_fujitsu_single()
    
    # Generate c-testsuite tests
    generate_c_testsuite()
    
    # Generate writing-c-compiler valid tests
    generate_writing_c_compiler()
    
    # Generate incremental C compiler tests
    generate_incremental()

def generate_fujitsu_single():
    """Generate gauntlet for Fujitsu single-source tests."""
    fujitsu_dir = DEPOT_DIR / "fujitsu" / "upstream" / "C"
    if not fujitsu_dir.exists():
        print("  Fujitsu not cloned, skipping")
        return
    
    tests = []
    test_dirs = sorted([d for d in fujitsu_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    
    for td in test_dirs:
        for test_path in sorted(td.iterdir()):
            if not test_path.is_dir():
                continue
            c_files = sorted(test_path.glob("*.c"))
            ref_files = sorted(test_path.glob("*.reference_output"))
            
            if len(c_files) != 1 or not ref_files:
                continue
            
            source = c_files[0].read_text()
            expected = ref_files[0].read_text().strip()
            
            if len(source) > 65536 or len(expected) > 16384:
                continue
            
            tests.append({
                "name": f"fujitsu_{test_path.name}",
                "source": source,
                "expected_output": expected,
            })
    
    write_gauntlet_file("gauntlet_fujitsu_proper", tests, "output_comparison")
    print(f"  Generated gauntlet_fujitsu_proper: {len(tests)} tests")

def generate_c_testsuite():
    """Generate gauntlet for c-testsuite single-exec tests."""
    ct_dir = DEPOT_DIR / "c-testsuite" / "upstream" / "tests" / "single-exec"
    if not ct_dir.exists():
        print("  c-testsuite not cloned, skipping")
        return
    
    tests = []
    for c_file in sorted(ct_dir.glob("*.c")):
        expected_file = c_file.with_suffix(".c.expected")
        if not expected_file.exists():
            continue
        
        source = c_file.read_text()
        expected = expected_file.read_text().strip()
        
        if len(source) > 65536:
            continue
        
        tests.append({
            "name": f"ct_{c_file.stem}",
            "source": source,
            "expected_output": expected,
        })
    
    write_gauntlet_file("gauntlet_c_testsuite_single", tests, "output_comparison")
    print(f"  Generated gauntlet_c_testsuite_single: {len(tests)} tests")

def generate_writing_c_compiler():
    """Generate gauntlet for writing-a-c-compiler-tests valid tests."""
    wcc_dir = DEPOT_DIR / "writing-c-compiler" / "upstream" / "tests"
    if not wcc_dir.exists():
        print("  writing-c-compiler not cloned, skipping")
        return
    
    # Load expected results
    results_file = DEPOT_DIR / "writing-c-compiler" / "upstream" / "expected_results.json"
    if not results_file.exists():
        print("  expected_results.json not found, skipping")
        return
    
    expected_results = json.loads(results_file.read_text())
    
    tests = []
    for chapter_dir in sorted(wcc_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue
        valid_dir = chapter_dir / "valid"
        if not valid_dir.exists():
            continue
        
        for test_file in sorted(valid_dir.glob("*.c")):
            # Key format in expected_results.json: chapter_XX/valid/name.c
            relative = test_file.relative_to(wcc_dir)
            test_key = str(relative)
            
            result = expected_results.get(test_key)
            if result is None:
                continue
            
            source = test_file.read_text()
            if len(source) > 65536:
                continue
            
            # result is {"return_code": N}
            return_code = result.get("return_code", 0)
            tests.append({
                "name": f"wcc_{test_key.replace('/', '_').replace('.c', '')}",
                "source": source,
                "expected_return": return_code,
            })
    
    write_gauntlet_file("gauntlet_writing_c_compiler_depot", tests, "return_value")
    print(f"  Generated gauntlet_writing_c_compiler_depot: {len(tests)} tests")

def generate_incremental():
    """Generate gauntlet for incremental C compiler tests (valid programs only)."""
    inc_dir = DEPOT_DIR / "incremental" / "upstream"
    if not inc_dir.exists():
        print("  incremental not cloned, skipping")
        return
    
    tests = []
    # Each stage_N/valid/*.c is a standalone program
    for stage_dir in sorted(inc_dir.glob("stage_*/valid")):
        if not stage_dir.is_dir():
            continue
        for test_file in sorted(stage_dir.rglob("*.c")):
            source = test_file.read_text()
            if len(source) > 65536:
                continue
            
            # Compute expected return value by compiling with host gcc
            # For now, we use gcc to get the expected result
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as tmp:
                tmp.write(source.encode())
                tmp_path = tmp.name
            
            try:
                # Compile with gcc
                result = subprocess.run(
                    ["gcc", "-w", "-o", tmp_path + ".out", tmp_path],
                    capture_output=True, timeout=10
                )
                if result.returncode != 0:
                    continue  # Skip tests that gcc can't compile
                
                # Run to get expected return value
                result = subprocess.run(
                    [tmp_path + ".out"],
                    capture_output=True, timeout=5
                )
                expected_return = result.returncode
                
                # Use relative path as name
                rel = test_file.relative_to(inc_dir)
                name = f"inc_{str(rel).replace('/', '_').replace('.c', '')}"
                
                tests.append({
                    "name": name,
                    "source": source,
                    "expected_return": expected_return,
                })
            except Exception:
                continue
            finally:
                # Cleanup
                for p in [tmp_path, tmp_path + ".out"]:
                    if os.path.exists(p):
                        os.unlink(p)
    
    write_gauntlet_file("gauntlet_incremental", tests, "return_value")
    print(f"  Generated gauntlet_incremental: {len(tests)} tests")

def write_gauntlet_file(name, tests, format_type):
    """Write a gauntlet suite C file."""
    path = SUITES_DIR / f"{name}.c"
    
    with open(path, "w") as f:
        f.write(f"/* Auto-generated by dmake.py — DO NOT EDIT */\n")
        f.write(f"/* Format: {format_type} */\n")
        f.write(f"/* Tests: {len(tests)} */\n\n")
        f.write('#include "wubu_test_gauntlet.h"\n\n')
        f.write(f"const test_entry_t {name}_tests[] = {{\n")
        
        for i, t in enumerate(tests):
            src = t["source"].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
            
            if format_type == "output_comparison":
                # For output comparison, we use expected_return=0 and store output separately
                # The runner handles this via the separate outputs file
                f.write(f'    {{"{t["name"]}", "{src}", 0, TEST_CAT_INTEGER, 0, 100}}')
            else:
                expected = t.get("expected_return", 0)
                f.write(f'    {{"{t["name"]}", "{src}", {expected}, TEST_CAT_INTEGER, 0, 100}}')
            
            if i < len(tests) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        f.write(f"const uint32_t {name}_test_count = {len(tests)};\n")
    
    # If output comparison, also write expected outputs file
    if format_type == "output_comparison":
        outputs_path = SUITES_DIR / f"{name}_outputs.c"
        with open(outputs_path, "w") as f:
            f.write(f"/* Auto-generated by dmake.py — expected outputs for {name} */\n\n")
            f.write(f"#include <stddef.h>\n\n")
            f.write(f"typedef struct {{ const char *name; const char *output; }} {name}_output_t;\n\n")
            f.write(f"const {name}_output_t {name}_outputs[] = {{\n")
            
            for i, t in enumerate(tests):
                out = t.get("expected_output", "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                f.write(f'    {{"{t["name"]}", "{out}"}}')
                if i < len(tests) - 1:
                    f.write(",")
                f.write("\n")
            
            f.write("}};\n\n")
            f.write(f"const size_t {name}_output_count = {len(tests)};\n")

def status():
    """Show depot status."""
    print("=== DMake Depot Status ===\n")
    
    print("Upstream sources:")
    for name, info in UPSTREAM_SOURCES.items():
        upstream_path = DEPOT_DIR / name / "upstream"
        if upstream_path.exists():
            result = run(f"cd {upstream_path} && git log --oneline -1 2>/dev/null", check=False)
            commit = result.stdout.strip()[:60] if result.returncode == 0 else "unknown"
            print(f"  {name}: {commit}")
        else:
            print(f"  {name}: NOT CLONED")
    
    print("\nGenerated gauntlet suites:")
    for name, info in GAUNTLETS.items():
        path = SUITES_DIR / f"{name}.c"
        if path.exists():
            size = path.stat().st_size
            print(f"  {name}: {size:,} bytes ({info['format']})")
        else:
            print(f"  {name}: NOT GENERATED")

def clean():
    """Remove generated gauntlet files."""
    print("Cleaning generated files...")
    for f in SUITES_DIR.glob("gauntlet_*.c"):
        if f.name not in ("gauntlet_comprehensive.c",):  # keep built-in
            f.unlink()
            print(f"  Removed {f.name}")
    
    # Remove generated binaries
    for binary in ["gauntlet_runner", "gauntlet_v2", "gauntlet_v3", "gauntlet_direct", "gauntlet_output"]:
        path = REPO_ROOT / binary
        if path.exists():
            path.unlink()
            print(f"  Removed {binary}")

def clean_all():
    """Remove everything including upstream clones."""
    clean()
    print("\nRemoving upstream clones...")
    if DEPOT_DIR.exists():
        shutil.rmtree(DEPOT_DIR)
        print(f"  Removed {DEPOT_DIR}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "acquire":
        acquire()
    elif command == "generate":
        generate()
    elif command == "status":
        status()
    elif command == "clean":
        clean()
    elif command == "clean-all":
        clean_all()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
