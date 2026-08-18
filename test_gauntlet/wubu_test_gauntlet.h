/*
 * wubu_test_gauntlet.h — The WuBuOS Universal Test Gauntlet.
 *
 * Every test from every compiler (GCC, LLVM, MSVC, CompCert, stdtests, Csmith)
 * runs through this harness. Memory-structured for speed. C11 pure.
 *
 * Design:
 *   - Tests are registered in a static array (no malloc at runtime)
   - Each test is a function pointer + name + expected result
   - Tests run across ALL ISA targets automatically
 *   - Results stored in a compact bitfield array
 *   - Summary printed as a scorecard
 */

#ifndef WUBU_TEST_GAUNTLET_H
#define WUBU_TEST_GAUNTLET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#error "C11 only. No C++."
#endif

/* ---- Test categories (bitmask) */
typedef enum {
    TEST_CAT_INTEGER     = 1 << 0,   /* Integer arithmetic correctness */
    TEST_CAT_FLOAT       = 1 << 1,   /* Floating point correctness */
    TEST_CAT_CONTROL     = 1 << 2,   /* Branches, loops, switches */
    TEST_CAT_MEMORY      = 1 << 3,   /* Load/store, structs, arrays, pointers */
    TEST_CAT_BITWISE     = 1 << 4,   /* Bit ops, shifts, masks */
    TEST_CAT_COMPARISON  = 1 << 5,   /* ==, !=, <, >, <=, >= */
    TEST_CAT_UNARY       = 1 << 6,   /* Neg, not, complement */
    TEST_CAT_LOGICAL     = 1 << 7,   /* &&, ||, ?: */
    TEST_CAT_ASSIGNMENT  = 1 << 8,   /* =, +=, -=, compound assign */
    TEST_CAT_CONVERSION  = 1 << 9,   /* int<->float, trunc, extend */
    TEST_CAT_STDLIB      = 1 << 10,  /* String ops, math, memory */
    TEST_CAT_CONCURRENCY = 1 << 11,  /* Atomics, fences */
    TEST_CAT_GPU         = 1 << 12,  /* GPU kernel correctness */
    TEST_CAT_RANDOM      = 1 << 13,  /* Fuzz/csmith-generated tests */
    TEST_CAT_PERF        = 1 << 14,  /* Performance benchmarks */
    TEST_CAT_STRESS      = 1 << 15,  /* Edge cases, torture tests */
} test_category_t;

/* ---- Test result codes */
typedef enum {
    TEST_PASS = 0,          /* Got expected result */
    TEST_FAIL = 1,          /* Wrong result */
    TEST_SKIP = 2,          /* Not supported on this target */
    TEST_ERROR = 3,         /* Compilation or runtime error */
    TEST_TIMEOUT = 4,       /* Took too long */
} test_result_t;

/* ---- Per-target result (packed into 1 byte) */
typedef struct {
    uint8_t result : 3;     /* test_result_t */
    uint8_t cycles : 5;     /* log2(cycle count), 0 = unknown */
} target_result_t;

/* ---- Test descriptor */
typedef struct {
    const char *name;           /* Test name (string interned) */
    const char *source;         /* C source code (or NULL for codegen tests) */
    int64_t     expected;       /* Expected return value */
    uint16_t    categories;     /* test_category_t bitmask */
    uint8_t     min_isa;        /* Minimum ISA level required */
    uint8_t     timeout_ms;     /* Timeout in milliseconds */
} test_entry_t;

/* ---- Test function (for codegen-only tests) */
typedef int64_t (*test_func_t)(void);

/* ---- Test descriptor (function-based) */
typedef struct {
    const char *name;
    test_func_t func;
    uint16_t    categories;
} codegen_test_entry_t;

/* ---- Gauntlet state */
typedef struct {
    /* Results: [test_idx][target_idx] */
    target_result_t *results;
    uint32_t         n_tests;
    uint32_t         n_targets;

    /* Per-category counts */
    uint32_t pass_by_cat[16];
    uint32_t fail_by_cat[16];
    uint32_t skip_by_cat[16];

    /* Per-target counts */
    uint32_t pass_by_target[16];
    uint32_t fail_by_target[16];

    /* Timing (ms) */
    uint64_t total_time_us;
    uint64_t compile_time_us;
    uint64_t run_time_us;

    /* Score: 0-10000 (percentage * 100) */
    uint32_t overall_score;
} gauntlet_state_t;

/* ---- API */

/* Initialize gauntlet. targets = array of ISA names. */
void gauntlet_init(gauntlet_state_t *g, const char **targets, uint32_t n_targets);

/* Register a source-based test */
void gauntlet_add_test(gauntlet_state_t *g, const test_entry_t *test);

/* Register a codegen-based test */
void gauntlet_add_codegen_test(gauntlet_state_t *g, const codegen_test_entry_t *test);

/* Run all tests across all targets */
void gauntlet_run_all(gauntlet_state_t *g);

/* Print scorecard */
void gauntlet_print_summary(const gauntlet_state_t *g);

/* Export results as CSV */
void gauntlet_export_csv(const gauntlet_state_t *g, const char *filename);

/* Run a specific category only */
void gauntlet_run_category(gauntlet_state_t *g, test_category_t cat);

/* Compare results between two targets (differential testing) */
uint32_t gauntlet_compare_targets(const gauntlet_state_t *g,
                                   uint32_t target_a, uint32_t target_b);

/* ---- Built-in test suites */

/* Integer arithmetic: 100 tests covering all ops */
extern const test_entry_t gauntlet_integer_tests[];
extern const uint32_t gauntlet_integer_test_count;

/* Control flow: 50 tests covering branches/loops/switches */
extern const test_entry_t gauntlet_control_tests[];
extern const uint32_t gauntlet_control_test_count;

/* Bitwise: 50 tests covering shifts/masks/bit ops */
extern const test_entry_t gauntlet_bitwise_tests[];
extern const uint32_t gauntlet_bitwise_test_count;

/* Memory: 50 tests covering structs/arrays/pointers */
extern const test_entry_t gauntlet_memory_tests[];
extern const uint32_t gauntlet_memory_test_count;

/* Comparison: 40 tests covering all comparison ops */
extern const test_entry_t gauntlet_comparison_tests[];
extern const uint32_t gauntlet_comparison_test_count;

/* Stress: 100 edge-case tests (signed overflow, boundary values, etc.) */
extern const test_entry_t gauntlet_stress_tests[];
extern const uint32_t gauntlet_stress_test_count;

/* Comprehensive: 90 auto-generated edge-case tests */
extern const test_entry_t gauntlet_comprehensive_tests[];
extern const uint32_t gauntlet_comprehensive_test_count;

/* Downloaded real compiler test suites */
extern const test_entry_t gauntlet_gcc_torture_tests[];
extern const uint32_t gauntlet_gcc_torture_test_count;

extern const test_entry_t gauntlet_gcc_dg_tests[];
extern const uint32_t gauntlet_gcc_dg_test_count;

extern const test_entry_t gauntlet_gcc_compile_tests[];
extern const uint32_t gauntlet_gcc_compile_test_count;

extern const test_entry_t gauntlet_fujitsu_tests[];
extern const uint32_t gauntlet_fujitsu_test_count;

extern const test_entry_t gauntlet_extern_gcc_tests[];
extern const uint32_t gauntlet_extern_gcc_test_count;

extern const test_entry_t gauntlet_compcert_tests[];
extern const uint32_t gauntlet_compcert_test_count;

extern const test_entry_t gauntlet_c_testsuite_tests[];
extern const uint32_t gauntlet_c_testsuite_test_count;

extern const test_entry_t gauntlet_llvm_tests[];
extern const uint32_t gauntlet_llvm_test_count;

extern const test_entry_t gauntlet_lacc_tests[];
extern const uint32_t gauntlet_lacc_test_count;

extern const test_entry_t gauntlet_tinycc_tests[];
extern const uint32_t gauntlet_tinycc_test_count;

extern const test_entry_t gauntlet_chibicc_tests[];
extern const uint32_t gauntlet_chibicc_test_count;

extern const test_entry_t gauntlet_writing_c_compiler_tests[];
extern const uint32_t gauntlet_writing_c_compiler_test_count;

extern const test_entry_t gauntlet_slimcc_tests[];
extern const uint32_t gauntlet_slimcc_test_count;

#endif /* WUBU_TEST_GAUNTLET_H */
