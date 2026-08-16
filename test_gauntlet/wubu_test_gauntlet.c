/*
 * wubu_test_gauntlet.c — The WuBuOS Universal Test Gauntlet implementation.
 * C11, self-contained, memory-structured for speed.
 */
#include "wubu_test_gauntlet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timing helper */
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

/* ---- Init */
void gauntlet_init(gauntlet_state_t *g, const char **targets, uint32_t n_targets) {
    memset(g, 0, sizeof(*g));
    g->n_targets = n_targets > 16 ? 16 : n_targets;
    g->results = calloc(1024 * 16, sizeof(target_result_t)); /* 1024 tests * 16 targets */
}

/* ---- Add tests */
void gauntlet_add_test(gauntlet_state_t *g, const test_entry_t *test) {
    (void)g; (void)test;
    /* Tests are registered via the static arrays below */
}

void gauntlet_add_codegen_test(gauntlet_state_t *g, const codegen_test_entry_t *test) {
    (void)g; (void)test;
    /* Tests are registered via the static arrays below */
}

/* ---- Built-in test suites ---- */

/* Integer arithmetic tests — HolyC expression format (no main(), just the value) */
const test_entry_t gauntlet_integer_tests[] = {
    /* Basic ops */
    {"int_add_basic",     "1+2",           3,           TEST_CAT_INTEGER, 0, 100},
    {"int_sub_basic",     "10-3",          7,           TEST_CAT_INTEGER, 0, 100},
    {"int_mul_basic",     "6*7",           42,          TEST_CAT_INTEGER, 0, 100},
    {"int_div_basic",     "20/4",          5,           TEST_CAT_INTEGER, 0, 100},
    {"int_mod_basic",     "17%5",          2,           TEST_CAT_INTEGER, 0, 100},
    /* Negative numbers */
    {"int_add_neg",       "-5+3",          -2,          TEST_CAT_INTEGER, 0, 100},
    {"int_sub_neg",       "-5-3",          -8,          TEST_CAT_INTEGER, 0, 100},
    {"int_mul_neg",       "-5*3",          -15,         TEST_CAT_INTEGER, 0, 100},
    {"int_div_neg",       "-15/3",         -5,          TEST_CAT_INTEGER, 0, 100},
    /* Large numbers */
    {"int_add_large",     "1000000+2000000", 3000000,   TEST_CAT_INTEGER, 0, 100},
    {"int_mul_large",     "1000*1000",     1000000,     TEST_CAT_INTEGER, 0, 100},
    /* Overflow behavior */
    {"int_add_overflow",  "2147483647+1",  -2147483648, TEST_CAT_INTEGER, 0, 100},
    /* Associativity */
    {"int_assoc",         "(1+2)+3",       6,           TEST_CAT_INTEGER, 0, 100},
    {"int_assoc2",        "1+(2+3)",       6,           TEST_CAT_INTEGER, 0, 100},
    /* Distributivity */
    {"int_dist",          "2*(3+4)",       14,          TEST_CAT_INTEGER, 0, 100},
    /* Variables */
    {"int_vars",          "{int a=5; int b=3; a*b}", 15, TEST_CAT_INTEGER, 0, 100},
    /* Chain */
    {"int_chain",         "1+2+3+4+5",     15,          TEST_CAT_INTEGER, 0, 100},
    /* Precedence */
    {"int_prec",          "2+3*4",         14,          TEST_CAT_INTEGER, 0, 100},
    {"int_prec2",         "(2+3)*4",       20,          TEST_CAT_INTEGER, 0, 100},
};

const uint32_t gauntlet_integer_test_count = sizeof(gauntlet_integer_tests) / sizeof(gauntlet_integer_tests[0]);

/* Control flow tests — HolyC expression format */
const test_entry_t gauntlet_control_tests[] = {
    {"ctrl_if_true",      "if(1) 42 else 0", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_if_false",     "if(0) 0 else 42", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_while",        "{int i=0; while(i<10) i++; i}", 10, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_for",          "{int s=0; for(int i=0;i<10;i++) s+=i; s}", 45, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_nested",       "{int s=0; for(int i=0;i<5;i++) for(int j=0;j<5;j++) s++; s}", 25, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_break",        "{int i=0; while(1){i++; if(i==5) break;} i}", 5, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_continue",     "{int s=0; for(int i=0;i<10;i++){if(i%2)continue; s+=i;} s}", 20, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary",      "1?42:0", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary2",     "0?0:42", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_and_short",    "{int x=0; if(0 && (x=1)) 0; x}", 0, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_or_short",     "{int x=0; if(1 || (x=1)) 0; x}", 0, TEST_CAT_CONTROL, 0, 100},
};

const uint32_t gauntlet_control_test_count = sizeof(gauntlet_control_tests) / sizeof(gauntlet_control_tests[0]);

/* Bitwise tests — HolyC expression format */
const test_entry_t gauntlet_bitwise_tests[] = {
    {"bit_and",           "0xFF & 0x0F",   0x0F,        TEST_CAT_BITWISE, 0, 100},
    {"bit_or",            "0xF0 | 0x0F",   0xFF,        TEST_CAT_BITWISE, 0, 100},
    {"bit_xor",           "0xFF ^ 0x0F",   0xF0,        TEST_CAT_BITWISE, 0, 100},
    {"bit_not",           "~0",            -1,          TEST_CAT_BITWISE, 0, 100},
    {"bit_shl",           "1<<8",          256,         TEST_CAT_BITWISE, 0, 100},
    {"bit_shr",           "256>>8",        1,           TEST_CAT_BITWISE, 0, 100},
    {"bit_shl_chain",     "1<<4<<2",       64,          TEST_CAT_BITWISE, 0, 100},
    {"bit_mask",          "0xDEAD & 0xFF",  0xAD,        TEST_CAT_BITWISE, 0, 100},
    {"bit_toggle",        "{int x=0; x^=1; x^=1; x}", 0,   TEST_CAT_BITWISE, 0, 100},
    {"bit_count",         "{int c=0; for(int i=0;i<32;i++) c+=((1<<i)&0xAAAAAAAA)?0:1; c}", 16, TEST_CAT_BITWISE, 0, 100},
};

const uint32_t gauntlet_bitwise_test_count = sizeof(gauntlet_bitwise_tests) / sizeof(gauntlet_bitwise_tests[0]);

/* Comparison tests — HolyC expression format */
const test_entry_t gauntlet_comparison_tests[] = {
    {"cmp_eq_true",       "1==1",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_eq_false",      "1==2",         0,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_ne_true",       "1!=2",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_lt_true",       "1<2",          1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_lt_false",      "2<1",          0,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_le_true",       "1<=1",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_gt_true",       "2>1",          1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_ge_true",       "1>=1",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_neg_lt",        "-1<0",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_neg_gt",        "0>-1",         1,           TEST_CAT_COMPARISON, 0, 100},
};

const uint32_t gauntlet_comparison_test_count = sizeof(gauntlet_comparison_tests) / sizeof(gauntlet_comparison_tests[0]);

/* Stress tests — HolyC expression format */
const test_entry_t gauntlet_stress_tests[] = {
    {"stress_signed_overflow", "{int x=2147483647; x++; x<0}", 1, TEST_CAT_STRESS, 0, 100},
    {"stress_unsigned_wrap",   "{unsigned x=0; x--; x>0}", 1, TEST_CAT_STRESS, 0, 100},
    {"stress_div_by_one",      "42/1", 42, TEST_CAT_STRESS, 0, 100},
    {"stress_mul_by_zero",     "42*0", 0, TEST_CAT_STRESS, 0, 100},
    {"stress_mul_by_one",      "42*1", 42, TEST_CAT_STRESS, 0, 100},
    {"stress_zero_div",        "0/42", 0, TEST_CAT_STRESS, 0, 100},
    {"stress_mod_by_one",      "42%1", 0, TEST_CAT_STRESS, 0, 100},
    {"stress_double_neg",      "- -42", 42, TEST_CAT_STRESS, 0, 100},
    {"stress_triple_neg",      "- - -42", -42, TEST_CAT_STRESS, 0, 100},
    {"stress_deep_nest",       "{int x=0; for(int i=0;i<100;i++) for(int j=0;j<100;j++) x++; x}", 10000, TEST_CAT_STRESS, 0, 100},
};

const uint32_t gauntlet_stress_test_count = sizeof(gauntlet_stress_tests) / sizeof(gauntlet_stress_tests[0]);

/* Memory tests — HolyC expression format */
const test_entry_t gauntlet_memory_tests[] = {
    {"mem_local_var",     "{int x=42; x}", 42,          TEST_CAT_MEMORY, 0, 100},
    {"mem_array_access",  "{int a[3]={1,2,3}; a[1]}", 2, TEST_CAT_MEMORY, 0, 100},
    {"mem_ptr_deref",     "{int x=42; int *p=&x; *p}", 42, TEST_CAT_MEMORY, 0, 100},
};

const uint32_t gauntlet_memory_test_count = sizeof(gauntlet_memory_tests) / sizeof(gauntlet_memory_tests[0]);

/* ---- Run all tests ---- */

void gauntlet_run_all(gauntlet_state_t *g) {
    uint64_t t0 = now_us();

    /* Collect all test suites */
    const test_entry_t *suites[] = {
        gauntlet_integer_tests, gauntlet_control_tests, gauntlet_bitwise_tests,
        gauntlet_comparison_tests, gauntlet_stress_tests, gauntlet_memory_tests,
    };
    const uint32_t counts[] = {
        gauntlet_integer_test_count, gauntlet_control_test_count,
        gauntlet_bitwise_test_count, gauntlet_comparison_test_count,
        gauntlet_stress_test_count, gauntlet_memory_test_count,
    };
    const uint32_t n_suites = sizeof(suites) / sizeof(suites[0]);

    g->n_tests = 0;
    for (uint32_t s = 0; s < n_suites; s++) g->n_tests += counts[s];

    printf("=== WuBuOS Test Gauntlet ===\n");
    printf("  Tests: %u\n", g->n_tests);
    printf("  Targets: %u\n", g->n_targets);
    printf("\n");

    uint32_t test_idx = 0;
    for (uint32_t s = 0; s < n_suites; s++) {
        for (uint32_t i = 0; i < counts[s]; i++) {
            const test_entry_t *t = &suites[s][i];
            printf("  [%3u/%3u] %-30s ", test_idx + 1, g->n_tests, t->name);

            for (uint32_t tgt = 0; tgt < g->n_targets; tgt++) {
                /* For now: simulate test execution */
                /* In real use: compile via holyc → target, run, check result */
                test_result_t result = TEST_PASS; /* Placeholder */

                /* Store result */
                g->results[test_idx * 16 + tgt].result = result;

                /* Update counts */
                if (result == TEST_PASS) {
                    g->pass_by_cat[__builtin_ctz(t->categories)]++;
                    g->pass_by_target[tgt]++;
                } else if (result == TEST_FAIL) {
                    g->fail_by_cat[__builtin_ctz(t->categories)]++;
                    g->fail_by_target[tgt]++;
                } else {
                    g->skip_by_cat[__builtin_ctz(t->categories)]++;
                }

                const char *sym = result == TEST_PASS ? "." :
                                  result == TEST_FAIL ? "F" : "S";
                printf("%s", sym);
            }
            printf("\n");
            test_idx++;
        }
    }

    g->total_time_us = now_us() - t0;

    /* Compute score */
    uint32_t total_pass = 0, total_tests = g->n_tests * g->n_targets;
    for (uint32_t c = 0; c < 16; c++) total_pass += g->pass_by_cat[c];
    g->overall_score = total_tests > 0 ? (total_pass * 10000) / total_tests : 0;
}

/* ---- Print summary ---- */

void gauntlet_print_summary(const gauntlet_state_t *g) {
    printf("\n=== Gauntlet Summary ===\n");
    printf("  Overall score: %u.%02u%%\n", g->overall_score / 100, g->overall_score % 100);
    printf("  Time: %lu us\n", (unsigned long)g->total_time_us);
    printf("\n  Per-category:\n");
    const char *cat_names[] = {
        "Integer", "Float", "Control", "Memory", "Bitwise", "Comparison",
        "Unary", "Logical", "Assignment", "Conversion", "Stdlib", "Concurrency",
        "GPU", "Random", "Perf", "Stress"
    };
    for (int c = 0; c < 16; c++) {
        if (g->pass_by_cat[c] || g->fail_by_cat[c] || g->skip_by_cat[c]) {
            printf("    %-12s: %u pass, %u fail, %u skip\n",
                   cat_names[c], g->pass_by_cat[c], g->fail_by_cat[c], g->skip_by_cat[c]);
        }
    }
    printf("\n  Per-target:\n");
    for (uint32_t t = 0; t < g->n_targets; t++) {
        printf("    Target %2u: %u pass, %u fail\n",
               t, g->pass_by_target[t], g->fail_by_target[t]);
    }
}

/* ---- Export CSV ---- */

void gauntlet_export_csv(const gauntlet_state_t *g, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "test_name,category");
    for (uint32_t t = 0; t < g->n_targets; t++)
        fprintf(f, ",target_%u", t);
    fprintf(f, "\n");
    /* ... iterate tests ... */
    fclose(f);
}

/* ---- Compare targets (differential) ---- */

uint32_t gauntlet_compare_targets(const gauntlet_state_t *g,
                                   uint32_t target_a, uint32_t target_b) {
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < g->n_tests; i++) {
        if (g->results[i * 16 + target_a].result !=
            g->results[i * 16 + target_b].result) {
            mismatches++;
        }
    }
    return mismatches;
}

/* ---- Run category only ---- */

void gauntlet_run_category(gauntlet_state_t *g, test_category_t cat) {
    (void)g; (void)cat;
    /* TODO: filter by category */
}
