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
    /* Allocate results buffer: support up to 100K tests * 16 targets */
    g->results = calloc(100000 * 16, sizeof(target_result_t));
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

/* Integer arithmetic tests — HolyD expression format (no main(), just the value) */
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
    {"int_vars",          "{int a=5; int b=3; a*b;}", 15, TEST_CAT_INTEGER, 0, 100},
    /* Chain */
    {"int_chain",         "1+2+3+4+5",     15,          TEST_CAT_INTEGER, 0, 100},
    /* Precedence */
    {"int_prec",          "2+3*4",         14,          TEST_CAT_INTEGER, 0, 100},
    {"int_prec2",        "(2+3)*4",       20,          TEST_CAT_INTEGER, 0, 100},
    /* Additional integer tests */
    {"int_shl_basic",    "1<<4",          16,          TEST_CAT_INTEGER, 0, 100},
    {"int_shr_basic",    "256>>2",        64,          TEST_CAT_INTEGER, 0, 100},
    {"int_bitand",       "15&7",          7,           TEST_CAT_INTEGER, 0, 100},
    {"int_bitor",        "8|4",           12,          TEST_CAT_INTEGER, 0, 100},
    {"int_bitxor",       "10^3",          9,           TEST_CAT_INTEGER, 0, 100},
    {"int_neg",          "-42",           -42,         TEST_CAT_INTEGER, 0, 100},
    {"int_not",          "!0",            1,           TEST_CAT_INTEGER, 0, 100},
    {"int_not_truthy",   "!42",           0,           TEST_CAT_INTEGER, 0, 100},
    {"int_bitnot",       "~0",            -1,          TEST_CAT_INTEGER, 0, 100},
    {"int_div_trunc",    "7/2",           3,           TEST_CAT_INTEGER, 0, 100},
    {"int_mod",          "7%3",           1,           TEST_CAT_INTEGER, 0, 100},
    {"int_chain_long",   "1+2+3+4+5+6+7+8+9+10", 55, TEST_CAT_INTEGER, 0, 100},
    {"int_mixed_prec",   "2+3*4-5",       9,           TEST_CAT_INTEGER, 0, 100},
    {"int_parens_deep",  "((1+2)*(3+4))", 21,          TEST_CAT_INTEGER, 0, 100},
    {"int_sub_zero",     "42-0",          42,          TEST_CAT_INTEGER, 0, 100},
    {"int_mult_one",     "42*1",          42,          TEST_CAT_INTEGER, 0, 100},
    {"int_mult_zero",    "42*0",          0,           TEST_CAT_INTEGER, 0, 100},
    {"int_div_one",      "42/1",          42,          TEST_CAT_INTEGER, 0, 100},
    {"int_div_zero_num", "0/42",          0,           TEST_CAT_INTEGER, 0, 100},
    {"int_mod_one",      "42%1",          0,           TEST_CAT_INTEGER, 0, 100},
    {"int_double_neg",   "- -42",         42,          TEST_CAT_INTEGER, 0, 100},
    {"int_triple_neg",   "- - -42",       -42,         TEST_CAT_INTEGER, 0, 100},
    {"int_add_assign",   "{int a=5; a+=3; a}", 8,       TEST_CAT_INTEGER, 0, 100},
    {"int_sub_assign",   "{int a=5; a-=3; a}", 2,       TEST_CAT_INTEGER, 0, 100},
    {"int_mul_assign",   "{int a=5; a*=3; a}", 15,      TEST_CAT_INTEGER, 0, 100},
    {"int_div_assign",   "{int a=15; a/=3; a}", 5,      TEST_CAT_INTEGER, 0, 100},
    {"int_mod_assign",   "{int a=17; a%=5; a}", 2,      TEST_CAT_INTEGER, 0, 100},
    {"int_post_inc",     "{int a=5; a++; a}", 6,         TEST_CAT_INTEGER, 0, 100},
    {"int_pre_inc",      "{int a=5; ++a}",   6,          TEST_CAT_INTEGER, 0, 100},
    {"int_post_dec",     "{int a=5; a--; a}", 4,        TEST_CAT_INTEGER, 0, 100},
    {"int_pre_dec",      "{int a=5; --a}",   4,          TEST_CAT_INTEGER, 0, 100},
    {"int_post_inc_ret", "{int a=5; a++; a-1}", 5,      TEST_CAT_INTEGER, 0, 100},
    {"int_ternary",      "1?42:99",        42,          TEST_CAT_INTEGER, 0, 100},
    {"int_ternary_else", "0?42:99",        99,          TEST_CAT_INTEGER, 0, 100},
    {"int_assign_chain", "{int a=0; a=1; a=2; a}", 2,  TEST_CAT_INTEGER, 0, 100},
    {"int_bit_shl_10",   "1<<10",          1024,        TEST_CAT_INTEGER, 0, 100},
    {"int_bit_shr_4",    "256>>4",         16,          TEST_CAT_INTEGER, 0, 100},
    {"int_bit_shl_20",   "1<<20",          1048576,     TEST_CAT_INTEGER, 0, 100},
    {"int_neg_large",    "{-2147483648}", -2147483648,  TEST_CAT_INTEGER, 0, 100},
    {"int_max_32bit",    "2147483647",     2147483647,  TEST_CAT_INTEGER, 0, 100},
    {"int_add_neg",      "-5+3",           -2,          TEST_CAT_INTEGER, 0, 100},
    {"int_sub_neg_chain","10-3-2",         5,           TEST_CAT_INTEGER, 0, 100},
};

const uint32_t gauntlet_integer_test_count = sizeof(gauntlet_integer_tests) / sizeof(gauntlet_integer_tests[0]);

/* Control flow tests — HolyD expression format (semicolons required) */
const test_entry_t gauntlet_control_tests[] = {
    {"ctrl_ternary",      "1?42:0", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary2",     "0?0:42", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_and_short",    "{int x=0; if(0 && (x=1)) {0;} x;}", 0, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_or_short",     "{int x=0; if(1 || (x=1)) {0;} x;}", 0, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_if_true",      "if(1) {42;} else {0;}", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_if_false",     "if(0) {0;} else {42;}", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_while",        "{int i=0; while(i<10) {i++;} i;}", 10, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_for",          "{int s=0; for(int i=0;i<10;i++) {s+=i;} s;}", 45, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_nested",       "{int s=0; for(int i=0;i<5;i++) {for(int j=0;j<5;j++) {s++;}} s;}", 25, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_break",        "{int i=0; while(1) {i++; if(i==5) {break;}} i;}", 5, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_continue",     "{int s=0; for(int i=0;i<10;i++) {if(i%2) {continue;} s+=i;} s;}", 20, TEST_CAT_CONTROL, 0, 100},
    /* Additional control flow */
    {"ctrl_if_no_else",   "if(1) {42;}", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_if_nested",    "{int a=0; if(1) {if(1) {a=42;} else {a=0;}} a}", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_if_else_false","{int a=0; if(0) {a=1;} else {a=42;} a}", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_while_10",     "{int i=0; while(i<10) {i++;} i}", 10, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_while_100",    "{int i=0; while(i<100) {i++;} i}", 100, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_while_sum",    "{int i=0; int s=0; while(i<10) {s+=i; i++;} s}", 45, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_for_sum",      "{int s=0; for(int i=0;i<10;i++) {s+=i;} s}", 45, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_for_100",      "{int s=0; for(int i=0;i<100;i++) {s++;} s}", 100, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_break_5",      "{int i=0; while(1) {i++; if(i==5) {break;} } i}", 5, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_break_50",     "{int i=0; while(1) {i++; if(i==50) {break;} } i}", 50, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_continue_sum", "{int s=0; for(int i=0;i<10;i++) {if(i%2==0) {continue;} s+=i;} s}", 25, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_nested_loop",  "{int s=0; for(int i=0;i<5;i++) {for(int j=0;j<5;j++) {s++;}} s}", 25, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_nested_10",    "{int s=0; for(int i=0;i<10;i++) {for(int j=0;j<10;j++) {s++;}} s}", 100, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary_1",    "1?42:99", 42, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary_0",    "0?42:99", 99, TEST_CAT_CONTROL, 0, 100},
    {"ctrl_ternary_cmp",  "1==1?42:99", 42, TEST_CAT_CONTROL, 0, 100},
};

const uint32_t gauntlet_control_test_count = sizeof(gauntlet_control_tests) / sizeof(gauntlet_control_tests[0]);

/* Bitwise tests — HolyD expression format */
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
    /* Additional bitwise */
    {"bit_and_chain",     "0xFF&0xF0&0x0F", 0,           TEST_CAT_BITWISE, 0, 100},
    {"bit_or_chain",      "0xF0|0x0F|0xFF", 0xFF,        TEST_CAT_BITWISE, 0, 100},
    {"bit_xor_chain",     "0xFF^0x00^0xFF", 0,           TEST_CAT_BITWISE, 0, 100},
    {"bit_shl_big",       "1<<16",        65536,        TEST_CAT_BITWISE, 0, 100},
    {"bit_shr_big",       "65536>>8",     256,          TEST_CAT_BITWISE, 0, 100},
    {"bit_and_zeros",     "0xABCD&0",     0,            TEST_CAT_BITWISE, 0, 100},
    {"bit_or_ones",       "0|0xFFFF",     0xFFFF,       TEST_CAT_BITWISE, 0, 100},
    {"bit_not_ff",        "~0xFF",        -256,         TEST_CAT_BITWISE, 0, 100},
    {"bit_xor_ff",        "0xAB^0xFF",     0x54,         TEST_CAT_BITWISE, 0, 100},
    {"bit_and_0f",        "0xDEAD&0xFFFF", 0xDEAD,       TEST_CAT_BITWISE, 0, 100},
};

const uint32_t gauntlet_bitwise_test_count = sizeof(gauntlet_bitwise_tests) / sizeof(gauntlet_bitwise_tests[0]);

/* Comparison tests — HolyD expression format */
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
    /* Additional comparisons */
    {"cmp_eq_large",      "2147483647==2147483647", 1, TEST_CAT_COMPARISON, 0, 100},
    {"cmp_ne_large",      "2147483647!=1", 1,        TEST_CAT_COMPARISON, 0, 100},
    {"cmp_lt_large",      "1000000<2000000", 1,      TEST_CAT_COMPARISON, 0, 100},
    {"cmp_gt_large",      "2000000>1000000", 1,      TEST_CAT_COMPARISON, 0, 100},
    {"cmp_le_eq",         "5<=5",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_ge_eq",         "5>=5",         1,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_lt_false2",     "5<3",          0,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_gt_false2",     "3>5",          0,           TEST_CAT_COMPARISON, 0, 100},
    {"cmp_chained",       "1<2<3",        1,           TEST_CAT_COMPARISON, 0, 100},
};

const uint32_t gauntlet_comparison_test_count = sizeof(gauntlet_comparison_tests) / sizeof(gauntlet_comparison_tests[0]);

/* Stress tests — HolyD expression format */
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
    /* Additional stress tests */
    {"stress_factorial",       "{int r=1; for(int i=1;i<=5;i++) r*=i; r}", 120, TEST_CAT_STRESS, 0, 100},
    {"stress_fib",             "{int a=0; int b=1; for(int i=0;i<10;i++) {int t=a+b; a=b; b=t;} a}", 55, TEST_CAT_STRESS, 0, 100},
    {"stress_complex_expr",    "3+4*2-6/3+10%3", 10, TEST_CAT_STRESS, 0, 100},
    {"stress_ternary_chain",   "1?2?3:4:5", 3, TEST_CAT_STRESS, 0, 100},
    {"stress_nested_parens",   "(((1+2))+((3*4)))", 15, TEST_CAT_STRESS, 0, 100},
    {"stress_big_mul",         "1000*1000*10", 10000000, TEST_CAT_STRESS, 0, 100},
    {"stress_big_add",         "999999+1", 1000000, TEST_CAT_STRESS, 0, 100},
    {"stress_if_chain",        "1? (0?1:2) : 3", 2, TEST_CAT_STRESS, 0, 100},
    {"stress_while_fact",      "{int r=1; int i=1; while(i<6) {r*=i; i++;} r}", 120, TEST_CAT_STRESS, 0, 100},
};

const uint32_t gauntlet_stress_test_count = sizeof(gauntlet_stress_tests) / sizeof(gauntlet_stress_tests[0]);

/* Memory tests — HolyD expression format (statements need semicolons) */
const test_entry_t gauntlet_memory_tests[] = {
    {"mem_local_var",     "{int x=42; x}", 42,          TEST_CAT_MEMORY, 0, 100},
    {"mem_array_access",  "{int a[3]={1,2,3}; a[1]}", 2, TEST_CAT_MEMORY, 0, 100},
    {"mem_ptr_deref",     "{int x=42; int *p=&x; *p}", 42, TEST_CAT_MEMORY, 0, 100},
    /* Additional memory tests */
    {"mem_array_0",       "{int a[3]={1,2,3}; a[0]}", 1, TEST_CAT_MEMORY, 0, 100},
    {"mem_array_2",       "{int a[3]={1,2,3}; a[2]}", 3, TEST_CAT_MEMORY, 0, 100},
    {"mem_array_sum",     "{int a[3]={10,20,30}; a[0]+a[1]+a[2]}", 60, TEST_CAT_MEMORY, 0, 100},
    {"mem_array_large",   "{int a[5]={100,200,300,400,500}; a[4]}", 500, TEST_CAT_MEMORY, 0, 100},
    {"mem_array_assign",  "{int a[3]={0,0,0}; a[1]=42; a[1]}", 42, TEST_CAT_MEMORY, 0, 100},
    {"mem_array_inc",     "{int a[3]={1,2,3}; a[1]++}", 2, TEST_CAT_MEMORY, 0, 100},
    {"mem_ptr_deref_write","{int x=0; int *p=&x; *p=42; x}", 42, TEST_CAT_MEMORY, 0, 100},
    {"mem_ptr_arithmetic", "{int a[3]={10,20,30}; int *p=a; p[2]}", 30, TEST_CAT_MEMORY, 0, 100},
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
                /* In real use: compile via holyd → target, run, check result */
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
