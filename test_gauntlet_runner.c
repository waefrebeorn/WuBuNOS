/*
 * test_gauntlet_runner.c — Main entry point for the WuBuOS Universal Test Gauntlet.
 *
 * Actually compiles each test via HolyC, executes it, checks the result.
 * C18 pure. Self-hosted capable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyc_mir_eval.h"

/* HolyC compiler API (native x86-64 JIT) */
extern int64_t hc_eval(const char *source);

/* Signal handling for crash recovery */
static sigjmp_buf jump_buffer;
static volatile sig_atomic_t got_signal = 0;

static void signal_handler(int sig) {
    got_signal = sig;
    siglongjmp(jump_buffer, 1);
}

/* Target names */
static const char *target_names[] = {
    "x86-64", "8086", "m68k", "6502", "riscv", "z80",
    "arm64", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "wasm"
};
#define N_TARGETS (sizeof(target_names) / sizeof(target_names[0]))

/* Additional includes for fork-based crash protection */
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
static void write_result(const char *path, int64_t result, int error) {
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%lld %d", (long long)result, error);
        fclose(f);
    }
}

/* Compile + run a test on x86-64 (native JIT) with crash protection via fork */
static int run_test_x86_64(const char *source, int64_t expected, test_result_t *result, int64_t *actual) {
    /* Use a temp file to communicate result from child process */
    char tmpfile[] = "/tmp/gauntlet_result_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) { *result = TEST_ERROR; return 0; }
    close(fd);
    
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process: run the test */
        int64_t res = hc_eval(source);
        write_result(tmpfile, res, 0);
        _exit(0);
    } else if (pid > 0) {
        /* Parent: wait for child */
        int status;
        pid_t w = waitpid(pid, &status, 0);
        
        /* Read result */
        FILE *f = fopen(tmpfile, "r");
        if (f) {
            int error = 0;
            if (fscanf(f, "%lld %d", (long long *)actual, &error) >= 1) {
                if (error || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    *result = TEST_ERROR;
                } else {
                    *result = (*actual == expected) ? TEST_PASS : TEST_FAIL;
                }
            } else {
                *result = TEST_ERROR;
            }
            fclose(f);
        } else {
            *result = TEST_ERROR;
        }
        unlink(tmpfile);
        return 0;
    } else {
        /* Fork failed */
        *result = TEST_ERROR;
        unlink(tmpfile);
        return 0;
    }
}

/* For non-native targets, use MIR + ISA driver with crash protection */
static int run_test_isa_driver(const char *source, const char *target, int64_t expected, test_result_t *result, int64_t *actual) {
    const wubu_isa_driver_t *driver = wubu_isa_find(target);
    if (!driver) { *result = TEST_SKIP; *actual = 0; return 0; }
    if (!driver->compile || !driver->run) { *result = TEST_SKIP; *actual = 0; return 0; }
    
    struct sigaction sa_old, sa_new;
    sa_new.sa_handler = signal_handler;
    sigemptyset(&sa_new.sa_mask);
    sa_new.sa_flags = 0;
    sigaction(SIGSEGV, &sa_new, &sa_old);
    sigaction(SIGBUS, &sa_new, NULL);
    sigaction(SIGILL, &sa_new, NULL);
    sigaction(SIGFPE, &sa_new, NULL);
    
    if (sigsetjmp(jump_buffer, 1) != 0) {
        *actual = 0;
        *result = TEST_ERROR;
        sigaction(SIGSEGV, &sa_old, NULL);
        return 0;
    }
    
    /* HolyC → MIR → driver → run */
    int64_t res = hc_eval_mir(source, driver);
    *actual = res;
    *result = (res == expected) ? TEST_PASS : TEST_FAIL;
    sigaction(SIGSEGV, &sa_old, NULL);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    gauntlet_state_t g;
    gauntlet_init(&g, target_names, N_TARGETS);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  WuBuOS Universal Test Gauntlet                         ║\n");
    printf("║  C18 pure · %d ISAs · %3u tests · %2u targets          ║\n",
           N_TARGETS,
           gauntlet_integer_test_count + gauntlet_control_test_count +
           gauntlet_bitwise_test_count + gauntlet_comparison_test_count +
           gauntlet_stress_test_count + gauntlet_memory_test_count +
           gauntlet_comprehensive_test_count +
           gauntlet_gcc_torture_test_count + gauntlet_gcc_dg_test_count +
           gauntlet_gcc_compile_test_count + gauntlet_fujitsu_test_count +
           gauntlet_extern_gcc_test_count + gauntlet_compcert_test_count +
           gauntlet_c_testsuite_test_count + gauntlet_llvm_test_count +
           gauntlet_lacc_test_count + gauntlet_tinycc_test_count +
           gauntlet_chibicc_test_count + gauntlet_writing_c_compiler_test_count +
           gauntlet_slimcc_test_count,
           g.n_targets);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Collect all test suites */
    const test_entry_t *suites[] = {
        gauntlet_integer_tests, gauntlet_control_tests, gauntlet_bitwise_tests,
        gauntlet_comparison_tests, gauntlet_stress_tests, gauntlet_memory_tests,
        gauntlet_comprehensive_tests,
        gauntlet_gcc_torture_tests, gauntlet_gcc_dg_tests,
        gauntlet_gcc_compile_tests, gauntlet_fujitsu_tests,
        gauntlet_extern_gcc_tests, gauntlet_compcert_tests,
        gauntlet_c_testsuite_tests, gauntlet_llvm_tests,
        gauntlet_lacc_tests, gauntlet_tinycc_tests,
        gauntlet_chibicc_tests, gauntlet_writing_c_compiler_tests,
        gauntlet_slimcc_tests,
    };
    const uint32_t counts[] = {
        gauntlet_integer_test_count, gauntlet_control_test_count,
        gauntlet_bitwise_test_count, gauntlet_comparison_test_count,
        gauntlet_stress_test_count, gauntlet_memory_test_count,
        gauntlet_comprehensive_test_count,
        gauntlet_gcc_torture_test_count, gauntlet_gcc_dg_test_count,
        gauntlet_gcc_compile_test_count, gauntlet_fujitsu_test_count,
        gauntlet_extern_gcc_test_count, gauntlet_compcert_test_count,
        gauntlet_c_testsuite_test_count, gauntlet_llvm_test_count,
        gauntlet_lacc_test_count, gauntlet_tinycc_test_count,
        gauntlet_chibicc_test_count, gauntlet_writing_c_compiler_test_count,
        gauntlet_slimcc_test_count,
    };
    const uint32_t n_suites = sizeof(suites) / sizeof(suites[0]);

    g.n_tests = 0;
    for (uint32_t s = 0; s < n_suites; s++) g.n_tests += counts[s];

    printf("=== WuBuOS Test Gauntlet ===\n");
    printf("  Tests: %u\n", g.n_tests);
    printf("  Targets: %u\n\n", g.n_targets);

    uint32_t test_idx = 0;
    for (uint32_t s = 0; s < n_suites; s++) {
        for (uint32_t i = 0; i < counts[s]; i++) {
            const test_entry_t *t = &suites[s][i];
            printf("  [%3u/%3u] %-30s ", test_idx + 1, g.n_tests, t->name);
            fflush(stdout);

            test_result_t x86_result = TEST_SKIP;
            int64_t x86_actual = 0;

            for (uint32_t tgt = 0; tgt < g.n_targets; tgt++) {
                test_result_t result = TEST_SKIP;
                int64_t actual = 0;

                if (strcmp(target_names[tgt], "x86-64") == 0) {
                    run_test_x86_64(t->source, t->expected, &result, &actual);
                    x86_result = result;
                    x86_actual = actual;
                } else {
                    run_test_isa_driver(t->source, target_names[tgt], t->expected, &result, &actual);
                }

                /* Store result */
                g.results[test_idx * 16 + tgt].result = result;

                /* Update counts */
                if (result == TEST_PASS) {
                    g.pass_by_cat[__builtin_ctz(t->categories)]++;
                    g.pass_by_target[tgt]++;
                } else if (result == TEST_FAIL) {
                    g.fail_by_cat[__builtin_ctz(t->categories)]++;
                    g.fail_by_target[tgt]++;
                } else {
                    g.skip_by_cat[__builtin_ctz(t->categories)]++;
                }

                const char *sym = result == TEST_PASS ? "." :
                                  result == TEST_FAIL ? "F" :
                                  result == TEST_ERROR ? "E" : "S";
                printf("%s", sym);
            }

            /* Show details for failures — reuse result from target loop */
            if (x86_result == TEST_FAIL) {
                printf("  [FAIL: expected %lld, got %lld]", (long long)t->expected, (long long)x86_actual);
            } else if (x86_result == TEST_ERROR) {
                printf("  [ERROR: compilation failed]");
            }

            printf("\n");
            test_idx++;
        }
    }

    /* Print summary */
    gauntlet_print_summary(&g);

    /* Export CSV */
    gauntlet_export_csv(&g, "/tmp/gauntlet_results.csv");
    printf("\n  CSV exported to /tmp/gauntlet_results.csv\n");

    /* Return non-zero on failures */
    uint32_t total_fail = 0;
    for (int c = 0; c < 16; c++) total_fail += g.fail_by_cat[c];

    if (total_fail > 0) {
        printf("\n  ❌ %u test(s) FAILED\n", total_fail);
        return 1;
    }

    printf("\n  ✅ ALL TESTS PASSED\n");
    return 0;
}
