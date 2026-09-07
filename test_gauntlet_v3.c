/*
 * test_gauntlet_v3.c — Multi-process gauntlet runner.
 * Forks N worker processes, each runs a subset of tests sequentially.
 * Results written to temp files, merged at the end.
 *
 * Usage: ./gauntlet_v3 [--suite <name>] [--jobs <N>]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <malloc.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

typedef struct {
    const char *name;
    const char *source;
    int64_t expected;
} test_info_t;

#define MAX_TESTS 25000
#define MAX_JOBS 12

static test_info_t all_tests[MAX_TESTS];
static int n_tests = 0;

static int run_single_test_nofree(const char *source, int64_t expected) {
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int build_result = hd_build_mir(source, &prog);
    if (build_result != 0) return 2;
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    int64_t result = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
    /* Do NOT free — we're in a forked child about to _exit().
     * Freeing in a forked child corrupts the parent's heap metadata. */
    return (result == expected) ? 0 : 1;
}

/* Worker process: run tests [start, end), write results to file */
static void worker_process(int start, int end, const char *result_file) {
/* Force all allocations through mmap to avoid lock contention */
mallopt(M_MMAP_THRESHOLD, 0);

FILE *f = fopen(result_file, "w");
if (!f) _exit(1);

for (int i = start; i < end; i++) {
    /* Run test in a forked child for crash isolation */
    pid_t pid = fork();
    if (pid == 0) {
        /* Child: run test, write result, exit */
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGILL, SIG_DFL);
        signal(SIGFPE, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        int rc = run_single_test_nofree(all_tests[i].source, all_tests[i].expected);
        /* Write result to a per-test file */
        char tmpfile[256];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/gv3_%d_%d.out", getpid(), i);
        FILE *tf = fopen(tmpfile, "w");
        if (tf) { fprintf(tf, "%d\n", rc); fclose(tf); }
        _exit(0);
    }
    if (pid < 0) {
        fprintf(f, "2 %s\n", all_tests[i].name);
        fflush(f);
        continue;
    }
    /* Wait with 5s timeout (most tests complete in <100ms) */
    int status, waited = 0;
    for (int t = 0; t < 50; t++) {
        pid_t wp = waitpid(pid, &status, WNOHANG);
        if (wp == pid) { waited = 1; break; }
        if (wp == -1) break;
        usleep(100000);
    }
    if (!waited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        fprintf(f, "2 %s\n", all_tests[i].name);
        fflush(f);
    } else {
        /* Read result from per-test file */
        char tmpfile[256];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/gv3_%d_%d.out", pid, i);
        FILE *tf = fopen(tmpfile, "r");
        int rc = 2; /* default: error */
        if (tf) {
            if (fscanf(tf, "%d", &rc) != 1) rc = 2;
            fclose(tf);
            unlink(tmpfile);
        }
        fprintf(f, "%d %s\n", rc, all_tests[i].name);
        fflush(f);
    }
    /* Progress every 100 tests */
    if ((i - start) % 100 == 0) {
        fprintf(stderr, "  worker [%d,%d): %d/%d\n", start, end, i - start, end - start);
    }
}
fclose(f);
_exit(0);
}

int main(int argc, char **argv) {
    const test_entry_t *suites[] = {
        gauntlet_gcc_torture_tests,
        gauntlet_extern_gcc_tests,
        gauntlet_c_testsuite_tests,
        gauntlet_llvm_tests,
        gauntlet_lacc_tests,
        gauntlet_fujitsu_proper_tests,
        gauntlet_chibicc_tests,
        gauntlet_compcert_tests,
        gauntlet_comprehensive_tests,
        gauntlet_gcc_compile_tests,
        gauntlet_gcc_dg_tests,
        gauntlet_slimcc_tests,
        gauntlet_tinycc_tests,
        gauntlet_writing_c_compiler_tests,
    };
    const uint32_t counts[] = {
        gauntlet_gcc_torture_test_count,
        gauntlet_extern_gcc_test_count,
        gauntlet_c_testsuite_test_count,
        gauntlet_llvm_test_count,
        gauntlet_lacc_test_count,
        gauntlet_fujitsu_proper_test_count,
        gauntlet_chibicc_test_count,
        gauntlet_compcert_test_count,
        gauntlet_comprehensive_test_count,
        gauntlet_gcc_compile_test_count,
        gauntlet_gcc_dg_test_count,
        gauntlet_slimcc_test_count,
        gauntlet_tinycc_test_count,
        gauntlet_writing_c_compiler_test_count,
    };
    const char *names[] = {
        "gcc_torture", "extern_gcc", "c_testsuite", "llvm", "lacc", "fujitsu_proper",
        "chibicc", "compcert", "comprehensive", "gcc_compile", "gcc_dg",
        "slimcc", "tinycc", "writing_c_compiler",
    };

    const char *only_suite = NULL;
    int n_jobs = 6;
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--suite") == 0 && ai + 1 < argc) only_suite = argv[ai + 1];
        if (strcmp(argv[ai], "--jobs") == 0 && ai + 1 < argc) n_jobs = atoi(argv[ai + 1]);
    }
    if (n_jobs < 1) n_jobs = 1;
    if (n_jobs > MAX_JOBS) n_jobs = MAX_JOBS;

    int n_suites = sizeof(suites) / sizeof(suites[0]);
    for (int s = 0; s < n_suites; s++) {
        if (only_suite && strcmp(names[s], only_suite) != 0) continue;
        for (uint32_t i = 0; i < counts[s] && n_tests < MAX_TESTS; i++) {
            all_tests[n_tests].name = suites[s][i].name;
            all_tests[n_tests].source = suites[s][i].source;
            all_tests[n_tests].expected = suites[s][i].expected;
            n_tests++;
        }
    }

    fprintf(stderr, "Loaded %d tests, running with %d workers\n", n_tests, n_jobs);

    /* Divide tests among workers */
    int tests_per_worker = n_tests / n_jobs;
    int remainder = n_tests % n_jobs;
    int start = 0;

    pid_t worker_pids[MAX_JOBS];
    char result_files[MAX_JOBS][64];

    for (int w = 0; w < n_jobs; w++) {
        int count = tests_per_worker + (w < remainder ? 1 : 0);
        int end = start + count;
        snprintf(result_files[w], sizeof(result_files[w]), "/tmp/gv3_results_%d.txt", w);

        pid_t pid = fork();
        if (pid == 0) {
            /* Child: run tests, write results to file */
            worker_process(start, end, result_files[w]);
            _exit(0); /* should not reach */
        }
        if (pid < 0) {
            perror("fork");
            exit(1);
        }

        worker_pids[w] = pid;
        start = end;
    }

    /* Collect results from all workers */
    uint32_t total_pass = 0, total_fail = 0, total_err = 0;
    char line[1024];

    for (int w = 0; w < n_jobs; w++) {
        /* Wait for worker to FINISH first, then read its result file */
        int status;
        waitpid(worker_pids[w], &status, 0);

        FILE *f = fopen(result_files[w], "r");
        if (!f) {
            fprintf(stderr, "  worker %d: no result file\n", w);
            continue;
        }

        int worker_pass = 0, worker_fail = 0, worker_err = 0;
        while (fgets(line, sizeof(line), f)) {
            int rc;
            char test_name[256];
            if (sscanf(line, " %d %255s", &rc, test_name) >= 1) {
                if (rc == 0) { worker_pass++; total_pass++; }
                else if (rc == 1) {
                    worker_fail++; total_fail++;
                    if (worker_fail <= 50)
                        printf("  FAIL %s\n", test_name);
                }
                else { worker_err++; total_err++; }
            }
        }
        fclose(f);
        unlink(result_files[w]);

        fprintf(stderr, "  worker %d done: pass=%d fail=%d err=%d\n",
                w, worker_pass, worker_fail, worker_err);
    }

    printf("\n{\"pass\":%u,\"fail\":%u,\"error\":%u,\"total\":%d}\n",
           total_pass, total_fail, total_err, n_tests);

    return 0;
}
