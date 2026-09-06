/*
 * test_gauntlet_v3.c — Multi-threaded gauntlet runner.
 * Uses pthreads to run tests in parallel. Each thread forks children
 * for individual tests (fork provides isolation from crashes).
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
#include <pthread.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

typedef struct {
    const char *name;
    const char *source;
    int64_t expected;
} test_info_t;

#define MAX_TESTS 20000
#define MAX_JOBS 12

static test_info_t all_tests[MAX_TESTS];
static int n_tests = 0;

/* Results accumulated per-thread, merged at the end */
typedef struct {
    uint32_t pass;
    uint32_t fail;
    uint32_t err;
    /* Store failing test names for reporting */
    char fail_names[100][HD_MAX_IDENT_LEN];
    int n_fails;
    char err_names[100][HD_MAX_IDENT_LEN];
    int n_errs;
} thread_result_t;

static int run_single_test(const char *source, int64_t expected) {
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    int build_result = hd_build_mir(source, &prog);
    if (build_result != 0) return 2;
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    int64_t result = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
    wubu_mir_free(&prog);
    return (result == expected) ? 0 : 1;
}

typedef struct {
    int thread_id;
    int start_idx, end_idx; /* [start, end) range into all_tests */
    thread_result_t result;
} worker_arg_t;

static void *worker_thread(void *arg) {
    worker_arg_t *w = (worker_arg_t *)arg;
    w->result.pass = w->result.fail = w->result.err = 0;
    w->result.n_fails = w->result.n_errs = 0;

    for (int i = w->start_idx; i < w->end_idx; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            /* Child: run test and exit */
            signal(SIGSEGV, SIG_DFL);
            signal(SIGBUS, SIG_DFL);
            signal(SIGILL, SIG_DFL);
            signal(SIGFPE, SIG_DFL);
            signal(SIGABRT, SIG_DFL);
            int rc = run_single_test(all_tests[i].source, all_tests[i].expected);
            _exit(rc);
        }
        if (pid < 0) {
            w->result.err++;
            continue;
        }
        /* Parent: wait with timeout */
        int status, waited = 0;
        for (int t = 0; t < 100; t++) { /* 10 seconds, 100ms polling */
            pid_t wp = waitpid(pid, &status, WNOHANG);
            if (wp == pid) { waited = 1; break; }
            if (wp == -1) break;
            usleep(100000);
        }
        if (!waited) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            w->result.err++;
            if (w->result.n_errs < 100)
                strncpy(w->result.err_names[w->result.n_errs++], all_tests[i].name, HD_MAX_IDENT_LEN - 1);
        } else if (WIFEXITED(status)) {
            int rc = WEXITSTATUS(status);
            if (rc == 0) {
                w->result.pass++;
            } else if (rc == 1) {
                w->result.fail++;
                if (w->result.n_fails < 100)
                    strncpy(w->result.fail_names[w->result.n_fails++], all_tests[i].name, HD_MAX_IDENT_LEN - 1);
            } else {
                w->result.err++;
                if (w->result.n_errs < 100)
                    strncpy(w->result.err_names[w->result.n_errs++], all_tests[i].name, HD_MAX_IDENT_LEN - 1);
            }
        } else {
            w->result.err++;
            if (w->result.n_errs < 100)
                strncpy(w->result.err_names[w->result.n_errs++], all_tests[i].name, HD_MAX_IDENT_LEN - 1);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    mallopt(M_MMAP_THRESHOLD, 0);

    const test_entry_t *suites[] = {
        gauntlet_gcc_torture_tests,
        gauntlet_extern_gcc_tests,
        gauntlet_c_testsuite_tests,
        gauntlet_llvm_tests,
        gauntlet_lacc_tests,
        gauntlet_fujitsu_tests,
    };
    const uint32_t counts[] = {
        gauntlet_gcc_torture_test_count,
        gauntlet_extern_gcc_test_count,
        gauntlet_c_testsuite_test_count,
        gauntlet_llvm_test_count,
        gauntlet_lacc_test_count,
        gauntlet_fujitsu_test_count,
    };
    const char *names[] = {
        "gcc_torture", "extern_gcc", "c_testsuite", "llvm", "lacc", "fujitsu",
    };

    const char *only_suite = NULL;
    int n_jobs = 6; /* default: match physical cores */
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--suite") == 0 && ai + 1 < argc) only_suite = argv[ai + 1];
        if (strcmp(argv[ai], "--jobs") == 0 && ai + 1 < argc) n_jobs = atoi(argv[ai + 1]);
    }
    if (n_jobs < 1) n_jobs = 1;
    if (n_jobs > MAX_JOBS) n_jobs = MAX_JOBS;

    for (int s = 0; s < 6; s++) {
        if (only_suite && strcmp(names[s], only_suite) != 0) continue;
        for (uint32_t i = 0; i < counts[s] && n_tests < MAX_TESTS; i++) {
            all_tests[n_tests].name = suites[s][i].name;
            all_tests[n_tests].source = suites[s][i].source;
            all_tests[n_tests].expected = suites[s][i].expected;
            n_tests++;
        }
    }

    fprintf(stderr, "Loaded %d tests, running with %d threads\n", n_tests, n_jobs);

    /* Divide tests among threads */
    pthread_t threads[MAX_JOBS];
    worker_arg_t worker_args[MAX_JOBS];
    int tests_per_thread = n_tests / n_jobs;
    int remainder = n_tests % n_jobs;
    int start = 0;

    for (int t = 0; t < n_jobs; t++) {
        int count = tests_per_thread + (t < remainder ? 1 : 0);
        worker_args[t].thread_id = t;
        worker_args[t].start_idx = start;
        worker_args[t].end_idx = start + count;
        start += count;
        pthread_create(&threads[t], NULL, worker_thread, &worker_args[t]);
    }

    /* Wait for all threads and merge results */
    uint32_t total_pass = 0, total_fail = 0, total_err = 0;
    for (int t = 0; t < n_jobs; t++) {
        pthread_join(threads[t], NULL);
        total_pass += worker_args[t].result.pass;
        total_fail += worker_args[t].result.fail;
        total_err += worker_args[t].result.err;
        /* Print per-thread failures */
        for (int f = 0; f < worker_args[t].result.n_fails; f++)
            printf("  FAIL %s\n", worker_args[t].result.fail_names[f]);
        for (int e = 0; e < worker_args[t].result.n_errs; e++)
            printf("  EROR %s\n", worker_args[t].result.err_names[e]);
    }

    printf("\n{\"pass\":%u,\"fail\":%u,\"error\":%u,\"total\":%d}\n",
           total_pass, total_fail, total_err, n_tests);

    return 0;
}
