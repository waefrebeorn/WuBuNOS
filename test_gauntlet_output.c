/*
 * test_gauntlet_output.c — Gauntlet runner for output-comparison tests.
 *
 * For each test:
 *   1. Compile source via HolyD -> MIR -> x86-64 JIT
 *   2. Redirect stdout to pipe
 *   3. Run JIT code
 *   4. Read back stdout, compare to expected output
 *
 * Supports both return-value tests (expected_return != -1) and
 * output-comparison tests (expected_output != NULL).
 *
 * Usage: ./gauntlet_output [--suite <name>] [--jobs <N>]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <malloc.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

#define MAX_TESTS 30000
#define MAX_JOBS 12
#define CAPTURE_SIZE (64 * 1024)

typedef struct {
    const char *name;
    const char *source;
    int64_t expected_return;   /* -1 = use output comparison */
    const char *expected_output; /* NULL = use return value comparison */
} test_info_t;

static test_info_t all_tests[MAX_TESTS];
static int n_tests = 0;

/* Normalize line endings: strip trailing whitespace, ensure single newline */
static int normalize_and_compare(const char *got, const char *expected) {
    /* Simple comparison: strip trailing whitespace from both */
    size_t glen = strlen(got);
    size_t elen = strlen(expected);

    /* Strip trailing whitespace */
    while (glen > 0 && (got[glen-1] == '\n' || got[glen-1] == '\r' || got[glen-1] == ' '))
        glen--;
    while (elen > 0 && (expected[elen-1] == '\n' || expected[elen-1] == '\r' || expected[elen-1] == ' '))
        elen--;

    if (glen != elen) return 1;
    return memcmp(got, expected, glen) != 0;
}

/* Run a single test in the current process (caller handles fork) */
static int run_single_test(const char *source, int64_t expected_return,
                           const char *expected_output, char *output_buf, size_t *output_len) {
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));

    int build_result = hd_build_mir(source, &prog);
    if (build_result != 0) return 2; /* ERROR */

    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    if (!drv) { wubu_mir_free(&prog); return 2; }

    /* Redirect stdout to pipe */
    int stdout_pipe[2];
    if (pipe(stdout_pipe) != 0) { wubu_mir_free(&prog); return 2; }

    int saved_stdout = dup(STDOUT_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdout_pipe[1]);

    int64_t result = hd_run_prog(&prog, drv);

    /* Flush and restore stdout */
    fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);

    /* Read captured output */
    char capture[CAPTURE_SIZE];
    ssize_t total = 0;
    ssize_t n;
    while ((n = read(stdout_pipe[0], capture + total, CAPTURE_SIZE - total - 1)) > 0) {
        total += n;
        if (total >= CAPTURE_SIZE - 1) break;
    }
    close(stdout_pipe[0]);
    capture[total] = '\0';

    wubu_mir_free(&prog);

    if (output_buf && output_len) {
        memcpy(output_buf, capture, total);
        *output_len = (size_t)total;
        output_buf[total] = '\0';
    }

    /* Compare */
    if (expected_output) {
        /* Output comparison mode */
        return (normalize_and_compare(capture, expected_output) == 0) ? 0 : 1;
    } else {
        /* Return value comparison mode */
        return (result == expected_return) ? 0 : 1;
    }
}

typedef struct {
    uint32_t pass;
    uint32_t fail;
    uint32_t err;
} thread_result_t;

typedef struct {
    int thread_id;
    int start_idx, end_idx;
    thread_result_t result;
} worker_arg_t;

static void *worker_thread(void *arg) {
    worker_arg_t *w = (worker_arg_t *)arg;
    w->result.pass = w->result.fail = w->result.err = 0;

    mallopt(M_MMAP_THRESHOLD, 0);

    for (int i = w->start_idx; i < w->end_idx; i++) {
        /* Fork for crash isolation */
        pid_t pid = fork();
        if (pid == 0) {
            signal(SIGSEGV, SIG_DFL);
            signal(SIGBUS, SIG_DFL);
            signal(SIGILL, SIG_DFL);
            signal(SIGFPE, SIG_DFL);
            signal(SIGABRT, SIG_DFL);
            char output[CAPTURE_SIZE];
            size_t outlen = 0;
            int rc = run_single_test(all_tests[i].source,
                                     all_tests[i].expected_return,
                                     all_tests[i].expected_output,
                                     output, &outlen);
            _exit(rc);
        }
        if (pid < 0) {
            w->result.err++;
            continue;
        }

        int status, waited = 0;
        for (int t = 0; t < 50; t++) { /* 5s timeout */
            pid_t wp = waitpid(pid, &status, WNOHANG);
            if (wp == pid) { waited = 1; break; }
            if (wp == -1) break;
            usleep(100000);
        }
        if (!waited) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            w->result.err++;
        } else if (WIFEXITED(status)) {
            int rc = WEXITSTATUS(status);
            if (rc == 0) w->result.pass++;
            else if (rc == 1) w->result.fail++;
            else w->result.err++;
        } else {
            w->result.err++;
        }
    }
    return NULL;
}

/* Include pthread for threading */
#include <pthread.h>

int main(int argc, char **argv) {
    /* TODO: load tests from command line or embedded suite */
    if (n_tests == 0) {
        fprintf(stderr, "No tests loaded. Use --source <file> for single test mode.\n");
        fprintf(stderr, "Usage: ./gauntlet_output --source '<c source>' --expected '<output>'\n");
        fprintf(stderr, "   or: ./gauntlet_output --source '<c source>' --return <value>\n");
        return 1;
    }

    fprintf(stderr, "Loaded %d tests\n", n_tests);

    int n_jobs = 6;
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

    uint32_t total_pass = 0, total_fail = 0, total_err = 0;
    for (int t = 0; t < n_jobs; t++) {
        pthread_join(threads[t], NULL);
        total_pass += worker_args[t].result.pass;
        total_fail += worker_args[t].result.fail;
        total_err += worker_args[t].result.err;
    }

    printf("\n{\"pass\":%u,\"fail\":%u,\"error\":%u,\"total\":%d}\n",
           total_pass, total_fail, total_err, n_tests);

    return 0;
}
