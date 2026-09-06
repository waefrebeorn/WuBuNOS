/*
 * test_gauntlet_v2.c — Run gauntlet tests with per-test fork+timeout.
 * Each test runs in a child process with a 10-second timeout.
 * Parent never accumulates memory because children exit immediately.
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

/* Max tests per suite */
#define MAX_TESTS 16000

static test_info_t all_tests[MAX_TESTS];
static int n_tests = 0;

static int load_suite(const test_entry_t *suite, uint32_t count) {
    for (uint32_t i = 0; i < count && n_tests < MAX_TESTS; i++) {
        all_tests[n_tests].name = suite[i].name;
        all_tests[n_tests].source = suite[i].source;
        all_tests[n_tests].expected = suite[i].expected;
        n_tests++;
    }
    return 0;
}

static int run_single_test(const char *source, int64_t expected) {
    /* Returns: 0=PASS, 1=FAIL, 2=ERROR/TIMEOUT/CRASH */
    wubu_mir_prog_t prog;
    memset(&prog, 0, sizeof(prog));
    
    int build_result = hd_build_mir(source, &prog);
    if (build_result != 0) {
        return 2; /* ERROR */
    }
    
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    int64_t result = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
    
    wubu_mir_free(&prog);
    
    return (result == expected) ? 0 : 1;
}

int main(int argc, char **argv) {
    /* Force all allocations through mmap to prevent OOM */
    mallopt(M_MMAP_THRESHOLD, 0);
    
    /* Collect test suites */
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
    int n_suites = 6;

    /* Filter by --suite */
    const char *only_suite = NULL;
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--suite") == 0 && ai + 1 < argc) {
            only_suite = argv[ai + 1];
        }
    }

    /* Load selected suites into flat array */
    for (int s = 0; s < n_suites; s++) {
        if (only_suite && strcmp(names[s], only_suite) != 0) continue;
        load_suite(suites[s], counts[s]);
    }

    fprintf(stderr, "Loaded %d tests\n", n_tests);
    
    uint32_t total_pass = 0, total_fail = 0, total_err = 0;
    
    for (int i = 0; i < n_tests; i++) {
        /* Fork a child for each test */
        pid_t pid = fork();
        if (pid == 0) {
            /* Child: run test and exit with result */
            signal(SIGSEGV, SIG_DFL);
            signal(SIGBUS, SIG_DFL);
            signal(SIGILL, SIG_DFL);
            signal(SIGFPE, SIG_DFL);
            signal(SIGABRT, SIG_DFL);
            
            int rc = run_single_test(all_tests[i].source, all_tests[i].expected);
            _exit(rc);
        }
        
        if (pid < 0) {
            total_err++;
            continue;
        }
        
        /* Parent: wait with timeout */
        int status;
        int waited = 0;
        int timeout = 10; /* seconds */
        
        for (int t = 0; t < timeout * 10; t++) {
            pid_t w = waitpid(pid, &status, WNOHANG);
            if (w == pid) {
                waited = 1;
                break;
            }
            if (w == -1) break;
            usleep(100000); /* 100ms */
        }
        
        if (!waited) {
            /* Timeout — kill child */
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            total_err++;
            if (total_err <= 100)
                printf("  EROR %-14s (timeout)\n", all_tests[i].name);
        } else if (WIFEXITED(status)) {
            int rc = WEXITSTATUS(status);
            if (rc == 0) {
                total_pass++;
            } else if (rc == 1) {
                total_fail++;
                if (total_fail <= 100)
                    printf("  FAIL %-14s\n", all_tests[i].name);
            } else {
                total_err++;
                if (total_err <= 100)
                    printf("  EROR %-14s (build error)\n", all_tests[i].name);
            }
        } else {
            /* Crashed */
            total_err++;
            if (total_err <= 100)
                printf("  EROR %-14s (crash signal=%d)\n", all_tests[i].name,
                       WIFSIGNALED(status) ? WTERMSIG(status) : 0);
        }
        
        /* Progress */
        if ((i + 1) % 100 == 0 || i + 1 == n_tests) {
            printf("  Progress: %d/%d (pass=%u fail=%u err=%u)\n",
                   i + 1, n_tests, total_pass, total_fail, total_err);
        }
    }

    printf("\n{\"pass\":%u,\"fail\":%u,\"error\":%u,\"total\":%d}\n",
           total_pass, total_fail, total_err, n_tests);

    return 0;
}
