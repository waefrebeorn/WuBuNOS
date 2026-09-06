/*
 * test_gauntlet_direct.c — Run gauntlet tests directly in the parent process.
 * No forking = no OOM from COW page table accumulation.
 * Uses signal handlers to catch crashes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

static sigjmp_buf jmp_env;
static volatile sig_atomic_t got_signal = 0;

static void sighandler(int sig) {
    got_signal = sig;
    siglongjmp(jmp_env, 1);
}

static void sigalarm_handler(int sig) {
    (void)sig;
    got_signal = SIGALRM;
    fprintf(stderr, "ALARM FIRED!\n");
    siglongjmp(jmp_env, 1);
}

int main(int argc, char **argv) {
    /* Install signal handlers */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sighandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* keep handler installed */
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);

    /* Watchdog timer for hangs */
    struct sigaction sa_alarm;
    memset(&sa_alarm, 0, sizeof(sa_alarm));
    sa_alarm.sa_handler = sigalarm_handler;
    sigemptyset(&sa_alarm.sa_mask);
    sa_alarm.sa_flags = 0; /* keep handler installed */
    sigaction(SIGALRM, &sa_alarm, NULL);

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

    /* Filter by --suite and --start/--count */
    const char *only_suite = NULL;
    int start_test = 0, count_test = 0;
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--suite") == 0 && ai + 1 < argc) {
            only_suite = argv[ai + 1];
        }
        if (strcmp(argv[ai], "--start") == 0 && ai + 1 < argc) {
            start_test = atoi(argv[ai + 1]);
        }
        if (strcmp(argv[ai], "--count") == 0 && ai + 1 < argc) {
            count_test = atoi(argv[ai + 1]);
        }
    }

    uint32_t total_pass = 0, total_fail = 0, total_err = 0;
    uint32_t total_tests = 0;

    for (int s = 0; s < n_suites; s++) {
        if (only_suite && strcmp(names[s], only_suite) != 0) continue;

        printf("=== Suite: %s (%u tests) ===\n", names[s], counts[s]);

        for (uint32_t i = 0; i < counts[s]; i++) {
            if ((int)i < start_test) continue;
            if (count_test > 0 && (int)i >= start_test + count_test) break;
            const test_entry_t *test = &suites[s][i];
            total_tests++;

            /* Build MIR */
            wubu_mir_prog_t prog;
            memset(&prog, 0, sizeof(prog));
            int build_result = hd_build_mir(test->source, &prog);
            if (build_result != 0) {
                total_err++;
                if (total_err <= 100)
                    printf("  EROR %-14s (parse/timeout)\n", test->name);
                continue;
            }

            /* Run with signal handling and watchdog */
            int64_t result = 0;
            int passed = 0;
            got_signal = 0;

            /* Set a 5-second alarm to detect hangs */
            alarm(5);
            
            if (sigsetjmp(jmp_env, 1) == 0) {
                const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
                result = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
                passed = (result == test->expected);
                alarm(0); /* cancel alarm */
            } else {
                /* Crashed or timed out — mark as error, skip cleanup */
                alarm(0); /* cancel alarm */
                total_err++;
                if (total_err <= 100)
                    printf("  EROR %-14s (crash/timeout signal=%d)\n", test->name, got_signal);
                /* Unblock SIGALRM in case siglongjmp restored a blocked mask */
                sigset_t unblock;
                sigemptyset(&unblock);
                sigaddset(&unblock, SIGALRM);
                sigprocmask(SIG_UNBLOCK, &unblock, NULL);
                continue; /* skip wubu_mir_free — heap may be corrupted */
            }

            if (passed) {
                total_pass++;
            } else {
                total_fail++;
                if (total_fail <= 100)
                    printf("  FAIL %-14s expected=%lld got=%lld\n",
                           test->name, (long long)test->expected, (long long)result);
            }

            wubu_mir_free(&prog);
        }
    }

    printf("\n{\"suite\":\"%s\",\"pass\":%u,\"fail\":%u,\"error\":%u,\"total\":%u}\n",
           only_suite ? only_suite : "all", total_pass, total_fail, total_err, total_tests);

    return 0;
}
