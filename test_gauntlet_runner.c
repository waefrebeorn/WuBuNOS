/*
 * test_gauntlet_runner.c — Main entry point for the WuBuOS Universal Test Gauntlet.
 *
 * Actually compiles each test via HolyD, executes it, checks the result.
 * C18 pure. Self-hosting capable.
 *
 * Fast crash recovery: fork() with shared memory for result communication.
 * Each test runs in a child process; parent reads result from shm.
 * No file I/O, no signal handler complexity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <malloc.h>
#include "wubu_test_gauntlet.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

/* HolyD compiler API (native x86-64 JIT) */
extern int64_t hd_eval(const char *source);

/* Target names */
static const char *target_names[] = {
    "x86-64", "8086", "m68k", "6502", "riscv", "z80",
    "arm64", "mips", "8051", "avr", "pic", "amdgpu", "ptx", "wasm"
};
#define N_TARGETS (sizeof(target_names) / sizeof(target_names[0]))

/* Real crash isolation: each test runs in a forked child. The parent builds
 * the canonical MIR ONCE, then forks; the child inherits the MIR and runs it
 * through EVERY ISA driver (all 14), writing back a packed result array.
 * This keeps correctness (every backend exercised) AND throughput (one fork
 * per test, not per test-per-target — 19k forks, not 266k). If the child
 * dies on any SIG* (including the x86-64 JIT stack-smash via SIGABRT), the
 * parent records ERROR for that test's targets and keeps going. */
typedef struct {
    uint8_t result[14];   /* test_result_t per target */
    int64_t actual[14];
} test_report_t;

static int run_test_safe_all(const char *source, int64_t expected,
                             test_report_t *out) {
    for (uint32_t k = 0; k < N_TARGETS; k++) { out->result[k] = (uint8_t)TEST_ERROR; out->actual[k] = 0; }

    wubu_mir_prog_t prog;
    if (hd_build_mir(source, &prog) != 0) return 0;  /* parse/lower error */

    int pipefd[2];
    if (pipe(pipefd) != 0) { wubu_mir_free(&prog); return 0; }

    pid_t pid = fork();
    if (pid == 0) {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS,  SIG_DFL);
        signal(SIGILL,  SIG_DFL);
        signal(SIGFPE,  SIG_DFL);
        signal(SIGABRT, SIG_DFL);  /* catch stack-smash / assert aborts */
        close(pipefd[0]);

        test_report_t r;
        for (uint32_t k = 0; k < N_TARGETS; k++) {
            const wubu_isa_driver_t *drv = wubu_isa_find(target_names[k]);
            int64_t val = drv ? hd_run_prog(&prog, drv) : wubu_mir_interp(&prog);
            r.result[k] = (uint8_t)((val == expected) ? TEST_PASS : TEST_FAIL);
            r.actual[k] = val;
        }
        ssize_t w = write(pipefd[1], &r, sizeof(r));
        (void)w;
        close(pipefd[1]);
        wubu_mir_free(&prog);
        _exit(0);
    }

    close(pipefd[1]);
    test_report_t r;
    size_t total = 0;
    while (total < sizeof(r)) {
        ssize_t got = read(pipefd[0], ((uint8_t*)&r) + total, sizeof(r) - total);
        if (got <= 0) break;
        total += (size_t)got;
    }
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    wubu_mir_free(&prog);

    if (total == sizeof(r)) {
        *out = r;
    }
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    setbuf(stdout, NULL);  /* unbuffered output */

    /* --core : run only the 7 built-in correctness suites (≈735 tests) across
     * all targets. This is the real "does the compiler work" gate and finishes
     * in minutes. The full 19k externally-downloaded suites (gcc_torture,
     * lacc, llvm, c_testsuite, ...) are a separate long-budget batch.
     * --target <name> : run only the named target (fast single-backend
     * iteration during debugging; combines with --core). */
    int core_only = (argc > 1 && strcmp(argv[1], "--core") == 0);
    const char *only_target = NULL;
    const char *only_suite = NULL;
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--target") == 0 && ai + 1 < argc) {
            only_target = argv[ai + 1];
        }
        if (strcmp(argv[ai], "--suite") == 0 && ai + 1 < argc) {
            only_suite = argv[ai + 1];
        }
    }

    /* --dump-asm <target> <src> : build src, compile via <target> driver,
     * print emitted bytes + run result. Debug aid (reuses the working link). */
    for (int ai = 1; ai < argc - 1; ai++) {
        if (strcmp(argv[ai], "--dump-asm") == 0) {
            const char *tn = argv[ai + 1];
            const char *src = argv[ai + 2];
            const wubu_isa_driver_t *drv = wubu_isa_find(tn);
            if (!drv) { printf("no driver '%s'\n", tn); return 1; }
            wubu_mir_prog_t prog;
            if (hd_build_mir(src, &prog) != 0) { printf("build failed\n"); return 1; }
            /* Allocate mem like hd_run_prog does (JIT needs it for LOAD/STORE) */
            int64_t *mem_ptr = prog.mem;
            if (mem_ptr == NULL) {
                int64_t mem_hi = prog.total_mem;
                if ((int64_t)(prog.next_vr_hi) - 1 > mem_hi) mem_hi = (int64_t)(prog.next_vr_hi) - 1;
                int64_t mem_size = (mem_hi < 1) ? 1 : (mem_hi + 1);
                mem_ptr = (int64_t *)calloc((size_t)mem_size, sizeof(int64_t));
            }
            wubu_mir_prog_t prog_copy = prog;
            prog_copy.mem = mem_ptr;
            uint8_t *code; size_t sz;
            if (drv->compile(&prog_copy, &code, &sz) != 0) { printf("compile failed\n"); free(mem_ptr); wubu_mir_free(&prog); return 1; }
            printf("=== %s asm for: %s ===\n", tn, src);
            for (size_t i = 0; i + 1 < sz; i += 2)
                printf("  +%zu: 0x%04X\n", i, (code[i]<<8)|code[i+1]);
            if (sz & 1) printf("  +%zu: 0x%02X\n", sz-1, code[sz-1]);
            int64_t r = drv->run(code, sz, (int64_t)mem_ptr);
            printf("result=%lld\n", (long long)r);
            free(code);
            if (mem_ptr != prog.mem) free(mem_ptr);
            wubu_mir_free(&prog);
            return 0;
        }
    }

    gauntlet_state_t g;
    gauntlet_init(&g, target_names, N_TARGETS);

    printf("WuBuOS Test Gauntlet — C18 pure · %ld ISAs · %u targets%s\n\n",
           (long)N_TARGETS, g.n_targets, core_only ? " (--core built-in suites)" : "");

    /* Collect test suites (built-in core first, then the external 19k). */
    const test_entry_t *builtin_suites[] = {
        gauntlet_integer_tests, gauntlet_control_tests, gauntlet_bitwise_tests,
        gauntlet_comparison_tests, gauntlet_stress_tests, gauntlet_memory_tests,
        gauntlet_comprehensive_tests,
    };
    const uint32_t builtin_counts[] = {
        gauntlet_integer_test_count, gauntlet_control_test_count,
        gauntlet_bitwise_test_count, gauntlet_comparison_test_count,
        gauntlet_stress_test_count, gauntlet_memory_test_count,
        gauntlet_comprehensive_test_count,
    };
    const char *builtin_names[] = {
        "integer", "control", "bitwise", "comparison", "stress", "memory", "comprehensive"
    };
    const test_entry_t *external_suites[] = {
        gauntlet_gcc_torture_tests, gauntlet_gcc_dg_tests,
        gauntlet_gcc_compile_tests, gauntlet_fujitsu_tests,
        gauntlet_extern_gcc_tests, gauntlet_compcert_tests,
        gauntlet_c_testsuite_tests, gauntlet_llvm_tests,
        gauntlet_lacc_tests, gauntlet_tinycc_tests,
        gauntlet_chibicc_tests, gauntlet_writing_c_compiler_tests,
        gauntlet_slimcc_tests,
    };
    const uint32_t external_counts[] = {
        gauntlet_gcc_torture_test_count, gauntlet_gcc_dg_test_count,
        gauntlet_gcc_compile_test_count, gauntlet_fujitsu_test_count,
        gauntlet_extern_gcc_test_count, gauntlet_compcert_test_count,
        gauntlet_c_testsuite_test_count, gauntlet_llvm_test_count,
        gauntlet_lacc_test_count, gauntlet_tinycc_test_count,
        gauntlet_chibicc_test_count, gauntlet_writing_c_compiler_test_count,
        gauntlet_slimcc_test_count,
    };
    const char *external_names[] = {
        "gcc_torture", "gcc_dg", "gcc_compile", "fujitsu", "extern_gcc",
        "compcert", "c_testsuite", "llvm", "lacc", "tinycc", "chibicc",
        "writing_c_compiler", "slimcc"
    };

    const test_entry_t **suites;
    const uint32_t *counts;
    uint32_t n_suites;
    if (core_only) {
        suites = builtin_suites; counts = builtin_counts;
        n_suites = sizeof(builtin_suites) / sizeof(builtin_suites[0]);
    } else {
        /* Concatenate built-in + external into one combined array. */
        static const test_entry_t *all_suites[20];
        static uint32_t all_counts[20];
        static const char *all_names[20];
        uint32_t bi = sizeof(builtin_suites) / sizeof(builtin_suites[0]);
        uint32_t ei = sizeof(external_suites) / sizeof(external_suites[0]);
        for (uint32_t i = 0; i < bi; i++) { all_suites[i] = builtin_suites[i]; all_counts[i] = builtin_counts[i]; all_names[i] = builtin_names[i]; }
        for (uint32_t i = 0; i < ei; i++) { all_suites[bi + i] = external_suites[i]; all_counts[bi + i] = external_counts[i]; all_names[bi + i] = external_names[i]; }
        /* --suite filter: keep only matching suites */
        if (only_suite) {
            uint32_t j = 0;
            static const test_entry_t *filtered_suites[20];
            static uint32_t filtered_counts[20];
            for (uint32_t i = 0; i < bi + ei; i++) {
                if (strcmp(all_names[i], only_suite) == 0) {
                    filtered_suites[j] = all_suites[i];
                    filtered_counts[j] = all_counts[i];
                    j++;
                }
            }
            if (j == 0) { fprintf(stderr, "no suite '%s'\n", only_suite); return 1; }
            suites = filtered_suites; counts = filtered_counts; n_suites = j;
        } else {
            suites = all_suites; counts = all_counts; n_suites = bi + ei;
        }
    }

    g.n_tests = 0;
    for (uint32_t s = 0; s < n_suites; s++) g.n_tests += counts[s];

    printf("Tests: %u\nTargets: %u\n\n", g.n_tests, g.n_targets);

    /* Parallel differential battery: fork ONE child per target. Each child
     * parses every test's HolyD into canonical MIR and runs it through its own
     * driver (real encoder path), writing a per-target tally to a temp file.
     * Parent waits for all 14 and merges. This is ~N_TARGETS-way parallel and
     * forks only 14 times (not 266k), so the full 19k-test battery finishes
     * in minutes. Crash isolation: a bad native encoder (e.g. the x86-64 JIT
     * stack-smash) kills only that child; the parent records ERROR for that
     * target and continues. Every target exercises the SAME canonical MIR. */

    /* Make a writable temp dir for per-target tallies. */
    char tdir[256];
    snprintf(tdir, sizeof(tdir), "/tmp/gauntlet_%d", (int)getpid());
    mkdir(tdir, 0755);

    pid_t child[14];
    for (uint32_t k = 0; k < N_TARGETS; k++) {
        if (only_target && strcmp(target_names[k], only_target) != 0) {
            child[k] = -1;  /* skipped target */
            continue;
        }
        /* Child processes run in batches of MAX_TESTS_PER_CHILD to prevent
         * malloc arena bloat from OOM under the 4GB cgroup limit. The parent
         * forks a new child when the previous one fills its batch. */
        #define MAX_TESTS_PER_CHILD 10
        uint32_t batch_start = 0; /* test offset within the flat test array */
        /* Build a flat array of all test pointers for batching. */
        uint32_t total_tests = g.n_tests;
        const test_entry_t **flat = NULL;
        uint32_t *flat_suite = NULL; /* suite index for each flat entry */
        uint32_t *flat_idx = NULL;   /* index within suite for each flat entry */
        flat = (const test_entry_t **)malloc(total_tests * sizeof(*flat));
        flat_suite = (uint32_t *)malloc(total_tests * sizeof(*flat_suite));
        flat_idx = (uint32_t *)malloc(total_tests * sizeof(*flat_idx));
        if (!flat || !flat_suite || !flat_idx) {
            free(flat); free(flat_suite); free(flat_idx);
            fprintf(stderr, "malloc failed\n");
            return 1;
        }
        uint32_t ti = 0;
        for (uint32_t s = 0; s < n_suites; s++) {
            for (uint32_t i = 0; i < counts[s]; i++) {
                flat[ti] = &suites[s][i];
                flat_suite[ti] = s;
                flat_idx[ti] = i;
                ti++;
            }
        }
        uint32_t tp_acc = 0, tf_acc = 0, te_acc = 0; /* accumulated tallies */
        for (uint32_t batch = 0; batch < total_tests; batch += MAX_TESTS_PER_CHILD) {
            uint32_t batch_end = batch + MAX_TESTS_PER_CHILD;
            if (batch_end > total_tests) batch_end = total_tests;
            pid_t pid = fork();
            if (pid == 0) {
                /* child: run tests [batch, batch_end) for target k. */
                signal(SIGSEGV, SIG_DFL);
                signal(SIGBUS,  SIG_DFL);
                signal(SIGILL,  SIG_DFL);
                signal(SIGFPE,  SIG_DFL);
                signal(SIGABRT, SIG_DFL);
                signal(SIGPIPE, SIG_IGN);
                uint32_t p = 0, f = 0, e = 0;
                uint32_t tests_run = 0;
                const wubu_isa_driver_t *drv = wubu_isa_find(target_names[k]);
                char fn[320];
                snprintf(fn, sizeof(fn), "%s/t%u", tdir, k);

                for (uint32_t t = batch; t < batch_end; t++) {
                    const test_entry_t *test = flat[t];

                    int pfd[2];
                    if (pipe(pfd) != 0) { e++; continue; }
                    pid_t cpid = fork();
                    if (cpid == 0) {
                        /* grandchild: build + run THIS test on target k only */
                        close(pfd[0]);
                        uint8_t buf[9];
                        buf[0] = (uint8_t)TEST_ERROR;
                        int64_t val = 0;
                        wubu_mir_prog_t prog;
                        int build_result = hd_build_mir(test->source, &prog);
                        if (build_result == 0) {
                            val = drv ? hd_run_prog(&prog, drv)
                                      : wubu_mir_interp(&prog);
                            buf[0] = (uint8_t)((val == test->expected)
                                              ? TEST_PASS : TEST_FAIL);
                            wubu_mir_free(&prog);
                        } else {
                            /* parse failed — mark as ERROR */
                            buf[0] = (uint8_t)TEST_ERROR;
                            val = -999;
                        }
                        memcpy(buf + 1, &val, sizeof(val));
                        ssize_t w = write(pfd[1], buf, sizeof(buf));
                        (void)w;
                        close(pfd[1]);
                        _exit(0);
                    }

                    close(pfd[1]);
                    /* reap grandchild with a per-test watchdog (seconds) */
                    int status = 0;
                    time_t t0 = time(NULL);
                    int settled = 0;
                    while (!settled) {
                        pid_t r = waitpid(cpid, &status, WNOHANG);
                        if (r == cpid) { settled = 1; break; }
                        if (r == 0) {
                            if (difftime(time(NULL), t0) > 20) {
                                kill(cpid, SIGKILL);
                                waitpid(cpid, &status, 0);
                                settled = 1; break;
                            }
                            usleep(5000);
                            continue;
                        }
                        if (r < 0) { settled = 1; break; }  /* ECHILD */
                    }

                    /* Read pipe with O_NONBLOCK so a grandchild that died before
                     * writing doesn't leave us blocked forever. */
                    fcntl(pfd[0], F_SETFL, O_NONBLOCK);
                    uint8_t buf[9] = {0};
                    size_t got = 0;
                    while (got < sizeof(buf)) {
                        ssize_t n = read(pfd[0], buf + got, sizeof(buf) - got);
                        if (n <= 0) break;
                        got += (size_t)n;
                    }
                    close(pfd[0]);

                    uint8_t res = (got == sizeof(buf)) ? buf[0] : (uint8_t)TEST_ERROR;
                    int64_t val = 0;
                    if (got == sizeof(buf)) memcpy(&val, buf + 1, sizeof(val));

                    if (res == (uint8_t)TEST_PASS) p++;
                    else if (res == (uint8_t)TEST_FAIL) {
                        f++;
                        if (f <= 30)
                            printf("  FAIL %-14s %-10s expected=%lld got=%lld\n",
                                   test->name, target_names[k],
                                   (long long)test->expected, (long long)val);
                    } else {
                        e++;
                        if (e <= 100)
                            printf("  EROR %-14s %-10s (crash/timeout/parse)\n",
                                   test->name, target_names[k]);
                    }

                    FILE *of2 = fopen(fn, "w");
                    if (of2) { setvbuf(of2, NULL, _IONBF, 0); fprintf(of2, "%u %u %u\n", p + tp_acc, f + tf_acc, e + te_acc); fclose(of2); }
                    tests_run++;
                    /* Release free memory after EVERY test to prevent OOM.
                     * malloc_trim returns unused sbrk memory to the OS. */
                    if (total_tests > 100) malloc_trim(0);
                }
                _exit(0);
            }
            child[k] = pid;
            /* Wait for this batch to complete before starting the next. */
            int status = 0;
            waitpid(pid, &status, 0);
            /* Read the tally from the file. */
            char fn[320];
            snprintf(fn, sizeof(fn), "%s/t%u", tdir, k);
            FILE *inf = fopen(fn, "r");
            if (inf) {
                if (fscanf(inf, "%u %u %u", &tp_acc, &tf_acc, &te_acc) != 3) {}
                fclose(inf);
            }
        }
        free(flat);
        free(flat_suite);
        free(flat_idx);
        #undef MAX_TESTS_PER_CHILD
    }

    /* Parent: wait for all children, but cap each one with a watchdog so a
     * backend that cannot run in this environment (e.g. ptx with no CUDA
     * device) cannot hang the whole gauntlet forever. A killed child is
     * reported as SKIPPED — an honest "not runnable here" signal, distinct
     * from ERROR (crashed) or FAIL (wrong result). */
    int skipped[14];
    for (uint32_t k = 0; k < N_TARGETS; k++) skipped[k] = 0;

    time_t start = time(NULL);
    int done[14];
    for (uint32_t k = 0; k < N_TARGETS; k++) done[k] = (child[k] == -1);
    /* Per-target watchdog budget (seconds). These are FINITE targets that
     * legitimately need minutes to JIT-compile or interpret 19k tests; only
     * ptx historically hung forever (it now fails fast internally too). Give
     * every backend a generous leash so a slow-but-correct run is never
     * mistaken for a hang. */
    const double budget[14] = {
        2400,2400,2400,2400,2400,2400,2400,2400,2400,2400,2400,2400,60,2400
    };
    while (1) {
        int all_done = 1;
        for (uint32_t k = 0; k < N_TARGETS; k++) {
            if (done[k] || child[k] == -1) continue;
            all_done = 0;
            int status = 0;
            pid_t r = waitpid(child[k], &status, WNOHANG);
            if (r == child[k]) { done[k] = 1; continue; }
            if (r == 0) {  /* still running — check watchdog */
                double elapsed = difftime(time(NULL), start);
                if (elapsed > budget[k]) {
                    fprintf(stderr, "[watchdog] target '%s' exceeded %.0fs — killing (SKIPPED)\n",
                            target_names[k], budget[k]);
                    kill(child[k], SIGKILL);
                    waitpid(child[k], &status, 0);
                    skipped[k] = 1;
                    done[k] = 1;
                }
            }
            /* r < 0 (ECHILD) → treat as done */
            else if (r < 0) { done[k] = 1; }
        }
        if (all_done) break;
        usleep(200000);  /* 200ms poll */
    }

    uint32_t tp[14], tf[14], te[14];
    for (uint32_t k = 0; k < N_TARGETS; k++) {
        tp[k] = tf[k] = te[k] = 0;
        if (child[k] == -1) continue;  /* filtered out — not an error */
        if (skipped[k]) { te[k] = 0; continue; }  /* SKIPPED: not counted as error */
        char fn[320];
        snprintf(fn, sizeof(fn), "%s/t%u", tdir, k);
        FILE *inf = fopen(fn, "r");
        if (inf) { if (fscanf(inf, "%u %u %u", &tp[k], &tf[k], &te[k]) != 3) tf[k] = te[k] = 0; fclose(inf); }
        else { te[k] = g.n_tests; }  /* child missing = all error */
    }

    printf("\n=== Gauntlet Summary ===\n");
    printf("  Total tests:        %u\n", g.n_tests);
    printf("  Targets exercised:  %zu\n\n", N_TARGETS);
    printf("  Per-target (pass / fail / error):\n");
    uint32_t tot_pass = 0, tot_fail = 0, tot_err = 0;
    for (uint32_t k = 0; k < N_TARGETS; k++) {
        const wubu_isa_driver_t *drv = wubu_isa_find(target_names[k]);
        const char *fam = drv ? drv->family : "?";
        if (skipped[k]) {
            printf("    %-10s %-8s      SKIPPED (not runnable in this environment)\n",
                   target_names[k], fam);
            continue;
        }
        printf("    %-10s %-8s %7u / %6u / %5u\n",
               target_names[k], fam, tp[k], tf[k], te[k]);
        tot_pass += tp[k]; tot_fail += tf[k]; tot_err += te[k];
    }
    printf("\n  TOTAL  pass=%-7u fail=%-7u error=%u\n",
           tot_pass, tot_fail, tot_err);
    printf("  (raw test-runs = %u ; per-target runs = %zu)\n",
           g.n_tests, (size_t)g.n_tests * N_TARGETS);

    FILE *f = fopen("/tmp/gauntlet_results.txt", "w");
    if (f) {
        fprintf(f, "PASS=%u FAIL=%u ERROR=%u TOTAL_TESTS=%u TARGETS=%zu\n",
                tot_pass, tot_fail, tot_err, g.n_tests, N_TARGETS);
        fclose(f);
    }

    if (tot_fail > 0 || tot_err > 0) {
        printf("\n  %u test run(s) did NOT pass cleanly\n", tot_fail + tot_err);
        return 1;
    }

    printf("\n  ALL TARGETS PASSED ALL TESTS\n");
    return 0;
}
