/*
 * test_single_runner.c — Run a single gauntlet test.
 * Usage: test_single_runner <source_file> <expected> <target>
 * Reads source from file, compiles with HolyD, runs on target, prints result.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern int hd_build_mir(const char *source, void *prog);
extern int64_t hd_run_prog(void *prog, void *drv);
extern void wubu_mir_free(void *prog);
extern const void *wubu_isa_find(const char *name);

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <source_file> <expected> <target>\n", argv[0]);
        return 2;
    }

    const char *source_file = argv[1];
    int64_t expected = atoll(argv[2]);
    const char *target = argv[3];

    /* Read source file */
    FILE *f = fopen(source_file, "r");
    if (!f) { perror("fopen"); return 2; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *source = malloc(len + 1);
    if (!source) { fclose(f); return 2; }
    fread(source, 1, len, f);
    source[len] = '\0';
    fclose(f);

    /* Build MIR */
    char prog[4096]; /* placeholder - actual struct is larger */
    memset(prog, 0, sizeof(prog));
    int build_result = hd_build_mir(source, prog);
    free(source);

    if (build_result != 0) {
        printf("ERROR\n");
        return 1;
    }

    /* Run */
    const void *drv = NULL;
    if (strcmp(target, "interp") != 0) {
        drv = wubu_isa_find(target);
    }
    int64_t result = hd_run_prog(prog, (void *)drv);
    wubu_mir_free(prog);

    if (result == expected) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL %lld\n", (long long)result);
        return 1;
    }
}
