#include <stdio.h>
#include <string.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

int gauntlet_c_testsuite_test_count = 0;
int gauntlet_extern_gcc_test_count = 0;
int gauntlet_gcc_compile_test_count = 0;
int gauntlet_gcc_dg_test_count = 0;
int gauntlet_gcc_torture_test_count = 0;
int gauntlet_fujitsu_test_count = 0;
int gauntlet_llvm_test_count = 0;
int gauntlet_lacc_test_count = 0;
int gauntlet_tinycc_test_count = 0;
int gauntlet_chibicc_test_count = 0;
int gauntlet_writing_c_compiler_test_count = 0;
int gauntlet_slimcc_test_count = 0;
void *gauntlet_c_testsuite_tests = NULL;
void *gauntlet_extern_gcc_tests = NULL;
void *gauntlet_gcc_compile_tests = NULL;
void *gauntlet_gcc_dg_tests = NULL;
void *gauntlet_gcc_torture_tests = NULL;
int wubu_run_program(const char *a, char **b, int c) { return 0; }

int main() {
    const char *src = "{int a=0; int b=1; for(int i=0;i<10;i++) {int t=a+b; a=b; b=t;} a}";
    wubu_mir_prog_t prog;
    hd_build_mir(src, &prog);
    if (!prog.mem) prog.mem = (int64_t*)calloc(8192, sizeof(int64_t));

    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    uint8_t *code = NULL;
    size_t sz = 0;
    int rc = drv->compile(&prog, &code, &sz);

    if (rc == 0 && code) {
        FILE *f = fopen("/tmp/jit_fib.bin", "wb");
        fwrite(code, 1, sz, f);
        fclose(f);
        printf("Wrote %zu bytes to /tmp/jit_fib.bin\n", sz);
        free(code);
    }

    wubu_mir_free(&prog);
    return 0;
}
