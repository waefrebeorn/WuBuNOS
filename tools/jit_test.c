#include <stdio.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

int main() {
    const char *src = "{int a=0; int b=1; for(int i=0;i<10;i++) {int t=a+b; a=b; b=t;} a}";
    wubu_mir_prog_t prog;
    hd_build_mir(src, &prog);
    if (!prog.mem) prog.mem = (int64_t*)calloc(8192, sizeof(int64_t));
    
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    uint8_t *code = NULL;
    size_t sz = 0;
    drv->compile(&prog, &code, &sz);
    
    FILE *f = fopen("/tmp/jit_fib.bin", "wb");
    fwrite(code, 1, sz, f);
    fclose(f);
    fprintf(stderr, "Wrote %zu bytes\n", sz);
    
    free(code);
    wubu_mir_free(&prog);
    return 0;
}
