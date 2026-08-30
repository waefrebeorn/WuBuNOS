#include <stdio.h>
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

int main() {
    const char *src = "{int a=0; for(int i=0;i<10;i++) a=a+1; a}";
    wubu_mir_prog_t prog;
    if (hd_build_mir(src, &prog) != 0) {
        fprintf(stderr, "build failed\n");
        return 1;
    }
    
    printf("MIR built: n=%zu\n", prog.n);
    
    const wubu_isa_driver_t *drv = wubu_isa_find("x86-64");
    if (!drv) { printf("no driver\n"); return 1; }
    
    uint8_t *code = NULL;
    size_t sz = 0;
    int rc = drv->compile(&prog, &code, &sz);
    printf("compile: rc=%d code=%p sz=%zu\n", rc, code, sz);
    
    if (rc == 0 && code) {
        printf("First 32 bytes: ");
        for (size_t i = 0; i < 32 && i < sz; i++) printf("%02x ", code[i]);
        printf("\n");
        
        int64_t result = drv->run(code, sz, (int64_t)prog.mem);
        printf("JIT result: %lld (expected 10)\n", (long long)result);
        free(code);
    }
    
    int64_t interp = wubu_mir_interp(&prog);
    printf("Interpreter result: %lld\n", (long long)interp);
    
    wubu_mir_free(&prog);
    return 0;
}
