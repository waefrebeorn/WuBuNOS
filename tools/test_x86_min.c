#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

int main(void) {
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t v = wubu_mir_const(&prog, 42);
    wubu_mir_ret(&prog, v);

    const wubu_isa_driver_t *d = wubu_isa_find("x86-64");
    uint8_t *code = NULL; size_t csize = 0;
    int rc = d->compile(&prog, &code, &csize);
    printf("compile=%d size=%zu\n", rc, csize);
    if (rc == 0) {
        int64_t r = d->run(code, csize, 0);
        printf("result=%lld\n", (long long)r);
    }
    free(code);
    wubu_mir_free(&prog);
    return 0;
}
