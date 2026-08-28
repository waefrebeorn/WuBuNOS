#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int n2 = N * N;
    
    wubu_mir_prog_t prog;
    wubu_mir_init(&prog);
    wubu_vr_t addr_a = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_b = wubu_mir_alloc(&prog, n2);
    wubu_vr_t addr_c = wubu_mir_alloc(&prog, n2);
    wubu_mir_tgemm_f32(&prog, addr_a, addr_b, addr_c, N, N, N);
    wubu_vr_t result = wubu_mir_load(&prog, addr_c);
    wubu_mir_ret(&prog, result);
    
    printf("MIR program: %zu instructions\n", prog.n);
    for (size_t i = 0; i < prog.n && i < 20; i++) {
        printf("  [%zu] op=%d dst=%d a=%d b=%d imm=%lld\n", i,
               prog.ins[i].op, prog.ins[i].dst, prog.ins[i].a, prog.ins[i].b,
               (long long)prog.ins[i].imm);
    }
    
    const wubu_isa_driver_t *jit = wubu_isa_find("x86-64");
    uint8_t *code = NULL;
    size_t code_size = 0;
    int rc = jit->compile(&prog, &code, &code_size);
    printf("JIT compile: rc=%d, code_size=%zu bytes\n", rc, code_size);
    
    if (rc == 0 && code) {
        /* Print first 64 bytes of JIT code */
        printf("JIT code (first 64 bytes): ");
        for (int i = 0; i < 64 && i < (int)code_size; i++)
            printf("%02x ", code[i]);
        printf("\n");
    }
    
    wubu_mir_free(&prog);
    return 0;
}
