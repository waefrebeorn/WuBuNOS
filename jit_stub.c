/* jit_stub.c -- minimal JIT memory-management stubs for standalone comp builds.
 * The full JIT implementation lives in the OS repo (src/jit/jit.c). These stubs
 * provide just enough for hd_eval / selfhost_battery to compile and run. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "holyd_codegen.h"

void *jit_alloc_exec(size_t size) {
    void *p = mmap(NULL, size, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    return p;
}

void jit_free_exec(void *ptr, size_t size) {
    if (ptr) munmap(ptr, size);
}

/* jit_lock_exec: in standalone mode, buffers are already RWX (mmap above).
 * The full JIT uses this to flip permissions after codegen. Here it's a no-op. */
void *jit_lock_exec(void *ptr, size_t size) {
    (void)size;
    return ptr;
}

/* Stub for the JIT's main entry -- not used in standalone mode. */
extern int wubu_run_program(const char *a, char **b, int c) {
    (void)a; (void)b; (void)c;
    return -1;
}

