/* std_jit_stub.c -- 3-symbol JIT stub for the `holyd_emit_elf` thin build.
 * The old codegen references jit_alloc_exec/free/lock_exec but the emit-elf
 * path never executes code, only writes bytes.  Serve those three with plain
 * mmap-backed executable memory so the compiler links standalone without
 * dragging in the whole OS JIT + minic + spawn chain. */
#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

void *jit_alloc_exec(size_t size) {
    size = (size + 4095) & ~(size_t)4095;
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
void jit_free_exec(void *p, size_t size) {
    if (p) munmap(p, size ? ((size + 4095) & ~(size_t)4095) : getpagesize());
}
void jit_lock_exec(void *p, size_t size) { (void)p; (void)size; }
