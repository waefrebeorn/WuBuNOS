/* jit_stub.c — standalone JIT exec-page shim for measurement builds.
 * The real JIT lives in the OS repo (jit.c). For compiler-repo tests that
 * actually EXECUTE compiled machine code (e.g. test_tgemm), we need
 * real RWX pages; mmap with PROT_EXEC gives that. NOT committed to OS repo
 * (measurement-only helper; the OS build links the real jit.c). */
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>

void *jit_alloc_exec(size_t n){
    void *p = mmap(NULL, n, PROT_READ|PROT_WRITE|PROT_EXEC,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    return p;
}
void  jit_lock_exec(void *p, size_t n){ (void)p; (void)n; }
void  jit_free_exec(void *p, size_t n){ (void)n; munmap(p, n); }
