/* Local shim so ../jit/jit.h resolves in the standalone compiler build.
 * Declares only what wubu_isa_x86_64.c uses; the real impl lives in the OS
 * repo's jit.c (or jit_stub.c for measurement builds). */
#ifndef WUBU_JIT_SHIM_H
#define WUBU_JIT_SHIM_H
#include <stddef.h>
void *jit_alloc_exec(size_t size);
void  jit_free_exec(void *ptr, size_t size);
void  wubu_clear_cache(void *addr, size_t size);
#endif
