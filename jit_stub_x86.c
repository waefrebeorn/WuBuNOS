/* jit_stub_x86.c — standalone shim: x86-64 has coherent I-cache. */
#include <stddef.h>
void wubu_clear_cache(void *code, size_t size) { (void)code; (void)size; }
