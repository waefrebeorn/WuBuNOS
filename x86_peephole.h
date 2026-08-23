/*
 * x86_peephole.h -- Peephole optimizer for x86-64 JIT code.
 * Public declarations for x86_peephole.c.
 */
#ifndef X86_PEEPHOLE_H
#define X86_PEEPHOLE_H
#include <stddef.h>
#include <stdint.h>

/* Apply peephole optimizations to `code` (in-place). Returns new size. */
size_t x86_peephole_optimize(uint8_t *code, size_t n);

#endif /* X86_PEEPHOLE_H */
