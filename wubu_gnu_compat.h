// wubu_gnu_compat.h - minimal GNU extension compatibility shim for C11 builds
#ifndef WUBU_GNU_COMPAT_H
#define WUBU_GNU_COMPAT_H

#include <stdlib.h>

#if defined(_WIN32)
/* MinGW/UCRT has no C11 aligned_alloc. _aligned_malloc memory MUST be
 * released with _aligned_free -- plain free() corrupts the heap
 * (0xC0000374). Every aligned_alloc site frees via wubu_aligned_free. */
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define wubu_aligned_free(p) _aligned_free(p)
/* MinGW has _putenv_s, not setenv; overwrite semantics match for our uses */
#define setenv(k, v, overwrite) _putenv_s((k), (v))
/* POSIX mkdir(path, mode) -> Win32 _mkdir(path) */
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#define wubu_aligned_free(p) free(p)
#endif

#endif /* WUBU_GNU_COMPAT_H */
