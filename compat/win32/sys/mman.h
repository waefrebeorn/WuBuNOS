/* compat/win32/sys/mman.h — mmap/munmap/mprotect over Win32 VirtualAlloc.
 * Only active on _WIN32; other platforms fall through to the real header.
 * JIT use case: MAP_PRIVATE|MAP_ANONYMOUS RWX pages for emitted code. */
#ifndef PORT_WIN32_SYS_MMAN_H
#define PORT_WIN32_SYS_MMAN_H

#if defined(_WIN32)

#include <windows.h>
#include <stddef.h>

#ifndef PROT_READ
#define PROT_READ  0x1
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 0x2
#endif
#ifndef PROT_EXEC
#define PROT_EXEC  0x4
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x02
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x20
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif

static inline DWORD port_mman_prot(int prot)
{
    if ((prot & PROT_EXEC) && (prot & PROT_WRITE)) return PAGE_EXECUTE_READWRITE;
    if (prot & PROT_EXEC)  return PAGE_EXECUTE_READ;
    if (prot & PROT_WRITE) return PAGE_READWRITE;
    if (prot & PROT_READ)  return PAGE_READONLY;
    return PAGE_NOACCESS;
}

static inline void *mmap(void *addr, size_t len, int prot, int flags,
                         int fd, long long off)
{
    (void)addr; (void)flags; (void)fd; (void)off;
    void *p = VirtualAlloc(NULL, len, MEM_COMMIT | MEM_RESERVE,
                           port_mman_prot(prot));
    return p ? p : MAP_FAILED;
}

static inline int munmap(void *addr, size_t len)
{
    (void)len;
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}

static inline int mprotect(void *addr, size_t len, int prot)
{
    DWORD old;
    return VirtualProtect(addr, len, port_mman_prot(prot), &old) ? 0 : -1;
}

#else
#include_next <sys/mman.h>
#endif

#endif /* PORT_WIN32_SYS_MMAN_H */