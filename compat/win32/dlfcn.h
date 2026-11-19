/* compat/win32/dlfcn.h -- dlopen/dlsym over Win32 modules.
 * Use case in this tree: the x86-64 JIT resolves external libc symbols.
 * RTLD_DEFAULT/RTLD_NEXT search the exe + common runtime DLLs; there is no
 * true interposition on Windows, only lookup (which is all this tree does). */
#ifndef PORT_WIN32_DLFCN_H
#define PORT_WIN32_DLFCN_H
#if defined(_WIN32)
#include <windows.h>
#define RTLD_LAZY   1
#define RTLD_GLOBAL 2
#define RTLD_DEFAULT ((void *)0)
#define RTLD_NEXT    ((void *)-1)
static inline void *dlopen(const char *name, int flags)
{
    (void)flags;
    return name ? (void *)LoadLibraryA(name) : (void *)GetModuleHandleA(NULL);
}
static inline void *dlsym(void *h, const char *name)
{
    if (h == RTLD_DEFAULT || h == RTLD_NEXT) {
        static const char *mods[] = { NULL, "ucrtbase.dll", "msvcrt.dll",
            "kernel32.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll" };
        for (unsigned i = 0; i < sizeof mods / sizeof mods[0]; i++) {
            HMODULE m = mods[i] ? GetModuleHandleA(mods[i]) : GetModuleHandleA(NULL);
            if (m) {
                void *p = (void *)GetProcAddress(m, name);
                if (p) return p;
            }
        }
        return NULL;
    }
    return (void *)GetProcAddress((HMODULE)h, name);
}
static inline int dlclose(void *h) { return FreeLibrary((HMODULE)h) ? 0 : -1; }
static inline const char *dlerror(void) { return "LoadLibrary/GetProcAddress failed"; }
#else
#include_next <dlfcn.h>
#endif
#endif
