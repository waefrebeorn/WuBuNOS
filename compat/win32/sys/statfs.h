/* compat/win32/sys/statfs.h -- include-seam stub for the Windows host build.
 * egojar.h hands every POSIX name to the tree; on _WIN32 this header
 * exists so the seam holds. Code that CALLS these APIs must carry its
 * own _WIN32 implementation (see wubu_spawn.c) or stay Linux-only. */
#ifndef PORT_WIN32_SYS_STATFS_H
#define PORT_WIN32_SYS_STATFS_H
#if !defined(_WIN32)
#include_next <sys/statfs.h>
#endif
#endif
