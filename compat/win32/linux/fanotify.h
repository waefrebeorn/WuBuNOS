/* compat/win32/linux/fanotify.h -- include-seam stub for the Windows host build.
 * egojar.h hands every POSIX name to the tree; on _WIN32 this header
 * exists so the seam holds. Code that CALLS these APIs must carry its
 * own _WIN32 implementation or stay Linux-only. */
#ifndef PORT_WIN32_LINUX_FANOTIFY_H
#define PORT_WIN32_LINUX_FANOTIFY_H
#if !defined(_WIN32)
#include_next <linux/fanotify.h>
#endif
#endif
