/* compat/win32/sys/wait.h -- status-decode macros; the spawn implementation
 * itself lives in wubu_spawn.c (_WIN32 branch uses CreateProcess). */
#ifndef PORT_WIN32_SYS_WAIT_H
#define PORT_WIN32_SYS_WAIT_H
#if defined(_WIN32)
#define WIFEXITED(s)   (1)
#define WEXITSTATUS(s) ((s) & 0xFF)
#define WIFSIGNALED(s) (0)
#define WTERMSIG(s)    (0)
#define WNOHANG 1
#else
#include_next <sys/wait.h>
#endif
#endif
