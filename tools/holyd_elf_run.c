/* run_holyd_elf.c -- execute a HolyD-emitted ELF's code on the native CPU.
 * Reads the PT_LOAD (code + data) from the emitted ELF, mmaps writable+exec,
 * patches the RIP-relative global refs already fixed at emit time, and calls
 * the entry via a set-up stack.  Returns the main() int64 (RAX).
 */
#define _POSIX_C_SOURCE 200809L
#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#if __x86_64__
__attribute__((naked))
static int64_t call_entry(void *fn, void *stack) {
    __asm__(
        "push %rbp\n"
        "mov %rsp, %rbp\n"      /* save the C caller frame */
        "mov %rsi, %rsp\n"      /* switch to a guarded throwaway stack */
        "sub $256, %rsp\n"
        "xor %rax, %rax\n"
        "call *%rdi\n"          /* rax = main() return value */
        "mov %rbp, %rsp\n"      /* restore caller frame */
        "pop %rbp\n"
        "ret\n");               /* back to C main, rax preserved */
}
#endif

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *img = malloc(n); fread(img, 1, n, f); fclose(f);

    Elf64_Ehdr *eh = (Elf64_Ehdr *)img;
    if (memcmp(eh->e_ident, ELFMAG, 4) || eh->e_machine != EM_X86_64) {
        fprintf(stderr, "not an x86-64 ELF\n"); return 2;
    }
    /* map code+data from the first PT_LOAD */
    void *code = NULL; size_t code_sz = 0; uintptr_t base = 0;
    for (int i = 0; i < eh->e_phnum; i++) {
        Elf64_Phdr *ph = (Elf64_Phdr *)(img + eh->e_phoff + i * eh->e_phentsize);
        if (ph->p_type == PT_LOAD) {
            base = (uintptr_t)ph->p_vaddr & ~(uintptr_t)4095;
            size_t sz = ph->p_memsz + ((uintptr_t)ph->p_vaddr - base);
            code = mmap((void *)base, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            code_sz = sz;
            memcpy((char *)code + ((uintptr_t)ph->p_vaddr - base),
                   img + ph->p_offset, ph->p_filesz);
            break;
        }
    }
    if (!code) { fprintf(stderr, "no PT_LOAD\n"); return 2; }
    uintptr_t entry = eh->e_entry;
    void *fn = (void *)(entry ? (entry - base + (uintptr_t)code) : code);
    /* throwaway stack above the mapping */
    void *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int64_t r = call_entry(fn, (char *)stack + 65536);
    /* raw write: no libc stdio, deterministic even after running emitted code
     * that mangles the caller frame beyond what stdio expects. */
    char buf[32];
    int wn = snprintf(buf, sizeof(buf), "exit=%lld\n", (long long)r);
    (void)write(1, buf, (size_t)(wn < 0 ? 0 : wn));
    return (int)r;
}
