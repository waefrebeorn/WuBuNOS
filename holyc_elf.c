/*
 * holyc_elf.c  --  ELF64 executable emitter for WuBuNOS HolyC compiler.
 *
 * Research-backed approach (Nathan Otterness "Tiny ELF", tchajed/minimal-elf):
 *   - Single PT_LOAD segment covering the ENTIRE file from offset 0
 *   - p_vaddr = ELF_BASE_ADDR, p_offset = 0, p_filesz = p_memsz = file size
 *   - Entry point = ELF_BASE_ADDR + code_start_offset
 *   - RWX permissions (simplest, works on all Linux kernels)
 *
 * Prepends a _start trampoline:
 *   call main     (e8 XX XX XX XX)  -- 5 bytes
 *   mov rdi, rax  (48 89 c7)        -- 3 bytes
 *   mov al, 60    (b0 3c)            -- 2 bytes
 *   syscall       (0f 05)            -- 2 bytes
 *   Total: 12 bytes
 *
 * C11, self-contained, no libelf.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define ELF_BASE_ADDR   0x400000

typedef struct __attribute__((packed)) {
    uint8_t  e_ident[16];
    uint16_t e_type;        /* 2 = ET_EXEC */
    uint16_t e_machine;     /* 0x3e = EM_X86_64 */
    uint32_t e_version;     /* 1 */
    uint64_t e_entry;       /* entry point VA */
    uint64_t e_phoff;       /* program header offset (64) */
    uint64_t e_shoff;       /* section header offset (0 = none) */
    uint32_t e_flags;       /* 0 */
    uint16_t e_ehsize;      /* 64 */
    uint16_t e_phentsize;   /* 56 */
    uint16_t e_phnum;       /* 1 */
    uint16_t e_shentsize;   /* 0 */
    uint16_t e_shnum;       /* 0 */
    uint16_t e_shstrndx;    /* 0 */
} elf64_ehdr_t;

typedef struct __attribute__((packed)) {
    uint32_t p_type;        /* 1 = PT_LOAD */
    uint32_t p_flags;       /* 7 = PF_R | PF_W | PF_X */
    uint64_t p_offset;      /* 0 */
    uint64_t p_vaddr;       /* ELF_BASE_ADDR */
    uint64_t p_paddr;       /* ELF_BASE_ADDR */
    uint64_t p_filesz;      /* total file size */
    uint64_t p_memsz;       /* total file size */
    uint64_t p_align;       /* 0x200000 */
} elf64_phdr_t;

/* _start trampoline: call main, exit via syscall */
static const uint8_t trampoline_template[12] = {
    0xe8, 0x00, 0x00, 0x00, 0x00,   /* call rel32 */
    0x48, 0x89, 0xc7,                  /* mov rdi, rax */
    0xb0, 0x3c,                        /* mov al, 60 */
    0x0f, 0x05                         /* syscall */
};
#define TRAMPOLINE_SIZE 12

int hc_write_elf(const char *filename,
                 const uint8_t *code, size_t code_size,
                 const uint8_t *data, size_t data_size,
                 const size_t *patch_offsets,
                 const size_t *patch_globals,
                 size_t n_patches)
{
    if (!filename || !code || code_size == 0) return -1;

    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    /* Mutable copy of code for patching globals */
    uint8_t *code_buf = (uint8_t *)malloc(code_size);
    if (!code_buf) { fclose(f); return -1; }
    memcpy(code_buf, code, code_size);

    /* Patch RIP-relative globals.
     * disp32 = code_size + global_offset - patch_pos - 4 */
    for (size_t i = 0; i < n_patches; i++) {
        size_t patch_pos = patch_offsets[i];
        size_t global_offset = patch_globals[i];
        if (patch_pos + 4 <= code_size) {
            int32_t disp32 = (int32_t)(code_size + global_offset - patch_pos - 4);
            memcpy(code_buf + patch_pos, &disp32, 4);
        }
    }

    /* Layout: [ELF header (64)] [PHDR (56)] [trampoline (12)] [user code] [data] */
    size_t ehdr_size = sizeof(elf64_ehdr_t);    /* 64 */
    size_t phdr_size = sizeof(elf64_phdr_t);    /* 56 */
    size_t hdr_size = ehdr_size + phdr_size;    /* 120 */

    size_t code_start = hdr_size + TRAMPOLINE_SIZE;  /* offset of user code in file */
    size_t user_code_va = ELF_BASE_ADDR + code_start;
    size_t data_start = code_start + code_size;
    size_t total_file_size = data_start + data_size;

    /* Build trampoline with correct call displacement.
     * call rel32: displacement = target - (rip_after_call)
     * rip_after_call = ELF_BASE_ADDR + ehdr_size + phdr_size + 5
     * target = user_code_va
     * displacement = user_code_va - (ELF_BASE_ADDR + hdr_size + 5)
     *               = TRAMPOLINE_SIZE - 5 = 7 */
    uint8_t trampoline[TRAMPOLINE_SIZE];
    memcpy(trampoline, trampoline_template, TRAMPOLINE_SIZE);
    int32_t call_disp = (int32_t)(TRAMPOLINE_SIZE - 5); /* = 7 */
    memcpy(trampoline + 1, &call_disp, 4);

    /* ELF header */
    elf64_ehdr_t ehdr;
    memset(&ehdr, 0, sizeof(ehdr));
    ehdr.e_ident[0] = 0x7f;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = 2;     /* 64-bit */
    ehdr.e_ident[5] = 1;     /* little-endian */
    ehdr.e_ident[6] = 1;     /* ELF version */
    ehdr.e_type = 2;         /* ET_EXEC */
    ehdr.e_machine = 0x3e;   /* EM_X86_64 */
    ehdr.e_version = 1;
    ehdr.e_entry = ELF_BASE_ADDR + ehdr_size + phdr_size;  /* entry = _start trampoline */
    ehdr.e_phoff = ehdr_size;
    ehdr.e_shoff = 0;
    ehdr.e_flags = 0;
    ehdr.e_ehsize = (uint16_t)ehdr_size;
    ehdr.e_phentsize = (uint16_t)phdr_size;
    ehdr.e_phnum = 1;
    ehdr.e_shentsize = 0;
    ehdr.e_shnum = 0;
    ehdr.e_shstrndx = 0;
    fwrite(&ehdr, 1, sizeof(ehdr), f);

    /* Single PT_LOAD covering entire file */
    elf64_phdr_t phdr;
    memset(&phdr, 0, sizeof(phdr));
    phdr.p_type = 1;                    /* PT_LOAD */
    phdr.p_flags = 7;                   /* PF_R | PF_W | PF_X */
    phdr.p_offset = 0;
    phdr.p_vaddr = ELF_BASE_ADDR;
    phdr.p_paddr = ELF_BASE_ADDR;
    phdr.p_filesz = total_file_size;
    phdr.p_memsz = total_file_size;
    phdr.p_align = 0x200000;
    fwrite(&phdr, 1, sizeof(phdr), f);

    /* Trampoline */
    fwrite(trampoline, 1, TRAMPOLINE_SIZE, f);

    /* User code (patched) */
    fwrite(code_buf, 1, code_size, f);
    free(code_buf);

    /* Data section */
    if (data_size > 0) {
        fwrite(data, 1, data_size, f);
    }

    fclose(f);
    return 0;
}
