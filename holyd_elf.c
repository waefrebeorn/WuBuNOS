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
 * Pure C11, no compiler extensions. Byte-level serialization for portability.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define ELF_BASE_ADDR   0x400000

/* Portable write helpers — no struct packing, no __attribute__ */
static void wb16(uint8_t *buf, uint16_t v) {
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)((v >> 8) & 0xFF);
}
static void wb32(uint8_t *buf, uint32_t v) {
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)((v >> 8) & 0xFF);
    buf[2] = (uint8_t)((v >> 16) & 0xFF);
    buf[3] = (uint8_t)((v >> 24) & 0xFF);
}
static void wb64(uint8_t *buf, uint64_t v) {
    buf[0] = (uint8_t)(v & 0xFF);
    buf[1] = (uint8_t)((v >> 8) & 0xFF);
    buf[2] = (uint8_t)((v >> 16) & 0xFF);
    buf[3] = (uint8_t)((v >> 24) & 0xFF);
    buf[4] = (uint8_t)((v >> 32) & 0xFF);
    buf[5] = (uint8_t)((v >> 40) & 0xFF);
    buf[6] = (uint8_t)((v >> 48) & 0xFF);
    buf[7] = (uint8_t)((v >> 56) & 0xFF);
}

#define ELF_HDR_SIZE 64
#define PHDR_SIZE 56

/* Build ELF64 header into buf (64 bytes) */
static void build_elf_header(uint8_t *buf, uint64_t entry, uint64_t filesz) {
    memset(buf, 0, ELF_HDR_SIZE);
    buf[0] = 0x7f; buf[1] = 'E'; buf[2] = 'L'; buf[3] = 'F';
    buf[4] = 2;     /* 64-bit */
    buf[5] = 1;     /* little-endian */
    buf[6] = 1;     /* ELF version */
    wb16(buf + 16, 2);          /* e_type = ET_EXEC */
    wb16(buf + 18, 0x3e);       /* e_machine = EM_X86_64 */
    wb32(buf + 20, 1);          /* e_version */
    wb64(buf + 24, entry);      /* e_entry */
    wb64(buf + 32, ELF_HDR_SIZE); /* e_phoff */
    wb64(buf + 40, 0);          /* e_shoff = 0 */
    wb32(buf + 48, 0);          /* e_flags */
    wb16(buf + 52, ELF_HDR_SIZE); /* e_ehsize */
    wb16(buf + 54, PHDR_SIZE);  /* e_phentsize */
    wb16(buf + 56, 1);          /* e_phnum */
    wb16(buf + 58, 0);          /* e_shentsize */
    wb16(buf + 60, 0);          /* e_shnum */
    wb16(buf + 62, 0);          /* e_shstrndx */
}

/* Build PT_LOAD program header into buf (56 bytes) */
static void build_phdr(uint8_t *buf, uint64_t filesz) {
    memset(buf, 0, PHDR_SIZE);
    wb32(buf + 0, 1);           /* p_type = PT_LOAD */
    wb32(buf + 4, 7);           /* p_flags = PF_R|PF_W|PF_X */
    wb64(buf + 8, 0);           /* p_offset */
    wb64(buf + 16, ELF_BASE_ADDR); /* p_vaddr */
    wb64(buf + 24, ELF_BASE_ADDR); /* p_paddr */
    wb64(buf + 32, filesz);     /* p_filesz */
    wb64(buf + 40, filesz);     /* p_memsz */
    wb64(buf + 48, 0x200000);   /* p_align */
}

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
    size_t hdr_size = ELF_HDR_SIZE + PHDR_SIZE;    /* 120 */
    size_t code_start = hdr_size + TRAMPOLINE_SIZE;
    size_t user_code_va = ELF_BASE_ADDR + code_start;
    size_t data_start = code_start + code_size;
    size_t total_file_size = data_start + data_size;

    /* Build trampoline with correct call displacement */
    uint8_t trampoline[TRAMPOLINE_SIZE];
    memcpy(trampoline, trampoline_template, TRAMPOLINE_SIZE);
    int32_t call_disp = (int32_t)(TRAMPOLINE_SIZE - 5); /* = 7 */
    memcpy(trampoline + 1, &call_disp, 4);

    /* Write ELF header — entry point is the _start trampoline */
    uint8_t ehdr[ELF_HDR_SIZE];
    uint64_t trampoline_va = ELF_BASE_ADDR + hdr_size;
    build_elf_header(ehdr, trampoline_va, total_file_size);
    fwrite(ehdr, 1, ELF_HDR_SIZE, f);

    /* Write program header */
    uint8_t phdr[PHDR_SIZE];
    build_phdr(phdr, total_file_size);
    fwrite(phdr, 1, PHDR_SIZE, f);

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
