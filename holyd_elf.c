/*
 * holyd_elf.c  --  ELF64 executable emitter for WuBuNOS HolyD compiler.
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

int hd_write_elf(const char *filename,
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

/* ── Multiboot-bootable variant ─────────────────────────────────────────
 * Same [ELF hdr][PHDR][user code][data] single PT_LOAD at ELF_BASE_ADDR, but
 * the 12-byte exit trampoline is replaced by a 32-bit->long-mode multiboot
 * prologue.  Multiboot/GRUB loads the ELF at p_vaddr (0x400000) and enters
 * the 32-bit prologue with paging OFF; the prologue sets up identity paging
 * (2MB huge pages), enters long mode, jumps to 64-bit code that puts a stack
 * up, calls the emitted main() body, writes the return value to COM1 as two
 * chars ('H' + low byte), then halts.  The user body and .data stay CONSECUTIVE
 * after the prologue, so RIP-relative global patches (relative to body start)
 * keep resolving regardless of prologue size.  Verified: boots under
 * `qemu -kernel`/GRUB and prints to the serial port -- no #UD.
 */

/* fixed prologue VA offsets relative to ELF_BASE_ADDR */
#define MB_HDR_OFF    0x78   /* = ELF_HDR_SIZE + PHDR_SIZE : multiboot header must sit in the first 8K */
#define GDPTR_OFF     (MB_HDR_OFF + 16)  /* the 6-byte {limit,base} lives here; lgdt reads [this] */
#define GDT_OFF       (GDPTR_OFF + 6)    /* the GDT entries follow immediately */
#define LM32_OFF      (GDT_OFF + 40)     /* then the 32-bit entry code */
#define PML4_VA_OFF   0x1000
#define PDPT_VA_OFF   0x2000
#define PD_VA_OFF     0x3000
#define LM64_VA_OFF   0x4000
#define STACK_TOP     0x90000

static void put8(uint8_t **p, uint8_t v) { *(*p)++ = v; }
static void put16(uint8_t **p, uint16_t v) { wb16(*p, v); *p += 2; }
static void put32(uint8_t **p, uint32_t v) { wb32(*p, v); *p += 4; }
static void put64(uint8_t **p, uint64_t v) { wb64(*p, v); *p += 8; }

static void passert(const char *msg, int ok) { if (!ok) { fprintf(stderr, "hd_write_boot_elf: %s\n", msg); exit(2); } }

int hd_write_boot_elf(const char *filename,
                      const uint8_t *code, size_t code_size,
                      const uint8_t *data, size_t data_size,
                      const size_t *patch_offsets,
                      const size_t *patch_globals,
                      size_t n_patches)
{
    if (!filename || !code || code_size == 0) return -1;

    /* Run the same RIP-relative global patching as hd_write_elf. */
    uint8_t *code_buf = (uint8_t *)malloc(code_size);
    if (!code_buf) return -1;
    memcpy(code_buf, code, code_size);
    for (size_t i = 0; i < n_patches; i++) {
        size_t pp = patch_offsets[i], go = patch_globals[i];
        if (pp + 4 <= code_size) {
            int32_t disp32 = (int32_t)(code_size + go - pp - 4);
            memcpy(code_buf + pp, &disp32, 4);
        }
    }

    char *pro = malloc(LM64_VA_OFF + 128);      /* prologue bytes (except tables) */
    uint8_t *pp = (uint8_t *)pro;
    uint8_t *gdtp, *gdptrdst, *lm32dst, *lm64dst, *calldst;

    /* multiboot header (16B) */
    uint8_t *hdr = (uint8_t *)pro;
    put32(&pp, 0x1BADB002);
    put32(&pp, 0x00000003);
    put32(&pp, (uint32_t)-(int32_t)(0x1BADB002 + 3));
    put32(&pp, 0);

    /* gdptr (6B) FIRST, at the exact address lgdt reads:
     *   limit 0x27 (5 descriptors), base = VA of the GDT that follows. */
    gdptrdst = pp;
    put16(&pp, 0x27);
    { uint8_t b[4]; wb32(b, (uint32_t)(ELF_BASE_ADDR + GDT_OFF)); memcpy(pp, b, 4); pp += 4; }

    /* GDT (40B, 5 entries):
     *   0x08 32-bit code, 0x10 32-bit code, 0x18 64-bit code, 0x20 data.
     * The bootloader enters with CS = 0x08 or 0x10; both must be code so the
     * fetch after lgdt is valid.  0x18 is the long-mode code segment; 0x20 the
     * flat data segment (L=0) for DS/ES/SS. */
    gdtp = pp;
    memset(pp, 0, 8); pp += 8;                       /* null */
    { /* 0x08 32-bit code */
        uint8_t *g = pp; (void)g;
        put8(&pp, 0xff); put8(&pp, 0xff); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x9a); put8(&pp, 0xcf); put8(&pp, 0x00);
    }
    { /* 0x10 32-bit code (matches GRUB's CS) */
        uint8_t *g = pp; (void)g;
        put8(&pp, 0xff); put8(&pp, 0xff); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x9a); put8(&pp, 0xcf); put8(&pp, 0x00);
    }
    { /* 0x18 long-mode code */
        uint8_t *g = pp; (void)g;
        put8(&pp, 0xff); put8(&pp, 0xff); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x9a); put8(&pp, 0xaf); put8(&pp, 0x00);
    }
    { /* 0x20 flat data (L=0) */
        uint8_t *g = pp; (void)g;
        put8(&pp, 0xff); put8(&pp, 0xff); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x00); put8(&pp, 0x92); put8(&pp, 0xcf); put8(&pp, 0x00);
    }

    /* 32-bit entry code */
    lm32dst = pp;
    /* stage marker 'Z': com1 0x3F8 */
    put8(&pp, 0xBA); put8(&pp, 0xF8); put8(&pp, 0x03); put8(&pp, 0); put8(&pp, 0);
    put8(&pp, 0xB0); put8(&pp, 0x5A);
    put8(&pp, 0xEE);
    put8(&pp, 0xFA);                                    /* cli */
    put8(&pp, 0x0F); put8(&pp, 0x01); put8(&pp, 0x15);  /* lgdt [disp32] = the gdptr */
    { uint8_t b[4]; wb32(b, (uint32_t)(ELF_BASE_ADDR + GDPTR_OFF)); memcpy(pp, b, 4); pp += 4; }
    /* Reload data segments to 0x20 (0x10 may now be code in OUR gdt) and give
     * a stack before SS swap. */
    put8(&pp, 0xBC);                                    /* mov esp, 0x90000 */
    { uint8_t b[4]; wb32(b, STACK_TOP); memcpy(pp, b, 4); pp += 4; }
    put8(&pp, 0x66); put8(&pp, 0xB8); put8(&pp, 0x20); put8(&pp, 0x00);  /* mov ax, 0x20 (66 = 16-bit) */
    put8(&pp, 0x8E); put8(&pp, 0xD8);                   /* mov ds, ax */
    put8(&pp, 0x8E); put8(&pp, 0xD0);                   /* mov ss, ax */
    put8(&pp, 0x8E); put8(&pp, 0xC0);                   /* mov es, ax */
    put8(&pp, 0x0F); put8(&pp, 0x20); put8(&pp, 0xE0);  /* mov eax,cr4 */
    put8(&pp, 0x83); put8(&pp, 0xC8); put8(&pp, 0x20);  /* or eax,0x20 (PAE) */
    put8(&pp, 0x0F); put8(&pp, 0x22); put8(&pp, 0xE0);  /* mov cr4,eax */
    put8(&pp, 0xB8);                                    /* mov eax, imm32 */
    { uint8_t b[4]; wb32(b, ELF_BASE_ADDR + PML4_VA_OFF); memcpy(pp, b, 4); pp += 4; }
    put8(&pp, 0x0F); put8(&pp, 0x22); put8(&pp, 0xD8);  /* mov cr3,eax */
    put8(&pp, 0xB9);                                    /* mov ecx, 0xC0000080 */
    { uint8_t b[4]; wb32(b, 0xC0000080); memcpy(pp, b, 4); pp += 4; }
    put8(&pp, 0x0F); put8(&pp, 0x32);                   /* rdmsr */
    put8(&pp, 0x0D);                                    /* or eax, 0x100 (LME) */
    { uint8_t b[4]; wb32(b, 0x100); memcpy(pp, b, 4); pp += 4; }
    put8(&pp, 0x0F); put8(&pp, 0x30);                   /* wrmsr */
    put8(&pp, 0x0F); put8(&pp, 0x20); put8(&pp, 0xC0);  /* mov eax,cr0 */
    put8(&pp, 0x0D);                                    /* or eax, 0x80000000 (PG) */
    { uint8_t b[4]; wb32(b, 0x80000000); memcpy(pp, b, 4); pp += 4; }
    put8(&pp, 0x0F); put8(&pp, 0x22); put8(&pp, 0xC0);  /* mov cr0,eax */
    put8(&pp, 0xEA);                                    /* far jmp 0x18:lm64 */
    { uint8_t b[4]; wb32(b, ELF_BASE_ADDR + LM64_VA_OFF); memcpy(pp, b, 4); pp += 4; }
    put16(&pp, 0x18);
    passert("prologue size > 32-bit code", 1); /* mark use of lm32dst */
    (void)lm32dst;

    /* user code starts right after the 32-bit prologue, BUT the page tables
     * need 4K alignment, so we place them at fixed offsets and let the PML4
     * at VA-offset 0x1000 be aligned.  We build the 64-bit block, then place
     * the tables, then the 64-bit block. */

    /* ---- assemble 64-bit block (len known) into a temp buffer ---- */
    char lm64[320];
    uint8_t *q = (uint8_t *)lm64;
    uint8_t *ret_q;
    put8(&q, 0x48); put8(&q, 0xBC);                /* mov rsp, imm64 */
    { uint8_t b[8]; wb64(b, STACK_TOP); memcpy(q, b, 8); q += 8; }
    /* 'Y' marker */
    put8(&q, 0xBA); put8(&q, 0xF8); put8(&q, 0x03); put8(&q, 0); put8(&q, 0);
    put8(&q, 0xB0); put8(&q, 0x59);
    put8(&q, 0xEE);
    ret_q = q;
    put8(&q, 0xE8); put8(&q, 0); put8(&q, 0); put8(&q, 0); put8(&q, 0); /* call rel32 (E8 + 4 rel bytes = 5) */
    put8(&q, 0x89); put8(&q, 0xC3);                /* mov ebx, eax (result) */
    /* print ebx as 8 hex digits to COM1 (0x3F8), unrolled so no loop/rel/reg
     * tricks can mis-encode: compiled result -> readable text ("0000002a" 42,
     * "00000143" 323). */
    for (int shift = 28; shift >= 0; shift -= 4) {
        put8(&q, 0x89); put8(&q, 0xD8);              /* mov eax, ebx */
        put8(&q, 0xC1); put8(&q, 0xE8); put8(&q, (uint8_t)shift); /* shr eax, <shift> */
        put8(&q, 0x24); put8(&q, 0x0F);              /* and al, 0x0F */
        put8(&q, 0x3C); put8(&q, 0x0A);              /* cmp al, 10 */
        put8(&q, 0x72); put8(&q, 0x02);              /* jb +2 */
        put8(&q, 0x04); put8(&q, 0x07);              /* add al, 7 (a-f) */
        put8(&q, 0x04); put8(&q, 0x30);              /* add al, '0' */
        put8(&q, 0xBA); put8(&q, 0xF8); put8(&q, 0x03); put8(&q, 0); put8(&q, 0); /* mov dx,0x3F8 */
        put8(&q, 0xEE);                              /* out dx, al */
    }
    put8(&q, 0xFA); put8(&q, 0xF4); put8(&q, 0xEB); put8(&q, 0xFE); /* cli;hlt;jmp -2 */
    size_t lm64_len = (size_t)(q - (uint8_t *)lm64);

    /* patch the call to main: target = code VA = ELF_BASE + code_start.
     * code_start(file) = LM64_VA_OFF + lm64_len; main called from ret_q. */
    {
        uint64_t code_va = ELF_BASE_ADDR + (uint64_t)LM64_VA_OFF + (uint64_t)lm64_len;
        int32_t rel = (int32_t)(code_va - ((uint64_t)ELF_BASE_ADDR + (uint64_t)LM64_VA_OFF + (uint64_t)(ret_q - (uint8_t *)lm64) + 5));
        uint8_t b[4]; wb32(b, (uint32_t)rel); memcpy(ret_q + 1, b, 4);  /* write the rel32 after the E8 opcode */
    }

    /* Build the page tables in a scratch (PML4 aligned at +0x1000). */
    uint8_t *tbl = calloc(LM64_VA_OFF, 1);          /* 0x4000 bytes covers 0x1000..0x4000 */
    uint8_t *pml4 = tbl + 0x1000;
    uint8_t *pdpt = tbl + 0x2000;
    uint8_t *pd   = tbl + 0x3000;
    wb64(pml4 + 0 * 8, (uint64_t)(ELF_BASE_ADDR + PDPT_VA_OFF) | 3u); /* PML4[0] -> PDPT, RW|P */
    wb64(pdpt + 0 * 8, (uint64_t)(ELF_BASE_ADDR + PD_VA_OFF)   | 3u); /* PDPT[0] -> PD  */
    for (int i = 0; i < 8; i++)                                   /* identity 0..16MB huge */
        wb64(pd + i * 8, (uint64_t)(i * 0x200000u) | 0x83u);
    /* fix up the 32-bit cr3 mov to point at pml4 VA (already emitted correctly). */

    /* Layout & write: [elf hdr 64][phdr 56][prologue: mb hdr + gdt + lm32]
     * then pad to 0x1000, then [PML4][PDPT][PD] (0x1000..0x4000), then
     * [lm64 block], then [user code][data]. */
    size_t prologue_file = LM64_VA_OFF + lm64_len;   /* file offset where user code begins */
    size_t code_start = prologue_file;               /* file offset of user code */
    size_t data_start = code_start + code_size;
    /* GRUB 2.12's multiboot ELF loader rejects e_shoff=0; append a minimal but
     * valid section-header table: [shdr0 SHT_NULL][shdr1 .shstrtab][strtab]. */
    const unsigned shstrlen = 10;                    /* ".shstrtab\0" */
    size_t sht_off = data_start + data_size;
    size_t total = sht_off + 2 * 64 + shstrlen;

    FILE *f = fopen(filename, "wb");
    if (!f) { free(pro); free(tbl); free(code_buf); return -1; }

    uint8_t ehdr[ELF_HDR_SIZE];
    build_elf_header(ehdr, ELF_BASE_ADDR + LM32_OFF, total);
    wb64(ehdr + 40, (uint64_t)sht_off);              /* e_shoff */
    wb16(ehdr + 58, 64);                             /* e_shentsize */
    wb16(ehdr + 60, 2);                              /* e_shnum */
    wb16(ehdr + 62, 1);                              /* e_shstrndx */
    fwrite(ehdr, 1, ELF_HDR_SIZE, f);
    uint8_t phdr[PHDR_SIZE];
    build_phdr(phdr, total);
    fwrite(phdr, 1, PHDR_SIZE, f);

    /* prologue bytes (MB hdr + gdt + gdptr + lm32), written at 0x78 */
    size_t pro_len = (size_t)(pp - (uint8_t *)pro);
    fwrite(pro, 1, pro_len, f);

    /* Explicit layout so every VA below matches the physical file layout:
     *   [0x1000..0x4000] page tables (PML4/PDPT/PD -- cr3 = 0x401000)
     *   [0x4000      ] lm64 block            (far jump target = 0x404000)
     *   [0x4000+len  ] user code + data
     * Fill gaps with zeros via fseek/fwrite. */
    if (fseek(f, 0x1000, SEEK_SET)) goto we;
    fwrite(tbl + 0x1000, 1, 0x3000, f);        /* exactly 0x1000..0x4000 */
    if (fseek(f, LM64_VA_OFF, SEEK_SET)) goto we;
    fwrite(lm64, 1, lm64_len, f);              /* at 0x4000 */
    if (fseek(f, code_start, SEEK_SET)) goto we;
    fwrite(code_buf, 1, code_size, f);
    if (data_size) fwrite(data, 1, data_size, f);
    if (fseek(f, sht_off, SEEK_SET)) goto we;
    /* section-header table (at sht_off): [SHT_NULL][.shstrtab] + bytes */
    {
        uint8_t sh[64] = {0};
        fwrite(sh, 1, 64, f);                          /* shdr0: SHT_NULL */
        memset(sh, 0, sizeof(sh));
        wb32(sh + 0, 3);                               /* sh_type = SHT_STRTAB */
        wb64(sh + 16, sht_off + 128);                  /* sh_offset */
        wb64(sh + 32, shstrlen);                       /* sh_size */
        wb32(sh + 40, 1);                              /* sh_link */
        wb32(sh + 44, 0);                              /* sh_info */
        wb64(sh + 48, 1);                              /* sh_addralign */
        fwrite(sh, 1, 64, f);                          /* shdr1: .shstrtab */
        fwrite(".shstrtab", 1, shstrlen, f);
    }
    fclose(f);

    free(pro); free(tbl); free(code_buf);
    (void)gdtp; (void)gdptrdst;
    return 0;
we:
    free(pro); free(tbl); free(code_buf);
    fclose(f);
    return -1;
}
