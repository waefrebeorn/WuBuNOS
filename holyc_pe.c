/*
 * holyc_pe.c  --  PE32+ executable emitter for WuBuNOS HolyC compiler.
 *
 * Research: TinyPE-on-Win10 (ayaka14732), "PE Format" (Microsoft),
 * "Anatomy of the Portable Executable" (blog.deephacking.tech).
 *
 * Emits a minimal PE32+ (64-bit) executable with:
 *   - MZ header (stub)
 *   - PE signature
 *   - COFF header (machine = 0x8664)
 *   - Optional header (PE32+ = magic 0x20B)
 *   - Single .text section (code)
 *
 * Pure C11, no compiler extensions. Byte-level serialization.
 */

#include "holyc_codegen.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* PE constants */
#define IMAGE_NT_SIGNATURE          0x00004550
#define IMAGE_FILE_MACHINE_AMD64    0x8664
#define IMAGE_FILE_EXECUTABLE_IMAGE 0x0002
#define IMAGE_FILE_LARGE_ADDRESS_AWARE 0x0020
#define IMAGE_NT_OPTIONAL_HDR64_MAGIC 0x20B
#define IMAGE_SUBSYSTEM_WINDOWS_CUI  3
#define IMAGE_SCN_CNT_CODE           0x00000020
#define IMAGE_SCN_MEM_EXECUTE        0x20000000
#define IMAGE_SCN_MEM_READ           0x40000000

#define PE_BASE_ADDR    0x140000000
#define PE_SECTION_ALIGN 0x1000
#define PE_FILE_ALIGN    0x200

/* Portable write helpers */
static void wb16(uint8_t *b, uint16_t v) { b[0]=v&0xFF; b[1]=(v>>8)&0xFF; }
static void wb32(uint8_t *b, uint32_t v) { b[0]=v&0xFF; b[1]=(v>>8)&0xFF; b[2]=(v>>16)&0xFF; b[3]=(v>>24)&0xFF; }
static void wb64(uint8_t *b, uint64_t v) { for(int i=0;i<8;i++) b[i]=(v>>(i*8))&0xFF; }

static uint32_t align_up(uint32_t val, uint32_t alignment) {
    if (val % alignment == 0) return val;
    return val + (alignment - (val % alignment));
}

#define MZ_HDR_SIZE 128
#define COFF_HDR_SIZE 20
#define OPT_HDR_SIZE 240
#define SECT_HDR_SIZE 40

int hc_write_pe(const char *filename,
                const uint8_t *code, size_t code_size,
                const uint8_t *data, size_t data_size,
                const size_t *patch_offsets,
                const size_t *patch_globals,
                size_t n_patches)
{
    (void)data; (void)data_size; /* unused for now */
    if (!filename || !code || code_size == 0) return -1;

    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    /* Mutable copy of code for patching */
    uint8_t *code_buf = (uint8_t *)malloc(code_size);
    if (!code_buf) { fclose(f); return -1; }
    memcpy(code_buf, code, code_size);

    /* Patch globals */
    for (size_t i = 0; i < n_patches; i++) {
        size_t pp = patch_offsets[i];
        size_t go = patch_globals[i];
        if (pp + 4 <= code_size) {
            int32_t disp32 = (int32_t)(code_size + go - pp - 4);
            memcpy(code_buf + pp, &disp32, 4);
        }
    }

    /* Layout */
    size_t pe_sig_offset = 0x80;
    size_t coff_offset = pe_sig_offset + 4;
    size_t opt_offset = coff_offset + COFF_HDR_SIZE;
    size_t sect_offset = opt_offset + OPT_HDR_SIZE;
    size_t headers_size = sect_offset + SECT_HDR_SIZE;

    size_t text_raw_offset = align_up((uint32_t)headers_size, PE_FILE_ALIGN);
    size_t text_virtual_offset = align_up((uint32_t)headers_size, PE_SECTION_ALIGN);
    size_t text_raw_size = align_up((uint32_t)code_size, PE_FILE_ALIGN);
    uint32_t entry_rva = (uint32_t)text_virtual_offset;

    uint8_t zero = 0;

    /* ---- MZ header (128 bytes) ---- */
    uint8_t mz[MZ_HDR_SIZE];
    memset(mz, 0, MZ_HDR_SIZE);
    wb16(mz + 0, 0x5A4D);   /* e_magic */
    wb16(mz + 2, 0x90);     /* e_cblp */
    wb16(mz + 4, 3);        /* e_cp */
    wb16(mz + 8, 4);        /* e_cparhdr */
    wb16(mz + 10, 0x10);    /* e_minalloc */
    wb16(mz + 12, 0xFFFF);  /* e_maxalloc */
    wb16(mz + 14, 0xB8);    /* e_sp */
    wb32(mz + 44, 0x40);    /* e_lfarlc */
    wb32(mz + 60, pe_sig_offset); /* e_lfanew */
    fwrite(mz, 1, MZ_HDR_SIZE, f);

    /* ---- PE signature (4 bytes) ---- */
    uint8_t pe_sig[4] = {0x50, 0x45, 0x00, 0x00};
    fwrite(pe_sig, 1, 4, f);

    /* ---- COFF header (20 bytes) ---- */
    uint8_t coff[COFF_HDR_SIZE];
    memset(coff, 0, COFF_HDR_SIZE);
    wb16(coff + 0, IMAGE_FILE_MACHINE_AMD64);
    wb16(coff + 2, 1);                  /* NumberOfSections */
    wb16(coff + 16, OPT_HDR_SIZE);      /* SizeOfOptionalHeader */
    wb16(coff + 18, IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_LARGE_ADDRESS_AWARE);
    fwrite(coff, 1, COFF_HDR_SIZE, f);

    /* ---- Optional header PE32+ (240 bytes) ---- */
    uint8_t opt[OPT_HDR_SIZE];
    memset(opt, 0, OPT_HDR_SIZE);
    wb16(opt + 0, IMAGE_NT_OPTIONAL_HDR64_MAGIC); /* Magic */
    opt[2] = 14;                     /* MajorLinkerVersion */
    wb32(opt + 16, (uint32_t)text_raw_size); /* SizeOfCode */
    wb32(opt + 24, entry_rva);       /* AddressOfEntryPoint */
    wb32(opt + 28, entry_rva);       /* BaseOfCode */
    wb64(opt + 32, PE_BASE_ADDR);    /* ImageBase */
    wb32(opt + 40, PE_SECTION_ALIGN); /* SectionAlignment */
    wb32(opt + 44, PE_FILE_ALIGN);   /* FileAlignment */
    wb16(opt + 48, 6);               /* MajorOperatingSystemVersion */
    wb16(opt + 56, 6);               /* MajorSubsystemVersion */
    wb32(opt + 64, (uint32_t)(text_virtual_offset + text_raw_size + PE_SECTION_ALIGN)); /* SizeOfImage */
    wb32(opt + 68, (uint32_t)headers_size); /* SizeOfHeaders */
    wb16(opt + 76, IMAGE_SUBSYSTEM_WINDOWS_CUI); /* Subsystem */
    wb16(opt + 78, 0x8160);          /* DllCharacteristics */
    wb64(opt + 80, 0x100000);        /* SizeOfStackReserve */
    wb64(opt + 88, 0x1000);          /* SizeOfStackCommit */
    wb64(opt + 96, 0x100000);        /* SizeOfHeapReserve */
    wb64(opt + 104, 0x1000);         /* SizeOfHeapCommit */
    wb32(opt + 112, 16);             /* NumberOfRvaAndSizes */
    fwrite(opt, 1, OPT_HDR_SIZE, f);

    /* ---- Section header .text (40 bytes) ---- */
    uint8_t sect[SECT_HDR_SIZE];
    memset(sect, 0, SECT_HDR_SIZE);
    memcpy(sect + 0, ".text\0\0\0", 8); /* Name */
    wb32(sect + 8, (uint32_t)code_size); /* VirtualSize */
    wb32(sect + 12, entry_rva);       /* VirtualAddress */
    wb32(sect + 16, (uint32_t)text_raw_size); /* SizeOfRawData */
    wb32(sect + 20, (uint32_t)text_raw_offset); /* PointerToRawData */
    wb32(sect + 36, IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ);
    fwrite(sect, 1, SECT_HDR_SIZE, f);

    /* ---- Pad to text_raw_offset ---- */
    size_t pad_needed = text_raw_offset - headers_size;
    for (size_t i = 0; i < pad_needed; i++)
        fwrite(&zero, 1, 1, f);

    /* ---- Write code ---- */
    fwrite(code_buf, 1, code_size, f);
    free(code_buf);

    /* Pad to text_raw_size */
    for (size_t i = code_size; i < text_raw_size; i++)
        fwrite(&zero, 1, 1, f);

    fclose(f);
    return 0;
}
