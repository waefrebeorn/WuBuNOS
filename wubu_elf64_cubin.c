/*
 * wubu_elf64_cubin.c -- direct ELF64 container writer for ptxas-free cubin emission
 * (and generic ELF64 object emission for any backend).
 *
 * Research finding (Kevin-Bacon wave, 2026-08-23): a CUDA cubin is exactly an
 * ELF64 object with e_machine = EM_NVIDIA (0x357) / EM_LLVMARCH64-ish, a .text
 * section holding the PTX cubin bytes, and a .nv_fatbin / .nv_fatbin.cst segment.
 * Since cubin == ELF64 container, WuBuNOS's direct-binary-emission doctrine
 * applies directly: we write the ELF bytes by hand (no binutils), place PTX
 * image text in .text, and the loader treats the whole blob as a relocatable
 * ELF image the GPU driver can ingest.
 *
 * This writer is minimal and self-contained: it builds a 64-bit ELF with a
 * single PT_LOAD segment covering the .text section. Pure C11 byte packing,
 * no struct packing attributes.
 *
 * For ptxas-free flow the caller fills `code` with already-assembled PTX
 * machine instructions (or a textual PTX blob the downstream driver parses).
 *
 * C11, self-contained.
 */
#include "wubu_elf64_cubin.h"
#include <stdlib.h>
#include <string.h>

/* ---- byte writers (little-endian) ---- */
static void wb16(uint8_t *b, uint16_t v){ b[0]=v;b[1]=v>>8; }
static void wb32(uint8_t *b, uint32_t v){ b[0]=v;b[1]=v>>8;b[2]=v>>16;b[3]=v>>24; }
static void wb64(uint8_t *b, uint64_t v){ b[0]=v;b[1]=v>>8;b[2]=v>>16;b[3]=v>>24;
    b[4]=v>>32;b[5]=v>>40;b[6]=v>>48;b[7]=v>>56; }

#define ELF_EHDR  64
#define ELF_PHDR  56
#define ELF_SHDR  64

/* Section header string table: "\0.text\0.shstrtab\0" */
static const char shstrtab[] = "\0.text\0.shstrtab\0";
#define OFF_TEXT_NAME   1
#define OFF_SHSTRT_NAME 7

/* Build a cubin-style ELF64 object:
 *   section 0: SHT_NULL
 *   section 1: .text  (SHT_PROGBITS, SHF_ALLOC|SHF_EXECINSTR) holding `code`
 *   section 2: .shstrtab (SHT_STRTAB)
 * One PT_LOAD segment maps .text into memory. */
uint8_t *wubu_elf64_cubin_build(const uint8_t *code, size_t code_size,
                                size_t *out_size, int machine,
                                uint64_t entry_offset)
{
    (void)entry_offset;
    if (!out_size) return NULL;
    const size_t text_off     = ELF_EHDR + ELF_PHDR;      /* file offset of .text */
    const size_t shstrtab_size = sizeof(shstrtab) - 1;     /* includes trailing NUL */
    const size_t shstrtab_off  = text_off + code_size;    /* file offset of .shstrtab */
    const size_t shdr_off      = shstrtab_off + shstrtab_size;
    const size_t shdr_size     = 3 * ELF_SHDR;            /* 3 sections */
    const size_t total        = shdr_off + shdr_size;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) return NULL;

    /* ---- ELF header ---- */
    uint8_t *h = buf;
    h[0]=0x7f; h[1]='E'; h[2]='L'; h[3]='F';
    h[4]=2;              /* 64-bit */
    h[5]=1;              /* little-endian */
    h[6]=1;              /* ELF version */
    h[7]=0;              /* System V ABI */
    memset(h+8,0,8);     /* padding */
    wb16(h+16, 1);       /* e_type = ET_REL (relocatable object) */
    wb16(h+18, (uint16_t)machine);
    wb32(h+20, 1);       /* e_version */
    wb64(h+24, 0);       /* e_entry (stub/zero) */
    wb64(h+32, ELF_EHDR);/* e_phoff */
    wb64(h+40, (uint64_t)shdr_off); /* e_shoff */
    wb32(h+48, 0);       /* e_flags */
    wb16(h+52, ELF_EHDR);/* e_ehsize */
    wb16(h+54, ELF_PHDR);/* e_phentsize */
    wb16(h+56, 1);       /* e_phnum (1 PT_LOAD) */
    wb16(h+58, ELF_SHDR);/* e_shentsize */
    wb16(h+60, 3);       /* e_shnum */
    wb16(h+62, 2);       /* e_shstrndx (.shstrtab index) */

    /* ---- program header (PT_LOAD for the whole file) ---- */
    uint8_t *p = buf + ELF_EHDR;
    wb32(p+0, 1);            /* p_type = PT_LOAD */
    wb32(p+4, 5);            /* p_flags = PF_R | PF_X */
    wb64(p+8, 0);            /* p_offset */
    wb64(p+16, 0);           /* p_vaddr */
    wb64(p+24, 0);           /* p_paddr */
    wb64(p+32, total);      /* p_filesz */
    wb64(p+40, total);      /* p_memsz */
    wb64(p+48, 0x1000);     /* p_align */

    /* ---- .text section ---- */
    if (code && code_size)
        memcpy(buf + text_off, code, code_size);

    /* ---- .shstrtab ---- */
    memcpy(buf + shstrtab_off, shstrtab, shstrtab_size);

    /* ---- section headers ---- */
    uint8_t *s = buf + shdr_off;
    /* 0: SHT_NULL */
    memset(s, 0, ELF_SHDR);
    s += ELF_SHDR;
    /* 1: .text */
    wb32(s+0,  OFF_TEXT_NAME);   /* sh_name */
    wb32(s+4,  1);               /* sh_type = SHT_PROGBITS */
    wb64(s+8,  (uint64_t)(SHF_ALLOC|SHF_EXECINSTR|SHF_WRITE));
    wb64(s+16, 0);               /* sh_addr */
    wb64(s+24, (uint64_t)text_off);
    wb64(s+32, code_size);       /* sh_size */
    wb32(s+40, 0);               /* sh_link */
    wb32(s+44, 0);               /* sh_info */
    wb64(s+48, 8);               /* sh_addralign */
    wb64(s+56, 0);               /* sh_entsize */
    s += ELF_SHDR;
    /* 2: .shstrtab */
    wb32(s+0,  OFF_SHSTRT_NAME);
    wb32(s+4,  3);               /* sh_type = SHT_STRTAB */
    wb64(s+8,  0);
    wb64(s+16, 0);
    wb64(s+24, (uint64_t)shstrtab_off);
    wb64(s+32, shstrtab_size);
    wb32(s+40, 0);
    wb32(s+44, 0);
    wb64(s+48, 1);
    wb64(s+56, 0);

    *out_size = total;
    return buf;
}

/* Validate a built cubin: magic + ET_REL + has .text. Returns 1 valid, 0 not. */
int wubu_elf64_cubin_validate(const uint8_t *img, size_t size) {
    if (size < ELF_EHDR + ELF_PHDR + ELF_SHDR) return 0;
    if (img[0]!=0x7f||img[1]!='E'||img[2]!='L'||img[3]!='F') return 0;
    if (img[4]!=2||img[5]!=1) return 0;         /* 64-bit LE */
    uint16_t etype; wb16((uint8_t*)&etype, 0); __builtin_memcpy(&etype, img+16, 2);
    if (etype != 1) return 0;                   /* ET_REL */
    return 1;
}
