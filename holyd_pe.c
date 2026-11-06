/*
 * holyd_pe.c  --  PE32+ executable emitter for WuBuNOS HolyD compiler.
 *
 * Research: TinyPE-on-Win10 (ayaka14732), "PE Format" (Microsoft),
 * "Anatomy of the Portable Executable" (blog.deephacking.tech).
 *
 * Emits a minimal PE32+ (64-bit) executable with:
 *   - MZ header (stub)
 *   - PE signature
 *   - COFF header (machine = 0x8664)
 *   - Optional header (PE32+ = magic 0x20B)
 *   - .text section (code)
 *   - .data section (module globals, when data_size > 0)
 *
 * SELF-HOSTING rung: a holyd program that references GLOBALS must be emitted
 * with its data placed at a known RVA and its RIP-relative global references
 * fixed against that RVA (not the old "data right after code" convention).
 * The caller passes the global patch list (code offsets of rip-relative disp32
 * slots + the data-section offset each points at); the emitter computes each
 * disp32 against the real .data RVA. Because the load-bearing references are
 * RIP-relative, code<->data distance is invariant across load base — the image
 * is fully position-independent and needs NO .reloc directory (a DIR64 reloc
 * inside the kernel loader would wrongly patch the disp32 as an absolute u64).
 *
 * Pure C11, no compiler extensions. Byte-level serialization.
 */

#include "holyd_codegen.h"
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
#define IMAGE_SCN_CNT_INITIALIZED_DATA 0x00000040
#define IMAGE_SCN_MEM_EXECUTE        0x20000000
#define IMAGE_SCN_MEM_READ           0x40000000
#define IMAGE_SCN_MEM_WRITE          0x80000000

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

int hd_write_pe(const char *filename,
                const uint8_t *code, size_t code_size,
                const uint8_t *data, size_t data_size,
                const size_t *patch_offsets,
                const size_t *patch_globals,
                size_t n_patches)
{
    if (!filename || !code || code_size == 0) return -1;

    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    /* Layout: .text always @ RVA 0x1000; .data (if any) aligned after it. */
    size_t text_rva = 0x1000;
    size_t data_rva = text_rva + (size_t)align_up((uint32_t)code_size, PE_SECTION_ALIGN);

    size_t pe_sig_offset = 0x80;
    size_t coff_offset = pe_sig_offset + 4;
    size_t opt_offset = coff_offset + COFF_HDR_SIZE;
    size_t sect_offset = opt_offset + OPT_HDR_SIZE;
    size_t n_sects = (data_size > 0 ? 1 : 0) + 1;
    size_t headers_size = align_up((uint32_t)(sect_offset + n_sects * SECT_HDR_SIZE), PE_FILE_ALIGN);

    size_t text_raw_off = headers_size;
    size_t text_raw_size = align_up((uint32_t)code_size, PE_FILE_ALIGN);
    size_t data_raw_off = text_raw_off + text_raw_size;
    size_t data_raw_size = data_size > 0 ? align_up((uint32_t)data_size, PE_FILE_ALIGN) : 0;

    size_t last_rva = (data_size > 0 ? data_rva : text_rva);
    size_t size_of_image = last_rva + 0x1000;

    uint32_t entry_rva = (uint32_t)text_rva;
    uint8_t zero = 0;

    /* ---- MZ header ---- */
    uint8_t mz[MZ_HDR_SIZE];
    memset(mz, 0, MZ_HDR_SIZE);
    wb16(mz + 0, 0x5A4D);
    wb16(mz + 2, 0x90);
    wb16(mz + 4, 3);
    wb16(mz + 8, 4);
    wb16(mz + 10, 0x10);
    wb16(mz + 12, 0xFFFF);
    wb16(mz + 14, 0xB8);
    wb32(mz + 44, 0x40);
    wb32(mz + 60, (uint32_t)pe_sig_offset);
    fwrite(mz, 1, MZ_HDR_SIZE, f);

    /* ---- PE signature ---- */
    uint8_t pe_sig[4] = {0x50, 0x45, 0x00, 0x00};
    fwrite(pe_sig, 1, 4, f);

    /* ---- COFF header ---- */
    uint8_t coff[COFF_HDR_SIZE];
    memset(coff, 0, COFF_HDR_SIZE);
    wb16(coff + 0, IMAGE_FILE_MACHINE_AMD64);
    wb16(coff + 2, (uint16_t)n_sects);
    wb16(coff + 16, OPT_HDR_SIZE);
    wb16(coff + 18, IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_LARGE_ADDRESS_AWARE);
    fwrite(coff, 1, COFF_HDR_SIZE, f);

    /* ---- Optional header PE32+ ---- */
    uint8_t opt[OPT_HDR_SIZE];
    memset(opt, 0, OPT_HDR_SIZE);
    wb16(opt + 0, IMAGE_NT_OPTIONAL_HDR64_MAGIC);
    opt[2] = 14;
    wb32(opt + 16, (uint32_t)text_raw_size);
    wb32(opt + 24, entry_rva);
    wb32(opt + 28, (uint32_t)text_rva);
    wb64(opt + 32, PE_BASE_ADDR);
    wb32(opt + 40, PE_SECTION_ALIGN);
    wb32(opt + 44, PE_FILE_ALIGN);
    wb16(opt + 48, 6);
    wb16(opt + 56, 6);
    wb32(opt + 64, (uint32_t)size_of_image);
    wb32(opt + 68, (uint32_t)headers_size);
    wb16(opt + 76, IMAGE_SUBSYSTEM_WINDOWS_CUI);
    wb16(opt + 78, 0x8160);
    wb64(opt + 80, 0x100000);
    wb64(opt + 88, 0x1000);
    wb64(opt + 96, 0x100000);
    wb64(opt + 104, 0x1000);
    wb32(opt + 112, 16);
    fwrite(opt, 1, OPT_HDR_SIZE, f);

    /* ---- Section headers ---- */
    uint8_t sect[SECT_HDR_SIZE];
    memset(sect, 0, SECT_HDR_SIZE);
    memcpy(sect, ".text\0\0\0", 8);
    wb32(sect + 8, (uint32_t)code_size);
    wb32(sect + 12, (uint32_t)text_rva);
    wb32(sect + 16, (uint32_t)text_raw_size);
    wb32(sect + 20, (uint32_t)text_raw_off);
    wb32(sect + 36, IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ);
    fwrite(sect, 1, SECT_HDR_SIZE, f);

    if (data_size > 0) {
        memset(sect, 0, SECT_HDR_SIZE);
        memcpy(sect, ".data\0\0\0", 8);
        wb32(sect + 8, (uint32_t)data_size);
        wb32(sect + 12, (uint32_t)data_rva);
        wb32(sect + 16, (uint32_t)data_raw_size);
        wb32(sect + 20, (uint32_t)data_raw_off);
        wb32(sect + 36, IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE);
        fwrite(sect, 1, SECT_HDR_SIZE, f);
    }

    /* ---- pad to text_raw_off ---- */
    long pos = ftell(f);
    if (pos < (long)headers_size) {
        for (long i = pos; i < (long)headers_size; i++) fwrite(&zero, 1, 1, f);
    }

    /* ---- write .text: fix globals against the .data RVA FIRST ---- */
    uint8_t *code_buf = (uint8_t *)malloc(code_size);
    if (!code_buf) { fclose(f); return -1; }
    memcpy(code_buf, code, code_size);
    for (size_t i = 0; i < n_patches; i++) {
        size_t pp = patch_offsets[i];
        size_t go = patch_globals[i];
        if (pp + 4 > code_size) continue;
        if (data_size == 0) continue;
        /* rip-relative disp32 at code+pp: disp32 = (data_rva + go) - (text_rva + pp + 4). */
        int32_t disp32 = (int32_t)((int64_t)(data_rva + go) - (int64_t)(text_rva + pp + 4));
        memcpy(code_buf + pp, &disp32, 4);
    }
    fwrite(code_buf, 1, code_size, f);
    free(code_buf);
    for (size_t i = code_size; i < text_raw_size; i++) fwrite(&zero, 1, 1, f);

    /* ---- write .data ---- */
    if (data_size > 0) {
        fwrite(data, 1, data_size, f);
        for (size_t i = data_size; i < data_raw_size; i++) fwrite(&zero, 1, 1, f);
    }

    fclose(f);
    return 0;
}

/* -- Program emitter (self-hosting rung 3) -------------------------------- */
/* Lays module code + function bodies contiguously as ONE .text section.
 * Entry = module code start. Fixes (a) module global disp32s, (b) each
 * function's global disp32s, (c) module > function call rel32s — all against
 * offsets WITHIN the single image + the .data RVA, so the result is fully
 * position-independent (base-invariant, no .reloc needed). */
static void pb16(uint8_t *b, uint16_t v){b[0]=v&255;b[1]=(v>>8)&255;}
static void pb32(uint8_t *b, uint32_t v){b[0]=v&255;b[1]=(v>>8)&255;b[2]=(v>>16)&255;b[3]=(v>>24)&255;}
static void pb64(uint8_t *b, uint64_t v){for(int i=0;i<8;i++)b[i]=(uint8_t)(v>>(i*8));}

int hd_write_pe_program(const char *filename,
                        const uint8_t *module_code, size_t module_size,
                        const size_t *module_global_offs, const size_t *module_global_goff,
                        size_t n_module_globals,
                        const size_t *call_offs, const size_t *call_fn, size_t n_calls,
                        const HDProgramFunc *funcs, size_t n_funcs,
                        const uint8_t *data, size_t data_size)
{
    if (!filename || !module_code || module_size == 0) return -1;

    /* per-function image offsets (relative to .text start) */
    size_t *f_off = n_funcs ? (size_t *)calloc(n_funcs, sizeof(size_t)) : NULL;
    if (n_funcs && !f_off) return -1;
    size_t text_size = module_size;
    for (size_t i = 0; i < n_funcs; i++) {
        text_size = align_up((uint32_t)text_size, 16);
        f_off[i] = text_size;
        text_size += funcs[i].code_size;
    }

    size_t text_rva = 0x1000;
    size_t data_rva = text_rva + (size_t)align_up((uint32_t)text_size, PE_SECTION_ALIGN);

    FILE *f = fopen(filename, "wb");
    if (!f) { free(f_off); return -1; }

    size_t pe_sig_offset = 0x80;
    size_t coff_offset = pe_sig_offset + 4;
    size_t opt_offset = coff_offset + COFF_HDR_SIZE;
    size_t sect_offset = opt_offset + OPT_HDR_SIZE;
    size_t n_sects = (data_size > 0 ? 1 : 0) + 1;
    size_t headers_size = align_up((uint32_t)(sect_offset + n_sects * SECT_HDR_SIZE), PE_FILE_ALIGN);

    size_t text_raw_off = headers_size;
    size_t text_raw_size = align_up((uint32_t)text_size, PE_FILE_ALIGN);
    size_t data_raw_off = text_raw_off + text_raw_size;
    size_t data_raw_size = data_size > 0 ? align_up((uint32_t)data_size, PE_FILE_ALIGN) : 0;
    size_t size_of_image = (data_size > 0 ? data_rva : text_rva) + 0x1000;

    uint8_t zero = 0;

    /* MZ */
    uint8_t mz[MZ_HDR_SIZE]; memset(mz, 0, MZ_HDR_SIZE);
    pb16(mz+0,0x5A4D); pb16(mz+2,0x90); pb16(mz+4,3); pb16(mz+8,4);
    pb16(mz+10,0x10); pb16(mz+12,0xFFFF); pb16(mz+14,0xB8); pb32(mz+44,0x40); pb32(mz+60,(uint32_t)pe_sig_offset);
    fwrite(mz,1,MZ_HDR_SIZE,f);
    uint8_t sig[4]={0x50,0x45,0,0}; fwrite(sig,1,4,f);
    /* COFF */
    uint8_t coff[COFF_HDR_SIZE]; memset(coff,0,COFF_HDR_SIZE);
    pb16(coff+0,0x8664); pb16(coff+2,(uint16_t)n_sects); pb16(coff+16,OPT_HDR_SIZE);
    pb16(coff+18,0x0020|0x0002);
    fwrite(coff,1,COFF_HDR_SIZE,f);
    /* Optional PE32+ */
    uint8_t opt[OPT_HDR_SIZE]; memset(opt,0,OPT_HDR_SIZE);
    pb16(opt+0,0x20B); opt[2]=14;
    pb32(opt+16,(uint32_t)text_raw_size);
    pb32(opt+24,(uint32_t)text_rva);          /* entry = module start */
    pb32(opt+28,(uint32_t)text_rva);          /* base of code */
    pb64(opt+32,PE_BASE_ADDR);
    pb32(opt+40,PE_SECTION_ALIGN); pb32(opt+44,PE_FILE_ALIGN);
    pb16(opt+48,6); pb16(opt+56,6);
    pb32(opt+64,(uint32_t)size_of_image); pb32(opt+68,(uint32_t)headers_size);
    pb16(opt+76,3); pb16(opt+78,0x8160);
    pb64(opt+80,0x100000); pb64(opt+88,0x1000); pb64(opt+96,0x100000); pb64(opt+104,0x1000);
    pb32(opt+112,16);
    fwrite(opt,1,OPT_HDR_SIZE,f);
    /* sections */
    uint8_t sect[SECT_HDR_SIZE];
    memset(sect,0,SECT_HDR_SIZE); memcpy(sect,".text\0\0\0",8);
    pb32(sect+8,(uint32_t)text_size); pb32(sect+12,(uint32_t)text_rva);
    pb32(sect+16,(uint32_t)text_raw_size); pb32(sect+20,(uint32_t)text_raw_off);
    pb32(sect+36,0x60000020);
    fwrite(sect,1,SECT_HDR_SIZE,f);
    if (data_size>0){
        memset(sect,0,SECT_HDR_SIZE); memcpy(sect,".data\0\0\0",8);
        pb32(sect+8,(uint32_t)data_size); pb32(sect+12,(uint32_t)data_rva);
        pb32(sect+16,(uint32_t)data_raw_size); pb32(sect+20,(uint32_t)data_raw_off);
        pb32(sect+36,0xC0000040);
        fwrite(sect,1,SECT_HDR_SIZE,f);
    }
    long pos=ftell(f);
    for (long i=pos;i<(long)headers_size;i++) fwrite(&zero,1,1,f);

    /* Build the combined .text buffer (module + funcs) with all fixes. */
    uint8_t *cb = (uint8_t *)calloc(1, text_size);
    if (!cb){ free(f_off); fclose(f); return -1; }
    memcpy(cb, module_code, module_size);
    for (size_t i=0;i<n_funcs;i++)
        memcpy(cb + f_off[i], funcs[i].code, funcs[i].code_size);

    /* fix module global disp32s against .data RVA (points within .text) */
    for (size_t i=0;i<n_module_globals;i++){
        size_t pp=module_global_offs[i], go=module_global_goff[i];
        if (pp+4>module_size || data_size==0) continue;
        int32_t d=(int32_t)((int64_t)(data_rva+go)-(int64_t)(text_rva+pp+4));
        memcpy(cb+pp,&d,4);
    }
    /* fix each function's global disp32s */
    for (size_t i=0;i<n_funcs;i++){
        for (size_t g=0;g<funcs[i].n_global_patches;g++){
            size_t pp=funcs[i].global_offs[g], go=funcs[i].global_goff[g];
            if (pp+4>funcs[i].code_size || data_size==0) continue;
            int32_t d=(int32_t)((int64_t)(data_rva+go)-(int64_t)(text_rva+f_off[i]+pp+4));
            memcpy(cb+f_off[i]+pp,&d,4);
        }
    }
    /* fix module > function call rel32s (call is inside module code). The
     * call_patch position is the DISP32 slot (opcode at pos-1), matching
     * hd_eval's convention: disp is stored AT pos, and the RIP after the
     * 5-byte call (E8+disp32) is pos+4. rel32 = target - rip_after. */
    for (size_t i = 0; i < n_calls; i++) {
        size_t cp = call_offs[i]; size_t fi = call_fn[i];
        if (fi >= n_funcs) continue;
        if (cp + 4 > module_size) continue;
        int32_t d = (int32_t)((int64_t)(text_rva + f_off[fi]) - (int64_t)(text_rva + cp + 4));
        memcpy(cb + cp, &d, 4);
    }

    fwrite(cb,1,text_size,f);
    for (size_t i=text_size;i<text_raw_size;i++) fwrite(&zero,1,1,f);
    if (data_size>0){ fwrite(data,1,data_size,f); for (size_t i=data_size;i<data_raw_size;i++) fwrite(&zero,1,1,f); }

    free(cb); free(f_off);
    fclose(f);
    return 0;
}
