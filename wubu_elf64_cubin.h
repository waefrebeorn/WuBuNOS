/*
 * wubu_elf64_cubin.h -- ELF64 container writer API for ptxas-free cubin
 * emission.
 *
 * A CUDA cubin is an ELF64 object (e_machine = EM_NVIDIA 0x357, or a
 * generic LLVM-arch machine). Since cubin == ELF64 container, this writer
 * produces a valid relocatable ELF64 object holding the backend's code
 * bytes in a single .text section — no ptxas, no libelf, no binutils.
 *
 * The caller supplies the already-emitted code bytes (e.g. the bytes PTX
 * assembler would have produced, or a raw machine-image blob) and this
 * wraps them in a loadable ELF64 container.
 *
 * C11, self-contained.
 */
#ifndef WUBU_ELF64_CUBIN_H
#define WUBU_ELF64_CUBIN_H

#include <stdint.h>
#include <stddef.h>

/* ELF64 section-flag / type constants used by the builder. */
#define SHF_WRITE        0x1u
#define SHF_ALLOC         0x2u
#define SHF_EXECINSTR     0x4u
#define SHF_MASKPROC     0xf0000000u

#define SHT_NULL          0u
#define SHT_PROGBITS      1u
#define SHT_STRTAB        3u

#define PT_LOAD            1u
#define PF_X               0x1u
#define PF_W               0x2u
#define PF_R               0x4u

#define ET_REL             1u   /* relocatable */
#define EM_NVIDIA          0x357u   /* CUDA / cubin */

/* Build a cubin-style ELF64 object around `code`.
 *   code      -- raw bytes for .text (may be NULL/0 for an empty skeleton)
 *   code_size -- length of code in bytes
 *   out_size  -- receives total file size
 *   machine   -- e_machine value (EM_NVIDIA for a cubin)
 *   entry_off -- entry offset (informational; stub returns 0 in e_entry)
 * Returns a malloc'd buffer the caller frees with free(). */
uint8_t *wubu_elf64_cubin_build(const uint8_t *code, size_t code_size,
                                size_t *out_size, int machine,
                                uint64_t entry_offset);

/* Validate that img looks like a well-formed ELF64 object of ours. */
int wubu_elf64_cubin_validate(const uint8_t *img, size_t size);

#endif /* WUBU_ELF64_CUBIN_H */
