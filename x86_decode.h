/* x86_decode.h -- minimal x86-64 instruction walker for JIT verification.
 * Decodes enough to report prefixes, opcode, ModRM and SIB, which is what the
 * ISA goals need; it is not a general disassembler. */
#ifndef WUBU_X86_DECODE_H
#define WUBU_X86_DECODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t   off;        /* offset of this instruction            */
    size_t   len;        /* bytes consumed                        */
    uint8_t  rex;        /* 0 if absent, else 0x40..0x4F          */
    int      rex_w, rex_r, rex_x, rex_b;
    uint8_t  opcode;     /* primary opcode byte                   */
    int      has_modrm;
    uint8_t  modrm;
    int      modrm_reg, modrm_rm, modrm_mod;
    int      has_sib;
    uint8_t  sib;
    int      imm_bytes;
} x86_insn_t;

/* Decode one instruction at code+off. Returns bytes consumed, 0 at the end of
 * the buffer, or -1 on an encoding this walker does not handle. */
int x86_decode_one(const uint8_t *code, size_t n, size_t off, x86_insn_t *out);

/* Walk the whole buffer, calling the callback for each instruction. Returns the
 * number of instructions decoded, or -1 on the first undecodable byte. */
typedef void (*x86_insn_cb)(const x86_insn_t *ins, void *user);
int x86_decode_all(const uint8_t *code, size_t n, x86_insn_cb cb, void *user);

#endif
