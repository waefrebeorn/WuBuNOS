/* x86_decode.c -- see x86_decode.h. */
#include "x86_decode.h"

#include <string.h>

static int modrm_len(const x86_insn_t *in, const uint8_t *p, size_t avail)
{
    if (!in->has_modrm) return 0;
    int mod = in->modrm_mod, rm = in->modrm_rm;
    int len = 0;
    if (mod != 3 && rm == 4) {            /* SIB */
        if (avail < 1) return -1;
        len += 1;
        int base = p[0] & 7;
        if (mod == 0 && base == 5) len += 4;   /* disp32 */
    }
    if (mod == 0 && rm == 5) len += 4;    /* RIP-relative disp32 */
    if (mod == 1) len += 1;
    if (mod == 2) len += 4;
    return len;
}

int x86_decode_one(const uint8_t *code, size_t n, size_t off, x86_insn_t *out)
{
    if (!code || !out || off >= n) return 0;
    memset(out, 0, sizeof *out);
    out->off = off;

    size_t i = off;
    /* Legacy prefixes we care about: operand-size (0x66). */
    while (i < n && (code[i] == 0x66 || code[i] == 0xF2 || code[i] == 0xF3 ||
                     code[i] == 0x2E || code[i] == 0x3E || code[i] == 0x26 ||
                     code[i] == 0x36 || code[i] == 0x64 || code[i] == 0x65))
        i++;

    if (i < n && code[i] >= 0x40 && code[i] <= 0x4F) {
        out->rex = code[i];
        out->rex_w = (code[i] >> 3) & 1;
        out->rex_r = (code[i] >> 2) & 1;
        out->rex_x = (code[i] >> 1) & 1;
        out->rex_b = code[i] & 1;
        i++;
    }
    if (i >= n) return -1;

    out->opcode = code[i++];

    uint8_t op = out->opcode;
    /* Opcodes carrying ModRM in the subset the JIT emits. */
    if ((op >= 0x00 && op <= 0x3F) || (op >= 0x80 && op <= 0x8F) ||
        (op >= 0xC0 && op <= 0xC1) || (op >= 0xC6 && op <= 0xC7) ||
        (op >= 0xD0 && op <= 0xD3) || (op >= 0xF6 && op <= 0xF7) ||
        (op >= 0x10 && op <= 0x17 && (out->rex || 1)) ||
        op == 0x6E || op == 0x7E || op == 0x2A || op == 0x2C ||
        op == 0x5A || op == 0x57 || op == 0x54) {
        out->has_modrm = 1;
    }
    /* Two-byte 0F xx: ModRM follows. */
    if (op == 0x0F) {
        if (i >= n) return -1;
        out->opcode = code[i++];
        out->has_modrm = 1;
    }

    if (out->has_modrm) {
        if (i >= n) return -1;
        out->modrm = code[i++];
        out->modrm_mod = (out->modrm >> 6) & 3;
        out->modrm_reg = ((out->modrm >> 3) & 7) | (out->rex_r << 3);
        out->modrm_rm = (out->modrm & 7) | (out->rex_b << 3);
    }

    size_t avail = n - i;
    int ml = modrm_len(out, code + i, avail);
    if (ml < 0) return -1;
    if ((size_t)ml > avail) return -1;
    if (out->has_modrm && out->modrm_mod != 3 && out->modrm_rm == 4 && ml >= 1) {
        out->has_sib = 1;
        out->sib = code[i];
    }
    i += (size_t)ml;

    /* Immediates. */
    int64_t imm = 0;
    if (op >= 0xB8 && op <= 0xBF) imm = out->rex_w ? 8 : 4;
    else if (op == 0xC7) imm = (out->modrm_mod == 3) ? 4 : 4;
    else if (op == 0xB0 || (op >= 0xB0 && op <= 0xB7)) imm = 1;
    else if (op >= 0x80 && op <= 0x83) imm = (op == 0x81) ? 4 : 1;
    else if (op >= 0x68 && op <= 0x69) imm = (op == 0x68) ? 4 : 4;
    else if (op == 0xC6) imm = 1;
    else if (op == 0xE9 || op == 0xE8) imm = 4;
    else if (op >= 0x70 && op <= 0x7F) imm = 1;
    else if (op == 0x0F) imm = 0;

    if (imm > 0) {
        if ((size_t)imm > n - i) return -1;
        out->imm_bytes = (int)imm;
        i += (size_t)imm;
    }

    out->len = i - off;
    return (int)out->len;
}

int x86_decode_all(const uint8_t *code, size_t n, x86_insn_cb cb, void *user)
{
    size_t off = 0;
    int count = 0;
    while (off < n) {
        x86_insn_t ins;
        int len = x86_decode_one(code, n, off, &ins);
        if (len <= 0) return -1;
        if (cb) cb(&ins, user);
        off += (size_t)len;
        count++;
    }
    return count;
}
