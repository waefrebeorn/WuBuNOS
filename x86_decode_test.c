#include "x86_decode.h"
#include <stdio.h>

static int pass, fail;
#define T(n) do { printf("  TEST %-52s", n); } while (0)
#define OK() do { printf("✅\n"); pass++; } while (0)
#define NO(m) do { printf("❌ %s\n", m); fail++; } while (0)

int main(void)
{
    x86_insn_t in;

    T("movabs r11, imm64 -> REX.B, B8+3, 8-byte immediate");
    const uint8_t a[] = {0x49, 0xBB, 1,2,3,4,5,6,7,8};
    int l = x86_decode_one(a, sizeof a, 0, &in);
    printf("[len=%d op=%02x rm=%d imm=%d] ", l, in.opcode, in.modrm_rm, in.imm_bytes);
    /* B8+rd carries the register in the opcode, not in a ModRM field */
    if (l == 10 && in.rex_b && in.opcode == 0xBB && ((in.opcode & 7) | (in.rex_b << 3)) == 11 && in.imm_bytes == 8) OK();
    else NO("movabs decode wrong");

    T("C7 /0 id -> REX.W, 4-byte immediate");
    const uint8_t b[] = {0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00};
    l = x86_decode_one(b, sizeof b, 0, &in);
    printf("[len=%d w=%d reg=%d imm=%d] ", l, in.rex_w, in.modrm_reg, in.imm_bytes);
    if (l == 7 && in.rex_w && in.modrm_reg == 0 && in.imm_bytes == 4) OK();
    else NO("C7 decode wrong");

    T("SIB form reports the SIB byte");
    const uint8_t c[] = {0x4C, 0x8B, 0x14, 0x24};   /* mov r10, [rsp] */
    l = x86_decode_one(c, sizeof c, 0, &in);
    printf("[len=%d sib=%d rm=%d] ", l, in.has_sib, in.modrm_rm);
    if (l == 4 && in.has_sib && in.modrm_rm == 4) OK(); else NO("SIB not detected");

    T("a whole stream walks cleanly");
    const uint8_t d[] = {0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00, 0xC3};
    printf("[count=%d] ", x86_decode_all(d, sizeof d, NULL, NULL));
    if (x86_decode_all(d, sizeof d, NULL, NULL) == 2) OK(); else NO("stream walk wrong");

    printf("\n=== x86_decode: %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
