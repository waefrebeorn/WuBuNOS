/*
 * wubu_pic_interp.c — the PIC interpreter (8-bit VR model).
 *
 * Executes virtual PIC bytecode emitted by wubu_isa_pic.c.
 * Virtual registers are 8-bit values stored in RAM at PIC_VR_BASE.
 *
 * C11, self-contained.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PIC_RAM_SIZE 256
#define PIC_VR_BASE 0x20

/* Virtual PIC opcodes — MUST match wubu_isa_pic.c exactly */
#define PIC_LIW   0x01 /* LIW imm     — W = imm       */
#define PIC_ADW   0x02 /* ADW fr      — W = W + RAM[fr] */
#define PIC_SUW   0x03 /* SUW fr      — W = W - RAM[fr] */
#define PIC_ANW   0x04 /* ANW fr      — W = W & RAM[fr] */
#define PIC_ORW   0x05 /* ORW fr      — W = W | RAM[fr] */
#define PIC_XRW   0x06 /* XRW fr      — W = W ^ RAM[fr] */
#define PIC_MVF   0x07 /* MVF fr      — RAM[fr] = W     */
#define PIC_MVW   0x08 /* MVW fr      — W = RAM[fr]     */
#define PIC_NEG   0x09 /* NEG         — W = -W          */
#define PIC_NOT   0x0A /* NOT         — W = ~W          */
#define PIC_CLR   0x0B /* CLR fr      — RAM[fr] = 0     */
#define PIC_INC   0x0C /* INC fr      — RAM[fr]++      */
#define PIC_DEC   0x0D /* DEC fr      — RAM[fr]--      */
#define PIC_RET   0x0E /* RET         — return W       */
#define PIC_MUL   0x0F /* MUL fr      — W = (W * RAM[fr]) & 0xFF  */
#define PIC_DIV   0x10 /* DIV fr      — W = W / RAM[fr] (unsigned) */
#define PIC_MOD   0x11 /* MOD fr      — W = W % RAM[fr] (unsigned) */
#define PIC_SHL   0x12 /* SHL fr      — W = (W << RAM[fr]) & 0xFF */
#define PIC_SHR   0x13 /* SHR fr      — W = W >> RAM[fr] (unsigned) */
#define PIC_GTU   0x14 /* GTU fr      — W = (W > RAM[fr]) ? 1 : 0 (unsigned) */
#define PIC_LEU   0x15 /* LEU fr      — W = (W <= RAM[fr]) ? 1 : 0 (unsigned) */
#define PIC_GTS   0x16 /* GTS fr      — W = (W > RAM[fr]) ? 1 : 0 (signed)   */
#define PIC_LES   0x17 /* LES fr      — W = (W <= RAM[fr]) ? 1 : 0 (signed)  */
#define PIC_EQ    0x18 /* EQ  fr      — W = (W == RAM[fr]) ? 1 : 0 */
#define PIC_NE    0x19 /* NE  fr      — W = (W != RAM[fr]) ? 1 : 0 */
#define PIC_AND   0x1A /* AND fr      — W = W & RAM[fr]  */
#define PIC_OR    0x1B /* OR  fr      — W = W | RAM[fr]  */
#define PIC_XR    0x1C /* XR  fr      — W = W ^ RAM[fr]  */

const char *wubu_pic_interp_name = "pic";

int64_t wubu_pic_interp(const uint8_t *code, size_t size, int64_t arg)
{
    int64_t ram[PIC_RAM_SIZE];
    memset(ram, 0, sizeof(ram));

    (void)arg;
    uint8_t W = 0;  /* W register (8-bit accumulator) */
    size_t pc = 0;

    while (pc < size) {
        uint8_t op = code[pc++];
        uint8_t fr;

        switch (op) {
        case PIC_LIW: /* W = imm */
            W = code[pc++];
            break;
        case PIC_ADW: /* W += ram[fr] */
            fr = code[pc++];
            W = (uint8_t)(W + ram[PIC_VR_BASE + fr]);
            break;
        case PIC_SUW: /* W -= ram[fr] */
            fr = code[pc++];
            W = (uint8_t)(W - ram[PIC_VR_BASE + fr]);
            break;
        case PIC_ANW: /* W &= ram[fr] */
            fr = code[pc++];
            W &= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_ORW: /* W |= ram[fr] */
            fr = code[pc++];
            W |= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_XRW: /* W ^= ram[fr] */
            fr = code[pc++];
            W ^= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_MVF: /* ram[fr] = W */
            fr = code[pc++];
            ram[PIC_VR_BASE + fr] = W;
            break;
        case PIC_MVW: /* W = ram[fr] */
            fr = code[pc++];
            W = (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_NEG: /* W = -W */
            W = (uint8_t)(-(int8_t)W);
            break;
        case PIC_NOT: /* W = ~W */
            W = (uint8_t)(~W);
            break;
        case PIC_CLR: /* ram[fr] = 0 */
            fr = code[pc++];
            ram[PIC_VR_BASE + fr] = 0;
            break;
        case PIC_INC: /* ram[fr]++ */
            fr = code[pc++];
            ram[PIC_VR_BASE + fr] = (uint8_t)(ram[PIC_VR_BASE + fr] + 1);
            break;
        case PIC_DEC: /* ram[fr]-- */
            fr = code[pc++];
            ram[PIC_VR_BASE + fr] = (uint8_t)(ram[PIC_VR_BASE + fr] - 1);
            break;
        case PIC_RET: /* return W (sign-extend 8-bit) */
            return (int64_t)(int8_t)W;
        case PIC_MUL: /* W = (W * ram[fr]) & 0xFF */
            fr = code[pc++];
            W = (uint8_t)(W * (uint8_t)ram[PIC_VR_BASE + fr]);
            break;
        case PIC_DIV: { /* W = W / ram[fr] (unsigned) */
            fr = code[pc++];
            uint8_t dv = (uint8_t)ram[PIC_VR_BASE + fr];
            W = dv ? (uint8_t)(W / dv) : 0;
            break;
        }
        case PIC_MOD: { /* W = W % ram[fr] (unsigned) */
            fr = code[pc++];
            uint8_t dv = (uint8_t)ram[PIC_VR_BASE + fr];
            W = dv ? (uint8_t)(W % dv) : 0;
            break;
        }
        case PIC_SHL: { /* W = (W << ram[fr]) & 0xFF */
            fr = code[pc++];
            uint8_t sh = (uint8_t)ram[PIC_VR_BASE + fr];
            if (sh >= 8) W = 0;
            else W = (uint8_t)(W << sh);
            break;
        }
        case PIC_SHR: { /* W = W >> ram[fr] (unsigned) */
            fr = code[pc++];
            uint8_t sh = (uint8_t)ram[PIC_VR_BASE + fr];
            if (sh >= 8) W = 0;
            else W = (uint8_t)(W >> sh);
            break;
        }
        case PIC_GTU: /* unsigned: W > ram[fr] */
            fr = code[pc++];
            W = (W > (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_LEU: /* unsigned: W <= ram[fr] */
            fr = code[pc++];
            W = (W <= (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_GTS: /* signed: W > ram[fr] */
            fr = code[pc++];
            W = ((int8_t)W > (int8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_LES: /* signed: W <= ram[fr] */
            fr = code[pc++];
            W = ((int8_t)W <= (int8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_EQ: /* W == ram[fr] */
            fr = code[pc++];
            W = (W == (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_NE: /* W != ram[fr] */
            fr = code[pc++];
            W = (W != (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_AND: /* W &= ram[fr] (same as ANW) */
            fr = code[pc++];
            W &= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_OR: /* W |= ram[fr] (same as ORW) */
            fr = code[pc++];
            W |= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_XR: /* W ^= ram[fr] (same as XRW) */
            fr = code[pc++];
            W ^= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        default:
            return 0;
        }
    }
    return (int64_t)(int8_t)W;
}
