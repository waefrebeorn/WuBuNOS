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
#include "wubu_softfloat.h"

#define PIC_RAM_SIZE 1024
#define PIC_XMEM_BASE 256   /* MIR flat memory: cell c -> ram[BASE+c] */
#define PIC_VR_BASE 0x20

/* Virtual PIC opcodes: ONE owner, wubu_pic_opcodes.h (the executor's
 * numbering is the contract; the encoder conforms). */
#include "wubu_pic_opcodes.h"

const char *wubu_pic_interp_name = "pic";

int64_t wubu_pic_interp(const uint8_t *code, size_t size, int64_t arg)
{
    int64_t ram[PIC_RAM_SIZE];
    uint16_t pic_call_stack[32];
    int pic_call_sp = 0;
    memset(ram, 0, sizeof(ram));

    (void)arg;
    uint8_t W = 0;
    uint32_t fret = 0;
    unsigned fret_valid = 0;
    size_t pc = 0;

    while (pc < size) {
        uint8_t op = code[pc++];
        uint8_t fn=0, fr=0, sb=0, sd=0;

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
            if (fret_valid) return (int64_t)(int32_t)fret;
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
        case PIC_SAR: { /* W = (int8_t)W >> ram[fr], sign-filling */
            fr = code[pc++];
            uint8_t sh = (uint8_t)ram[PIC_VR_BASE + fr];
            int8_t sv = (int8_t)W;
            if (sh >= 8) W = (uint8_t)(sv < 0 ? 0xFF : 0x00);
            else W = (uint8_t)(sv >> sh);
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
        case PIC_GEU: /* unsigned: W >= ram[fr] */
            fr = code[pc++];
            W = (W >= (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_LEQ: /* unsigned: W < ram[fr]  (LTU complement for ULE building) */
            fr = code[pc++];
            W = (W < (uint8_t)ram[PIC_VR_BASE + fr]) ? 1 : 0;
            break;
        case PIC_ULEQ: /* unsigned: W <= ram[fr] */
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
        case PIC_FOP: /* fn, fr(a), sb, sd (ram indices) */
            fn = code[pc++]; fr = code[pc++]; sb = code[pc++]; sd = code[pc++];
            {
                uint32_t fa = (uint32_t)(ram[fr] & 0xFFFFFFFF);
                uint32_t fb = (uint32_t)(ram[sb] & 0xFFFFFFFF);
                uint32_t r = 0;
                switch (fn) {
            case 0:  r = wubu_sf_f32_add((uint32_t)fa, (uint32_t)fb); break;
            case 1:  r = wubu_sf_f32_sub((uint32_t)fa, (uint32_t)fb); break;
            case 2:  r = wubu_sf_f32_mul((uint32_t)fa, (uint32_t)fb); break;
            case 3:  r = wubu_sf_f32_div((uint32_t)fa, (uint32_t)fb); break;
            case 4:  r = (uint32_t)wubu_sf_i64_to_f32(fa); break;
            case 5:  r = (uint32_t)wubu_sf_f32_to_i64((uint32_t)fa); break;
            case 6:  r = (wubu_sf_f32_cmp((uint32_t)fa,(uint32_t)fb)==0)?0xFFFFFFFFu:0; break;
            case 7:  r = (wubu_sf_f32_cmp((uint32_t)fa,(uint32_t)fb)!=0)?0xFFFFFFFFu:0; break;
            case 8:  r = (wubu_sf_f32_cmp((uint32_t)fa,(uint32_t)fb) <0)?0xFFFFFFFFu:0; break;
            case 9:  r = (wubu_sf_f32_cmp((uint32_t)fa,(uint32_t)fb)<=0)?0xFFFFFFFFu:0; break;
            case 10: r = fa ^ 0x80000000u; break;
            case 11: r = fa; fret = fa; fret_valid = 1; break;
            default: break;
            }
                ram[sd] = (int64_t)r;
            }
            break;
        case PIC_FRET:
            fr = code[pc++];
            fret = (uint32_t)(ram[fr] & 0xFFFFFFFF);
            fret_valid = 1;
            break;
        case PIC_MEMOP: { /* fn, cell_slot, s2 (dst|val) */
            uint8_t mfn = code[pc++];
            uint8_t cell_slot = code[pc++];
            uint8_t s2 = code[pc++];
            if (mfn == 25)      /* MEM_LOAD */
                ram[PIC_VR_BASE + s2] =
                    ram[PIC_XMEM_BASE + (ram[PIC_VR_BASE + cell_slot] & 0xFF)] & 0xFF;
            else if (mfn == 26) /* MEM_STORE */
                ram[PIC_XMEM_BASE + (ram[PIC_VR_BASE + cell_slot] & 0xFF)] =
                    ram[PIC_VR_BASE + s2] & 0xFF;
            break;
        }
        case PIC_OR: /* W |= ram[fr] (same as ORW) */
            fr = code[pc++];
            W |= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_XR: /* W ^= ram[fr] (same as XRW) */
            fr = code[pc++];
            W ^= (uint8_t)ram[PIC_VR_BASE + fr];
            break;
        case PIC_JMP:
            pc = (uint16_t)(code[pc] | ((uint16_t)code[pc+1] << 8));
            break;
        case PIC_CALL: {
            if (pic_call_sp < 32) {
                uint16_t target = (uint16_t)(code[pc] | ((uint16_t)code[pc+1] << 8));
                pc += 2;
                pic_call_stack[pic_call_sp++] = (uint16_t)pc;
                pc = target;
            }
            break;
        }
        case PIC_FUNC_RET: {
            uint8_t rslot = code[pc++];
            ram[PIC_VR_BASE + 0] = ram[PIC_VR_BASE + rslot];
            if (pic_call_sp > 0) {
                pc = pic_call_stack[--pic_call_sp];
            } else {
                /* top-level return: park value in vr0 slot; main's RET reads it */
                ram[PIC_VR_BASE + 0] = ram[rslot];
            }
            break;
        }
        default:
            return 0;
        }
    }
    if (fret_valid) return (int64_t)(int32_t)fret;
    return (int64_t)(int8_t)W;
}
