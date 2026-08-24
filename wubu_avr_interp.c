/* 
 * wubu_avr_interp.c -- the AVR interpreter (64-bit VR model).
 *
 * Executes virtual AVR-style operations emitted by wubu_isa_avr.c.
 * Virtual registers are full 64-bit ints; the 8-bit ops mask to 8 bits.
 *
 * C11, self-contained.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_softfloat.h"

#define AVR_VR_BASE 0x30
#define AVR_RAM_SIZE 1024
#define AVR_XMEM_BASE 256

/* Virtual AVR opcodes (must match wubu_isa_avr.c) */
#define AVR_LDI  0x01
#define AVR_ADD  0x02
#define AVR_SUB  0x03
#define AVR_AND  0x04
#define AVR_OR   0x05
#define AVR_XOR  0x06
#define AVR_MOV  0x07
#define AVR_NEG  0x08
#define AVR_NOT  0x09
#define AVR_RET  0x0A
#define AVR_MUL  0x0B
#define AVR_DIV  0x0C
#define AVR_MOD  0x0D
#define AVR_SHL  0x0E
#define AVR_SHR  0x0F
#define AVR_GT   0x10
#define AVR_LT   0x11
#define AVR_GE   0x12
#define AVR_LE   0x13
#define AVR_EQ   0x14
#define AVR_NE   0x15
#define AVR_UGT  0x16
#define AVR_ULT  0x17
#define AVR_UGE  0x18
#define AVR_ULE  0x19
#define AVR_FOP  0x20
#define AVR_MEMOP 0x23   /* MIR memory LOAD/STORE */
#define AVR_CALL 0x24
#define AVR_FUNC_RET 0x25
#define AVR_JMP 0x26   /* soft-float hostcall */
#define AVR_FRET 0x21   /* soft-float return */

int64_t wubu_avr_interp(const uint8_t *code, size_t size, int64_t arg)
{
    int64_t vr[AVR_RAM_SIZE];
    memset(vr, 0, sizeof(vr));
    uint16_t avr_call_stack[32];
    int avr_call_sp = 0;

    /* arg → vr0 */
    vr[AVR_VR_BASE + 0] = arg;

    size_t pc = 0;
    while (pc < size) {
        uint8_t op = code[pc++];
        uint8_t a, b, fn, sa, sb, sd;

        switch (op) {
        case AVR_LDI: /* LDI vr, imm8 */
            a = code[pc++];
            vr[AVR_VR_BASE + a] = (int64_t)(code[pc++] & 0xFF);
            break;
        case AVR_ADD: /* ADD vr_a, vr_b */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = (int64_t)((int64_t)(vr[AVR_VR_BASE + a] & 0xFF) +
                                              (int64_t)(vr[AVR_VR_BASE + b] & 0xFF)) & 0xFF;
            break;
        case AVR_SUB: /* SUB vr_a, vr_b */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = (int64_t)((int64_t)(vr[AVR_VR_BASE + a] & 0xFF) -
                                              (int64_t)(vr[AVR_VR_BASE + b] & 0xFF)) & 0xFF;
            break;
        case AVR_AND:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = (vr[AVR_VR_BASE + a] & 0xFF) &
                                   (vr[AVR_VR_BASE + b] & 0xFF);
            break;
        case AVR_OR:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = (vr[AVR_VR_BASE + a] & 0xFF) |
                                   (vr[AVR_VR_BASE + b] & 0xFF);
            break;
        case AVR_XOR:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = (vr[AVR_VR_BASE + a] & 0xFF) ^
                                   (vr[AVR_VR_BASE + b] & 0xFF);
            break;
        case AVR_MOV:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = vr[AVR_VR_BASE + b];
            break;
        case AVR_NEG:
            a = code[pc++];
            vr[AVR_VR_BASE + a] = (-((vr[AVR_VR_BASE + a] & 0xFF))) & 0xFF;
            break;
        case AVR_NOT:
            a = code[pc++];
            vr[AVR_VR_BASE + a] = (~(vr[AVR_VR_BASE + a] & 0xFF)) & 0xFF;
            break;
        case AVR_RET:
            a = code[pc++];
            return (int64_t)(int8_t)(vr[AVR_VR_BASE + a] & 0xFF);
        case AVR_MUL:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) *
                                    (vr[AVR_VR_BASE + b] & 0xFF)) & 0xFF;
            break;
        case AVR_DIV:
            a = code[pc++]; b = code[pc++];
            { int64_t dv = vr[AVR_VR_BASE + b] & 0xFF;
              if (dv != 0)
                  vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) / dv) & 0xFF;
              else
                  vr[AVR_VR_BASE + a] = 0; }
            break;
        case AVR_MOD:
            a = code[pc++]; b = code[pc++];
            { int64_t dv = vr[AVR_VR_BASE + b] & 0xFF;
              if (dv != 0)
                  vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) % dv) & 0xFF;
              else
                  vr[AVR_VR_BASE + a] = 0; }
            break;
        case AVR_SHL:
            a = code[pc++]; b = code[pc++];
            { int shift = vr[AVR_VR_BASE + b] & 0xFF;
              vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) << shift) & 0xFF; }
            break;
        case AVR_SHR:
            a = code[pc++]; b = code[pc++];
            { int shift = vr[AVR_VR_BASE + b] & 0xFF;
              vr[AVR_VR_BASE + a] = (vr[AVR_VR_BASE + a] & 0xFF) >> shift; }
            break;
        case AVR_GT: /* signed comparison */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((int8_t)(vr[AVR_VR_BASE + a] & 0xFF) >
                                    (int8_t)(vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_LT: /* signed comparison */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((int8_t)(vr[AVR_VR_BASE + a] & 0xFF) <
                                    (int8_t)(vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_GE: /* signed comparison */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((int8_t)(vr[AVR_VR_BASE + a] & 0xFF) >=
                                    (int8_t)(vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_LE: /* signed comparison */
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((int8_t)(vr[AVR_VR_BASE + a] & 0xFF) <=
                                    (int8_t)(vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_EQ:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) ==
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_NE:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) !=
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;

        /* unsigned compares: plain uint8 */
        case AVR_UGT:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) >
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_ULT:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) <
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_UGE:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) >=
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_ULE:
            a = code[pc++]; b = code[pc++];
            vr[AVR_VR_BASE + a] = ((vr[AVR_VR_BASE + a] & 0xFF) <=
                                    (vr[AVR_VR_BASE + b] & 0xFF)) ? 1 : 0;
            break;
        case AVR_FOP: /* fn, sa, sb, sd (vr slots) */
            fn = code[pc++]; sa = code[pc++]; sb = code[pc++]; sd = code[pc++];
            {
                uint32_t fa = (uint32_t)(vr[AVR_VR_BASE + sa] & 0xFFFFFFFF);
                uint32_t fb = (uint32_t)(vr[AVR_VR_BASE + sb] & 0xFFFFFFFF);
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
            case 11: /* FRET: result already in r (fa); driver emits AVR_FRET after */ break;
            default: break;
            }
                vr[AVR_VR_BASE + sd] = (int64_t)r;
            }
            break;
        case AVR_FRET:
            a = code[pc++];
            return (int64_t)(int32_t)(vr[AVR_VR_BASE + a] & 0xFFFFFFFF);
        case AVR_MEMOP: { /* fn, cell_slot, s2 (dst|val) */
            uint8_t mfn = code[pc++];
            uint8_t cell_slot = code[pc++];
            uint8_t s2 = code[pc++];
            if (mfn == 25)      /* MEM_LOAD */
                vr[AVR_VR_BASE + s2] =
                    vr[AVR_XMEM_BASE + (vr[AVR_VR_BASE + cell_slot] & 0xFF)] & 0xFF;
            else if (mfn == 26) /* MEM_STORE */
                vr[AVR_XMEM_BASE + (vr[AVR_VR_BASE + cell_slot] & 0xFF)] =
                    vr[AVR_VR_BASE + s2] & 0xFF;
            break;
        }
        case AVR_JMP:
            pc = (uint16_t)(code[pc] | ((uint16_t)code[pc+1] << 8));
            break;
        case AVR_CALL: {
            if (avr_call_sp < 32) {
                uint16_t target = (uint16_t)(code[pc] | ((uint16_t)code[pc+1] << 8));
                pc += 2;
                avr_call_stack[avr_call_sp++] = (uint16_t)pc;
                pc = target;
            }
            break;
        }
        case AVR_FUNC_RET: {
            a = code[pc++];
            vr[AVR_VR_BASE + 0] = vr[AVR_VR_BASE + a];
            if (avr_call_sp > 0) {
                pc = avr_call_stack[--avr_call_sp];
            } else {
                return vr[AVR_VR_BASE + a];
            }
            break;
        }
        default:
            return 0;
        }
    }
    return 0;
}
