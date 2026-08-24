/*
 * wubu_8051_interp.c -- the Intel 8051 interpreter (64-bit VR model).
 *
 * Executes virtual opcodes emitted by wubu_isa_8051.c.
 * Virtual registers are full 64-bit ints; the 8-bit ops mask to 8 bits.
 * Fixed: Added all comparison ops, SHL, SHR, MOD, NEG, NOT, EXT_RET.
 *
 * C11, self-contained.
 */
#include <stdint.h>
#include "wubu_softfloat.h"
#include <stdlib.h>
#include <string.h>

#define I8051_VR_BASE 0x30
#define I8051_RAM_SIZE 1024
#define I8051_XMEM_BASE 256   /* MIR flat memory: cell c lives at ram[BASE+c] */

#define I8051_MOV_A_IMM 0x74
#define I8051_MOV_A_DIR 0xE5
#define I8051_MOV_DIR_A 0xF5
#define I8051_ADD_A_DIR 0x25
#define I8051_SUBB_A_DIR 0x95
#define I8051_ANL_A_DIR 0x55
#define I8051_ORL_A_DIR 0x45
#define I8051_XRL_A_DIR 0x65
#define I8051_CLR_A 0xE4
#define I8051_CPL_A 0xF4
#define I8051_RET 0x22
#define I8051_EXT 0xFF

#define EXT_MUL  0x01
#define EXT_DIV  0x02
#define EXT_MOD  0x03
#define EXT_SHL  0x04
#define EXT_SHR  0x05
#define EXT_NEG  0x06
#define EXT_NOT  0x07
#define EXT_GT   0x08
#define EXT_LT   0x09
#define EXT_GE   0x0A
#define EXT_LE   0x0B
#define EXT_EQ   0x0C
#define EXT_NE   0x0D
#define EXT_UGT  0x0F
#define EXT_ULT  0x10
#define EXT_UGE  0x11
#define EXT_ULE  0x12
#define EXT_RET  0x0E
#define EXT_FOP  0x20   /* soft-float: fn, slot_a, slot_b, slot_dst */
#define EXT_FRET 0x21
#define EXT_FCONST 0x22   /* soft-float return: slot */
#define EXT_MEMOP 0x23   /* MIR memory LOAD/STORE */

int64_t wubu_8051_interp_exec(const uint8_t *code, size_t size, int64_t arg)
{
    int64_t ram[I8051_RAM_SIZE];
    memset(ram, 0, sizeof(ram));
    ram[I8051_VR_BASE] = arg;

    size_t pc = 0;
    while (pc < size) {
        uint8_t op = code[pc++];
        uint8_t a, b;
        switch (op) {
        case I8051_MOV_A_IMM:
            ram[0xE0] = (int64_t)(code[pc++] & 0xFF);
            break;
        case I8051_MOV_A_DIR:
            ram[0xE0] = ram[code[pc++]];
            break;
        case I8051_MOV_DIR_A:
            ram[code[pc++]] = ram[0xE0];
            break;
        case I8051_ADD_A_DIR:
            { ram[0xE0] = ((ram[0xE0] & 0xFF) + (ram[code[pc++]] & 0xFF)) & 0xFF; }
            break;
        case I8051_SUBB_A_DIR:
            { ram[0xE0] = ((ram[0xE0] & 0xFF) - (ram[code[pc++]] & 0xFF)) & 0xFF; }
            break;
        case I8051_ANL_A_DIR:
            ram[0xE0] = (ram[0xE0] & 0xFF) & (ram[code[pc++]] & 0xFF);
            break;
        case I8051_ORL_A_DIR:
            ram[0xE0] = (ram[0xE0] & 0xFF) | (ram[code[pc++]] & 0xFF);
            break;
        case I8051_XRL_A_DIR:
            ram[0xE0] = (ram[0xE0] & 0xFF) ^ (ram[code[pc++]] & 0xFF);
            break;
        case I8051_CLR_A:
            ram[0xE0] = 0;
            break;
        case 0xC3: /* CLR C (no-op in our 8-bit model) */
            break;
        case I8051_CPL_A:
            ram[0xE0] = (~(ram[0xE0] & 0xFF)) & 0xFF;
            break;
        case 0xA4: /* MUL AB: A = A * B (low byte), B = high byte */
            { int64_t a = ram[0xE0] & 0xFF;
              int64_t b = ram[0xF0] & 0xFF;
              int64_t prod = a * b;
              ram[0xE0] = prod & 0xFF;
              ram[0xF0] = (prod >> 8) & 0xFF; }
            break;
        case 0x84: /* DIV AB: A = A / B (quotient), B = A % B (remainder) */
            { int64_t a = ram[0xE0] & 0xFF;
              int64_t b = ram[0xF0] & 0xFF;
              if (b != 0) {
                  ram[0xE0] = (a / b) & 0xFF;
                  ram[0xF0] = (a % b) & 0xFF;
              } else {
                  ram[0xE0] = 0xFF;
                  ram[0xF0] = 0xFF; }
            }
            break;
        case I8051_EXT:
            {
                uint8_t ext_op = code[pc++];
                switch (ext_op) {
                case EXT_MUL:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) *
                                  (ram[I8051_VR_BASE + b] & 0xFF)) & 0xFF;
                    break;
                case EXT_DIV:
                    a = code[pc++]; b = code[pc++];
                    { int64_t dv = ram[I8051_VR_BASE + b] & 0xFF;
                      if (dv != 0) ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) / dv) & 0xFF;
                      else ram[0xE0] = 0; }
                    break;
                case EXT_MOD:
                    a = code[pc++]; b = code[pc++];
                    { int64_t dv = ram[I8051_VR_BASE + b] & 0xFF;
                      if (dv != 0) ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) % dv) & 0xFF;
                      else ram[0xE0] = 0; }
                    break;
                case EXT_SHL:
                    b = code[pc++];
                    { int shift = ram[I8051_VR_BASE + b] & 0xFF;
                      ram[0xE0] = ((ram[0xE0] & 0xFF) << shift) & 0xFF; }
                    break;
                case EXT_SHR:
                    b = code[pc++];
                    { int shift = ram[I8051_VR_BASE + b] & 0xFF;
                      ram[0xE0] = (ram[0xE0] & 0xFF) >> shift; }
                    break;
                case EXT_NEG:
                    ram[0xE0] = (-(ram[0xE0] & 0xFF)) & 0xFF;
                    break;
                case EXT_NOT:
                    ram[0xE0] = (~(ram[0xE0] & 0xFF)) & 0xFF;
                    break;
                case EXT_GT:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((int8_t)(ram[I8051_VR_BASE + a] & 0xFF) >
                                  (int8_t)(ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_LT:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((int8_t)(ram[I8051_VR_BASE + a] & 0xFF) <
                                  (int8_t)(ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_GE:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((int8_t)(ram[I8051_VR_BASE + a] & 0xFF) >=
                                  (int8_t)(ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_LE:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((int8_t)(ram[I8051_VR_BASE + a] & 0xFF) <=
                                  (int8_t)(ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                /* unsigned compares: plain uint8 compare */
                case EXT_UGT:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) >
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_ULT:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) <
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_UGE:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) >=
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_ULE:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) <=
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_EQ:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) ==
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_NE:
                    a = code[pc++]; b = code[pc++];
                    ram[0xE0] = ((ram[I8051_VR_BASE + a] & 0xFF) !=
                                  (ram[I8051_VR_BASE + b] & 0xFF)) ? 1 : 0;
                    break;
                case EXT_RET:
                    a = code[pc++];
                    return (int64_t)(int8_t)(ram[I8051_VR_BASE + a] & 0xFF);
                case EXT_FOP: {
                    /* fn, sa, sb, sd — slots index ram[] directly; the
                     * f32 bits live in the low 32 bits of each cell */
                    uint8_t fn = code[pc++];
                    int64_t fa = ram[code[pc++]] & 0xFFFFFFFFLL;
                    int64_t fb = ram[code[pc++]] & 0xFFFFFFFFLL;
                    uint8_t sd = code[pc++];
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
            case 11: /* FRET handled by EXT_FRET case */ break;
            default: break;
            }
                    ram[sd] = (int64_t)r;   /* zero-extend into cell */
                    break;
                }
                case EXT_FCONST: {
                    uint8_t sd = code[pc+4];
                    int64_t v = (int64_t)code[pc] | ((int64_t)code[pc+1]<<8)
                              | ((int64_t)code[pc+2]<<16) | ((int64_t)code[pc+3]<<24);
                    pc += 5;
                    ram[sd] = v;
                    break;
                }
                case EXT_MEMOP: {
                    /* fn, cell_slot, second_slot (dst for LOAD, val for STORE).
                     * MIR flat memory: cell c -> ram[I8051_XMEM_BASE + c]. */
                    uint8_t fn = code[pc++];
                    uint8_t cell_slot = code[pc++];
                    uint8_t s2 = code[pc++];
                    if (fn == 25)      /* MEM_LOAD */
                        ram[s2] = ram[I8051_XMEM_BASE + (ram[cell_slot] & 0xFF)] & 0xFF;
                    else if (fn == 26) /* MEM_STORE */
                        ram[I8051_XMEM_BASE + (ram[cell_slot] & 0xFF)] = ram[s2] & 0xFF;
                    break;
                }
                case EXT_FRET: {
                    uint8_t sa = code[pc++];
                    return (int64_t)(int32_t)(ram[sa] & 0xFFFFFFFFLL);
                }
                default:
                    return 0;
                }
                break;
            }
        case I8051_RET:
            return (int64_t)(int8_t)(ram[0xE0] & 0xFF);
        default:
            return 0;
        }
    }
    return (int64_t)(int8_t)(ram[0xE0] & 0xFF);
}
