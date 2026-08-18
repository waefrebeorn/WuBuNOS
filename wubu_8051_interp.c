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
#include <stdlib.h>
#include <string.h>

#define I8051_VR_BASE 0x30
#define I8051_RAM_SIZE 256

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
#define EXT_RET  0x0E

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
