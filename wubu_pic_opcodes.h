/*
 * wubu_pic_opcodes.h -- the virtual PIC opcode contract. ONE owner.
 *
 * The interpreter (wubu_pic_interp.c) is the executor: its numbering IS
 * the contract, and the encoder (wubu_isa_pic.c) conforms. Both include
 * this header; neither keeps a private table.
 *
 * Created by the org-wide hardcoding census (2026-11-19). The two private
 * tables had drifted: the encoder emitted EQ=0x16 where the executor runs
 * GTS, and LT=0x15 where the executor runs LEU — equality and less-than
 * compiled to the wrong operations. The encoder's own note already said
 * "must match the INTERP's table"; now it is structurally impossible not to.
 *
 * Model: accumulator W + file registers RAM[fr]; comparisons yield 0/1.
 */
#ifndef WUBU_PIC_OPCODES_H
#define WUBU_PIC_OPCODES_H

#define PIC_LIW      0x01 /* LIW imm  — W = imm */
#define PIC_ADW      0x02 /* ADW fr   — W = W + RAM[fr] */
#define PIC_SUW      0x03 /* SUW fr   — W = W - RAM[fr] */
#define PIC_ANW      0x04 /* ANW fr   — W = W & RAM[fr] */
#define PIC_ORW      0x05 /* ORW fr   — W = W | RAM[fr] */
#define PIC_XRW      0x06 /* XRW fr   — W = W ^ RAM[fr] */
#define PIC_MVF      0x07 /* MVF fr   — RAM[fr] = W */
#define PIC_MVW      0x08 /* MVW fr   — W = RAM[fr] */
#define PIC_NEG      0x09 /* NEG      — W = -W */
#define PIC_NOT      0x0A /* NOT      — W = ~W */
#define PIC_CLR      0x0B /* CLR fr   — RAM[fr] = 0 */
#define PIC_INC      0x0C /* INC fr   — RAM[fr]++ */
#define PIC_DEC      0x0D /* DEC fr   — RAM[fr]-- */
#define PIC_RET      0x0E /* RET      — return W */
#define PIC_MUL      0x0F /* MUL fr   — W = (W * RAM[fr]) & 0xFF */
#define PIC_DIV      0x10 /* DIV fr   — W = W / RAM[fr] (unsigned) */
#define PIC_MOD      0x11 /* MOD fr   — W = W % RAM[fr] (unsigned) */
#define PIC_SHL      0x12 /* SHL fr   — W = (W << RAM[fr]) & 0xFF */
#define PIC_SHR      0x13 /* SHR fr   — W = W >> RAM[fr] (unsigned) */
#define PIC_GTU      0x14 /* GTU fr   — W = (W >  RAM[fr]) unsigned */
#define PIC_LEU      0x15 /* LEU fr   — W = (W <= RAM[fr]) unsigned */
#define PIC_GTS      0x16 /* GTS fr   — W = (W >  RAM[fr]) signed */
#define PIC_LES      0x17 /* LES fr   — W = (W <= RAM[fr]) signed */
#define PIC_EQ       0x18 /* EQ  fr   — W = (W == RAM[fr]) */
#define PIC_NE       0x19 /* NE  fr   — W = (W != RAM[fr]) */
#define PIC_AND      0x1A /* AND fr   — W = W & RAM[fr] (alias ANW) */
#define PIC_OR       0x1B /* OR  fr   — W = W | RAM[fr] (alias ORW) */
#define PIC_SAR      0x27 /* SAR fr   — W = (int8_t)W >> RAM[fr], ARITHMETIC.
                           * 0x13 (SHR) is the logical one; MIR_SHR is sar and
                           * MIR_SHR_U is shr, and this model had only the
                           * logical op, so a negative value came back large. */

#define PIC_XR       0x1C /* XR  fr   — W = W ^ RAM[fr] (alias XRW) */
#define PIC_GEU      0x1D /* GEU fr   — W = (W >= RAM[fr]) unsigned */
#define PIC_LEQ      0x1E /* LEQ fr   — W = (W <  RAM[fr]) unsigned */
#define PIC_ULEQ     0x1F /* ULEQ fr  — W = (W <= RAM[fr]) unsigned */
#define PIC_FOP      0x20 /* FOP fn,a,b,dst — soft-float hostcall */
#define PIC_FRET     0x21 /* FRET fr  — soft-float return slot */
#define PIC_MEMOP    0x23 /* MEMOP fn,cell,s2 — MIR LOAD/STORE */
#define PIC_CALL     0x24 /* CALL abs16 */
#define PIC_FUNC_RET 0x25 /* FUNC_RET — result slot */
#define PIC_JMP      0x26 /* JMP abs16 */
#define PIC_JZ       0x28 /* JZ  fr, abs16 -- branch to addr when ram[BASE+fr] == 0 */
#define PIC_JNZ      0x29 /* JNZ fr, abs16 -- branch to addr when ram[BASE+fr] != 0 */

#endif /* WUBU_PIC_OPCODES_H */