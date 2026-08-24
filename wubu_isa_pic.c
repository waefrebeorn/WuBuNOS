/*
 * wubu_isa_pic.c -- the Microchip PIC ISA driver.
 *
 * The PIC: Microchip's 8-bit Harvard-architecture microcontroller
 * (PIC16F877A = the most deployed MCU in automotive/industrial).
 * 8-bit data bus, 14-bit instruction word, banked register file,
 * W accumulator + 80 general-purpose registers, hardware stack.
 * The "AGI on a PIC" — the chip that runs your car's windshield wipers.
 *
 * Strategy: SAME MIR as every driver. Each vr lives at RAM[vr+0x20].
 * Emits virtual PIC-style opcodes interpreted by wubu_pic_interp.c.
 * Accumulator-based model: W register + file registers + bank select.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define PIC_VR_BASE 0x20

typedef struct {
    uint8_t *code;
    size_t n, cap;
    size_t *label_offsets;
    size_t n_labels;
} pic_emitter_t;

static void pic_ep8(pic_emitter_t *e, uint8_t b) {
    if (e->n == e->cap) { e->cap = e->cap ? e->cap * 2 : 256; e->code = realloc(e->code, e->cap); }
    e->code[e->n++] = b;
}

static void pic_note_label(pic_emitter_t *e, uint32_t label, size_t off) {
    if (label >= e->n_labels) {
        size_t old = e->n_labels;
        e->n_labels = label + 1;
        e->label_offsets = realloc(e->label_offsets, e->n_labels * sizeof(size_t));
        for (size_t i = old; i < e->n_labels; i++) e->label_offsets[i] = (size_t)-1;
    }
    e->label_offsets[label] = off;
}

/* Virtual PIC opcodes — accumulator/W + file-register model */
#define PIC_LIW  0x01 /* LIW imm     — W = imm */
#define PIC_ADW  0x02 /* ADW fr      — W = W + RAM[fr] */
#define PIC_SUW  0x03 /* SUW fr      — W = W - RAM[fr] */
#define PIC_ANW  0x04 /* ANW fr      — W = W & RAM[fr] */
#define PIC_ORW  0x05 /* ORW fr      — W = W | RAM[fr] */
#define PIC_XRW  0x06 /* XRW fr      — W = W ^ RAM[fr] */
#define PIC_MVF  0x07 /* MVF fr      — RAM[fr] = W */
#define PIC_MVW  0x08 /* MVW fr      — W = RAM[fr] */
#define PIC_NEG  0x09 /* NEG         — W = -W */
#define PIC_NOT  0x0A /* NOT         — W = ~W */
#define PIC_CLR  0x0B /* CLR fr      — RAM[fr] = 0 */
#define PIC_INC  0x0C /* INC fr      — RAM[fr]++ */
#define PIC_DEC  0x0D /* DEC fr      — RAM[fr]-- */
#define PIC_MUL  0x0F /* MUL fr      — W = (W * RAM[fr]) & 0xFF */
#define PIC_DIV  0x10 /* DIV fr      — W = W / RAM[fr] (unsigned) */
#define PIC_MOD  0x11 /* MOD fr      — W = W % RAM[fr] (unsigned) */
#define PIC_SHL  0x12 /* SHL fr      — W = (W << RAM[fr]) & 0xFF */
#define PIC_SHR  0x13 /* SHR fr      — W = W >> RAM[fr] (unsigned) */
#define PIC_GTU  0x14 /* GTU fr      — W = (W > RAM[fr]) ? 1 : 0 */
#define PIC_LTU  0x15 /* LTU fr      — W = (W < RAM[fr]) ? 1 : 0 */
#define PIC_EQ   0x16
/* NOTE: opcode values must match the INTERP's table (wubu_pic_interp.c),
 * not the compiler's local historical numbering. GEU=0x1D, LT-complement=0x1E. */
#define PIC_GEU  0x1D
#define PIC_ULEQ 0x1F /* EQ  fr      — W = (W == RAM[fr]) ? 1 : 0 */
#define PIC_RET  0x0E /* RET         — return W */
#define PIC_FOP  0x20 /* soft-float hostcall */
#define PIC_MEMOP 0x23 /* MIR memory LOAD/STORE */
#define PIC_FRET 0x21 /* soft-float return */

static int pic_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size) {
    pic_emitter_t e;
    memset(&e, 0, sizeof(e));
    e.n_labels = p->n_labels;
    e.label_offsets = calloc(e.n_labels, sizeof(size_t));
    for (size_t i = 0; i < e.n_labels; i++) e.label_offsets[i] = (size_t)-1;

    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];

        if (in->op == MIR_LABEL) { pic_note_label(&e, in->label, e.n); continue; }

        switch (in->op) {
        case MIR_CONST:
            /* LIW imm */
            pic_ep8(&e, PIC_LIW);
            pic_ep8(&e, (uint8_t)(in->imm & 0xFF));
            /* store to dst */
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_MOV:
            /* W = RAM[src]; RAM[dst] = W */
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_ADD:
            /* W = RAM[a]; W += RAM[b]; RAM[dst] = W */
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_ADW);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_SUB:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_SUW);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_AND:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_ANW);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_OR:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_ORW);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_XOR:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_XRW);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_NEG:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_NEG);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_NOT:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_NOT);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_MUL:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_MUL);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_DIV:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_DIV);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_MOD:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_MOD);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_SHL:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_SHL);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_SHR:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_SHR);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_GT:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_GTU);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_LT:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_LTU);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_EQ:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_EQ);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;

        case MIR_UGT:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_GTU);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_ULT:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_LTU);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_UGE:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_GEU);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_ULE:
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_ULEQ);
            pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, PIC_MVF);
            pic_ep8(&e, (uint8_t)in->dst);
            break;        case MIR_FADD:
        case MIR_FSUB:
        case MIR_FMUL:
        case MIR_FDIV: {
            uint8_t fn = (in->op==MIR_FADD)?0:(in->op==MIR_FSUB)?1:(in->op==MIR_FMUL)?2:3;
            pic_ep8(&e, PIC_FOP); pic_ep8(&e, fn);
            pic_ep8(&e, (uint8_t)in->a); pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        }
        case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE: {
            uint8_t fn = (in->op==MIR_FEQ)?6:(in->op==MIR_FNE)?7:(in->op==MIR_FLT)?8:9;
            pic_ep8(&e, PIC_FOP); pic_ep8(&e, fn);
            pic_ep8(&e, (uint8_t)in->a); pic_ep8(&e, (uint8_t)in->b);
            pic_ep8(&e, (uint8_t)in->dst);
            break;
        }
        case MIR_FRET:
            pic_ep8(&e, PIC_FRET);
            pic_ep8(&e, (uint8_t)in->a);
            break;

        case MIR_LOAD:
            /* dst = mem[addr]; cell c -> ram[BASE + c] */
            pic_ep8(&e, PIC_MEMOP); pic_ep8(&e, 25);
            pic_ep8(&e, (uint8_t)in->a); pic_ep8(&e, (uint8_t)in->dst);
            break;

        case MIR_STORE:
            pic_ep8(&e, PIC_MEMOP); pic_ep8(&e, 26);
            pic_ep8(&e, (uint8_t)in->a); pic_ep8(&e, (uint8_t)in->b);
            break;

        case MIR_RET:
            /* return RAM[a] */
            pic_ep8(&e, PIC_MVW);
            pic_ep8(&e, (uint8_t)in->a);
            pic_ep8(&e, PIC_RET);
            break;
        default:
            break;
        }
    }

    free(e.label_offsets);
    *out = e.code;
    *out_size = e.n;
    return 0;
}

static int64_t pic_run(const uint8_t *code, size_t size, int64_t arg) {
    extern int64_t wubu_pic_interp(const uint8_t *code, size_t size, int64_t arg);
    return wubu_pic_interp(code, size, arg);
}

static void pic_describe(void) {
    printf("Microchip PIC driver (8-bit Harvard, PIC16F877A): W+80 regs,\n"
           "14-bit ISA, banked memory; runs via the bundled interpreter —\n"
           "the AGI runs on the chip that runs your car.\n");
}

const wubu_isa_driver_t wubu_isa_pic = {
    .name = "pic",
    .family = "interpreter",
    .exec = WUBU_ISA_INTERPRETED,
    .compile = pic_compile,
    .run = pic_run,
    .describe = pic_describe,
};
