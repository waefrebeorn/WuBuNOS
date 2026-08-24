/*
 * wubu_isa_avr.c -- the Atmel AVR ISA driver.
 *
 * The AVR: Atmel's 8-bit RISC microcontroller (ATmega328P = Arduino Uno).
 * 32 8-bit registers (R0-R31), SREG flags, 16-bit instructions.
 * The "AGI on an Arduino" — the most popular MCU in the world.
 *
 * Strategy: SAME MIR as every driver. Each vr lives at RAM[vr+0x30].
 * Emits virtual AVR-style opcodes interpreted by wubu_avr_interp.c.
 * Accumulator-based model: W register + file registers.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define AVR_VR_BASE 0x30

typedef struct {
    uint8_t *code;
    size_t n, cap;
    size_t *label_offsets;
    size_t n_labels;
} avr_emitter_t;

static void ep8(avr_emitter_t *e, uint8_t b) {
    if (e->n == e->cap) { e->cap = e->cap ? e->cap * 2 : 256; e->code = realloc(e->code, e->cap); }
    e->code[e->n++] = b;
}

static void ep16(avr_emitter_t *e, uint16_t w) {
    ep8(e, (uint8_t)(w & 0xFF));
    ep8(e, (uint8_t)((w >> 8) & 0xFF));
}

static void note_label(avr_emitter_t *e, uint32_t label, size_t off) {
    if (label >= e->n_labels) {
        size_t old = e->n_labels;
        e->n_labels = label + 1;
        e->label_offsets = realloc(e->label_offsets, e->n_labels * sizeof(size_t));
        for (size_t i = old; i < e->n_labels; i++) e->label_offsets[i] = (size_t)-1;
    }
    e->label_offsets[label] = off;
}

/* Virtual AVR opcodes */
#define AVR_LDI  0x01
#define AVR_ADD  0x02
#define AVR_FOP  0x20   /* soft-float hostcall */
#define AVR_FRET 0x21   /* soft-float return */
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
/* unsigned compares */
#define AVR_UGT  0x16
#define AVR_ULT  0x17
#define AVR_UGE  0x18
#define AVR_ULE  0x19

static int avr_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size) {
    avr_emitter_t e;
    memset(&e, 0, sizeof(e));
    e.n_labels = p->n_labels;
    e.label_offsets = calloc(e.n_labels, sizeof(size_t));
    for (size_t i = 0; i < e.n_labels; i++) e.label_offsets[i] = (size_t)-1;

    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];

        if (in->op == MIR_LABEL) { note_label(&e, in->label, e.n); continue; }

        switch (in->op) {
        case MIR_CONST:
            ep8(&e, AVR_LDI);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)(in->imm & 0xFF));
            break;
        case MIR_MOV:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            break;
        case MIR_ADD:
            /* dst = a + b: MOV dst, a; ADD dst, b */
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_ADD);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->b);
            break;
        case MIR_SUB:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_SUB);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->b);
            break;
        case MIR_AND:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_AND);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->b);
            break;
        case MIR_OR:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_OR);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->b);
            break;
        case MIR_XOR:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_XOR);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->b);
            break;
        case MIR_NEG:
            ep8(&e, AVR_MOV);
            ep8(&e, (uint8_t)in->dst);
            ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_NEG);
            ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_NOT:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_NOT); ep8(&e, (uint8_t)in->dst);
            break;
        case MIR_MUL:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_MUL); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b);
            break;
        case MIR_DIV:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_DIV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b);
            break;
        case MIR_MOD:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_MOD); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b);
            break;
        case MIR_SHL:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_SHL); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b);
            break;
        case MIR_SHR:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_SHR); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b);
            break;
        case MIR_GT:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_GT); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_LT:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_LT); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_GE:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_GE); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_LE:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_LE); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_EQ:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_EQ); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_NE:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_NE); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;

        case MIR_UGT:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_UGT); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_ULT:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_ULT); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_UGE:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_UGE); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;
        case MIR_ULE:
            ep8(&e, AVR_MOV); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->a);
            ep8(&e, AVR_ULE); ep8(&e, (uint8_t)in->dst); ep8(&e, (uint8_t)in->b); break;        case MIR_FADD:
        case MIR_FSUB:
        case MIR_FMUL:
        case MIR_FDIV: {
            uint8_t fn = (in->op==MIR_FADD)?0:(in->op==MIR_FSUB)?1:(in->op==MIR_FMUL)?2:3;
            ep8(&e, AVR_FOP); ep8(&e, fn);
            ep8(&e, (uint8_t)in->a); ep8(&e, (uint8_t)in->b);
            ep8(&e, (uint8_t)in->dst);
            break;
        }
        case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE: {
            uint8_t fn = (in->op==MIR_FEQ)?6:(in->op==MIR_FNE)?7:(in->op==MIR_FLT)?8:9;
            ep8(&e, AVR_FOP); ep8(&e, fn);
            ep8(&e, (uint8_t)in->a); ep8(&e, (uint8_t)in->b);
            ep8(&e, (uint8_t)in->dst);
            break;
        }
        case MIR_FRET:
            ep8(&e, AVR_FRET);
            ep8(&e, (uint8_t)in->a);
            break;

        case MIR_RET:
            ep8(&e, AVR_RET);
            ep8(&e, (uint8_t)in->a);
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

static int64_t avr_run(const uint8_t *code, size_t size, int64_t arg) {
    extern int64_t wubu_avr_interp(const uint8_t *code, size_t size, int64_t arg);
    return wubu_avr_interp(code, size, arg);
}

static void avr_describe(void) {
    printf("Atmel AVR driver (8-bit RISC): W+32 regs, 16-bit ISA, Arduino heritage.\n"
           "Accumulator-based model; runs via the bundled interpreter —\n"
           "the AGI runs on an Arduino.\n");
}

const wubu_isa_driver_t wubu_isa_avr = {
    .name = "avr",
    .family = "interpreter",
    .exec = WUBU_ISA_INTERPRETED,
    .compile = avr_compile,
    .run = avr_run,
    .describe = avr_describe,
};
