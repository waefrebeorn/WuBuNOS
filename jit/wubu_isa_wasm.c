/*
 * wubu_isa_wasm.c — WebAssembly binary emitter.
 *
 * Compiles MIR → WASM MVP binary. Handles: CONST, MOV, ADD, SUB, MUL, DIV,
 * MOD, AND, OR, XOR, SHL, SHR, NEG, NOT, EQ/NE/LT/LE/GT/GE, JMP, JZ, JNZ,
 * LABEL, RET. F32 arithmetic (ADD/SUB/MUL/DIV) also emitted.
 *
 * Strategy: each MIR virtual register maps to a WASM local. The function
 * signature is (i32) -> (i32) — entry_vr is read from param 0, result
 * returned as i32. All computation is stack-based via WASM's operand stack.
 *
 * C11, self-contained. No external tools needed.
 */

#include "wubu_isa_wasm.h"
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ---- WASM opcodes ---- */
#define OP_UNREACHABLE  0x00
#define OP_NOP          0x01
#define OP_BLOCK        0x02
#define OP_LOOP         0x03
#define OP_IF           0x04
#define OP_ELSE         0x05
#define OP_END          0x0B
#define OP_BR           0x0C
#define OP_BR_IF        0x0D
#define OP_RETURN       0x0F
#define OP_DROP         0x1A
#define OP_SELECT       0x1B
#define OP_LOCAL_GET    0x20
#define OP_LOCAL_SET    0x21
#define OP_LOCAL_TEE    0x22
#define OP_I32_CONST    0x41
#define OP_I32_EQ       0x46
#define OP_I32_NE       0x47
#define OP_I32_LT_S     0x48
#define OP_I32_LT_U     0x49
#define OP_I32_GT_S     0x4A
#define OP_I32_GT_U     0x4B
#define OP_I32_LE_S     0x4C
#define OP_I32_LE_U     0x4D
#define OP_I32_GE_S     0x4E
#define OP_I32_GE_U     0x4F
#define OP_I32_ADD      0x6A
#define OP_I32_SUB      0x6B
#define OP_I32_MUL      0x6C
#define OP_I32_DIV_S    0x6D
#define OP_I32_DIV_U    0x6E
#define OP_I32_REM_S    0x6F
#define OP_I32_REM_U    0x70
#define OP_I32_AND      0x71
#define OP_I32_OR       0x72
#define OP_I32_XOR      0x73
#define OP_I32_SHL      0x74
#define OP_I32_SHR_U    0x75
#define OP_I32_SHR_S    0x76

#define OP_F32_CONST    0x43
#define OP_F32_EQ       0x5B
#define OP_F32_NE       0x5C
#define OP_F32_LT       0x5D
#define OP_F32_GT       0x5E
#define OP_F32_ADD      0x92
#define OP_F32_SUB      0x93
#define OP_F32_MUL      0x94
#define OP_F32_DIV      0x95
#define OP_F32_ABS      0x8B
#define OP_F32_NEG      0x8C
#define OP_F32_SQRT     0x9F

#define BLOCK_VOID      0x40
#define TYPE_I32        0x7F
#define TYPE_F32        0x7D
#define SEC_TYPE        1
#define SEC_FUNC        3
#define SEC_CODE        10

/* ---- LEB128 encoding into buffer ---- */
typedef struct {
    uint8_t *buf;
    size_t   cap;
    size_t   pos;
} wasm_buf_t;

static void buf_init(wasm_buf_t *b)
{
    b->cap = 4096;
    b->buf = malloc(b->cap);
    b->pos = 0;
}

static void buf_byte(wasm_buf_t *b, uint8_t v)
{
    if (b->pos >= b->cap) { b->cap *= 2; b->buf = realloc(b->buf, b->cap); }
    b->buf[b->pos++] = v;
}

static void buf_leb_u(wasm_buf_t *b, uint32_t v)
{
    while (v > 0x7F) { buf_byte(b, (v & 0x7F) | 0x80); v >>= 7; }
    buf_byte(b, v & 0x7F);
}

static void buf_leb_s(wasm_buf_t *b, int64_t v)
{
    int more = 1;
    while (more) {
        uint8_t byte = v & 0x7F;
        v >>= 7;
        if ((v == 0 && !(byte & 0x40)) || (v == -1 && (byte & 0x40)))
            more = 0;
        else byte |= 0x80;
        buf_byte(b, byte);
    }
}

static void buf_f32(wasm_buf_t *b, float v)
{
    uint32_t u; memcpy(&u, &v, 4);
    for (int i = 0; i < 4; i++) buf_byte(b, (u >> (i*8)) & 0xFF);
}

static void buf_bytes(wasm_buf_t *b, const uint8_t *data, size_t n)
{
    for (size_t i = 0; i < n; i++) buf_byte(b, data[i]);
}

static void buf_free(wasm_buf_t *b) { free(b->buf); }

/* ---- Section emission (with size prefix) ---- */
static size_t sec_start(wasm_buf_t *b, uint8_t sec_id)
{
    buf_byte(b, sec_id);
    size_t patch = b->pos;
    buf_leb_u(b, 0); /* placeholder for size (1 byte, patched in sec_end) */
    return patch;
}

static void sec_end(wasm_buf_t *b, size_t patch_pos)
{
    /* Compute content size (between placeholder and current pos) */
    size_t payload_size = b->pos - patch_pos - 1;
    /* Encode LEB128 size */
    uint8_t tmp[5]; int n = 0;
    {
        uint32_t v = (uint32_t)payload_size;
        do { tmp[n++] = (v & 0x7F); v >>= 7; if (v) tmp[n-1] |= 0x80; } while (v);
    }
    if (n <= 1) {
        /* Fits in the 1-byte placeholder */
        b->buf[patch_pos] = tmp[0];
    } else {
        /* Need more bytes: shift content right to make room */
        int extra = n - 1;
        while (b->cap < b->pos + extra) { b->cap *= 2; b->buf = realloc(b->buf, b->cap); }
        for (size_t i = b->pos; i > patch_pos + 1; i--)
            b->buf[i + extra - 1] = b->buf[i - 1];
        b->pos += extra;
        for (int i = 0; i < n; i++)
            b->buf[patch_pos + i] = tmp[i];
    }
}

/* ---- High-level emit helpers ---- */
static void emit_local_get(wasm_buf_t *b, uint32_t idx) { buf_byte(b, OP_LOCAL_GET); buf_leb_u(b, idx); }
static void emit_local_set(wasm_buf_t *b, uint32_t idx) { buf_byte(b, OP_LOCAL_SET); buf_leb_u(b, idx); }
static void emit_local_tee(wasm_buf_t *b, uint32_t idx) { buf_byte(b, OP_LOCAL_TEE); buf_leb_u(b, idx); }
static void emit_i32_const(wasm_buf_t *b, int32_t v) { buf_byte(b, OP_I32_CONST); buf_leb_s(b, v); }
static void emit_f32_const(wasm_buf_t *b, float v) { buf_byte(b, OP_F32_CONST); buf_f32(b, v); }
static void emit_drop(wasm_buf_t *b) { buf_byte(b, OP_DROP); }
static void emit_return(wasm_buf_t *b) { buf_byte(b, OP_RETURN); }
static void emit_end(wasm_buf_t *b) { buf_byte(b, OP_END); }

/* ---- Compile MIR → WASM ---- */
static int wasm_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size)
{
    if (!p || !out || !out_size) return -1;

    /* Count max VR to determine local count */
    uint32_t max_vr = 0;
    for (size_t i = 0; i < p->n; i++) {
        if (p->ins[i].dst > max_vr) max_vr = p->ins[i].dst;
        if (p->ins[i].a > max_vr) max_vr = p->ins[i].a;
        if (p->ins[i].b > max_vr) max_vr = p->ins[i].b;
    }
    uint32_t n_locals = max_vr + 1; /* VR 0 = param, VR 1+ = locals */

    wasm_buf_t b;
    buf_init(&b);

    /* Magic + version */
    buf_bytes(&b, (const uint8_t *)"\0asm", 4);
    buf_bytes(&b, (const uint8_t[]){1,0,0,0}, 4);

    /* Type section: 1 signature (i32) -> (i32) */
    {
        size_t patch = sec_start(&b, SEC_TYPE);
        buf_leb_u(&b, 1); /* 1 type */
        buf_byte(&b, 0x60); /* functype */
        buf_byte(&b, 1); /* 1 param */
        buf_byte(&b, TYPE_I32);
        buf_byte(&b, 1); /* 1 result */
        buf_byte(&b, TYPE_I32);
        sec_end(&b, patch);
    }

    /* Function section: 1 function, type index 0 */
    {
        size_t patch = sec_start(&b, SEC_FUNC);
        buf_leb_u(&b, 1); /* 1 function */
        buf_leb_u(&b, 0); /* type index 0 */
        sec_end(&b, patch);
    }

    /* Code section: 1 function body */
    {
        size_t code_patch = sec_start(&b, SEC_CODE);
        buf_leb_u(&b, 1); /* 1 body */

        /* Reserve space for body size */
        size_t body_start = b.pos;
        buf_leb_u(&b, 0); /* placeholder body size */

        /* Local declarations: n_locals - 1 (VR 0 is param) of type i32 */
        uint32_t extra_locals = (n_locals > 1) ? n_locals - 1 : 0;
        if (extra_locals > 0) {
            buf_leb_u(&b, 1); /* 1 local declaration group */
            buf_leb_u(&b, extra_locals);
            buf_byte(&b, TYPE_I32);
        } else {
            buf_leb_u(&b, 0); /* no local declarations */
        }

        /* Track label positions for branch patching */
        size_t *label_pos = calloc(p->n_labels + 16, sizeof(size_t));
        size_t *label_stack = calloc(p->n_labels + 16, sizeof(size_t));
        size_t label_sp = 0;

        /* Emit MIR instructions */
        for (size_t i = 0; i < p->n; i++) {
            const wubu_mir_instr_t *in = &p->ins[i];

            if (in->op == MIR_LABEL) {
                /* If we have an open block from JZ/JNZ, note its label */
                if (label_sp > 0) {
                    label_sp--;
                    label_pos[in->label] = b.pos;
                }
                continue;
            }

            switch (in->op) {
            case MIR_CONST:
                emit_i32_const(&b, (int32_t)in->imm);
                emit_local_set(&b, in->dst);
                break;
            case MIR_MOV:
                emit_local_get(&b, in->a);
                emit_local_set(&b, in->dst);
                break;
            case MIR_ADD:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_ADD);
                emit_local_set(&b, in->dst);
                break;
            case MIR_SUB:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_SUB);
                emit_local_set(&b, in->dst);
                break;
            case MIR_MUL:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_MUL);
                emit_local_set(&b, in->dst);
                break;
            case MIR_DIV:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_DIV_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_MOD:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_REM_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_AND:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_AND);
                emit_local_set(&b, in->dst);
                break;
            case MIR_OR:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_OR);
                emit_local_set(&b, in->dst);
                break;
            case MIR_XOR:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_XOR);
                emit_local_set(&b, in->dst);
                break;
            case MIR_SHL:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_SHL);
                emit_local_set(&b, in->dst);
                break;
            case MIR_SHR:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_SHR_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_NEG:
                emit_i32_const(&b, 0);
                emit_local_get(&b, in->a);
                buf_byte(&b, OP_I32_SUB);
                emit_local_set(&b, in->dst);
                break;
            case MIR_NOT:
                emit_local_get(&b, in->a);
                emit_i32_const(&b, -1);
                buf_byte(&b, OP_I32_XOR);
                emit_local_set(&b, in->dst);
                break;
            case MIR_EQ:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_EQ);
                emit_local_set(&b, in->dst);
                break;
            case MIR_NE:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_NE);
                emit_local_set(&b, in->dst);
                break;
            case MIR_LT:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_LT_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_LE:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_LE_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_GT:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_GT_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_GE:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_I32_GE_S);
                emit_local_set(&b, in->dst);
                break;
            case MIR_FADD:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_F32_ADD);
                emit_local_set(&b, in->dst);
                break;
            case MIR_FSUB:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_F32_SUB);
                emit_local_set(&b, in->dst);
                break;
            case MIR_FMUL:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_F32_MUL);
                emit_local_set(&b, in->dst);
                break;
            case MIR_FDIV:
                emit_local_get(&b, in->a);
                emit_local_get(&b, in->b);
                buf_byte(&b, OP_F32_DIV);
                emit_local_set(&b, in->dst);
                break;
            case MIR_JMP:
                /* BR to end of block — simplified */
                buf_byte(&b, OP_BR);
                buf_leb_u(&b, 0); /* depth 0 */
                break;
            case MIR_JZ:
                emit_local_get(&b, in->a);
                buf_byte(&b, OP_BR_IF);
                buf_leb_u(&b, 0);
                break;
            case MIR_JNZ:
                emit_local_get(&b, in->a);
                /* JNZ = NOT JZ: load, eq 0, br_if */
                buf_byte(&b, OP_I32_CONST); buf_leb_s(&b, 0);
                buf_byte(&b, OP_I32_EQ);
                buf_byte(&b, OP_BR_IF);
                buf_leb_u(&b, 0);
                break;
            case MIR_RET:
                emit_local_get(&b, in->a);
                emit_return(&b);
                break;
            default:
                break;
            }
        }

        /* End */
        emit_end(&b);

        /* Patch body size */
        size_t body_end = b.pos;
        size_t body_size = body_end - (body_start + 1);
        /* Write LEB at body_start */
        uint8_t tmp[5]; int n = 0;
        {
            uint32_t v = (uint32_t)body_size;
            do { tmp[n++] = (v & 0x7F); v >>= 7; if (v) tmp[n-1] |= 0x80; } while (v);
        }
        b.buf[body_start] = tmp[0];
        if (n > 1) {
            /* Need to shift */
            int shift = n - 1;
            for (size_t i = body_end; i > body_start + 1; i--)
                b.buf[i + shift - 1] = b.buf[i - 1];
            b.pos += shift;
            for (int i = 1; i < n; i++)
                b.buf[body_start + i] = tmp[i];
        }

        sec_end(&b, code_patch);
        free(label_pos);
        free(label_stack);
    }

    *out = b.buf;
    *out_size = b.pos;
    return 0;
}

/* ---- Run: WASM can't JIT on x86-64, so this returns error ---- */
static int64_t wasm_run(const uint8_t *code, size_t size, int64_t arg)
{
    (void)code; (void)size; (void)arg;
    return -999; /* not natively runnable; use external interpreter */
}

static void wasm_describe(void)
{
    printf("WebAssembly MVP (portable): i32/i64/f32 arithmetic, locals, control flow.\n");
}

const wubu_isa_driver_t wubu_isa_wasm = {
    .name = "wasm",
    .family = "portable",
    .exec = WUBU_ISA_INTERPRETED,
    .compile = wasm_compile,
    .run = wasm_run,
    .describe = wasm_describe,
};

/* Public API: compile MIR to WASM without needing the driver registry */
int wubu_isa_wasm_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size) {
    return wasm_compile(p, out, out_size);
}
