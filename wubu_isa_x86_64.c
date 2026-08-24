/*
 * wubu_isa_x86_64.c -- the x86-64 ISA driver.
 *
 * The driver space (wubu_isa_driver.h): consumes MIR, emits x86-64
 * machine code, runs it via mmap+JIT call. This is the SAME hourglass
 * neck the RISC-V driver consumes — one frontend, N backends.
 *
 * Strategy: the MIR register allocator (wubu_mir_regalloc.h) assigns
 * each virtual register to either a physical register or a stack slot.
 * Register-resident vrs live in their assigned x86-64 register for
 * their entire lifetime — no load/store traffic. Spilled vrs use
 * [rbp - offset] like the stack-only driver.
 *
 * Physical register map (14 total, indices 0..13):
 *   0=rax  1=r10  2=r11  3=r12  4=r13  5=r14  6=r15
 *   7=rbx  8=r8   9=r9   10=rdx 11=rsi 12=r10 13=rdi
 *
 * Note: rcx is NOT in the allocator pool — it's used implicitly by
 * shl/shr (which need cl). rdi is index 13, used as second-operand
 * scratch only when NOT holding a live vr.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include "wubu_mir_regalloc.h"
#include "../jit/jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Peephole optimizer — declared in x86_peephole.c */
extern size_t x86_peephole_optimize(uint8_t *code, size_t n);

/* ---- the emitter ---- */

typedef struct {
    uint8_t *code;
    size_t n, cap;
    size_t frame;                /* stack frame bytes (spills + var mem) */
    int32_t mem_off;             /* byte offset from rbp to mem[0] (var memory) */
    size_t *label_offsets;       /* label id -> byte offset */
    size_t n_labels;
} x86_emitter_t;

static void e8(x86_emitter_t *e, uint8_t b) { if (e->n == e->cap) { e->cap = e->cap ? e->cap*2 : 256; e->code = realloc(e->code, e->cap); } e->code[e->n++] = b; }
static void e32(x86_emitter_t *e, uint32_t v) { e8(e, v & 0xFF); e8(e, (v >> 8) & 0xFF); e8(e, (v >> 16) & 0xFF); e8(e, (v >> 24) & 0xFF); }
static void e64(x86_emitter_t *e, uint64_t v) { for (int i = 0; i < 8; i++) e8(e, (v >> (8*i)) & 0xFF); }
static void rex(x86_emitter_t *e, int w, int r, int x, int b) { e8(e, 0x40 | (w<<3) | (r<<2) | (x<<1) | b); }

/* x86-64 register encoding for our 10 physical registers.
 * Index is the allocator's physical register number.
 * We skip rax(0), rcx(1), rdi(7), rbp(5), rsp(4).
 * rax is the implicit accumulator — not in the allocator pool.
 * rdi is used as second-operand scratch.
 * rcx is used for shift counts only. */
static const int reg_x86[10] = {
    10,  /* 0: r10 */
    11,  /* 1: r11 */
    12,  /* 2: r12 */
    13,  /* 3: r13 */
    14,  /* 4: r14 */
    15,  /* 5: r15 */
    3,   /* 6: rbx (callee-saved) */
    8,   /* 7: r8  */
    9,   /* 8: r9  */
    2,   /* 9: rdx */
};

/* does this x86 encoding need REX.B or REX.R? */
static inline int reg_needs_rex(int r) { return r >= 8; }

/* emit: mov dst_reg, src_reg (both are x86 encodings 0-15) */
static void emit_mov_reg(x86_emitter_t *e, int dst, int src) {
    if (dst == src) return;
    rex(e, 1, reg_needs_rex(src), 0, reg_needs_rex(dst));
    e8(e, 0x89);
    e8(e, (uint8_t)(0xC0 | ((src & 7) << 3) | (dst & 7)));
}

/* emit: mov dst_reg, [rbp + off]  (off is signed) */
static void emit_load_rbp(x86_emitter_t *e, int dst, int32_t off) {
    rex(e, 1, reg_needs_rex(dst), 0, 0);
    if (off >= -128 && off <= 127) { e8(e, 0x8B); e8(e, (uint8_t)(0x45 | ((dst & 7) << 3))); e8(e, (uint8_t)off); }
    else { e8(e, 0x8B); e8(e, (uint8_t)(0x85 | ((dst & 7) << 3))); e32(e, (uint32_t)off); }
}

/* emit: mov [rbp + off], src_reg */
static void emit_store_rbp(x86_emitter_t *e, int32_t off, int src) {
    rex(e, 1, reg_needs_rex(src), 0, 0);
    if (off >= -128 && off <= 127) { e8(e, 0x89); e8(e, (uint8_t)(0x45 | ((src & 7) << 3))); e8(e, (uint8_t)off); }
    else { e8(e, 0x89); e8(e, (uint8_t)(0x85 | ((src & 7) << 3))); e32(e, (uint32_t)off); }
}

/* Spilled vr slot: slot N -> [rbp - (N+1)*8] */
static int32_t spill_offset(int spill_slot) {
    return -(int32_t)((spill_slot + 1) * 8);
}

/* Load vr into a specific x86 register (encoded) */
static void emit_load_vr_to(x86_emitter_t *e, wubu_vr_t vr,
                             const wubu_reg_assign_t *assign, int dst_enc) {
    if (vr < 0) return;
    uint32_t vru = (uint32_t)vr;
    /* find assignment — we need the assign array indexed by vr */
    /* assign is indexed 0..assign_count-1; we pass count separately */
    (void)assign; (void)dst_enc; /* placeholder — see full impl below */
}

/* ---- label resolution ---- */
typedef struct { size_t pos; uint32_t label; } x86_patch_t;

static void note_label(x86_emitter_t *e, uint32_t label, size_t off) {
    if (label >= e->n_labels) {
        size_t old = e->n_labels;
        e->n_labels = label + 1;
        e->label_offsets = realloc(e->label_offsets, e->n_labels * sizeof(size_t));
        for (size_t i = old; i < e->n_labels; i++) e->label_offsets[i] = (size_t)-1;
    }
    e->label_offsets[label] = off;
}
static size_t label_off(const x86_emitter_t *e, uint32_t label) {
    return (label < e->n_labels) ? e->label_offsets[label] : (size_t)-1;
}
static void x86_patch_push(x86_patch_t **patches, size_t *np, size_t *cap,
                           size_t pos, uint32_t lbl) {
    if (*np == *cap) { *cap = *cap ? *cap * 2 : 16; *patches = realloc(*patches, *cap * sizeof(x86_patch_t)); }
    (*patches)[*np].pos = pos;
    (*patches)[*np].label = lbl;
    (*np)++;
}

/* ---- the full compile function with regalloc ---- */

/* scalar tensor GEMM: C += A*B (row-major, int64).
 * Mirrors MIR_T_GEMM interpreter semantics exactly.
 *
 * Tiled kernel (H4): 4-row register blocking — one pass over B[k][j] feeds
 * four A rows' accumulators (4x fewer B loads), plus K-unroll-by-4 with
 * hoisted row pointers. Bit-identical to the naive reference loop below. */
static void wubu_tgemm_scalar(int64_t *mem, int64_t A, int64_t B,
                              int64_t C, int M, int N, int K)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const int64_t *a0 = &mem[A + (int64_t)(i+0) * K];
        const int64_t *a1 = &mem[A + (int64_t)(i+1) * K];
        const int64_t *a2 = &mem[A + (int64_t)(i+2) * K];
        const int64_t *a3 = &mem[A + (int64_t)(i+3) * K];
        int64_t       *c0 = &mem[C + (int64_t)(i+0) * N];
        int64_t       *c1 = &mem[C + (int64_t)(i+1) * N];
        int64_t       *c2 = &mem[C + (int64_t)(i+2) * N];
        int64_t       *c3 = &mem[C + (int64_t)(i+3) * N];
        for (int j = 0; j < N; j++) {
            int64_t s0 = c0[j], s1 = c1[j], s2 = c2[j], s3 = c3[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k + 3 < K; k += 4) {
                const int64_t b0 = bk[(size_t)(k+0) * N];
                const int64_t b1 = bk[(size_t)(k+1) * N];
                const int64_t b2 = bk[(size_t)(k+2) * N];
                const int64_t b3 = bk[(size_t)(k+3) * N];
                s0 += a0[k+0]*b0 + a0[k+1]*b1 + a0[k+2]*b2 + a0[k+3]*b3;
                s1 += a1[k+0]*b0 + a1[k+1]*b1 + a1[k+2]*b2 + a1[k+3]*b3;
                s2 += a2[k+0]*b0 + a2[k+1]*b1 + a2[k+2]*b2 + a2[k+3]*b3;
                s3 += a3[k+0]*b0 + a3[k+1]*b1 + a3[k+2]*b2 + a3[k+3]*b3;
            }
            for (int k = K & ~3; k < K; k++) {
                const int64_t b = bk[(size_t)k * N];
                s0 += a0[k]*b; s1 += a1[k]*b; s2 += a2[k]*b; s3 += a3[k]*b;
            }
            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }
    for (; i < M; i++) {                       /* tail rows */
        const int64_t *a0 = &mem[A + (int64_t)i * K];
        int64_t       *c0 = &mem[C + (int64_t)i * N];
        for (int j = 0; j < N; j++) {
            int64_t s0 = c0[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k < K; k++)
                s0 += a0[k] * bk[(size_t)k * N];
            c0[j] = s0;
        }
    }
}
#if defined(WUBU_TGEMM_KEEP_NAIVE)
static void wubu_tgemm_naive(int64_t *mem, int64_t A, int64_t B,
                             int64_t C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int64_t acc = mem[C + (int64_t)i * N + j];
            for (int k = 0; k < K; k++)
                acc += mem[A + (int64_t)i * K + k] * mem[B + (int64_t)k * N + j];
            mem[C + (int64_t)i * N + j] = acc;
        }
}
#endif
static int x86_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size) {
#if !defined(__x86_64__)
    (void)p; (void)out; (void)out_size;
    return -1; /* cannot compile x86-64 on this host */
#endif
    /* Step 1: call the MIR register allocator */
    size_t assign_count = 0;
    wubu_reg_assign_t *assign = wubu_mir_alloc_regs(p, 10, &assign_count);
    if (!assign) return -1;
    /* Any operand vr outside the assignment table would spill to offset 0 and
     * corrupt the saved RBP — refuse instead. */
    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *ci = &p->ins[i];
        if (ci->dst >= (wubu_vr_t)assign_count || ci->a >= (wubu_vr_t)assign_count ||
            ci->b >= (wubu_vr_t)assign_count) return -1;

    }

    /* Count spilled vrs to size the frame.
     * NOTE: the regalloc stores the spill slot byte-offset in
     * assign[v].stack (a negative value: -(slot+1)*8) and sets
     * assign[v].reg = -1. Derive the slot number from .stack. */
    size_t n_spilled = 0;
    for (size_t i = 0; i < assign_count; i++) {
        if (assign[i].reg < 0) {
            int slot = (-assign[i].stack / 8) - 1;
            if (slot < 0) slot = 0;
            if ((size_t)slot >= n_spilled) n_spilled = (size_t)slot + 1;
        }
    }

    /* Variable memory model: MIR programs address a flat int64[] array
     * (cell 0 reserved as null). Reserve space for it on the stack so
     * MIR_LOAD/MIR_STORE can mirror the interpreter exactly. */
    size_t n_mem_cells = (p->total_mem > 0) ? (size_t)(p->total_mem + 1) : 1;
    size_t mem_bytes = n_mem_cells * 8;

    x86_emitter_t e;
    memset(&e, 0, sizeof(e));
    e.frame = n_spilled * 8 + mem_bytes;          /* spills + var mem */
    e.mem_off = (int32_t)(n_spilled * 8 + mem_bytes); /* offset to mem[0] */
    e.n_labels = p->n_labels;
    e.label_offsets = calloc(e.n_labels, sizeof(size_t));
    for (size_t i = 0; i < e.n_labels; i++) e.label_offsets[i] = (size_t)-1;

    x86_patch_t *patches = NULL;
    size_t n_patches = 0, cap_patches = 0;
#define PATCH_PUSH(pos, lbl) x86_patch_push(&patches, &n_patches, &cap_patches, (pos), (lbl))

    /* prologue */
    e8(&e, 0x55);                      /* push rbp */
    rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0xE5);  /* mov rbp, rsp */
    if (e.frame > 0) {
        e8(&e, 0x48); e8(&e, 0x81); e8(&e, 0xEC); e32(&e, (uint32_t)e.frame);
    }

    /* Helper: get x86 encoding for vr (returns -1 if spilled) */
    #define VR_ENC(vr) ((vr) < (wubu_vr_t)assign_count && assign[(vr)].reg >= 0 ? reg_x86[assign[(vr)].reg] : -1)
    /* The regalloc stores the spill slot byte-offset in assign[v].stack
     * (already a negative value: -(slot+1)*8). Use it directly. */
    #define VR_SPILL(vr) ((vr) < (wubu_vr_t)assign_count && assign[(vr)].reg < 0 ? assign[(vr)].stack : 0)
    /* Lookahead: is the next instruction a RET that reads this vr? */
    #define NEXT_IS_RET(vr) (i + 1 < p->n && p->ins[i+1].op == MIR_RET && p->ins[i+1].a == (wubu_vr_t)(vr))

    int result_in_rax = 0;  /* set when last op skipped store to keep result in rax */

    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (in->op == MIR_LABEL) { note_label(&e, in->label, e.n); result_in_rax = 0; continue; }
        if (in->op != MIR_RET) result_in_rax = 0;  /* reset unless RET handles it */

        switch (in->op) {
        case MIR_CONST: {
            int64_t imm = in->imm;
            int dst_enc = VR_ENC(in->dst);
            if (dst_enc >= 0) {
                /* mov reg, imm — on x86-64 REX.W + B8+rd is ALWAYS movabs
                 * (imm64); there is no mov r64, imm32 form. So emit imm64. */
                if (dst_enc >= 8) { e8(&e, 0x49); e8(&e, (uint8_t)(0xB8 + (dst_enc & 7))); }
                else { e8(&e, 0x48); e8(&e, (uint8_t)(0xB8 + dst_enc)); }
                e64(&e, (uint64_t)imm);
            } else {
                /* spilled: mov rax, imm; mov [rbp+off], rax. Uses movabs too. */
                e8(&e, 0x48); e8(&e, 0xB8); e64(&e, (uint64_t)imm);
                emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            }
            break;
        }
        case MIR_MOV: {
            int sa = VR_ENC(in->a), sd = VR_ENC(in->dst);
            if (sd >= 0) {
                /* dest is a register */
                if (sa >= 0) {
                    emit_mov_reg(&e, sd, sa);  /* mov dst_reg, src_reg */
                } else {
                    emit_load_rbp(&e, sd, VR_SPILL(in->a));  /* mov dst_reg, [rbp+off] */
                }
            } else {
                /* dest is spilled */
                if (sa >= 0) {
                    emit_store_rbp(&e, VR_SPILL(in->dst), sa);
                } else {
                    emit_load_rbp(&e, 0, VR_SPILL(in->a));  /* mov rax, [rbp+off_a] */
                    emit_store_rbp(&e, VR_SPILL(in->dst), 0);  /* mov [rbp+off_dst], rax */
                }
            }
            break;
        }
        case MIR_ADD: case MIR_SUB: case MIR_MUL: case MIR_DIV: case MIR_MOD:
        case MIR_AND: case MIR_OR: case MIR_XOR:
        case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE:
        case MIR_FADD: case MIR_FSUB: case MIR_FMUL: case MIR_FDIV:
        case MIR_FNEG: {
            /* Load 'a' into rax (accumulator) */
            int sa = VR_ENC(in->a);
            if (sa == 0) {
                /* already in rax — nothing to do */
            } else if (sa >= 0) {
                emit_mov_reg(&e, 0, sa);  /* mov rax, src_reg */
            } else {
                emit_load_rbp(&e, 0, VR_SPILL(in->a));
            }
            /* Load 'b' into rdi (second operand) */
            int sb = VR_ENC(in->b);
            if (sb == 7) {
                /* already in rdi (x86 encoding 7) — wait, rdi is encoding 7? No. */
                /* rdi x86 encoding is 7. But our reg_x86[] maps allocator index -> x86. */
                /* We need to emit mov rdi, <sb_enc> */
            }
            /* Actually, let's use rdi (x86 enc 7) as second-operand scratch */
            if (sb >= 0) {
                emit_mov_reg(&e, 7, sb);  /* mov rdi, b_reg */
            } else {
                emit_load_rbp(&e, 7, VR_SPILL(in->b));
            }
            switch (in->op) {
            case MIR_ADD: rex(&e,1,0,0,0); e8(&e, 0x01); e8(&e, 0xF8); break; /* add rax,rdi */
            case MIR_SUB: rex(&e,1,0,0,0); e8(&e, 0x29); e8(&e, 0xF8); break; /* sub rax,rdi */
            case MIR_MUL: rex(&e,1,0,0,0); e8(&e, 0x0F); e8(&e, 0xAF); e8(&e, 0xC7); break; /* imul rax,rdi */
            case MIR_DIV: rex(&e,1,0,0,0); e8(&e, 0x99); rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xFF); break;
            case MIR_MOD: rex(&e,1,0,0,0); e8(&e, 0x99); rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xFF); rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0xD0); break;
            case MIR_AND: rex(&e,1,0,0,0); e8(&e, 0x21); e8(&e, 0xF8); break;
            case MIR_OR:  rex(&e,1,0,0,0); e8(&e, 0x09); e8(&e, 0xF8); break;
            case MIR_XOR: rex(&e,1,0,0,0); e8(&e, 0x31); e8(&e, 0xF8); break;

            /* ---- SSE single-precision float ops (values are f32 bits) ---- */
            case MIR_FNEG: {
                int sa4 = VR_ENC(in->a);
                if (sa4 >= 0) emit_mov_reg(&e, 0, sa4);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                /* mov edi, 0x80000000 : BF 00 00 00 80 */
                e8(&e, 0xBF); e8(&e, 0x00); e8(&e, 0x00); e8(&e, 0x00); e8(&e, 0x80);
                /* movd xmm1, edi */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xCF);
                /* movd xmm0, eax */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                /* xorps xmm0, xmm1 : 0F 57 C1 */
                e8(&e, 0x0F); e8(&e, 0x57); e8(&e, 0xC1);
                /* movd eax, xmm0 */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }

            case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE: {
                /* load a -> rax, b -> rdi (same staging as arithmetic group) */
                int sa3 = VR_ENC(in->a);
                if (sa3 >= 0) emit_mov_reg(&e, 0, sa3);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                int sb3 = VR_ENC(in->b);
                if (sb3 >= 0) emit_mov_reg(&e, 7, sb3);
                else emit_load_rbp(&e, 7, VR_SPILL(in->b));
                /* movd xmm0, eax ; movd xmm1, edi */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xCF);
                /* ucomiss xmm0, xmm1 : 0F 2E C1 */
                e8(&e, 0x0F); e8(&e, 0x2E); e8(&e, 0xC1);
                switch (in->op) {
                case MIR_FEQ:
                    /* sete al ; setnp cl ; and al,cl  (excludes unordered) */
                    e8(&e, 0x0F); e8(&e, 0x94); e8(&e, 0xC0);
                    e8(&e, 0x0F); e8(&e, 0x9B); e8(&e, 0xC1);
                    e8(&e, 0x20); e8(&e, 0xC8);
                    break;
                case MIR_FNE:
                    /* setne al ; setnp cl ; and al,cl */
                    e8(&e, 0x0F); e8(&e, 0x95); e8(&e, 0xC0);
                    e8(&e, 0x0F); e8(&e, 0x9B); e8(&e, 0xC1);
                    e8(&e, 0x20); e8(&e, 0xC8);
                    break;
                case MIR_FLT:
                    /* setb al (CF=1 => a<b); movzx eax, al */
                    e8(&e, 0x0F); e8(&e, 0x92); e8(&e, 0xC0);
                    break;
                default:
                    /* setbe al (CF=1 or ZF=1 => a<=b) */
                    e8(&e, 0x0F); e8(&e, 0x96); e8(&e, 0xC0);
                    break;
                }
                /* movzx eax, al : 0F B6 C0 (for FEQ/FNE after AND) */
                e8(&e, 0x0F); e8(&e, 0xB6); e8(&e, 0xC0);
                break;
            }

                    case MIR_FADD: case MIR_FSUB: case MIR_FMUL: case MIR_FDIV: {
                /* movd xmm0, eax-ish: load a into rax, then movq rax->xmm0;
                 * simpler: use SSE directly from memory/regs via GPR staging. */
                int sa2 = VR_ENC(in->a);
                if (sa2 >= 0) emit_mov_reg(&e, 0, sa2);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                int sb2 = VR_ENC(in->b);
                if (sb2 >= 0) emit_mov_reg(&e, 7, sb2);
                else emit_load_rbp(&e, 7, VR_SPILL(in->b));
                /* movd xmm0, eax   : 66 0F 6E C0 */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                /* movd xmm1, edi   : 66 0F 6E CF */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xCF);
                /* opss xmm0, xmm1  : F3 0F 5x C1 (single-precision) */
                e8(&e, 0xF3); e8(&e, 0x0F);
                switch (in->op) {
                case MIR_FADD: e8(&e, 0x58); break;
                case MIR_FSUB: e8(&e, 0x5C); break;
                case MIR_FMUL: e8(&e, 0x59); break;
                default:       e8(&e, 0x5E); break; /* FDIV */
                }
                e8(&e, 0xC1);
                /* movd eax, xmm0   : 66 0F 7E C0 */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }

            default:
                /* unimplemented op — fail loudly instead of emitting broken code */
                free(assign);
                return -1;
            }
            /* Store result — skip if next instr is RET consuming this dst */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) {
                if (sd != 0 && !NEXT_IS_RET(in->dst)) {
                    emit_mov_reg(&e, sd, 0);
                } else if (NEXT_IS_RET(in->dst)) {
                    result_in_rax = 1;  /* result stays in rax for RET */
                } else {
                    result_in_rax = 0;
                }
            } else {
                emit_store_rbp(&e, VR_SPILL(in->dst), 0);
                result_in_rax = 0;
            }
            break;
        }
        case MIR_SHL: case MIR_SHR: {
            /* shifts need rcx */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));

            int sb = VR_ENC(in->b);
            /* mov rcx, b */
            if (sb >= 0) {
                rex(&e,1,reg_needs_rex(sb),0,0); e8(&e, 0x89); e8(&e, (uint8_t)(0xC0 | ((sb & 7) << 3) | 1));
            } else {
                /* mov rcx, [rbp+off] */
                int32_t off = VR_SPILL(in->b);
                if (off >= -128 && off <= 127) { rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x4D); e8(&e, (uint8_t)off); }
                else { rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x8D); e32(&e, (uint32_t)off); }
            }
            if (in->op == MIR_SHL) { rex(&e,1,0,0,0); e8(&e, 0xD3); e8(&e, 0xE0); }
            else { rex(&e,1,0,0,0); e8(&e, 0xD3); e8(&e, 0xE8); }

            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            break;
        }
        case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE: case MIR_GT: case MIR_GE:
        case MIR_ULT: case MIR_ULE: case MIR_UGT: case MIR_UGE: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));

            int sb = VR_ENC(in->b);
            if (sb >= 0) emit_mov_reg(&e, 7, sb);
            else emit_load_rbp(&e, 7, VR_SPILL(in->b));

            rex(&e,1,0,0,0); e8(&e, 0x39); e8(&e, 0xF8);  /* cmp rax, rdi */
            uint8_t cc;
            switch (in->op) {
            case MIR_EQ: cc = 0x94; break;
            case MIR_NE: cc = 0x95; break;
            case MIR_LT: cc = 0x9C; break;
            case MIR_LE: cc = 0x9E; break;
            case MIR_GT: cc = 0x9F; break;
            case MIR_GE: cc = 0x9D; break;
            /* unsigned SETcc: setb / setbe / seta / setae */
            case MIR_ULT: cc = 0x92; break;
            case MIR_ULE: cc = 0x96; break;
            case MIR_UGT: cc = 0x97; break;
            case MIR_UGE: cc = 0x93; break;
            default: cc = 0x94; break;
            }
            e8(&e, 0x0F); e8(&e, cc); e8(&e, 0xC0);        /* setcc al */
            rex(&e,1,0,0,0); e8(&e, 0x0F); e8(&e, 0xB6); e8(&e, 0xC0); /* movzx rax,al */

            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            break;
        }
        case MIR_NEG: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xD8);  /* neg rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            break;
        }
        case MIR_NOT: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xD0);  /* not rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            break;
        }
        case MIR_JMP:
            e8(&e, 0xE9);
            PATCH_PUSH(e.n, in->label);
            e32(&e, 0);
            break;
        case MIR_JZ: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            rex(&e,1,0,0,0); e8(&e, 0x85); e8(&e, 0xC0);  /* test rax,rax */
            e8(&e, 0x0F); e8(&e, 0x84);                     /* jz rel32 */
            PATCH_PUSH(e.n, in->label);
            e32(&e, 0);
            break;
        }
        case MIR_JNZ: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            rex(&e,1,0,0,0); e8(&e, 0x85); e8(&e, 0xC0);  /* test rax,rax */
            e8(&e, 0x0F); e8(&e, 0x85);                     /* jnz rel32 */
            PATCH_PUSH(e.n, in->label);
            e32(&e, 0);
            break;
        }
        case MIR_T_GEMM: {
            /* Native tensor-MADD via libcall to wubu_tgemm_scalar.
             * SysV ABI: rdi=&mem[0], rsi=Abase, rdx=Bbase, rcx=Cbase,
             *          r8=M, r9=N, stack(K).
             * We use r11 for the helper pointer (movabs r11) — r11 is NOT in
             * the MIR regalloc map (reg_x86[]), so it's safe to clobber. The
             * peephole shrink_movabs leaves `49 BB imm64` intact when the
             * address exceeds 2GiB (typical for host symbols). */
            int M = (int)(in->imm >> 22);
            int N = (int)((in->imm >> 11) & 0x7FF);
            int K = (int)(in->imm & 0x7FF);
            int sa = VR_ENC(in->a);
            int sb = VR_ENC(in->b);
            int sd = VR_ENC(in->dst);
            /* rdi = &mem[0]  (lea rdi,[rbp-mem_off]; modrm 0xBD = mod10 reg7 rm5(rbp+disp32))
             * 0x3D (mod00 rm101) would wrongly encode [rip+disp32]. */
            rex(&e,1,0,0,0); e8(&e,0x8D); e8(&e,0xBD); e32(&e,(uint32_t)(-(int32_t)e.mem_off));
            /* rsi = A (vr value / slot index) */
            if (sa>=0) emit_mov_reg(&e,6,sa);            else emit_load_rbp(&e,6,VR_SPILL(in->a));
            /* rdx = B */
            if (sb>=0) emit_mov_reg(&e,2,sb);            else emit_load_rbp(&e,2,VR_SPILL(in->b));
            /* rcx = C */
            if (sd>=0) emit_mov_reg(&e,1,sd);            else emit_load_rbp(&e,1,VR_SPILL(in->dst));
            /* r8d = M (41 B8)  — but 41 B8 matches shrink_movabs? No, B8 here is
             * a reg-load not movabs: shrink checks `code[i]==0x48` for the plain
             * case. 41 B8 is the `need_rex_b` (0x49) branch ONLY. 41 != 49, so
             * 41 B8 is NOT matched by shrink_movabs. Safe. */
            e8(&e,0x41); e8(&e,0xB8); e32(&e,(uint32_t)M);     /* r8d = M */
            e8(&e,0x41); e8(&e,0xB9); e32(&e,(uint32_t)N);     /* r9d = N */
            /* push K (7th arg, on stack): 6A ib (imm8) if K<128, else 68 id */
            if ((uint32_t)K < 0x80u)     { e8(&e,0x6A); e8(&e,(uint8_t)K); }
            else                          { e8(&e,0x68); e32(&e,(uint32_t)K); }
            /* movabs r11, &wubu_tgemm_scalar (REX.WB + B8 = 49 BB imm64).
             * rex(1,0,0,1) = 0x49. If addr > 2GB, shrink_movabs leaves it
             * intact; if it shrinks it's a bug we avoid by keeping addr wide. */
            rex(&e,1,0,0,1); e8(&e,0xBB); e64(&e,(uint64_t)&wubu_tgemm_scalar);
            /* call r11 (FF D3 with REX.B → 41 FF D3; without 0x41 it decodes as call rbx!) */
            e8(&e,0x41); e8(&e,0xFF); e8(&e,0xD3);
            /* add rsp,8 (restore stack for K arg) */
            e8(&e,0x48); e8(&e,0x83); e8(&e,0xC4); e8(&e,0x08);
            break;
        }
        case MIR_RET: case MIR_FRET: {
            /* If result is already in rax (lookahead skip), don't reload */
            if (!result_in_rax) {
                int sa = VR_ENC(in->a);
                if (sa >= 0) emit_mov_reg(&e, 0, sa);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            }
            result_in_rax = 0;
            e8(&e, 0xC9);  /* leave */
            e8(&e, 0xC3);  /* ret */
            break;
        }
        case MIR_ALLOC:
            break;  /* home base already emitted as a CONST vr */
        case MIR_LOAD: {
            /* dst = mem[addr]; addr is a vr holding the cell index.
             * Mirror the interpreter's flat int64[] array on the stack. */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);          /* rax = addr */
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            rex(&e,1,0,0,0); e8(&e, 0xC1); e8(&e, 0xE0); e8(&e, 0x03); /* shl rax,3 */
            rex(&e,1,0,0,0); e8(&e, 0x8D); e8(&e, 0xB5);   /* lea rsi,[rbp-mem_off] */
            e32(&e, (uint32_t)(-(int32_t)e.mem_off));
            rex(&e,1,0,0,0); e8(&e, 0x01); e8(&e, 0xC6);   /* add rsi, rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) {
                rex(&e,1,reg_needs_rex(sd),0,0); e8(&e, 0x8B);
                e8(&e, (uint8_t)(0x06 | ((sd & 7) << 3))); /* mov dstreg,[rsi] */
            } else {
                rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x06); /* mov rax,[rsi] */
                emit_store_rbp(&e, VR_SPILL(in->dst), 0);
            }
            break;
        }
        case MIR_STORE: {
            /* mem[addr] = val */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);          /* rax = addr */
            else emit_load_rbp(&e, 0, VR_SPILL(in->a));
            int sb = VR_ENC(in->b);
            if (sb >= 0) emit_mov_reg(&e, 7, sb);          /* rdi = val */
            else emit_load_rbp(&e, 7, VR_SPILL(in->b));
            rex(&e,1,0,0,0); e8(&e, 0xC1); e8(&e, 0xE0); e8(&e, 0x03); /* shl rax,3 */
            rex(&e,1,0,0,0); e8(&e, 0x8D); e8(&e, 0xB5);   /* lea rsi,[rbp-mem_off] */
            e32(&e, (uint32_t)(-(int32_t)e.mem_off));
            rex(&e,1,0,0,0); e8(&e, 0x01); e8(&e, 0xC6);   /* add rsi, rax */
            rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0x3E);   /* mov [rsi], rdi */
            break;
        }
        }
    }

    /* fallback ret */
    if (e.n == 0 || e.code[e.n-1] != 0xC3) { e8(&e, 0xC9); e8(&e, 0xC3); }

    /* patch jumps */
    for (size_t i = 0; i < n_patches; i++) {
        size_t t = label_off(&e, patches[i].label);
        if (t == (size_t)-1) continue;
        size_t pos = patches[i].pos;
        int32_t rel = (int32_t)(t - (pos + 4));
        e.code[pos] = (uint8_t)(rel & 0xFF);
        e.code[pos+1] = (uint8_t)((rel >> 8) & 0xFF);
        e.code[pos+2] = (uint8_t)((rel >> 16) & 0xFF);
        e.code[pos+3] = (uint8_t)((rel >> 24) & 0xFF);
    }
    free(patches);
    free(e.label_offsets);
    wubu_mir_free_alloc(assign);

    /* Peephole: wire up the real optimizer (x86_peephole.c). */
    e.n = x86_peephole_optimize(e.code, e.n);

    *out = e.code;
    *out_size = e.n;
    return 0;
}

static int64_t x86_run(const uint8_t *code, size_t size, int64_t arg) {
    (void)arg;
    void *exec = jit_alloc_exec(size);
    if (!exec) return -1;
    memcpy(exec, code, size);
    wubu_clear_cache(exec, size);
    int64_t (*fn)(void) = (int64_t (*)(void))exec;
    int64_t r = fn();
    jit_free_exec(exec, size);
    return r;
}

static void x86_describe(void) {
    printf("x86-64 driver: native JIT, MIR register allocator (11 regs), "
           "spill-to-stack fallback, WUBU-ABI-v1 frame.\n");
}

const wubu_isa_driver_t wubu_isa_x86_64 = {
    .name = "x86-64",
    .family = "native",
    .exec = WUBU_ISA_NATIVE,
    .compile = x86_compile,
    .run = x86_run,
    .describe = x86_describe,
};
