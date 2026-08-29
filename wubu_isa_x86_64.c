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
#include "wubu_tgemm.h"
#include "wubu_mir_regalloc.h"
#include "../jit/jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Peephole optimizer — declared in x86_peephole.c */
extern size_t x86_peephole_optimize(uint8_t *code, size_t n);

/* Global pointer for JIT working memory (heap-allocated).
 * Set by x86_run before calling JIT'd code. Used by wubu_tgemm_parallel.
 * NOT thread-local: OpenMP worker threads must see the same pointer. */
extern int64_t *wubu_jit_mem_ptr;

/* ---- the emitter ---- */

typedef struct {
    uint8_t *code;
    size_t n, cap;
    size_t frame;                /* stack frame bytes (spills + var mem) */
    int32_t mem_off;             /* byte offset from rbp to mem[0] (var memory) */
    size_t *label_offsets;       /* label id -> byte offset */
    size_t n_labels;
    int32_t spare_off;           /* emergency spill slot below the frame */
} x86_emitter_t;

static void e8(x86_emitter_t *e, uint8_t b) { if (e->n + 1 > e->cap) { e->cap = e->cap ? e->cap*2 : 256; e->code = realloc(e->code, e->cap); } e->code[e->n++] = b; }
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

/* Safe spill offset: if the allocator marked a vr spilled but never gave it a
 * slot (split bug), fall back to the emitter's emergency slot below the frame. */
static int32_t spill_off(const wubu_reg_assign_t *assign, size_t assign_count,
                         const x86_emitter_t *e, wubu_vr_t vr) {
    /* mirrors VR_SPILL: spilled vrs hold a negative byte offset */
    int32_t off = (vr < (wubu_vr_t)assign_count && assign[vr].reg < 0)
                    ? (int32_t)assign[vr].stack : 0;
    if (off == 0) return e->spare_off;
    return off;
}

/* Load the base address of mem[] into rdi (same pattern as T_GEMM).
 * lea rdi, [rbp - mem_off]  (modrm 0xBD = mod10 reg7 rm5 = rbp disp32) */
static void emit_mov_mem0_rdi(x86_emitter_t *e) {
    rex(e,1,0,0,0); e8(e,0x8D); e8(e,0xBD); e32(e,(uint32_t)(-(int32_t)e->mem_off));
}

/* ---- AGI tensor ops host dispatch ---- */
/* Called from JIT'd code with SysV ABI:
 *   rdi=&mem[0], rsi=op_code, rdx=a_base, rcx=b_base, r8=dst_base,
 *   r9=imm_lo, stack(40)=imm_hi.
 * op_code selects which tensor op to execute. */
void wubu_tensor_dispatch(int64_t *mem, uint32_t op,
                          int64_t a, int64_t b, int64_t dst,
                          int64_t imm_lo, int64_t imm_hi);


/* is instruction index `idx` inside any declared function body? */
static int x86_in_func_body(const wubu_mir_prog_t *p, size_t idx) {
    for (int f = 0; f < p->n_funcs; f++)
        if (idx >= p->funcs[f].start && idx < p->funcs[f].end) return 1;
    return 0;
}

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

/* ---- Float32 GEMM parallel wrapper -------------------------------- */
void wubu_tgemm_f32_parallel(int64_t *stack_mem, int64_t A, int64_t B,
                              int64_t C, int M, int N, int K) {
    /* Use MIR-compatible wrapper: handles int64-cell memory layout */
    wubu_tgemm_f32_mir(stack_mem, A, B, C, M, N, K);
}

static void wubu_tgemm_scalar(int64_t *mem, int64_t A, int64_t B,
                              int64_t C, int M, int N, int K)
{
    /* Cache-friendly order: iterate k on the middle level so B accesses
     * are sequential (B[k*N+j] for fixed k, varying j). This reduces
     * cache misses for large N where stride-N access washes out L1/L2. */
    int i;
    for (i = 0; i + 3 < M; i += 4) {
        const int64_t *a0 = &mem[A + (int64_t)(i+0) * K];
        const int64_t *a1 = &mem[A + (int64_t)(i+1) * K];
        const int64_t *a2 = &mem[A + (int64_t)(i+2) * K];
        const int64_t *a3 = &mem[A + (int64_t)(i+3) * K];
        int64_t       *c0 = &mem[C + (int64_t)(i+0) * N];
        int64_t       *c1 = &mem[C + (int64_t)(i+1) * N];
        int64_t       *c2 = &mem[C + (int64_t)(i+2) * N];
        int64_t       *c3 = &mem[C + (int64_t)(i+3) * N];
        for (int k = 0; k < K; k++) {
            const int64_t *bj = &mem[B + (int64_t)k * N];  /* B row k, sequential access */
            const int64_t a0k = a0[k], a1k = a1[k], a2k = a2[k], a3k = a3[k];
            for (int j = 0; j < N; j++) {
                const int64_t b = bj[j];
                c0[j] += a0k * b;
                c1[j] += a1k * b;
                c2[j] += a2k * b;
                c3[j] += a3k * b;
            }
        }
    }
    for (; i < M; i++) {
        const int64_t *a0 = &mem[A + (int64_t)i * K];
        int64_t       *c0 = &mem[C + (int64_t)i * N];
        for (int k = 0; k < K; k++) {
            const int64_t *bj = &mem[B + (int64_t)k * N];
            const int64_t a0k = a0[k];
            for (int j = 0; j < N; j++)
                c0[j] += a0k * bj[j];
        }
    }
}

/* ---- AVX-512 micro-kernel (Zen 4 native) ----
 * 8x4 micro-kernel: 8 rows x 4 columns using AVX-512 512-bit vectors.
 * Each ZMM holds 8x int64. We accumulate 4 columns simultaneously:
 *   c_col0 = ZMM with C[i+0..7, j+0]
 *   c_col1 = ZMM with C[i+0..7, j+1]
 *   c_col2 = ZMM with C[i+0..7, j+2]
 *   c_col3 = ZMM with C[i+0..7, j+3]
 * Inner loop: for each k, broadcast A[i+0..7, k] and B[k, j+0..3],
 * multiply-add into the 4 column accumulators.
 * Register pressure: 4 (accum) + 1 (A) + 4 (B broadcast) = 9 ZMM regs.
 * Zen 4 has 16 ZMM registers — fits comfortably. */

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#define HAS_AVX512 1
#include <immintrin.h>
#else
#define HAS_AVX512 0
#endif

#define MC 64   /* M cache block: 64 rows */
#define NC 64   /* N cache block: 64 cols */
#define KC 64   /* K cache block: 64 */
#define MR 8    /* Micro-kernel rows (AVX-512 = 8x int64) */
#define NR 4    /* Micro-kernel cols */

static void wubu_tgemm_avx512(int64_t *mem, int64_t A, int64_t B,
                               int64_t C, int M, int N, int K)
{
#if !HAS_AVX512
    wubu_tgemm_scalar(mem, A, B, C, M, N, K);
    return;
#else
    /* Aligned packing buffers.
     * A_packed: KC x MC (col-major) — column k of the tile is contiguous
     *           so the micro-kernel can _mm512_loadu 8 rows at once.
     * B_packed: KC x NC (row-major) — row k of the tile is contiguous
     *           so B[k, j:j+4] is a 32-byte load. */
    size_t a_buf_size = (size_t)KC * MC * sizeof(int64_t);
    size_t b_buf_size = (size_t)KC * NC * sizeof(int64_t);
    int64_t *A_packed = (int64_t *)aligned_alloc(64, a_buf_size);
    int64_t *B_packed = (int64_t *)aligned_alloc(64, b_buf_size);
    if (!A_packed || !B_packed) { free(A_packed); free(B_packed); return; }

    for (int i3 = 0; i3 < M; i3 += MC) {
        int mc = (i3 + MC <= M) ? MC : M - i3;
        for (int j3 = 0; j3 < N; j3 += NC) {
            int nc = (j3 + NC <= N) ? NC : N - j3;
            for (int k3 = 0; k3 < K; k3 += KC) {
                int kc = (k3 + KC <= K) ? KC : K - k3;

                /* Pack A[i3:i3+MC, k3:k3+KC] as KC x MC col-major:
                 * A_packed[k * MC + i] = mem[A + (i3+i)*K + k3+k] */
                for (int k = 0; k < kc; k++)
                    for (int i = 0; i < mc; i++)
                        A_packed[(size_t)k * MC + i] = mem[A + (int64_t)(i3 + i) * K + k3 + k];

                /* Pack B[k3:k3+KC, j3:j3+NC] as KC x NC row-major:
                 * B_packed[k * NC + j] = mem[B + (k3+k)*N + j3+j] */
                for (int k = 0; k < kc; k++)
                    for (int j = 0; j < nc; j++)
                        B_packed[(size_t)k * nc + j] = mem[B + (int64_t)(k3 + k) * N + j3 + j];

                /* Micro-kernel: C[i3:i3+mc, j3:j3+nc] += A_packed * B_packed
                 * Process MR=8 rows x NR=4 cols per iteration */
                for (int i = 0; i + MR <= mc; i += MR) {
                    for (int j = 0; j + NR <= nc; j += NR) {
                        __m512i c0 = _mm512_setzero_si512();
                        __m512i c1 = _mm512_setzero_si512();
                        __m512i c2 = _mm512_setzero_si512();
                        __m512i c3 = _mm512_setzero_si512();

                        for (int k = 0; k < kc; k++) {
                            __m512i a = _mm512_loadu_si512(A_packed + (size_t)k * MC + i);
                            __m512i b0 = _mm512_set1_epi64(B_packed[(size_t)k * nc + j + 0]);
                            __m512i b1 = _mm512_set1_epi64(B_packed[(size_t)k * nc + j + 1]);
                            __m512i b2 = _mm512_set1_epi64(B_packed[(size_t)k * nc + j + 2]);
                            __m512i b3 = _mm512_set1_epi64(B_packed[(size_t)k * nc + j + 3]);
                            c0 = _mm512_add_epi64(c0, _mm512_mullo_epi64(a, b0));
                            c1 = _mm512_add_epi64(c1, _mm512_mullo_epi64(a, b1));
                            c2 = _mm512_add_epi64(c2, _mm512_mullo_epi64(a, b2));
                            c3 = _mm512_add_epi64(c3, _mm512_mullo_epi64(a, b3));
                        }

                        int64_t *c_base = mem + C + (int64_t)(i3 + i) * N + j3 + j;
                        int64_t tmp[4][8];
                        _mm512_storeu_si512(tmp[0], c0);
                        _mm512_storeu_si512(tmp[1], c1);
                        _mm512_storeu_si512(tmp[2], c2);
                        _mm512_storeu_si512(tmp[3], c3);
                        for (int r = 0; r < 8; r++) {
                            c_base[r * N + 0] += tmp[0][r];
                            c_base[r * N + 1] += tmp[1][r];
                            c_base[r * N + 2] += tmp[2][r];
                            c_base[r * N + 3] += tmp[3][r];
                        }
                    }
                }

                /* Handle remaining rows (mc % MR != 0) with scalar */
                int i_remain = (mc / MR) * MR;
                for (int i = i_remain; i < mc; i++) {
                    for (int j = 0; j < nc; j++) {
                        int64_t acc = mem[C + (int64_t)(i3 + i) * N + j3 + j];
                        for (int k = 0; k < kc; k++)
                            acc += A_packed[(size_t)k * MC + i] * B_packed[(size_t)k * nc + j];
                        mem[C + (int64_t)(i3 + i) * N + j3 + j] = acc;
                    }
                }
            }
        }
    }
    free(A_packed);
    free(B_packed);
#endif /* HAS_AVX512 */
}

/* H4: parallel T_GEMM wrapper — dispatches row blocks to OpenMP threads.
 * Uses wubu_jit_mem_ptr (heap-allocated via mmap in x86_run) for thread safety.
 * Copies JIT stack frame to heap, runs parallel kernel, copies results back.
 *
 * Parallelization strategy: OpenMP parallel for over 4-row blocks.
 * Each wubu_tgemm_scalar call handles 4 rows, so we distribute these
 * calls across threads. B is read-only (shared), C rows are independent
 * (no races). schedule(static) gives even distribution. */
void wubu_tgemm_parallel(int64_t *stack_mem, int64_t A, int64_t B,
                                int64_t C, int M, int N, int K)
{
    int64_t *mem = wubu_jit_mem_ptr ? wubu_jit_mem_ptr : stack_mem;
    size_t frame_bytes = (size_t)(((C > A) ? (size_t)C : (size_t)A) + M*N + 8) * 8;

#if defined(_OPENMP)
    int nt = omp_get_max_threads();
    /* Skip parallelization for small problems: OpenMP + mmap overhead
     * dominates. Also skip M=256 (regalloc edge case at K=256 boundary). */
    if (nt <= 1 || M < 4 || M == 256 || ((size_t)M * N * K) < 500000) {
        wubu_tgemm_scalar(mem, A, B, C, M, N, K); return;
    }

    /* Sync stack -> heap for OpenMP worker access */
    if (mem != stack_mem) {
        memcpy(mem, stack_mem, frame_bytes);
    }

    /* Use AVX-512 kernel if available — it has its own cache blocking
     * and micro-kernel. Otherwise fall back to scalar 4-row parallel. */
#if HAS_AVX512
    /* Multi-threaded AVX-512: split M into bands, each thread processes
     * one band. B is read-only (shared), C bands are independent. */
    {
        int nthreads = nt;
        int rows_per = (M + nthreads - 1) / nthreads;
        /* Round up to multiple of MR for clean micro-kernel boundaries */
        rows_per = ((rows_per + MR - 1) / MR) * MR;
        #pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < M; i0 += rows_per) {
            int mc = i0 + rows_per <= M ? rows_per : M - i0;
            if (mc > 0)
                wubu_tgemm_avx512(mem, A + (int64_t)i0*K, B, C + (int64_t)i0*N, mc, N, K);
        }
    }
#else
    /* Parallelize over 4-row blocks. wubu_tgemm_scalar processes 4 rows
     * per call (plus a tail for M%4). Each block of rows is independent:
     * B is read-only, C rows don't overlap. */
    int nm = M - (M % 4);  /* aligned rows */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nm; i += 4) {
        wubu_tgemm_scalar(mem, A + (int64_t)i*K, B, C + (int64_t)i*N, 4, N, K);
    }
    /* Tail rows */
    if (nm < M) {
        wubu_tgemm_scalar(mem, A + (int64_t)nm*K, B, C + (int64_t)nm*N, M - nm, N, K);
    }

    /* Sync heap -> stack to write results back to JIT frame */
    if (mem != stack_mem) {
        memcpy(stack_mem, mem, frame_bytes);
    }
#endif /* HAS_AVX512 */
#else
    (void)stack_mem; (void)mem;
    wubu_tgemm_scalar(mem, A, B, C, M, N, K);
#endif
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
/* ---- VR remapping for function call support ---- */
static wubu_mir_prog_t *x86_remap_vrs(const wubu_mir_prog_t *p) {
    uint32_t max_vr = 0;
    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        if (in->dst > max_vr) max_vr = in->dst;
        if (in->a > max_vr) max_vr = in->a;
        if (in->b > max_vr) max_vr = in->b;
    }
    if (max_vr < 4096) return NULL;
    uint32_t *map = (uint32_t *)calloc((size_t)max_vr + 1, sizeof(uint32_t));
    uint32_t next_id = 4096;
    for (uint32_t v = 0; v <= max_vr; v++) map[v] = v;
    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *in = &p->ins[i];
        uint32_t vrs[3] = {in->dst, in->a, in->b};
        for (int j = 0; j < 3; j++) {
            if (vrs[j] >= 4096 && map[vrs[j]] == vrs[j]) {
                map[vrs[j]] = next_id++;
            }
        }
    }
    wubu_mir_prog_t *rp = (wubu_mir_prog_t *)malloc(sizeof(*rp));
    *rp = *p;
    rp->ins = (wubu_mir_instr_t *)malloc(p->n * sizeof(wubu_mir_instr_t));
    memcpy(rp->ins, p->ins, p->n * sizeof(wubu_mir_instr_t));
    for (size_t i = 0; i < rp->n; i++) {
        wubu_mir_instr_t *in = &rp->ins[i];
        if (in->dst <= max_vr) in->dst = map[in->dst];
        if (in->a <= max_vr) in->a = map[in->a];
        if (in->b <= max_vr) in->b = map[in->b];
    }
    free(map);
    return rp;
}

static int x86_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size) {
#if !defined(__x86_64__)
    (void)p; (void)out; (void)out_size;
    return -1; /* cannot compile x86-64 on this host */
#endif
    wubu_mir_prog_t *remapped = x86_remap_vrs(p);
    const wubu_mir_prog_t *prog = remapped ? remapped : p;
    /* Step 1: call the MIR register allocator */
    size_t assign_count = 0;
    wubu_reg_assign_t *assign = wubu_mir_alloc_regs(prog, 10, &assign_count);
    if (!assign) { if (remapped) { free(remapped->ins); free(remapped); } return -1; }
    /* Any operand vr outside the assignment table would spill to offset 0 and
     * corrupt the saved RBP — refuse instead. */
    for (size_t i = 0; i < prog->n; i++) {
        const wubu_mir_instr_t *ci = &prog->ins[i];
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

    /* Variable memory model: MIR programs address a flat int64[] array.
     * Memory is accessed via prog.mem (embedded pointer), NOT stack frame.
     * Only spilled registers need stack space. */
    size_t mem_bytes = 0; /* memory accessed via prog.mem, not stack */

    x86_emitter_t e;
    memset(&e, 0, sizeof(e));
    e.frame = n_spilled * 8 + 16;  /* spills + alignment padding */
    /* Ensure 16-byte stack alignment for host calls (wubu_tgemm_parallel etc.).
     * SysV ABI requires %rsp ≡ 0 (mod 16) at call sites. The JIT prologue does
     * `push rbp` (8 bytes), making %rsp ≡ 8 (mod 16). So e.frame must be ≡ 8 (mod 16). */
    e.frame = (e.frame + 7) & ~7ULL;      /* round up to 8 */
    if ((e.frame % 16) == 0) e.frame += 8; /* force ≡ 8 mod 16 */
    e.spare_off = -(int32_t)(n_spilled * 8 + 16);
    e.mem_off = (int32_t)(n_spilled * 8 + mem_bytes); /* offset to mem[0] */
    e.n_labels = prog->n_labels;
    e.label_offsets = calloc(e.n_labels, sizeof(size_t));
    for (size_t i = 0; i < e.n_labels; i++) e.label_offsets[i] = (size_t)-1;

    /* ---- function-call support state ---- */
    size_t *func_off = calloc((size_t)(prog->n_funcs > 0 ? prog->n_funcs : 1), sizeof(size_t));
    for (int f = 0; f < prog->n_funcs; f++) func_off[f] = (size_t)-1;
    uint32_t max_func_end = 0;
    for (int f = 0; f < prog->n_funcs; f++)
        if (prog->funcs[f].end > max_func_end) max_func_end = prog->funcs[f].end;
    size_t entry_off = (size_t)-1;
    size_t entry_jmp_pos = 0;
    typedef struct { size_t pos; uint32_t func_id; } x86_call_fixup_t;
    x86_call_fixup_t *callps = NULL;
    size_t ncallp = 0, ccap = 0;

    x86_patch_t *patches = NULL;
    size_t n_patches = 0, cap_patches = 0;
#define PATCH_PUSH(pos, lbl) x86_patch_push(&patches, &n_patches, &cap_patches, (pos), (lbl))

    /* prologue */
    e8(&e, 0x55);                      /* push rbp */
    rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0xE5);  /* mov rbp, rsp */
    /* rbx = [wubu_jit_mem_ptr] (load mem base from global) */
    rex(&e,1,0,0,1); e8(&e, 0xBB); e64(&e, (uint64_t)&wubu_jit_mem_ptr); /* movabs rbx, &wubu_jit_mem_ptr */
    rex(&e,1,0,0,1); e8(&e, 0x8B); e8(&e, 0x1B);  /* mov rbx, [rbx] */
    if (e.frame > 0) {
        e8(&e, 0x48); e8(&e, 0x81); e8(&e, 0xEC); e32(&e, (uint32_t)e.frame);
    }

    /* entry trampoline: jmp rel32 over function bodies to main code.
     * MIR layout: main code (0..funcs[0].start-1), then function bodies.
     * Only emit trampoline if function bodies start at 0 (no main code before them). */
    if (prog->n_funcs > 0 && prog->funcs[0].start == 0) {
        entry_jmp_pos = e.n;
        for (int z = 0; z < 5; z++) e8(&e, 0x90);   /* NOPs, patched to jmp rel32 */
    }

    /* Helper: get x86 encoding for vr (returns -1 if spilled) */
    #define VR_ENC(vr) ((vr) < (wubu_vr_t)assign_count && assign[(vr)].reg >= 0 ? reg_x86[assign[(vr)].reg] : -1)
    /* The regalloc stores the spill slot byte-offset in assign[v].stack
     * (already a negative value: -(slot+1)*8). Use it directly. */
    #define VR_SPILL(vr) ((vr) < (wubu_vr_t)assign_count && assign[(vr)].reg < 0 ? assign[(vr)].stack : 0)

    /* Lookahead: is the next instruction a RET that reads this vr? */
    #define NEXT_IS_RET(vr) (i + 1 < prog->n && prog->ins[i+1].op == MIR_RET && prog->ins[i+1].a == (wubu_vr_t)(vr))

    int result_in_rax = 0;  /* set when last op skipped store to keep result in rax */

    for (size_t i = 0; i < prog->n; i++) {
        const wubu_mir_instr_t *in = &prog->ins[i];
        if (in->op == MIR_LABEL) { note_label(&e, in->label, e.n); result_in_rax = 0; continue; }
        if (in->op != MIR_RET) result_in_rax = 0;  /* reset unless RET handles it */

        for (int f = 0; f < prog->n_funcs; f++)
            if ((size_t)prog->funcs[f].start == i && func_off[f] == (size_t)-1)
                func_off[f] = e.n;
        if ((uint32_t)i == max_func_end && entry_off == (size_t)-1)
            entry_off = e.n;

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
                emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
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
                    emit_load_rbp(&e, sd, spill_off(assign, assign_count, &e, in->a));  /* mov dst_reg, [rbp+off] */
                }
            } else {
                /* dest is spilled */
                if (sa >= 0) {
                    emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), sa);
                } else {
                    emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));  /* mov rax, [rbp+off_a] */
                    emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);  /* mov [rbp+off_dst], rax */
                }
            }
            break;
        }
        case MIR_ADD: case MIR_SUB: case MIR_MUL: case MIR_DIV: case MIR_MOD:
        case MIR_AND: case MIR_OR: case MIR_XOR:
        case MIR_FEQ: case MIR_FNE: case MIR_FLT: case MIR_FLE:
        case MIR_FADD: case MIR_FSUB: case MIR_FMUL: case MIR_FDIV:
        case MIR_ITOF: case MIR_FTOI:
        case MIR_BF16_TO_F32: case MIR_F32_TO_BF16:
        case MIR_DITOF: case MIR_DTOI:
        case MIR_F32_TO_F64: case MIR_F64_TO_F32:
        case MIR_DADD: case MIR_DSUB: case MIR_DMUL: case MIR_DDIV: case MIR_DNEG:
        case MIR_FNEG: {
            /* Load 'a' into rax (accumulator) */
            int sa = VR_ENC(in->a);
            if (sa == 0) {
                /* already in rax — nothing to do */
            } else if (sa >= 0) {
                emit_mov_reg(&e, 0, sa);  /* mov rax, src_reg */
            } else {
                emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
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
                emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));
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
            case MIR_DADD: case MIR_DSUB: case MIR_DMUL: case MIR_DDIV: {
                int da = VR_ENC(in->a);
                if (da >= 0) emit_mov_reg(&e, 0, da);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                int db = VR_ENC(in->b);
                if (db >= 0) emit_mov_reg(&e, 7, db);
                else emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));
                /* movq xmm0, rax : 66 48 0F 6E C0 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                /* movq xmm1, rdi : 66 48 0F 6E CF */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xCF);
                /* addsd/subsd/mulsd/divsd xmm0, xmm1 : F2 0F 5x C1 */
                e8(&e, 0xF2); e8(&e, 0x0F);
                switch (in->op) {
                case MIR_DADD: e8(&e, 0x58); break;
                case MIR_DSUB: e8(&e, 0x5C); break;
                case MIR_DMUL: e8(&e, 0x59); break;
                default:       e8(&e, 0x5E); break;
                }
                e8(&e, 0xC1);
                /* movq rax, xmm0 : 66 48 0F 7E C0 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }

            case MIR_DNEG: {
                int da = VR_ENC(in->a);
                if (da >= 0) emit_mov_reg(&e, 0, da);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                /* movq xmm0, rax */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                /* mov rdi, 0x8000000000000000 : 48 BF + imm64 */
                e8(&e, 0x48); e8(&e, 0xBF);
                e32(&e, 0x00000000); e32(&e, 0x80000000);
                /* movq xmm1, rdi */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xCF);
                /* xorps xmm0, xmm1 */
                e8(&e, 0x0F); e8(&e, 0x57); e8(&e, 0xC1);
                /* movq rax, xmm0 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }

            case MIR_F32_TO_F64: case MIR_F64_TO_F32: {
                int sc = VR_ENC(in->a);
                if (sc >= 0) emit_mov_reg(&e, 0, sc);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                if (in->op == MIR_F32_TO_F64) {
                    /* movd xmm0, eax ; cvtss2sd xmm0, xmm0 : F3 0F 5A C1 */
                    e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                    e8(&e, 0xF3); e8(&e, 0x0F); e8(&e, 0x5A); e8(&e, 0xC0);
                    /* movq rax, xmm0 : 66 48 0F 7E C0 */
                    e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                } else {
                    /* movq xmm0, rax : 66 48 0F 6E C0 */
                    e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                    /* cvtsd2ss xmm0, xmm0 : F2 0F 5A C1 */
                    e8(&e, 0xF2); e8(&e, 0x0F); e8(&e, 0x5A); e8(&e, 0xC0);
                    /* movd eax, xmm0 */
                    e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                }
                break;
            }

            case MIR_BF16_TO_F32: {
                /* widen: f32 bits = bf16 << 16 (exact) */
                int sc = VR_ENC(in->a);
                if (sc >= 0) emit_mov_reg(&e, 0, sc);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                rex(&e,1,0,0,0); e8(&e, 0xC1); e8(&e, 0xE0); e8(&e, 0x10); /* shl rax,16 */
                break;
            }

            case MIR_F32_TO_BF16: {
                /* narrow RNE: (x + 0x7FFF + ((x>>16)&1)) >> 16 */
                int sc = VR_ENC(in->a);
                if (sc >= 0) emit_mov_reg(&e, 0, sc);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                e8(&e, 0x89); e8(&e, 0xC1);                              /* mov ecx,eax */
                e8(&e, 0xC1); e8(&e, 0xE9); e8(&e, 0x10);                /* shr ecx,16 */
                e8(&e, 0x83); e8(&e, 0xE1); e8(&e, 0x01);                /* and ecx,1 */
                e8(&e, 0x01); e8(&e, 0xC8);                              /* add eax,ecx */
                e8(&e, 0x05); e8(&e, 0xFF); e8(&e, 0x7F); e8(&e, 0x00); e8(&e, 0x00);
                e8(&e, 0xC1); e8(&e, 0xE8); e8(&e, 0x10);                /* shr eax,16 */
                break;
            }

            case MIR_F16_TO_F32: case MIR_F32_TO_F16:
            case MIR_F16_ADD: case MIR_F16_MUL: case MIR_F16_DIV:
            case MIR_DITOF: case MIR_DTOI: {
                int sc = VR_ENC(in->a);
                if (sc >= 0) emit_mov_reg(&e, 0, sc);
                else emit_load_rbp(&e, 0, VR_SPILL(in->a));
                if (in->op == MIR_DITOF) {
                    /* cvtsi2sd xmm0, eax : F2 0F 2A C0 */
                    e8(&e, 0xF2); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC0);
                    /* movq rax, xmm0 */
                    e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                } else {
                    /* movq xmm0, rax */
                    e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                    /* cvttsd2si rax, xmm0 : F2 48 0F 2C C0 */
                    e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2C); e8(&e, 0xC0);
                }
                break;
            }

            case MIR_ITOF: case MIR_FTOI: {
                int sc = VR_ENC(in->a);
                if (sc >= 0) emit_mov_reg(&e, 0, sc);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                if (in->op == MIR_ITOF) {
                    /* cvtsi2ss xmm0, eax : F3 0F 2A C0 */
                    e8(&e, 0xF3); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC0);
                    /* movd eax, xmm0 */
                    e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                } else {
                    /* movd xmm0, eax */
                    e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xC0);
                    /* cvttss2si eax, xmm0 : F3 0F 2C C0 */
                    e8(&e, 0xF3); e8(&e, 0x0F); e8(&e, 0x2C); e8(&e, 0xC0);
                }
                break;
            }

            case MIR_FNEG: {
                int sa4 = VR_ENC(in->a);
                if (sa4 >= 0) emit_mov_reg(&e, 0, sa4);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
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
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                int sb3 = VR_ENC(in->b);
                if (sb3 >= 0) emit_mov_reg(&e, 7, sb3);
                else emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));
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
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                int sb2 = VR_ENC(in->b);
                if (sb2 >= 0) emit_mov_reg(&e, 7, sb2);
                else emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));
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
                emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
                result_in_rax = 0;
            }
            break;
        }
        case MIR_SHL: case MIR_SHR: {
            /* shifts need rcx */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));

            int sb = VR_ENC(in->b);
            /* mov rcx, b */
            if (sb >= 0) {
                rex(&e,1,reg_needs_rex(sb),0,0); e8(&e, 0x89); e8(&e, (uint8_t)(0xC0 | ((sb & 7) << 3) | 1));
            } else {
                /* mov rcx, [rbp+off] */
                int32_t off = spill_off(assign, assign_count, &e, in->b);
                if (off >= -128 && off <= 127) { rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x4D); e8(&e, (uint8_t)off); }
                else { rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x8D); e32(&e, (uint32_t)off); }
            }
            if (in->op == MIR_SHL) { rex(&e,1,0,0,0); e8(&e, 0xD3); e8(&e, 0xE0); }
            else { rex(&e,1,0,0,0); e8(&e, 0xD3); e8(&e, 0xE8); }

            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
            break;
        }
        case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE: case MIR_GT: case MIR_GE:
        case MIR_ULT: case MIR_ULE: case MIR_UGT: case MIR_UGE: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));

            int sb = VR_ENC(in->b);
            if (sb >= 0) emit_mov_reg(&e, 7, sb);
            else emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));

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
            else emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
            break;
        }
        case MIR_NEG: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xD8);  /* neg rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
            break;
        }
        case MIR_NOT: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            rex(&e,1,0,0,0); e8(&e, 0xF7); e8(&e, 0xD0);  /* not rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) { if (sd != 0) emit_mov_reg(&e, sd, 0); }
            else emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
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
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            rex(&e,1,0,0,0); e8(&e, 0x85); e8(&e, 0xC0);  /* test rax,rax */
            e8(&e, 0x0F); e8(&e, 0x84);                     /* jz rel32 */
            PATCH_PUSH(e.n, in->label);
            e32(&e, 0);
            break;
        }
        case MIR_JNZ: {
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            rex(&e,1,0,0,0); e8(&e, 0x85); e8(&e, 0xC0);  /* test rax,rax */
            e8(&e, 0x0F); e8(&e, 0x85);                     /* jnz rel32 */
            PATCH_PUSH(e.n, in->label);
            e32(&e, 0);
            break;
        }
        case MIR_T_GEMM: {
            /* Native tensor-MADD via parallel libcall to wubu_tgemm_parallel.
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
            /* rdi = prog.mem (embedded pointer) */
            rex(&e,1,0,0,0); e8(&e, 0xBF); e64(&e, (uint64_t)prog->mem);
            /* rsi = A (vr value / slot index) */
            if (sa>=0) emit_mov_reg(&e,6,sa);            else emit_load_rbp(&e,6,spill_off(assign, assign_count, &e, in->a));
            /* rdx = B */
            if (sb>=0) emit_mov_reg(&e,2,sb);            else emit_load_rbp(&e,2,spill_off(assign, assign_count, &e, in->b));
            /* rcx = C */
            if (sd>=0) emit_mov_reg(&e,1,sd);            else emit_load_rbp(&e,1,spill_off(assign, assign_count, &e, in->dst));
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
            rex(&e,1,0,0,1); e8(&e,0xBB); e64(&e,(uint64_t)&wubu_tgemm_parallel);
            /* call r11 (FF D3 with REX.B → 41 FF D3; without 0x41 it decodes as call rbx!) */
            e8(&e,0x41); e8(&e,0xFF); e8(&e,0xD3);
            /* add rsp,8 (restore stack for K arg) */
            e8(&e,0x48); e8(&e,0x83); e8(&e,0xC4); e8(&e,0x08);
            break;
        }
        case MIR_T_GEMM_F32: {
            /* Float32 GEMM via optimized AVX2+FMA kernel.
             * Same calling convention as T_GEMM but calls wubu_tgemm_f32_parallel. */
            int M = (int)(in->imm >> 22);
            int N = (int)((in->imm >> 11) & 0x7FF);
            int K = (int)(in->imm & 0x7FF);
            int sa = VR_ENC(in->a);
            int sb = VR_ENC(in->b);
            int sd = VR_ENC(in->dst);
            /* rdi = prog->mem (base of MIR memory) */
            rex(&e,1,0,0,0); e8(&e, 0xBF); e64(&e, (uint64_t)prog->mem);
            /* rsi = A */
            if (sa>=0) emit_mov_reg(&e,6,sa);  else emit_load_rbp(&e,6,spill_off(assign, assign_count, &e, in->a));
            /* rdx = B */
            if (sb>=0) emit_mov_reg(&e,2,sb);  else emit_load_rbp(&e,2,spill_off(assign, assign_count, &e, in->b));
            /* rcx = C */
            if (sd>=0) emit_mov_reg(&e,1,sd);  else emit_load_rbp(&e,1,spill_off(assign, assign_count, &e, in->dst));
            /* r8d = M, r9d = N */
            e8(&e,0x41); e8(&e,0xB8); e32(&e,(uint32_t)M);
            e8(&e,0x41); e8(&e,0xB9); e32(&e,(uint32_t)N);
            /* push K */
            if ((uint32_t)K < 0x80u) { e8(&e,0x6A); e8(&e,(uint8_t)K); }
            else                     { e8(&e,0x68); e32(&e,(uint32_t)K); }
            /* movabs r11, &wubu_tgemm_f32_parallel; call r11 */
            rex(&e,1,0,0,1); e8(&e,0xBB); e64(&e,(uint64_t)&wubu_tgemm_f32_parallel);
            e8(&e,0x41); e8(&e,0xFF); e8(&e,0xD3);
            /* add rsp,8 */
            e8(&e,0x48); e8(&e,0x83); e8(&e,0xC4); e8(&e,0x08);
            break;
        }
        case MIR_CALL: {
            /* native call rel32 to the callee body (patched after emission).
             * Flat-register model matches the retro backends: args in v1..vN,
             * result in v0, no caller-save. */
            e8(&e, 0xE8);                       /* call rel32 */
            if (ncallp == ccap) {
                ccap = ccap ? ccap * 2 : 8;
                callps = realloc(callps, ccap * sizeof(*callps));
            }
            callps[ncallp].pos = e.n;
            callps[ncallp].func_id = in->func_id;
            ncallp++;
            e.n += 4;
            break;
        }

        case MIR_RET: case MIR_FRET: {
            if (x86_in_func_body(prog, i)) {
                /* callee return under the flat-register model: park the value
                 * in vr0's home (v0 == the call result), then hardware `ret`
                 * pops the call-pushed return address. rsp untouched. */
                int sret = VR_ENC(in->a);
                int v0h  = VR_ENC(0);
                if (sret >= 0 && v0h >= 0) {
                    if (sret != v0h) emit_mov_reg(&e, v0h, sret);
                } else if (sret >= 0) {
                    emit_store_rbp(&e, spill_off(assign, assign_count, &e, 0), sret);
                } else {
                    emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                    if (v0h >= 0) emit_mov_reg(&e, v0h, 0);
                    else emit_store_rbp(&e, spill_off(assign, assign_count, &e, 0), 0);
                }
                e8(&e, 0xC3);   /* ret */
                break;
            }
            /* If result is already in rax (lookahead skip), don't reload */
            if (!result_in_rax) {
                int sa = VR_ENC(in->a);
                if (sa >= 0) emit_mov_reg(&e, 0, sa);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            }
            result_in_rax = 0;
            e8(&e, 0xC9);  /* leave */
            e8(&e, 0xC3);  /* ret */
            break;
        }
        case MIR_ALLOC:
            break;  /* home base already emitted as a CONST vr */
        case MIR_LOAD: {
            /* dst = mem[addr]; use prog.mem directly (no stack frame) */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);          /* rax = addr */
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            rex(&e,1,0,0,0); e8(&e, 0xC1); e8(&e, 0xE0); e8(&e, 0x03); /* shl rax,3 */
            /* rsi = prog.mem + rax */
            rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0xDE);   /* mov rsi, rbx */
            rex(&e,1,0,0,0); e8(&e, 0x01); e8(&e, 0xC6);   /* add rsi, rax */
            int sd = VR_ENC(in->dst);
            if (sd >= 0) {
                rex(&e,1,reg_needs_rex(sd),0,0); e8(&e, 0x8B);
                e8(&e, (uint8_t)(0x06 | ((sd & 7) << 3))); /* mov dstreg,[rsi] */
            } else {
                rex(&e,1,0,0,0); e8(&e, 0x8B); e8(&e, 0x06); /* mov rax,[rsi] */
                emit_store_rbp(&e, spill_off(assign, assign_count, &e, in->dst), 0);
            }
            break;
        }
        case MIR_STORE: {
            /* mem[addr] = val; use prog.mem directly */
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 0, sa);          /* rax = addr */
            else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
            int sb = VR_ENC(in->b);
            if (sb >= 0) emit_mov_reg(&e, 7, sb);          /* rdi = val */
            else emit_load_rbp(&e, 7, spill_off(assign, assign_count, &e, in->b));
            rex(&e,1,0,0,0); e8(&e, 0xC1); e8(&e, 0xE0); e8(&e, 0x03); /* shl rax,3 */
            /* rsi = prog.mem + rax */
            rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0xDE);   /* mov rsi, rbx */
            rex(&e,1,0,0,0); e8(&e, 0x01); e8(&e, 0xC6);   /* add rsi, rax */
            rex(&e,1,0,0,0); e8(&e, 0x89); e8(&e, 0x3E);   /* mov [rsi], rdi */
            break;
        }
        case MIR_T_SOFTMAX:
        case MIR_T_TANH:
        case MIR_T_SIGMOID:
        case MIR_T_GELU:
        case MIR_T_RELU:
        case MIR_T_EXP:
        case MIR_T_SQRT:
        case MIR_T_SUM:
        case MIR_T_RMS_NORM:
        case MIR_T_EMBEDDING:
        case MIR_T_ROPE:
        case MIR_T_LAYERNORM:
        case MIR_T_CONV2D:
        case MIR_T_DROPOUT:
        case MIR_T_ARGMAX:
        case MIR_T_SWIGLU:
        case MIR_T_CLAMP: {
            /* Tensor ops: emit call to wubu_tensor_dispatch(op, mem, a, b, dst, N).
             * Use same pattern as T_GEMM: movabs r11, &wubu_tensor_dispatch; call r11. */
            /* rdi = prog.mem (embedded pointer) */
            rex(&e,1,0,0,0); e8(&e, 0xBF); e64(&e, (uint64_t)prog->mem);
            int sa = VR_ENC(in->a);
            if (sa >= 0) emit_mov_reg(&e, 6, sa);
            else emit_load_rbp(&e, 6, spill_off(assign, assign_count, &e, in->a));
            int sb = VR_ENC(in->b);
            if (sb >= 0) emit_mov_reg(&e, 2, sb);
            else emit_load_rbp(&e, 2, spill_off(assign, assign_count, &e, in->b));
            int sd = VR_ENC(in->dst);
            if (sd >= 0) emit_mov_reg(&e, 1, sd);
            else emit_load_rbp(&e, 1, spill_off(assign, assign_count, &e, in->dst));
            /* push op, then push N (7th and 8th args) */
            if ((uint32_t)in->op < 0x80u) { e8(&e,0x6A); e8(&e,(uint8_t)in->op); }
            else { e8(&e,0x68); e32(&e,(uint32_t)in->op); }
            if ((uint32_t)in->imm < 0x80u) { e8(&e,0x6A); e8(&e,(uint8_t)in->imm); }
            else { e8(&e,0x68); e32(&e,(uint32_t)in->imm); }
            /* movabs r11, &wubu_tensor_dispatch; call r11 */
            rex(&e,1,0,0,1); e8(&e,0xBB); e64(&e,(uint64_t)&wubu_tensor_dispatch);
            e8(&e,0x41); e8(&e,0xFF); e8(&e,0xD3);
            /* add rsp,16 (restore stack for 2 args) */
            e8(&e,0x48); e8(&e,0x83); e8(&e,0xC4); e8(&e,0x10);
            break;
        }
        }
    }

    /* fallback ret */
    if (e.n == 0 || e.code[e.n-1] != 0xC3) { e8(&e, 0xC9); e8(&e, 0xC3); }

    /* patch jumps */
    if (getenv("WUBU_CALL_DEBUG"))
        fprintf(stderr, "[fixups] ncallp=%zu entry_jmp_pos=%zu entry_off=%zu func0=%zu\n",
                ncallp, entry_jmp_pos, entry_off,
                prog->n_funcs > 0 ? func_off[0] : 0);
    for (size_t i = 0; i < n_patches; i++) {
        size_t t = label_off(&e, patches[i].label);
        if (t == (size_t)-1) continue;
        size_t pos = patches[i].pos;
        int32_t rel = (int32_t)(t - (pos + 5));   /* E9 rel32: next insn at pos+5 */
        e.code[pos] = (uint8_t)(rel & 0xFF);
        e.code[pos+1] = (uint8_t)((rel >> 8) & 0xFF);
        e.code[pos+2] = (uint8_t)((rel >> 16) & 0xFF);
        e.code[pos+3] = (uint8_t)((rel >> 24) & 0xFF);
    }
    free(patches);
    free(e.label_offsets);
    wubu_mir_free_alloc(assign);

    /* CALL target fixups (MUST run before peephole — it shifts offsets) */
    for (size_t fi2 = 0; fi2 < ncallp; fi2++) {
        size_t t = (size_t)-1;
        for (int f = 0; f < prog->n_funcs; f++)
            if ((uint32_t)f == callps[fi2].func_id && func_off[f] != (size_t)-1) { t = func_off[f]; break; }
        if (t == (size_t)-1) continue;
        size_t pos = callps[fi2].pos;
        int32_t rel = (int32_t)(t - (pos + 5));
        e.code[pos]   = (uint8_t)(rel & 0xFF);
        e.code[pos+1] = (uint8_t)((rel >> 8) & 0xFF);
        e.code[pos+2] = (uint8_t)((rel >> 16) & 0xFF);
        e.code[pos+3] = (uint8_t)((rel >> 24) & 0xFF);
    }
    /* entry trampoline patch: jmp rel32 from entry_jmp_pos to entry_off */
    if (prog->n_funcs > 0 && entry_off != (size_t)-1 && entry_jmp_pos) {
        int32_t rel = (int32_t)(entry_off - (entry_jmp_pos + 5));
        e.code[entry_jmp_pos]     = 0xE9;
        e.code[entry_jmp_pos + 1] = (uint8_t)(rel & 0xFF);
        e.code[entry_jmp_pos + 2] = (uint8_t)((rel >> 8) & 0xFF);
        e.code[entry_jmp_pos + 3] = (uint8_t)((rel >> 16) & 0xFF);
        e.code[entry_jmp_pos + 4] = (uint8_t)((rel >> 24) & 0xFF);
    }
    free(callps);
    free(func_off);

    /* Peephole: wire up the real optimizer (x86_peephole.c).
     * Skipped for multi-function programs: shrinking movabs shifts byte
     * offsets and would invalidate the already-applied CALL/trampoline
     * rel32 fixups. */
    if (prog->n_funcs == 0)
        e.n = x86_peephole_optimize(e.code, e.n);

    *out = e.code;
    *out_size = e.n;
    if (remapped) { free(remapped->ins); free(remapped); }
    return 0;
}

/* Global thread-local pointer for JIT working memory (heap-allocated).
 * Set by x86_run before calling JIT'd code. Used by wubu_tgemm_parallel. */
int64_t *wubu_jit_mem_ptr = NULL;

static int64_t x86_run(const uint8_t *code, size_t size, int64_t arg) {
    void *exec = jit_alloc_exec(size);
    if (!exec) return -1;
    memcpy(exec, code, size);
    wubu_clear_cache(exec, size);

    /* Set global mem base for JIT code */
    wubu_jit_mem_ptr = (int64_t*)arg;

    /* Patch the emitted code: scan entire buffer for movabs rsi, imm64
     * (the JIT mem base embedded at compile time) and replace with
     * mov rsi, rbx (which holds the runtime mem base from SysV arg rdi). */
    uint8_t *patch = (uint8_t *)exec;
    for (size_t i = 0; i + 9 < size; i++) {
        if (patch[i] == 0x48 && patch[i+1] == 0xBE) {
            patch[i+1] = 0x89; patch[i+2] = 0xDE; /* mov rsi, rbx */
        }
    }

    int64_t (*fn)(void) = (int64_t (*)(void))exec;
    if (getenv("WUBU_CALL_DEBUG") && size > 40) {
        fprintf(stderr, "[run] size=%zu code:", size);
        for (size_t q = 0; q < size && q < 90; q++) fprintf(stderr, " %02X", ((const uint8_t*)code)[q]);
        fprintf(stderr, "\n");
    }
    int64_t r = fn();

    if (wubu_jit_mem_ptr) { munmap(wubu_jit_mem_ptr, 64*1024*1024); wubu_jit_mem_ptr = NULL; }
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
