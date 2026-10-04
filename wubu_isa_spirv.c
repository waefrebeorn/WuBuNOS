/*
 * wubu_isa_spirv.c -- the Vulkan/SPIR-V ISA driver (the borg leg).
 *
 * Emits SPIR-V compute modules directly from MIR — hand-encoded binary,
 * no shader compiler in our codegen path (spirv-val used as test oracle
 * only, same relationship ptxas has to the PTX driver).
 *
 * One emitter runs on EVERY Vulkan device: NVIDIA dGPU, AMD APU iGPU,
 * old recycled cards, llvmpipe CPU fallback.
 *
 * Kernel ABI:
 *   binding 0 SSBO: struct { u64 mem[N+1]; }   cell 0 = return slot
 *   push constants: { u64 arg; }
 *   LocalSize x = 64 (grid-stride loop handles any N)
 *
 * C11.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"

/* ---- word buffer ---- */
typedef struct { uint32_t *w; size_t n, cap; } sbuf_t;
typedef struct { long long imm; uint32_t id; } cment_t;
static void sb_word(sbuf_t *b, uint32_t v) {
    if (b->n + 1 > b->cap) { b->cap = b->cap ? b->cap*2 : 1024;
                             b->w = realloc(b->w, b->cap*4); }
    b->w[b->n++] = v;
}
static void spv_ins(sbuf_t *b, uint16_t op, const uint32_t *ops, size_t n) {
    sb_word(b, ((uint32_t)(n+1)) << 16 | op);
    for (size_t i=0;i<n;i++) sb_word(b, ops[i]);
}

/* opcodes we use */
#define OP_SOURCE                 3
#define OP_NAME                   5
#define OP_MEMORYMODEL           14
#define OP_ENTRYPOINT            15
#define OP_EXEC_MODE             16
#define OP_CAPABILITY            17
#define OP_TYPE_FLOAT            22
#define OP_TYPE_VOID             19
#define OP_TYPE_BOOL             20
#define OP_TYPE_INT              21
#define OP_TYPE_VECTOR           23
#define OP_TYPE_ARRAY            28
#define OP_TYPE_STRUCT           30
#define OP_TYPE_POINTER          32
#define OP_TYPE_FUNCTION         33
#define OP_CONSTANT              43
#define OP_FUNCTION              54
#define OP_FUNCTION_END          56
#define OP_VARIABLE              59
#define OP_LOAD                  61
#define OP_STORE                 62
#define OP_ACCESS_CHAIN          65
#define OP_COMPOSITE_EXTRACT     81
#define OP_DECORATE              71
#define OP_RETURN               253
#define OP_LABEL                248

/* opcode numbers verified against spirv-as round-trip */
#define OPCODE_IADD                 128
#define OPCODE_ISUB                 130
#define OPCODE_IMUL                 132
#define OPCODE_BITWISE_AND          199
#define OPCODE_BITWISE_OR           197
#define OPCODE_BITWISE_XOR          198
#define OPCODE_NOT                  200
#define OPCODE_SNEGATE              126
#define OPCODE_SHIFT_LEFT           196
#define OPCODE_SHIFT_RIGHT_LOGICAL 194
#define OPCODE_SHIFT_RIGHT_ARITH    195
#define OPCODE_SDIV                 135
#define OPCODE_SREM                 138
#define OPCODE_I_EQ                 170
#define OPCODE_I_NE                 171
#define OPCODE_S_LT                 177
#define OPCODE_S_LE                 179
#define OPCODE_U_LT                 176
#define OPCODE_U_LE                 178
#define OPCODE_S_GT                 173
#define OPCODE_S_GE                 175
#define OPCODE_U_GT                 172
#define OPCODE_U_GE                 174
#define OPCODE_SELECT               169
#define OPCODE_LABEL                248
#define OPCODE_BRANCH               249
#define OPCODE_BRANCH_COND          250
#define OPCODE_SELECTION_MERGE      247
#define OPCODE_LOOP_MERGE           246
#define OPCODE_RETURN               253
#define OPCODE_FUNCTION_END         56
#define OPCODE_FUNCTION             54
#define OPCODE_UCONVERT             113

typedef struct {
    sbuf_t bin;
    uint32_t next_id;
    uint32_t scratch_base;   /* first SSBO scratch elem (T_GEMM loop vars) */
    /* pinned ids */
    uint32_t t_void, t_i32, t_v3i32, t_u64, t_bool;
    uint32_t c_i32_1, c_i32_2, c_i32_sem;              /* i32 consts for barriers */
    uint32_t t_arr_mem, t_struct_ssbo, t_ptr_ssbo;     /* storage class 5 */
    uint32_t t_i32_in, t_gid_input;                    /* input v3i32 ptr */
    uint32_t t_pushblk, t_ptr_push;
    uint32_t t_fn_void, t_res_u64;
    uint32_t t_v2i32, t_f32, t_f64;
    uint32_t c_zero32, len_mem_cells;
    uint32_t c_zero64, c_one64, c_three64;
    uint32_t var_ssbo, var_push, var_gid;
    uint32_t fn_main, lbl_entry;
} S;

static uint32_t nid(S *s){ return s->next_id++; }

/* u64 SSA -> f32 bits in low 32 (Bitcast v2i32 -> Extract.0 -> Bitcast f32) */
static uint32_t spirv_unpack_f32(S *s, uint32_t src64)
{
    uint32_t v2 = nid(s), lo = nid(s), fl = nid(s);
    { uint32_t o[]={s->t_v2i32,v2,src64}; spv_ins(&s->bin,124,o,3);}
    { uint32_t o[]={s->t_i32,lo,v2,0};    spv_ins(&s->bin,81,o,4);}
    { uint32_t o[]={s->t_f32,fl,lo};      spv_ins(&s->bin,124,o,3);}
    return fl;
}

/* ---- SPIR-V T_GEMM: grid-stride structured triple loop with OpPhi ---- */
static uint32_t spirv_find_const(const cment_t *cmts, size_t ncm, long long v)
{
    for (size_t w = 0; w < ncm; w++)
        if (cmts[w].imm == v) return cmts[w].id;
    return 0;
}

static void TG_ACX(S *s, uint32_t res, uint32_t idx){
    uint32_t o[]={s->t_res_u64,res,s->var_ssbo,s->c_zero32,idx};
    spv_ins(&s->bin,65,o,5);
}
static void TG_STX(S *s, uint32_t ptr, uint32_t val){
    uint32_t o[]={ptr,val}; spv_ins(&s->bin,62,o,2);
}
static void TG_LDX(S *s, uint32_t res, uint32_t ptr){
    uint32_t o[]={s->t_u64,res,ptr}; spv_ins(&s->bin,61,o,3);
}
static void emit_tgemm_spirv(S *s, const wubu_mir_instr_t *in,
                             const cment_t *cmts, size_t ncm,
                             uint32_t id_gid, uint32_t preheader,
                             uint32_t *vrmap, uint32_t maxvr,
                             int *jt)
{
    /* SINGLE flat loop over all M*K*N work items (grid-stride by gid):
     *   w = gid; while (w < M*K*N): decompose to i,k,j; GEMM; w += 64 */
    #define TG_VRG(k) ((uint32_t)((k) < maxvr && vrmap[k] ? vrmap[k] : s->c_zero64))
    #define TG_AC(res_, idx_) do { \
        (res_) = nid(s); \
        uint32_t o_[]={s->t_res_u64,(res_),s->var_ssbo,s->c_zero32,(idx_)}; \
        spv_ins(&s->bin,65,o_,5); } while(0)
    #define TG_LD(res_, ptr_) do { \
        (res_) = nid(s); \
        uint32_t o_[]={s->t_u64,(res_),(ptr_)}; spv_ins(&s->bin,61,o_,3);} while(0)
    #define TG_ST(ptr_, val_) do { \
        uint32_t o_[]={ptr_,val_}; spv_ins(&s->bin,62,o_,2);} while(0)

    int M  = (int)(in->imm >> 22);
    int Nn = (int)((in->imm >> 11) & 0x7FF);
    int K  = (int)(in->imm & 0x7FF);
    uint32_t cM   = spirv_find_const(cmts, ncm, M);
    uint32_t cN   = spirv_find_const(cmts, ncm, Nn);
    uint32_t cK   = spirv_find_const(cmts, ncm, K);
    /* TRUE grid-stride: out=gid; out<M*N; out+=64. Each lane owns whole cells,
     * full K-reduction per cell — every C write has exactly one writer. */
    uint32_t cS   = spirv_find_const(cmts, ncm, 64);
    uint32_t cKN  = spirv_find_const(cmts, ncm, K*Nn); (void)cKN;
    uint32_t cMN  = spirv_find_const(cmts, ncm, M*Nn); (void)cMN;
    uint32_t c4L  = spirv_find_const(cmts, ncm, 4);
    uint32_t aBase0 = TG_VRG(in->a), bBase0 = TG_VRG(in->b), cBase0 = TG_VRG(in->dst);
    /* MIR cell i lives at SSBO element i+1 (cell 0 = return slot) — match LOAD/STORE */
    uint32_t aBase=nid(s), bBase=nid(s), cBase=nid(s);
    { uint32_t o[]={s->t_u64,aBase,aBase0,s->c_one64}; spv_ins(&s->bin,OPCODE_IADD,o,4);}
    { uint32_t o[]={s->t_u64,bBase,bBase0,s->c_one64}; spv_ins(&s->bin,OPCODE_IADD,o,4);}
    { uint32_t o[]={s->t_u64,cBase,cBase0,s->c_one64}; spv_ins(&s->bin,OPCODE_IADD,o,4);}

    uint32_t sc_i   = spirv_find_const(cmts, ncm, (long long)s->scratch_base);
    /* Multi-WG grid-stride: stride = total lanes across WGs (gx*64), so each cell
     * is owned by exactly one lane of one WG. Baked at emit time from env. */
    unsigned gx_ = 1;
    { const char *ge = getenv("WUBU_VK_GROUPS"); if (ge) gx_=(unsigned)atoi(ge); if (gx_<1) gx_=1; }
    uint32_t nlanes = gx_*64;
    uint32_t cSTRIDE = spirv_find_const(cmts, ncm, (long long)nlanes);
    uint32_t cPASS  = spirv_find_const(cmts, ncm, ((long long)M*Nn + nlanes - 1)/nlanes);
    uint32_t c8C    = spirv_find_const(cmts, ncm, 8); /* lane stride: 4 slots */
    uint32_t head=nid(s), cont=nid(s), body=nid(s), merge=nid(s);
    uint32_t w = nid(s);

    /* lane = gid; slot0(gid) = lane*4+scratch_base, slot1(pass) = slot0+4 */
    { uint32_t o[]={s->t_u64,w,id_gid}; spv_ins(&s->bin,83,o,3); }   /* w = lane */
    uint32_t lane4=nid(s), slt=nid(s), s1=nid(s), spo_=nid(s);
    { uint32_t o[]={s->t_u64,lane4,w,c8C}; spv_ins(&s->bin,OPCODE_IMUL,o,4);} /* lane*8 (4 slots) */
    { uint32_t o[]={s->t_u64,slt,lane4,sc_i};   spv_ins(&s->bin,OPCODE_IADD,o,4);} /* slot0 */
    { uint32_t o[]={s->t_u64,s1,slt,c4L};      spv_ins(&s->bin,OPCODE_IADD,o,4);} /* slot1 */
    { uint32_t o[]={s->t_res_u64,spo_,s->var_ssbo,s->c_zero32,slt};
      spv_ins(&s->bin,65,o,5); TG_ST(spo_,w); }                         /* slot0 = gid */
    { uint32_t op1[]={s->t_res_u64,nid(s),s->var_ssbo,s->c_zero32,s1};
      uint32_t st1=op1[1]; spv_ins(&s->bin,65,op1,5); TG_ST(st1,s->c_zero64); } /* slot1 = 0 */
    { uint32_t o[]={head}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
    *jt = 1;

    { uint32_t o[]={head}; spv_ins(&s->bin,OP_LABEL,o,1); }
    {
        uint32_t cmp=nid(s);
        { uint32_t oh[]={s->t_res_u64,nid(s),s->var_ssbo,s->c_zero32,s1};
          uint32_t hp_=oh[1]; spv_ins(&s->bin,65,oh,5); TG_LD(w,hp_); }  /* w = pass# */
        { uint32_t o[]={s->t_bool,cmp,w,cPASS}; spv_ins(&s->bin,176,o,4); } /* p < passes */
        { uint32_t om[]={merge,cont,0}; spv_ins(&s->bin,246,om,3); }
        { uint32_t o3[]={cmp,body,merge}; spv_ins(&s->bin,OPCODE_BRANCH_COND,o3,3); }
    }

    { uint32_t o[]={body}; spv_ins(&s->bin,OP_LABEL,o,1); }
    {
        /* out = gid + pass*stride (multi-WG uniform grid-stride; guarded store) */
        uint32_t p64=nid(s), out=nid(s);
        { uint32_t o[]={s->t_u64,p64,w,cSTRIDE}; spv_ins(&s->bin,OPCODE_IMUL,o,4);} /* pass*stride */
        { uint32_t o[]={s->t_u64,out,id_gid,p64}; spv_ins(&s->bin,OPCODE_IADD,o,4); }
        /* i = out/N; j = out%N */
        uint32_t pi_=nid(s), pj_=nid(s);
        { uint32_t o[]={s->t_u64,pi_,out,cN};  spv_ins(&s->bin,134,o,4);}
        { uint32_t o[]={s->t_u64,pj_,out,cN};  spv_ins(&s->bin,137,o,4);}
        /* K reduction as structured k-loop — glslang-exact block layout that dzn handles:
         *   kpre  = Label, LoopMerge(kmer,kcont), Branch khdr
         *   khdr  = Label, Load k(slot2), ULessThan, BranchCond(kbody|kmer)
         *   kbody = Label, FMA + Store acc(slot3), Branch kcont
         *   kcont = Label, k+1→slot2, Branch khdr (back-edge)
         *   kmer  = Label, Load final acc(slot3)
         * LoopMerge lives on kpre (preheader), NOT on the header. dzn silently
         * miscompiles headers where instructions precede LoopMerge. */
        uint32_t kpre=nid(s), khdr=nid(s), kbody=nid(s), kcont=nid(s), kmer=nid(s);
        uint32_t s8=nid(s), s3=nid(s);
        { uint32_t o[]={s->t_u64,s8,slt,c8C};  spv_ins(&s->bin,128,o,4);}  /* slot2 = slt+8 */
        { uint32_t o[]={s->t_u64,s3,s8,c4L};   spv_ins(&s->bin,128,o,4);}  /* slot3 = slt+12 */
        /* init: k=0, acc=0 */
        { uint32_t o[]={s->t_res_u64,nid(s),s->var_ssbo,s->c_zero32,s8};
          uint32_t kp_=o[1]; spv_ins(&s->bin,65,o,5); TG_ST(kp_, s->c_zero64); }
        { uint32_t o[]={s->t_res_u64,nid(s),s->var_ssbo,s->c_zero32,s3};
          uint32_t ap_=o[1]; spv_ins(&s->bin,65,o,5); TG_ST(ap_, s->c_zero64); }
        /* body → kpre */
        { uint32_t o[]={kpre}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
        /* kpre: preheader with LoopMerge + Branch to header */
        { uint32_t o[]={kpre}; spv_ins(&s->bin,OPCODE_LABEL,o,1); }
        { uint32_t om2[]={kmer,kcont,0}; spv_ins(&s->bin,OPCODE_LOOP_MERGE,om2,3); }
        { uint32_t o[]={khdr}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
        /* khdr: header - load k, test k<K, BranchCond */
        { uint32_t o[]={khdr}; spv_ins(&s->bin,OPCODE_LABEL,o,1); }
        uint32_t kptr=nid(s), kload=nid(s), klt=nid(s);
        { uint32_t o[]={s->t_res_u64,kptr,s->var_ssbo,s->c_zero32,s8}; spv_ins(&s->bin,65,o,5); }
        TG_LD(kload, kptr);
        { uint32_t o[]={s->t_bool,klt,kload,cK}; spv_ins(&s->bin,176,o,4);} /* k < K */
        { uint32_t o3[]={klt,kbody,kmer}; spv_ins(&s->bin,OPCODE_BRANCH_COND,o3,3); }
        /* kbody: FMA + store acc */
        { uint32_t o[]={kbody}; spv_ins(&s->bin,OPCODE_LABEL,o,1); }
        uint32_t tiK=nid(s), tik=nid(s), ta_=nid(s), aptr=nid(s), aval=nid(s);
        { uint32_t o[]={s->t_u64,tiK,pi_,cK};    spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,tik,tiK,kload}; spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,ta_,tik,aBase}; spv_ins(&s->bin,128,o,4);}
        TG_AC(aptr, ta_); TG_LD(aval, aptr);
        uint32_t tkN=nid(s), tkj=nid(s), tb_=nid(s), bptr=nid(s), bval=nid(s);
        { uint32_t o[]={s->t_u64,tkN,kload,cN};  spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,tkj,tkN,pj_};   spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,tb_,tkj,bBase}; spv_ins(&s->bin,128,o,4);}
        TG_AC(bptr, tb_); TG_LD(bval, bptr);
        uint32_t prod=nid(s), olda=nid(s), accn=nid(s), acptr=nid(s);
        { uint32_t o[]={s->t_u64,prod,aval,bval}; spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_res_u64,acptr,s->var_ssbo,s->c_zero32,s3}; spv_ins(&s->bin,65,o,5); }
        TG_LD(olda, acptr);
        { uint32_t o[]={s->t_u64,accn,olda,prod}; spv_ins(&s->bin,128,o,4);}
        TG_ST(acptr, accn);
        { uint32_t o[]={kcont}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
        /* kcont: k+1 → slot2, back-edge to header */
        { uint32_t o[]={kcont}; spv_ins(&s->bin,OPCODE_LABEL,o,1); }
        { uint32_t kn1=nid(s);
          { uint32_t o[]={s->t_u64,kn1,kload,s->c_one64}; spv_ins(&s->bin,128,o,4);}
          TG_ST(kptr, kn1); }
        { uint32_t o[]={kpre}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
        /* kmer: load final acc from slot3 */
        { uint32_t o[]={kmer}; spv_ins(&s->bin,OPCODE_LABEL,o,1); }
        uint32_t acc=nid(s), afinal=nid(s);
        { uint32_t o[]={s->t_res_u64,afinal,s->var_ssbo,s->c_zero32,s3}; spv_ins(&s->bin,65,o,5); }
        TG_LD(acc, afinal);

        /* writeback: if (out < M*N) C[i*N+j] += acc — uniform-loop, guarded store.
         * Uses a structured if (SelectionMerge) instead of OpSelect: Lavapipe
         * mishandles OpSelect feeding an SSBO store when lanes diverge on data. */
        uint32_t ciN=nid(s), cij=nid(s), coff=nid(s), cptr=nid(s);
        uint32_t oldc=nid(s), cond=nid(s), wb_then=nid(s), wb_end=nid(s);
        { uint32_t o[]={s->t_bool,cond,out,cMN};  spv_ins(&s->bin,176,o,4);} /* out<M*N */
        { uint32_t o[]={s->t_u64,ciN,pi_,cN};    spv_ins(&s->bin,132,o,4);} /* i*N */
        { uint32_t o[]={s->t_u64,cij,ciN,pj_};   spv_ins(&s->bin,128,o,4);} /* +j */
        { uint32_t o[]={s->t_u64,coff,cij,cBase};spv_ins(&s->bin,128,o,4);}
        { uint32_t osm[]={wb_end,0}; spv_ins(&s->bin,OPCODE_SELECTION_MERGE,osm,2); }
        { uint32_t obc[]={cond,wb_then,wb_end}; spv_ins(&s->bin,OPCODE_BRANCH_COND,obc,3); }
        { uint32_t ot[]={wb_then}; spv_ins(&s->bin,OP_LABEL,ot,1); }
        TG_AC(cptr, coff); TG_LD(oldc, cptr);
        uint32_t added=nid(s);
        { uint32_t o[]={s->t_u64,added,oldc,acc}; spv_ins(&s->bin,128,o,4);}
        TG_ST(cptr, added);
        { uint32_t oe[]={wb_end}; spv_ins(&s->bin,OPCODE_BRANCH,oe,1); }
        { uint32_t oend[]={wb_end}; spv_ins(&s->bin,OP_LABEL,oend,1); }
        { uint32_t o[]={cont}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
    }

    /* cont: increment pass counter (slot1), branch header */
    { uint32_t o[]={cont}; spv_ins(&s->bin,OP_LABEL,o,1); }
    { uint32_t o_[]={s->t_res_u64,nid(s),s->var_ssbo,s->c_zero32,s1};
      uint32_t cp_=o_[1]; spv_ins(&s->bin,65,o_,5); TG_LD(w,cp_);  /* w = pass */
      uint32_t nw2=nid(s);
      { uint32_t o[]={s->t_u64,nw2,w,s->c_one64}; spv_ins(&s->bin,OPCODE_IADD,o,4);} /* pass+1 */
      TG_ST(cp_,nw2); }
    { uint32_t o[]={head}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
    *jt = 1;

    { uint32_t o[]={merge}; spv_ins(&s->bin,OP_LABEL,o,1); }
}

int wubu_spirv_emit(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_n)
{
    S s; memset(&s,0,sizeof s);
    s.next_id = 1;

    /* ---- pin ALL type/const/var ids up front (logical layout order) ---- */
    /* pinned in EXACTLY the order they are emitted below (types first,
     * then constants, then globals) so ids never collide. */
    s.t_void      = nid(&s);
    s.t_bool      = nid(&s);
    s.t_i32       = nid(&s);
    s.t_u64       = nid(&s);
    s.t_v3i32     = nid(&s);
    s.t_f32       = nid(&s);
    s.t_f64       = nid(&s);
    s.t_v2i32     = nid(&s);
    s.len_mem_cells = nid(&s);
    s.t_arr_mem   = nid(&s);
    s.t_struct_ssbo = nid(&s);
    s.t_ptr_ssbo  = nid(&s);
    s.t_i32_in    = nid(&s);
    s.t_gid_input = nid(&s);
    s.t_pushblk   = nid(&s);
    s.t_ptr_push  = nid(&s);
    s.t_res_u64   = nid(&s);
    s.t_fn_void   = nid(&s);
    s.c_zero32    = nid(&s);
    s.c_i32_1     = nid(&s);
    s.c_i32_2     = nid(&s);
    s.c_i32_sem   = nid(&s);
    s.c_zero64    = nid(&s);
    s.c_one64     = nid(&s);
    s.c_three64   = nid(&s);
    s.var_ssbo    = nid(&s);
    s.var_push    = nid(&s);
    s.var_gid     = nid(&s);
    s.fn_main     = nid(&s);

    /* Pre-pass: collect DISTINCT i64 immediates and pin an id for each.
     * Constants cannot appear inside function bodies (section 09 rule), so
     * every MIR_CONST materializes as OpCopyObject from its section-09 def. */
    cment_t *cmts = NULL; size_t ncm = 0, cap_cm = 0;
    /* helper to register an immediate */
    #define ADD_CONST(v_) do { \
        long long v__ = (long long)(v_); \
        size_t w_; for (w_ = 0; w_ < ncm; w_++) if (cmts[w_].imm == v__) break; \
        if (w_ == ncm) { \
            if (ncm + 1 > cap_cm) { cap_cm = cap_cm ? cap_cm*2 : 32; \
                                    cmts = realloc(cmts, cap_cm * sizeof *cmts); } \
            cmts[ncm].imm = v__; \
            cmts[ncm].id  = nid(&s); \
            ncm++; \
        } \
    } while (0)
    for (size_t q = 0; q < p->n; q++) {
        if (p->ins[q].op == MIR_CONST) {
            ADD_CONST(p->ins[q].imm);
        } else if (p->ins[q].op == MIR_SEXT32 || p->ins[q].op == MIR_ZEXT32
                || p->ins[q].op == MIR_SEXT16 || p->ins[q].op == MIR_SEXT8) {
            /* Pre-register the width mask and sign bit. OpConstant must live
             * in the module's global constants section, which is emitted
             * before the instruction loop -- registering here is what gives
             * them a definition. Emitting the OpConstant inline in the
             * instruction switch instead yields "Constant cannot appear in a
             * function declaration" from spirv-val. */
            int bits_ = (p->ins[q].op == MIR_SEXT16) ? 16 :
                        (p->ins[q].op == MIR_SEXT8)  ? 8  : 32;
            ADD_CONST((long long)((1ULL << bits_) - 1ULL));   /* width mask */
            ADD_CONST((long long)(1ULL << (bits_ - 1)));      /* sign bit   */
            ADD_CONST((long long)(bits_ - 1));                 /* shift amt  */
            ADD_CONST(1LL);                                    /* & 1        */
            /* 0xFFFFFFFF00000000: multiply 0/1 by this to broadcast the sign */
            ADD_CONST((long long)(int64_t)(0xFFFFFFFFULL << 32));
        } else if (p->ins[q].op == MIR_T_GEMM) {
            int M_ = (int)(p->ins[q].imm >> 22);
            int K_ = (int)((p->ins[q].imm >> 11) & 0x7FF);
            int Nn = (int)(p->ins[q].imm & 0x7FF);
            ADD_CONST(M_); ADD_CONST(K_); ADD_CONST(Nn);
            ADD_CONST(M_*Nn); ADD_CONST(M_*K_*Nn);   /* cKN, cMKN — used by helper */
            ADD_CONST(64); ADD_CONST(1); ADD_CONST(0); ADD_CONST(4); ADD_CONST(8);
            { int _k; for (_k = 0; _k < K_; _k++) ADD_CONST(_k); } /* unroll consts */
            /* multi-WG: bake stride + pass count */
            { unsigned _gx=1; const char *_ge=getenv("WUBU_VK_GROUPS");
              if (_ge) _gx=(unsigned)atoi(_ge); if (_gx<1) _gx=1;
              unsigned _st=_gx*64;
              ADD_CONST((long long)_st);
              ADD_CONST((long long)(((long long)M_*Nn + _st - 1)/_st)); }
            long long sb_ = (long long)((p->total_mem>0?p->total_mem:1)+1);
            s.scratch_base = (uint32_t)sb_;
            ADD_CONST(sb_+0); ADD_CONST(sb_+1); ADD_CONST(sb_+2);
            ADD_CONST(sb_+3); ADD_CONST(sb_+4);
        }
    }
    #undef ADD_CONST
    uint32_t vr_base = s.next_id;
    /* vrmap sized from the program's actual VR high-water mark (scan all
     * operands). The old hardcoded 256 silently dropped writes for bigger
     * programs — every read then returned c_zero64 (wrong results at scale). */
    uint32_t maxvr = 256;
    for (size_t q = 0; q < p->n; q++) {
        const wubu_mir_instr_t *qi = &p->ins[q];
        if (qi->dst + 1u > maxvr) maxvr = qi->dst + 1u;
        if (qi->a  + 1u > maxvr) maxvr = qi->a  + 1u;
        if (qi->b  + 1u > maxvr) maxvr = qi->b  + 1u;
    }
    maxvr += 8; /* headroom */

    /* ---- header ---- */
    sb_word(&s.bin, 0x07230203u);
    sb_word(&s.bin, 0x00010300u);         /* SPIR-V 1.3 */
    sb_word(&s.bin, 0x00080077u);         /* generator wubu */
    sb_word(&s.bin, vr_base + maxvr + 4096); /* bound placeholder */
    sb_word(&s.bin, 0);

    /* ---- capabilities / model / entry ---- */
    { uint32_t c[]={1}; spv_ins(&s.bin,OP_CAPABILITY,c,1);}        /* Shader */
    { uint32_t c[]={11}; spv_ins(&s.bin,OP_CAPABILITY,c,1);}        /* Int64 */
    /* Float64 (10) is required by OpTypeFloat 64. Declaring it only when the
     * program actually uses a double keeps f32-only modules byte-identical;
     * without it spirv-val reports "Using a 64-bit floating point type
     * requires the Float64 capability" even though dzn tolerates it. */
    { int uses_f64_ = 0;
      for (size_t q = 0; q < p->n; q++) {
          uint16_t o_ = (uint16_t)p->ins[q].op;
          if (o_==MIR_DADD||o_==MIR_DSUB||o_==MIR_DMUL||o_==MIR_DDIV||
              o_==MIR_DNEG||o_==MIR_DLT ||o_==MIR_DLE ||o_==MIR_DGT ||
              o_==MIR_DGE ||o_==MIR_DEQ||o_==MIR_DNE ||o_==MIR_DITOF||
              o_==MIR_DTOI||o_==MIR_DTOI_U) { uses_f64_ = 1; break; }
      }
      if (uses_f64_) { uint32_t c[]={10}; spv_ins(&s.bin,OP_CAPABILITY,c,1); } }
    { uint32_t m[]={0,0}; spv_ins(&s.bin,OP_MEMORYMODEL,m,2);}     /* Simple,None */

    /* OpEntryPoint GLCompute %main "main" (no interface vars needed: SSBO) */
    {
        const char nm[]="main";
        size_t ws=(sizeof(nm)+3)/4;
        /* content words: execmodel + entry-id + string-words + 1 interface var */
        sb_word(&s.bin,(uint32_t)((4+ws)<<16)|OP_ENTRYPOINT);
        sb_word(&s.bin,5); sb_word(&s.bin,s.fn_main);
        uint8_t tmp[8]={0}; memcpy(tmp,nm,sizeof(nm)-1);
        for(size_t i=0;i<ws;i++) sb_word(&s.bin,(uint32_t)tmp[i*4]|((uint32_t)tmp[i*4+1]<<8)|((uint32_t)tmp[i*4+2]<<16)|((uint32_t)tmp[i*4+3]<<24));
        sb_word(&s.bin, s.var_gid);   /* interface: GlobalInvocationId input */
    }
    /* LocalSize 1 for scalar programs, 64 when a MIR_T_GEMM needs the lanes
     * for its cross-lane reduction. With 64 lanes a SCALAR program would run
     * the whole body in every invocation and all 64 would store to mem cell 0,
     * so the result would be whichever lane finished last. */
    { int scalar_prog_ = 1;
      for (size_t q = 0; q < p->n; q++) if (p->ins[q].op == MIR_T_GEMM) scalar_prog_ = 0;
      { uint32_t e[]={s.fn_main,17,scalar_prog_?1:64,1,1};
        spv_ins(&s.bin,OP_EXEC_MODE,e,5); } }
    { uint32_t e[]={1,450}; spv_ins(&s.bin,OP_SOURCE,e,2);}

    /* debug names help spirv-val diagnostics */
    { /* "main\0" padded to 8 bytes = 2 words; content = id + 2 = 3; wc = 4 */
      sb_word(&s.bin,(uint32_t)(4<<16)|OP_NAME); sb_word(&s.bin,s.fn_main);
      sb_word(&s.bin,0x6e69616du); sb_word(&s.bin,0x00000000u); }

    /* ---- annotations (section 08: ALL decorations BEFORE types) ---- */
    { uint32_t o[]={s.t_arr_mem,6,8}; spv_ins(&s.bin,OP_DECORATE,o,3);}   /* ArrayStride */
    { uint32_t o[]={s.t_struct_ssbo,2}; spv_ins(&s.bin,OP_DECORATE,o,2);} /* Block */
    { uint32_t o[]={s.t_pushblk,2}; spv_ins(&s.bin,OP_DECORATE,o,2);}     /* Block */
    { uint32_t o[]={s.var_ssbo,34,0}; spv_ins(&s.bin,OP_DECORATE,o,3);} /* DescriptorSet 0 */
    { uint32_t o[]={s.var_ssbo,33,0}; spv_ins(&s.bin,OP_DECORATE,o,3);} /* Binding 0 */
    { uint32_t o[]={s.var_gid,11,28}; spv_ins(&s.bin,OP_DECORATE,o,3);} /* BuiltIn GlobalInvocationId */
    /* member layout: Block structs need explicit member Offsets */
    { uint32_t o[]={s.t_struct_ssbo,0,35,0}; spv_ins(&s.bin,72 /*MemberDecorate*/,o,4);}
    { uint32_t o[]={s.t_pushblk,0,35,0};     spv_ins(&s.bin,72,o,4);}

    /* ---- types & constants ---- */
    { uint32_t o[]={s.t_void}; spv_ins(&s.bin,OP_TYPE_VOID,o,1);}
    { uint32_t o[]={s.t_bool}; spv_ins(&s.bin,20,o,1);}
    { uint32_t o[]={s.t_i32,32,1}; spv_ins(&s.bin,OP_TYPE_INT,o,3);}
    { uint32_t o[]={s.t_u64,64,0}; spv_ins(&s.bin,OP_TYPE_INT,o,3);}
    { uint32_t o[]={s.t_v3i32,s.t_i32,3}; spv_ins(&s.bin,OP_TYPE_VECTOR,o,3);}
    { uint32_t o[]={s.t_f32,32};    spv_ins(&s.bin,OP_TYPE_FLOAT,o,2);}
    { uint32_t o[]={s.t_f64,64};    spv_ins(&s.bin,OP_TYPE_FLOAT,o,2);}
    { uint32_t o[]={s.t_v2i32,s.t_i32,2}; spv_ins(&s.bin,OP_TYPE_VECTOR,o,3);}

    /* OpConstant i32 N (mem cells incl. result slot) */
    { uint32_t has_tg=0;
      for (size_t q=0;q<p->n;q++) if (p->ins[q].op==MIR_T_GEMM) has_tg=1;
      (void)has_tg;
      /* 64 lanes x 4 per-lane scratch slots for parallel T_GEMM */
      uint32_t ncells=(uint32_t)(s.scratch_base + 64*4 + 1);
      sb_word(&s.bin,(uint32_t)(4<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_i32); sb_word(&s.bin,s.len_mem_cells); sb_word(&s.bin,ncells); }
    { uint32_t o[]={s.t_arr_mem,s.t_u64,s.len_mem_cells}; spv_ins(&s.bin,OP_TYPE_ARRAY,o,3);}
{ uint32_t o[]={s.t_struct_ssbo,s.t_arr_mem}; spv_ins(&s.bin,OP_TYPE_STRUCT,o,2);}
{ uint32_t o[]={s.t_ptr_ssbo,12,s.t_struct_ssbo}; spv_ins(&s.bin,OP_TYPE_POINTER,o,3);}
    /* decorate array stride + block offsets */
            { uint32_t o[]={s.t_i32_in,1,s.t_i32}; spv_ins(&s.bin,OP_TYPE_POINTER,o,3);}
    { uint32_t o[]={s.t_gid_input,1,s.t_v3i32}; spv_ins(&s.bin,OP_TYPE_POINTER,o,3);}
    { uint32_t o[]={s.t_pushblk,s.t_u64}; spv_ins(&s.bin,OP_TYPE_STRUCT,o,2);}

        { uint32_t o[]={s.t_ptr_push,9,s.t_pushblk}; spv_ins(&s.bin,OP_TYPE_POINTER,o,3);}
    { uint32_t o[]={s.t_res_u64,12,s.t_u64}; spv_ins(&s.bin,OP_TYPE_POINTER,o,3);}
    { uint32_t o[]={s.t_fn_void,s.t_void}; spv_ins(&s.bin,OP_TYPE_FUNCTION,o,2);}
    /* type decorations (logical layout: before variables) */

/* constants zero32/zero64/one64 */
    { sb_word(&s.bin,(uint32_t)(4<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_i32); sb_word(&s.bin,s.c_zero32); sb_word(&s.bin,0); }
    { sb_word(&s.bin,(uint32_t)(4<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_i32); sb_word(&s.bin,s.c_i32_1); sb_word(&s.bin,1); }
    { sb_word(&s.bin,(uint32_t)(4<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_i32); sb_word(&s.bin,s.c_i32_2); sb_word(&s.bin,2); }
    { sb_word(&s.bin,(uint32_t)(4<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_i32); sb_word(&s.bin,s.c_i32_sem); sb_word(&s.bin,0x48); }
    { sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_u64); sb_word(&s.bin,s.c_zero64); sb_word(&s.bin,0); sb_word(&s.bin,0);}
    { sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_u64); sb_word(&s.bin,s.c_one64); sb_word(&s.bin,1); sb_word(&s.bin,0);}
    /* cell width in bytes: MIR addresses are BYTE offsets (see the x86-64
     * backend's "No shl rax,3 -- addresses are byte offsets"), so a MIR
     * address must be shifted right by 3 to become an SSBO element index. */
    { sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_u64); sb_word(&s.bin,s.c_three64); sb_word(&s.bin,3); sb_word(&s.bin,0);}
    /* remaining cmts[] constants */
    for (size_t q = 0; q < ncm; q++) {
        uint32_t lo = (uint32_t)(uint64_t)cmts[q].imm;
        uint32_t hi = (uint32_t)((uint64_t)cmts[q].imm >> 32);
        sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
        sb_word(&s.bin,s.t_u64); sb_word(&s.bin,cmts[q].id);
        sb_word(&s.bin,lo); sb_word(&s.bin,hi);
    }


    /* globals */
    { uint32_t o[]={s.t_ptr_ssbo,s.var_ssbo,12}; spv_ins(&s.bin,OP_VARIABLE,o,3);}
    { uint32_t o[]={s.t_ptr_push,s.var_push,9}; spv_ins(&s.bin,OP_VARIABLE,o,3);}
    { uint32_t o[]={s.t_gid_input,s.var_gid,1}; spv_ins(&s.bin,OP_VARIABLE,o,3);}


    /* ---- function ---- */
    { uint32_t o[]={s.t_void,s.fn_main,0,s.t_fn_void}; spv_ins(&s.bin,OP_FUNCTION,o,4);}
    s.lbl_entry = nid(&s);
    { uint32_t o[]={s.lbl_entry}; spv_ins(&s.bin,OP_LABEL,o,1);}

    /* gid.x -> i64 row/thread index into VR space (we use SSA ids >= vr_base) */
    uint32_t id_v3 = nid(&s), id_x32 = nid(&s), id_gid = nid(&s);
    { uint32_t o[]={s.t_v3i32,id_v3,s.var_gid}; spv_ins(&s.bin,OP_LOAD,o,3);}
    { uint32_t o[]={s.t_i32,id_x32,id_v3,0}; spv_ins(&s.bin,OP_COMPOSITE_EXTRACT,o,4);}
    { uint32_t o[]={s.t_u64,id_gid,id_x32}; spv_ins(&s.bin,OPCODE_UCONVERT,o,3);}

    /* MIR VR -> current SSA id map (each write kills the prior version) */
    uint32_t *vrmap = calloc(maxvr, 4);
    #define VRMAP_GET(k) ((uint32_t)((k) < maxvr && vrmap[k] ? vrmap[k] : s.c_zero64))
    #define VRMAP_SET(k,id_) do{ if((size_t)(k) < maxvr) vrmap[(k)]=(id_); }while(0)
    VRMAP_SET(0, id_gid);   /* arg/thread register */

    /* ---- control flow pre-pass ----
     * Map each MIR label id to an SSA label id, and classify jumps:
     * forward (target pc > jump pc) or backward (loop). Backward jumps
     * need SPIR-V structured loops (OpLoopMerge); wave A handles
     * forward-only control flow. */
    uint32_t lbl_ctr = 5000;
    int just_terminator = 0;
    uint32_t cur_block = s.lbl_entry;
    uint32_t last_ret_src = s.c_zero64;


    /* Pre-assign SSA block ids to every MIR label so forward jumps can
     * reference blocks before they are emitted. */
    #define MAXLBL 256
    uint32_t lbl_ssa[MAXLBL];
    for (int z = 0; z < MAXLBL; z++) lbl_ssa[z] = 0;
    int has_backward = 0;
    for (size_t q = 0; q < p->n; q++) {
        const wubu_mir_instr_t *in2 = &p->ins[q];
        if (in2->op == MIR_LABEL && in2->label < MAXLBL && !lbl_ssa[in2->label])
            lbl_ssa[in2->label] = nid(&s);
    }
    /* Loop analysis: find labels that are targets of backward jumps.
     * For each loop header label we pre-assign exit + continue blocks. */
    int lbl_is_loop_header[MAXLBL];
    uint32_t lbl_exit[MAXLBL], lbl_cont[MAXLBL];
    for (int z = 0; z < MAXLBL; z++) { lbl_is_loop_header[z] = 0; lbl_exit[z] = 0; lbl_cont[z] = 0; }
    for (size_t q = 0; q < p->n; q++) {
        const wubu_mir_instr_t *in2 = &p->ins[q];
        if (in2->op != MIR_JZ && in2->op != MIR_JNZ && in2->op != MIR_JMP)
            continue;
        size_t t;
        for (t = 0; t < p->n; t++)
            if (p->ins[t].op == MIR_LABEL && p->ins[t].label == in2->label)
                break;
        if (t < q && in2->label < MAXLBL) {
            has_backward = 1;
            if (!lbl_is_loop_header[in2->label]) {
                lbl_is_loop_header[in2->label] = 1;
                lbl_exit[in2->label] = nid(&s);
                lbl_cont[in2->label] = nid(&s);
            }
        }
    }

    for (size_t pc = 0; pc < p->n; pc++) {
        const wubu_mir_instr_t *in = &p->ins[pc];
        switch (in->op) {
        case MIR_CONST: {
            size_t w; for (w = 0; w < ncm; w++) if (cmts[w].imm == in->imm) break;
            uint32_t csrc = (w < ncm) ? cmts[w].id : s.c_zero64;
            uint32_t dst_id = nid(&s);
            { uint32_t o[]={s.t_u64,dst_id,csrc}; spv_ins(&s.bin,83,o,3); }  /* OpCopyObject */
            VRMAP_SET(in->dst, dst_id);
            break;
        }
        /* Copy and integer width casts. These had NO case at all, so the
         * emitted SPIR-V silently dropped them. `127 + 1` lowers to
         * CONST/SEXT32/MOV/RET; without the MOV the RET read an unmapped
         * (zero) value and the Vulkan backend returned 0 for every program.
         *
         * Same failure mode as the ptx backend and the interpreter's
         * positional dispatch table: an opcode with no handler vanishes
         * instead of failing loudly. Add the opcode to wubu_mir_op_t and this
         * switch will silently miscompile -- grep every backend. */
        case MIR_MOV: {
            uint32_t dst_id = nid(&s);
            { uint32_t o[]={s.t_u64,dst_id,VRMAP_GET(in->a)};
              spv_ins(&s.bin,83,o,3); }                              /* OpCopyObject */
            VRMAP_SET(in->dst, dst_id);
            break;
        }
        /* Unary negate. Like the cast opcodes this had NO case, so `-1`
         * silently produced 0 on the Vulkan backend. SPIR-V has no OpSNegate;
         * negate with two's-complement subtraction from zero. */
        case MIR_NEG: {
            uint32_t d = nid(&s);
            /* 130 = OpISub (128 is OpIAdd -- verified against the
             * MIR_ADD/MIR_SUB/MIR_MUL mapping in this same file) */
            { uint32_t o[]={s.t_u64,d,s.c_zero64,VRMAP_GET(in->a)};
              spv_ins(&s.bin,130,o,4); }                             /* OpISub 0 - a */
            VRMAP_SET(in->dst, d);
            break;
        }
        /* Sign/zero-extension on 64-bit values: truncate to the narrower
         * type, then reinterpret back to u64. OpUConvert 126 handles
         * truncation; the shift/mask below is the portable spelling that
         * does not depend on signedness rules. */
        case MIR_SEXT32: case MIR_SEXT16: case MIR_SEXT8:
        case MIR_ZEXT32: {
            int bits = (in->op==MIR_SEXT32 || in->op==MIR_ZEXT32) ? 32 :
                       (in->op==MIR_SEXT16) ? 16 : 8;
            uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
            uint32_t src  = VRMAP_GET(in->a);
            /* A negative MIR_CONST is already stored sign-extended in its
             * 64-bit OpConstant field, so narrowing it again is wrong.
             * `100 - 250` folds to MIR_CONST(-150) and still carries a
             * trailing MIR_SEXT32; re-truncating turned -150 into 4294967146.
             * Detect that case by checking whether the source id is one of the
             * negative constants we registered, and pass it through. */
            int src_is_neg_const = 0;
            for (size_t q_ = 0; q_ < ncm; q_++)
                if (cmts[q_].id == src && cmts[q_].imm < 0) { src_is_neg_const = 1; break; }
            if (src_is_neg_const) {
                uint32_t d0 = nid(&s);
                { uint32_t o[]={s.t_u64,d0,src}; spv_ins(&s.bin,83,o,3); }
                VRMAP_SET(in->dst, d0);
                break;
            }
            /* The mask and sign ids were registered in the prepass above, so
             * they are already defined in the global constants section. Look
             * them up in cmts[] rather than emitting a new OpConstant here. */
            uint32_t cid_mask = 0, cid_31 = 0, cid_one = 0, cid_himask = 0;
            { size_t w_;
              long long himask = (long long)(int64_t)(0xFFFFFFFFULL << 32);
              for (w_ = 0; w_ < ncm; w_++) {
                  if (cmts[w_].imm == (long long)mask)     cid_mask   = cmts[w_].id;
                  if (cmts[w_].imm == 1LL)                 cid_one    = cmts[w_].id;
                  if (cmts[w_].imm == himask)              cid_himask = cmts[w_].id;
              }
              /* shift amount is bits-1 (e.g. 31 for a 32-bit cast), which is
               * NOT the same constant as the sign bit value 2^31 */
              for (w_ = 0; w_ < ncm; w_++)
                  if (cmts[w_].imm == (long long)(bits - 1)) { cid_31 = cmts[w_].id; break; }
            }
            if (!cid_mask || !cid_31 || !cid_one || !cid_himask) {
                fprintf(stderr, "[spirv] missing cast constant bits=%d "
                        "(mask=%u shift=%u one=%u himask=%u)\n",
                        bits, cid_mask, cid_31, cid_one, cid_himask);
                break;
            }
            uint32_t masked = nid(&s);
            { uint32_t o[]={s.t_u64,masked,src,cid_mask}; spv_ins(&s.bin,199,o,4); } /* and */
            if (in->op == MIR_ZEXT32) {
                VRMAP_SET(in->dst, masked);
            } else {
                /* Sign-extend 32 -> 64 on a value held in a u64 register.
                 *
                 * The usual (x ^ sign) - sign trick is WRONG here. The
                 * u64's bits 32..63 are already zero, so XOR-ing bit 31 in
                 * just sets bit 31; the borrow does not propagate back down
                 * from bit 63, so the result keeps bit 31 set. Measured: every
                 * value in 0..2^31-1 came out as 2147483648.
                 *
                 * Correct sequence: mask to 32 bits, then broadcast bit 31
                 * into the top 32 bits with a shift-and-subtract that is
                 * exact in u64 arithmetic:
                 *     hi    = (masked >> 31) & 1        // 0 or 1
                 *     hi64  = hi * 0xFFFFFFFF00000000   // 0 or all-ones
                 *     result= (masked & 0xFFFFFFFF) | hi64
                 * OpShiftRightLogical(194), OpAnd(199), OpIMul(132),
                 * OpBitwiseOr(197). No final mask on the result -- masking
                 * it would discard the sign extension again. */
                uint32_t hi = nid(&s), hib = nid(&s),
                         hi64 = nid(&s), lo64 = nid(&s), res = nid(&s);
                { uint32_t o0[]={s.t_u64,hi,masked,cid_31};
                  spv_ins(&s.bin,194,o0,4); }                    /* lshr 31 */
                { uint32_t o1[]={s.t_u64,hib,hi,cid_one};
                  spv_ins(&s.bin,199,o1,4); }                    /* and 1 */
                { uint32_t o2[]={s.t_u64,hi64,hib,cid_himask};
                  spv_ins(&s.bin,132,o2,4); }                    /* mul */
                { uint32_t o3[]={s.t_u64,lo64,masked,cid_mask};
                  spv_ins(&s.bin,199,o3,4); }                    /* and mask */
                { uint32_t o4[]={s.t_u64,res,lo64,hi64};
                  spv_ins(&s.bin,197,o4,4); }                    /* or */
                VRMAP_SET(in->dst, res);
            }
            break;
        }
        case MIR_TO_PTR:
            /* Cell offset -> pointer into the mem SSBO. SPIR-V here is
             * single-address-space, so the offset is already the value the
             * loads/stores use; a copy keeps dst defined for the VR map. */
        {
            uint32_t dst_id = nid(&s);
            { uint32_t o[]={s.t_u64,dst_id,VRMAP_GET(in->a)};
              spv_ins(&s.bin,83,o,3); }
            VRMAP_SET(in->dst, dst_id);
            break;
        }
        case MIR_ADD: case MIR_SUB: case MIR_MUL:
        case MIR_AND: case MIR_OR: case MIR_XOR: {
            uint16_t opc =
                in->op==MIR_ADD ? 128 : in->op==MIR_SUB ? 130 :
                in->op==MIR_MUL ? 132 : in->op==MIR_AND ? 199 :
                in->op==MIR_OR  ? 197 : 198;
            uint32_t d = nid(&s);
            { uint32_t o[]={s.t_u64,d,VRMAP_GET(in->a),VRMAP_GET(in->b)};
              spv_ins(&s.bin,opc,o,4); }
            VRMAP_SET(in->dst, d);
            break;
        }
        /* Integer divide / remainder / shifts. These had no case, so every
         * one silently evaluated to whatever the destination already held --
         * the same "missing opcode = plausible wrong answer" failure as
         * MIR_MOV and MIR_NEG. The gauntlet does not catch it because
         * hd_run_prog silently falls back to wubu_mir_interp, and the Vulkan
         * driver is not in the gauntlet target list at all.
         *
         * Opcode numbers are the SPIR-V core ones. Verified by assembling a
         * reference module with spirv-as and decoding the emitted words,
         * cross-checked against the ADD/SUB/MUL mapping just above
         * (128 OpIAdd / 130 OpISub / 132 OpIMul):
         *   134 OpUDiv  135 OpSDiv  137 OpUMod  139 OpSMod
         *   195 OpShiftRightArithmetic  196 OpShiftLeftLogical  200 OpNot
         *
         * C's % is TRUNCATING (the remainder takes the sign of the dividend):
         *   -100 % 7 == -2   because -100 / 7 == -14
         * SPIR-V's OpSMod (139) is FLOOR-based and gives 5 instead, so it is
         * the wrong instruction. OpSRem (138) truncates toward zero and
         * matches C. Verified on both dzn and lavapipe: with 139 the backend
         * returned 5 for `(-100) % 7` where x86-64 returns -2.
         *
         * Do not "fix" this to 139. Same trap as OpSDiv: SPIR-V's SDiv is
         * truncating (matches C), but its SMod is not. */
        case MIR_DIV: case MIR_UDIV: case MIR_MOD: case MIR_UMOD: {
            uint32_t opc = (in->op==MIR_DIV)  ? 135 :   /* OpSDiv  - truncating */
                           (in->op==MIR_UDIV) ? 134 :   /* OpUDiv  */
                           (in->op==MIR_MOD)  ? 138 :   /* OpSRem  - truncating */
                                            137;      /* OpUMod  */
            uint32_t d = nid(&s);
            { uint32_t o[]={s.t_u64,d,VRMAP_GET(in->a),VRMAP_GET(in->b)};
              spv_ins(&s.bin,opc,o,4); }
            VRMAP_SET(in->dst, d);
            break;
        }
        case MIR_SHL: case MIR_SHR: {
            uint32_t opc = (in->op==MIR_SHL) ? 196      /* OpShiftLeftLogical */
                                           : 195;     /* OpShiftRightArithmetic */
            uint32_t d = nid(&s);
            { uint32_t o[]={s.t_u64,d,VRMAP_GET(in->a),VRMAP_GET(in->b)};
              spv_ins(&s.bin,opc,o,4); }
            VRMAP_SET(in->dst, d);
            break;
        }
        case MIR_NOT: {
            /* SPIR-V OpNot(200) is a unary bitwise complement. */
            uint32_t d = nid(&s);
            { uint32_t o[]={s.t_u64,d,VRMAP_GET(in->a)};
              spv_ins(&s.bin,200,o,3); }
            VRMAP_SET(in->dst, d);
            break;
        }
        /* f64 arithmetic and conversions. MIR holds an f64 as its raw 64-bit
         * pattern in a u64 VR, so unlike the f32 path there is no v2i32
         * unpack/pack -- OpBitcast between ulong and double is all that is
         * needed. These had NO case at all, so every double-precision program
         * returned 0 on the GPU: 1.5 + 2.5 and 3.0 * 1.5 both gave 0 while
         * x86-64 gave the correct 4.0 and 4.5.
         *
         * Opcodes verified by assembling a reference module with spirv-as and
         * decoding the emitted words:
         *   FAdd=129 FSub=131 FMul=133 FDiv=136 FNegate=127
         *   FOrdEqual=180 FOrdNotEqual=182 FOrdLessThan=184
         *   FOrdGreaterThan=186 FOrdLessThanEqual=188 FOrdGreaterThanEqual=190
         * (the <= / >= forms are 188/190, NOT 185/187 -- verified by
         *  assembling and decoding, since 185/187 do not exist)
         *   ConvertUToF=112 ConvertFToS=109 ConvertFToU=110
         *   Bitcast=124
         * NOTE: SPIR-V's FOrd* comparisons return bool; the 1/0 the rest of
         * this backend works in is produced by OpSelect, not by the compare. */
        case MIR_DADD: case MIR_DSUB: case MIR_DMUL: case MIR_DDIV:
        case MIR_DNEG: {
            uint32_t fa = nid(&s);
            { uint32_t o[]={s.t_f64, fa, VRMAP_GET(in->a)}; spv_ins(&s.bin,124,o,3); }
            uint32_t dstf = nid(&s);
            if (in->op == MIR_DNEG) {
                { uint32_t o[]={s.t_f64,dstf,fa}; spv_ins(&s.bin,127,o,3); }
            } else {
                uint32_t fb = nid(&s);
                { uint32_t o[]={s.t_f64, fb, VRMAP_GET(in->b)}; spv_ins(&s.bin,124,o,3); }
                uint16_t opc = in->op==MIR_DADD ? 129 :
                               in->op==MIR_DSUB ? 131 :
                               in->op==MIR_DMUL ? 133 : 136;
                { uint32_t o[]={s.t_f64,dstf,fa,fb}; spv_ins(&s.bin,opc,o,4); }
            }
            uint32_t pk = nid(&s);
            { uint32_t o[]={s.t_u64, pk, dstf}; spv_ins(&s.bin,124,o,3); }
            VRMAP_SET(in->dst, pk);
            break;
        }
        case MIR_DLT: case MIR_DLE: case MIR_DGT: case MIR_DGE:
        case MIR_DEQ: case MIR_DNE: {
            uint32_t fa = nid(&s), fb = nid(&s);
            { uint32_t o[]={s.t_f64, fa, VRMAP_GET(in->a)}; spv_ins(&s.bin,124,o,3); }
            { uint32_t o[]={s.t_f64, fb, VRMAP_GET(in->b)}; spv_ins(&s.bin,124,o,3); }
            uint16_t opc = in->op==MIR_DLT ? 184 : in->op==MIR_DLE ? 188 :
                           in->op==MIR_DGT ? 186 : in->op==MIR_DGE ? 190 :
                           in->op==MIR_DEQ ? 180 : 182;
            uint32_t b = nid(&s);
            { uint32_t o[]={s.t_bool, b, fa, fb}; spv_ins(&s.bin,opc,o,4); }
            /* bool -> 1/0 u64 via OpSelect */
            uint32_t sel = nid(&s);
            { uint32_t o[]={s.t_u64, sel, b, s.c_one64, s.c_zero64};
              spv_ins(&s.bin,169 /*OpSelect*/,o,5); }
            VRMAP_SET(in->dst, sel);
            break;
        }
        case MIR_DITOF: {   /* int -> double */
            uint32_t f = nid(&s);
            { uint32_t o[]={s.t_f64, f, VRMAP_GET(in->a)}; spv_ins(&s.bin,112 /*ConvertUToF*/,o,3); }
            uint32_t pk = nid(&s);
            { uint32_t o[]={s.t_u64, pk, f}; spv_ins(&s.bin,124,o,3); }
            VRMAP_SET(in->dst, pk);
            break;
        }
        case MIR_DTOI: case MIR_DTOI_U: {   /* double -> int */
            uint32_t f = nid(&s);
            { uint32_t o[]={s.t_f64, f, VRMAP_GET(in->a)}; spv_ins(&s.bin,124,o,3); }
            uint32_t iv = nid(&s);
            /* 110 ConvertFToU (unsigned), 109 ConvertFToS (signed) --
             * verified by spirv-as round-trip, NOT 115/116 */
            { uint32_t o[]={s.t_u64, iv, f};
              spv_ins(&s.bin, in->op==MIR_DTOI_U ? 110 : 109, o, 3); }
            VRMAP_SET(in->dst, iv);
            break;
        }
        case MIR_FADD: case MIR_FSUB: case MIR_FMUL: case MIR_FDIV:
        case MIR_FNEG: {
            /* MIR keeps f32 bits in the low 32 of the i64 VR. Unpack:
             * u64 -Bitcast-> v2i32 -Extract.0-> i32 -Bitcast-> f32; op;
             * repack via Bitcast i32, CompositeConstruct {lo,0}, Bitcast i64.
             * Opcodes verified by spirv-as round-trip:
             *   FAdd=129 FSub=131 FMul=133 FDiv=136 FNegate=127
             *   Bitcast=124 CompositeExtract=81 CompositeConstruct=80 */
            uint32_t fa = spirv_unpack_f32(&s, VRMAP_GET(in->a));
            uint32_t dst_id = nid(&s);
            if (in->op == MIR_FNEG) {
                uint32_t o[]={s.t_f32,dst_id,fa};
                spv_ins(&s.bin,127,o,3);
            } else {
                uint32_t fb = spirv_unpack_f32(&s, VRMAP_GET(in->b));
                uint16_t opc = in->op==MIR_FADD ? 129 :
                               in->op==MIR_FSUB ? 131 :
                               in->op==MIR_FMUL ? 133 : 136;
                uint32_t o[]={s.t_f32,dst_id,fa,fb};
                spv_ins(&s.bin,opc,o,4);
            }
            /* pack: Bitcast f32->i32, CompositeConstruct {lo,0}, Bitcast i64 */
            {
                uint32_t ib = nid(&s), cc = nid(&s), pk = nid(&s);
                { uint32_t o[]={s.t_i32,ib,dst_id};          spv_ins(&s.bin,124,o,3);}
                { uint32_t o[]={s.t_v2i32,cc,ib,s.c_zero32}; spv_ins(&s.bin,80,o,4);}
                { uint32_t o[]={s.t_u64,pk,cc};              spv_ins(&s.bin,124,o,3);}
                VRMAP_SET(in->dst, pk);
            }
            break;
        }
        case MIR_LABEL: {
            if (lbl_ssa[in->label]) {
                if (!just_terminator) {
                    uint32_t o[]={lbl_ssa[in->label]};
                    spv_ins(&s.bin,OPCODE_BRANCH,o,1);
                }
                just_terminator = 0;
                uint32_t o[]={lbl_ssa[in->label]};
                spv_ins(&s.bin,OP_LABEL,o,1);
                cur_block = lbl_ssa[in->label];
                /* loop header: structured loop with continue + exit blocks */
                if (lbl_is_loop_header[in->label]) {
                    uint32_t om[]={lbl_exit[in->label], lbl_cont[in->label], 0};
                    spv_ins(&s.bin,246 /*OpLoopMerge*/,om,3);
                    uint32_t ob[]={lbl_cont[in->label]};
                    spv_ins(&s.bin,OPCODE_BRANCH,ob,1);
                    just_terminator = 1;
                    /* the continue block is where the body resumes */
                    { uint32_t oc[]={lbl_cont[in->label]}; spv_ins(&s.bin,OP_LABEL,oc,1); }
                }
            }
            break;
        }
        case MIR_JMP: {
            if (in->label < MAXLBL && lbl_is_loop_header[in->label]) {
                uint32_t o[]={lbl_ssa[in->label]};   /* back-edge to header */
                spv_ins(&s.bin,OPCODE_BRANCH,o,1);
                just_terminator = 1;
                /* emit exit label for post-loop code */
                { uint32_t oe[]={lbl_exit[in->label]}; spv_ins(&s.bin,OP_LABEL,oe,1); }
                break;
            }
            if (in->label < MAXLBL && lbl_ssa[in->label]) {
                uint32_t o[]={lbl_ssa[in->label]};
                spv_ins(&s.bin,OPCODE_BRANCH,o,1);
                just_terminator = 1;
            }
            break;
        }
        case MIR_JZ: case MIR_JNZ: {
            /* backward conditional = loop latch */
            if (in->label < MAXLBL && lbl_is_loop_header[in->label]) {
                uint32_t cond = nid(&s);
                uint16_t opc = in->op == MIR_JZ ? OPCODE_I_EQ : OPCODE_I_NE;
                { uint32_t o[]={s.t_bool,cond,VRMAP_GET(in->a),s.c_zero64};
                  spv_ins(&s.bin,opc,o,4); }
                /* taken -> branch back to loop HEADER (the back-edge);
                 * not-taken -> exit. Header holds the OpLoopMerge so this
                 * forms the legal continue->header edge. */
                uint32_t o2[]={cond,lbl_ssa[in->label],lbl_exit[in->label]};
                spv_ins(&s.bin,OPCODE_BRANCH_COND,o2,3);
                just_terminator = 1;
                /* post-loop code lands in the exit block */
                { uint32_t oe[]={lbl_exit[in->label]}; spv_ins(&s.bin,OP_LABEL,oe,1); }
                just_terminator = 0;
                break;
            }
            if (in->label < MAXLBL && lbl_ssa[in->label]) {
                /* cond = (a == 0) for JZ, (a != 0) for JNZ */
                uint32_t cond = nid(&s);
                uint16_t opc = in->op == MIR_JZ ? 170 /*INotEqual: a!=0 -> taken*/ :
                                                  170;
                /* JZ: branch if zero => INotEqual(a, 0); JNZ: IEqual? No:
                 * JZ taken when a==0 -> OpIEqual; JNZ taken when a!=0 ->
                 * OpINotEqual. */
                opc = in->op == MIR_JZ ? OPCODE_I_EQ : OPCODE_I_NE;
                { uint32_t o[]={s.t_bool,cond,VRMAP_GET(in->a),s.c_zero64};
                  spv_ins(&s.bin,opc,o,4); }
                uint32_t fall = nid(&s);
                uint32_t true_tgt  = in->op == MIR_JZ ? fall : lbl_ssa[in->label];
                uint32_t false_tgt = in->op == MIR_JZ ? lbl_ssa[in->label] : fall;
                /* structured selection: merge block is the true target when
                 * jumping to a later label, else the fall-through */
                uint32_t merge = in->op == MIR_JZ ? lbl_ssa[in->label] : fall;
                { uint32_t o3[]={merge,0 /*None*/}; spv_ins(&s.bin,OPCODE_SELECTION_MERGE,o3,2); }
                uint32_t o2[]={cond,true_tgt,false_tgt};
                spv_ins(&s.bin,OPCODE_BRANCH_COND,o2,3);
                just_terminator = 0;  /* fall-through block is now current */
                { uint32_t o[]={fall}; spv_ins(&s.bin,OP_LABEL,o,1); }
            }
            break;
        }
        case MIR_LOAD: {
            /* MIR cell i -> SSBO element i+1 (cell 0 = return slot).
             * idx64 = (a >> 3) + 1; val = load(ptr_u64); dst = val
             * The >>3 is REQUIRED: MIR addresses are BYTE offsets into the
             * interpreter's cell array, so address 8 is cell 1. Indexing the
             * SSBO with the raw byte address made every load read 8 cells too
             * far -- which is why memory-using programs returned 0 while the
             * straight-line arithmetic cases (no MIR_LOAD) passed. Matches
             * wubu_isa_x86_64.c:1632 ("No shl rax,3 -- addresses are byte
             * offsets") and wubu_mir_interp's mem_load64(mem, byte_addr). */
            uint32_t cell = nid(&s);
            { uint32_t o[]={s.t_u64,cell,VRMAP_GET(in->a),s.c_three64};
              spv_ins(&s.bin,OPCODE_SHIFT_RIGHT_LOGICAL,o,4); }
            uint32_t idx64 = nid(&s);
            { uint32_t o[]={s.t_u64,idx64,cell,s.c_one64};
              spv_ins(&s.bin,OPCODE_IADD,o,4); }
            uint32_t uptr = nid(&s);
            { uint32_t o[]={s.t_res_u64,uptr,s.var_ssbo,s.c_zero32,idx64};
              spv_ins(&s.bin,OP_ACCESS_CHAIN,o,5); }
            uint32_t val = nid(&s);
            { uint32_t o[]={s.t_u64,val,uptr}; spv_ins(&s.bin,OP_LOAD,o,3); }
            VRMAP_SET(in->dst, val);
            break;
        }
        case MIR_STORE: {
            /* Same byte-offset -> cell conversion as MIR_LOAD. */
            uint32_t cell = nid(&s);
            { uint32_t o[]={s.t_u64,cell,VRMAP_GET(in->a),s.c_three64};
              spv_ins(&s.bin,OPCODE_SHIFT_RIGHT_LOGICAL,o,4); }
            uint32_t idx64 = nid(&s);
            { uint32_t o[]={s.t_u64,idx64,cell,s.c_one64};
              spv_ins(&s.bin,OPCODE_IADD,o,4); }
            uint32_t uptr = nid(&s);
            { uint32_t o[]={s.t_res_u64,uptr,s.var_ssbo,s.c_zero32,idx64};
              spv_ins(&s.bin,OP_ACCESS_CHAIN,o,5); }
            { uint32_t o[]={uptr,VRMAP_GET(in->b)}; spv_ins(&s.bin,OP_STORE,o,2); }
            break;
        }
        case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE:
        case MIR_GT: case MIR_GE:
        case MIR_ULT: case MIR_ULE: case MIR_UGT: case MIR_UGE: {
            /* int compare -> bool -> select 1/0 (u64) */
            uint16_t opc =
                in->op==MIR_EQ ? 170 : in->op==MIR_NE ? 171 :
                in->op==MIR_LT ? 177 : in->op==MIR_LE ? 179 :
                in->op==MIR_GT ? 173 : in->op==MIR_GE ? 175 :
                in->op==MIR_ULT ? 176 : in->op==MIR_ULE ? 178 :
                in->op==MIR_UGT ? 172 : 174;
            uint32_t bl = nid(&s);
            { uint32_t o[]={s.t_bool,bl,VRMAP_GET(in->a),VRMAP_GET(in->b)};
              spv_ins(&s.bin,opc,o,4); }
            uint32_t pk = nid(&s);
            { uint32_t o[]={s.t_u64,pk,bl,s.c_one64,s.c_zero64};
              spv_ins(&s.bin,169,o,5); }
            VRMAP_SET(in->dst, pk);
            break;
        }
        case MIR_FEQ: case MIR_FLT: {
            /* unpack both to f32, FOrdEqual(180)/FOrdLessThan(184) -> bool,
             * OpSelect(169) u64 1/0. MIR semantics: FEQ/FLT return 0/1. */
            uint32_t fa = spirv_unpack_f32(&s, VRMAP_GET(in->a));
            uint32_t fb = spirv_unpack_f32(&s, VRMAP_GET(in->b));
            uint16_t opc = in->op == MIR_FEQ ? 180 : 184;
            uint32_t bl = nid(&s);
            { uint32_t o[]={s.t_bool,bl,fa,fb}; spv_ins(&s.bin,opc,o,4); }
            uint32_t pk = nid(&s);
            { uint32_t o[]={s.t_u64,pk,bl,s.c_one64,s.c_zero64};
              spv_ins(&s.bin,169,o,5); }
            VRMAP_SET(in->dst, pk);
            break;
        }
        case MIR_FNE: case MIR_FLE: {
            /* FUnordNotEqual=183 for NE; FOrdGreaterThanEqual=175 for LE? no:
               LE = 179 per earlier table (S_LE). Float LE = FOrdLessThanEqual
               which we verified as 184's sibling... use round-tripped numbers:
               FOrdNotEqual handled later wave; keep FLT-only now. */
            break;
        }
        case MIR_RET:
            last_ret_src = VRMAP_GET(in->a);
            just_terminator = 1;
            break;
        case MIR_T_GEMM:
            emit_tgemm_spirv(&s, in, cmts, ncm,
                             id_gid, cur_block, vrmap, maxvr,
                             &just_terminator);
            /* helper leaves us in an open labeled block */
            just_terminator = 0;
            break;
        /* Deliberately loud. An opcode with no case here is DROPPED, and the
         * program then returns a plausible wrong answer instead of failing.
         * That is how MIR_MOV, MIR_NEG, the casts, the integer div/mod/shift
         * group, the f64 core and MIR_BREAK/MIR_CONTINUE all went unnoticed.
         *
         * Do not restore a silent `break` here. If an opcode is genuinely not
         * implemented yet, say so in the log so the gap is visible; keep the
         * list in sync with the coverage audit in
         * references/backend-opcode-coverage.md. */
        default:
            fprintf(stderr, "[spirv] UNIMPLEMENTED opcode %d at pc=%zu -- "
                            "dropped; result will be wrong\n", (int)in->op, pc);
            break;
        }
    }

    /* cross-WG + intra-WG barrier: all T_GEMM C-stores must be visible before any
     * lane loads the result cell for the RET value. OpControlBarrier: exec=Workgroup(2),
     * mem=Device(1), sem=AcquireRelease(0x8)|UniformMemory(0x40). */
    { uint32_t ob[]={s.c_i32_2,s.c_i32_1,s.c_i32_sem}; spv_ins(&s.bin,224,ob,3); }

    /* store RET value into mem cell 0 (the return slot) */
    {
        uint32_t res_ptr = nid(&s);
        { uint32_t o[]={s.t_res_u64,res_ptr,s.var_ssbo,s.c_zero32,s.c_zero32};
          spv_ins(&s.bin,OP_ACCESS_CHAIN,o,5); }   /* result,base,idx,idx */
        { uint32_t o[]={res_ptr,last_ret_src}; spv_ins(&s.bin,OP_STORE,o,2); }
    }

    { uint32_t o[1]; spv_ins(&s.bin,OP_RETURN,o,0);}
    { uint32_t o[1]; spv_ins(&s.bin,OP_FUNCTION_END,o,0);}

    /* patch bound */
    s.bin.w[3] = s.next_id;
    *out = (uint8_t*)s.bin.w;
    *out_n = s.bin.n*4;
    return 0;
}

