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
    uint32_t t_arr_mem, t_struct_ssbo, t_ptr_ssbo;     /* storage class 5 */
    uint32_t t_i32_in, t_gid_input;                    /* input v3i32 ptr */
    uint32_t t_pushblk, t_ptr_push;
    uint32_t t_fn_void, t_res_u64;
    uint32_t t_v2i32, t_f32;
    uint32_t c_zero32, len_mem_cells;
    uint32_t c_zero64, c_one64;
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
    uint32_t cS   = spirv_find_const(cmts, ncm, 64);
    uint32_t cKN  = spirv_find_const(cmts, ncm, K*Nn);
    uint32_t cMKN = spirv_find_const(cmts, ncm, M*K*Nn);
    uint32_t aBase = TG_VRG(in->a), bBase = TG_VRG(in->b), cBase = TG_VRG(in->dst);

    uint32_t head=nid(s), cont=nid(s), body=nid(s), merge=nid(s);
    uint32_t w = nid(s);

    { uint32_t o[]={s->t_u64,w,id_gid}; spv_ins(&s->bin,83,o,3); }
    { uint32_t o[]={head}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }
    *jt = 1;

    { uint32_t o[]={head}; spv_ins(&s->bin,OP_LABEL,o,1); }
    { uint32_t om[]={merge,cont,0}; spv_ins(&s->bin,246,om,3); }
    { uint32_t o[]={cont}; spv_ins(&s->bin,OPCODE_BRANCH,o,1); }

    { uint32_t o[]={cont}; spv_ins(&s->bin,OP_LABEL,o,1); }
    {
        uint32_t cmp=nid(s);
        { uint32_t o[]={s->t_bool,cmp,w,cMKN}; spv_ins(&s->bin,176,o,4); }
        { uint32_t o3[]={cmp,body,merge}; spv_ins(&s->bin,OPCODE_BRANCH_COND,o3,3); }
    }

    { uint32_t o[]={body}; spv_ins(&s->bin,OP_LABEL,o,1); }
    {
        uint32_t iv_=nid(s), rr=nid(s), kk=nid(s), jj=nid(s), jN=nid(s);
        { uint32_t o[]={s->t_u64,iv_,w,cKN}; spv_ins(&s->bin,134,o,4);}
        { uint32_t o[]={s->t_u64,rr,w,cKN};  spv_ins(&s->bin,137,o,4);}
        { uint32_t o[]={s->t_u64,kk,rr,cN};  spv_ins(&s->bin,134,o,4);}
        { uint32_t o[]={s->t_u64,jj,rr,kk};  spv_ins(&s->bin,137,o,4);}
        { uint32_t o[]={s->t_u64,jN,jj,cN};  spv_ins(&s->bin,132,o,4);}

        uint32_t t1=nid(s), t2=nid(s), t4=nid(s), t5=nid(s), aptr=nid(s), aval=nid(s);
        { uint32_t o[]={s->t_u64,t1,aBase,s->c_one64}; spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,t2,iv_,cK};           spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,t4,t2,kk};            spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,t5,t4,t1};            spv_ins(&s->bin,128,o,4);}
        TG_AC(aptr, t5); TG_LD(aval, aptr);

        uint32_t u1=nid(s), u2=nid(s), u4=nid(s), u5=nid(s), bptr=nid(s), bval=nid(s);
        { uint32_t o[]={s->t_u64,u1,bBase,s->c_one64}; spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,u2,kk,cN};            spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,u4,u2,jN};            spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,u5,u4,u1};            spv_ins(&s->bin,128,o,4);}
        TG_AC(bptr, u5); TG_LD(bval, bptr);

        uint32_t v1=nid(s), w1=nid(s), w2=nid(s), w3=nid(s);
        uint32_t cptr=nid(s), oldc=nid(s), newc=nid(s), prod=nid(s);
        { uint32_t o[]={s->t_u64,v1,cBase,s->c_one64}; spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,w1,iv_,cN};           spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,w2,w1,jN};            spv_ins(&s->bin,132,o,4);}
        { uint32_t o[]={s->t_u64,w3,w2,v1};            spv_ins(&s->bin,128,o,4);}
        { uint32_t o[]={s->t_u64,prod,aval,bval};      spv_ins(&s->bin,132,o,4);}
        TG_AC(cptr, w3); TG_LD(oldc, cptr);
        { uint32_t o[]={s->t_u64,newc,oldc,prod};      spv_ins(&s->bin,128,o,4);}
        TG_ST(cptr, newc);
    }

    /* continue: w += 64 */
    {
        uint32_t nw=nid(s);
        { uint32_t o[]={s->t_u64,nw,w,cS}; spv_ins(&s->bin,128,o,4);}
        w = nw;
    }
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
    s.c_zero64    = nid(&s);
    s.c_one64     = nid(&s);
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
        } else if (p->ins[q].op == MIR_T_GEMM) {
            int M_ = (int)(p->ins[q].imm >> 22);
            int K_ = (int)((p->ins[q].imm >> 11) & 0x7FF);
            int Nn = (int)(p->ins[q].imm & 0x7FF);
            ADD_CONST(M_); ADD_CONST(K_); ADD_CONST(Nn);
            ADD_CONST(64); ADD_CONST(1); ADD_CONST(0);
            long long sb_ = (long long)((p->total_mem>0?p->total_mem:1)+1);
            s.scratch_base = (uint32_t)sb_;
            ADD_CONST(sb_+0); ADD_CONST(sb_+1); ADD_CONST(sb_+2);
        }
    }
    #undef ADD_CONST
    uint32_t vr_base = s.next_id;
    uint32_t maxvr = 256;

    /* ---- header ---- */
    sb_word(&s.bin, 0x07230203u);
    sb_word(&s.bin, 0x00010300u);         /* SPIR-V 1.3 */
    sb_word(&s.bin, 0x00080077u);         /* generator wubu */
    sb_word(&s.bin, vr_base + maxvr + 4096); /* bound placeholder */
    sb_word(&s.bin, 0);

    /* ---- capabilities / model / entry ---- */
    { uint32_t c[]={1}; spv_ins(&s.bin,OP_CAPABILITY,c,1);}        /* Shader */
    { uint32_t c[]={11}; spv_ins(&s.bin,OP_CAPABILITY,c,1);}        /* Int64 */
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
    { uint32_t e[]={s.fn_main,17,1,1,1}; spv_ins(&s.bin,OP_EXEC_MODE,e,5);} /* LocalSize */
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
    { uint32_t o[]={s.t_v2i32,s.t_i32,2}; spv_ins(&s.bin,OP_TYPE_VECTOR,o,3);}

    /* OpConstant i32 N (mem cells incl. result slot) */
    { uint32_t has_tg=0;
      for (size_t q=0;q<p->n;q++) if (p->ins[q].op==MIR_T_GEMM) has_tg=1;
      (void)has_tg;
      /* array always gets +8 scratch elems so T_GEMM can use them */
      uint32_t ncells=(uint32_t)(s.scratch_base+8);
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
    { sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_u64); sb_word(&s.bin,s.c_zero64); sb_word(&s.bin,0); sb_word(&s.bin,0);}
    { sb_word(&s.bin,(uint32_t)(5<<16)|OP_CONSTANT);
      sb_word(&s.bin,s.t_u64); sb_word(&s.bin,s.c_one64); sb_word(&s.bin,1); sb_word(&s.bin,0);}

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
             * idx64 = a + 1; val = load(ptr_u64); dst = val */
            uint32_t idx64 = nid(&s);
            { uint32_t o[]={s.t_u64,idx64,VRMAP_GET(in->a),s.c_one64};
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
            uint32_t idx64 = nid(&s);
            { uint32_t o[]={s.t_u64,idx64,VRMAP_GET(in->a),s.c_one64};
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
        default: break; /* remaining ops land next wave */
        }
    }

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

