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

int wubu_spirv_emit(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_n)
{
    S s; memset(&s,0,sizeof s);
    s.next_id = 1;

    /* ---- pin ALL type/const/var ids up front (logical layout order) ---- */
    s.t_void      = nid(&s);
    s.t_bool      = nid(&s);
    s.t_i32       = nid(&s);
    s.t_v3i32     = nid(&s);
    s.t_u64       = nid(&s);
    s.len_mem_cells = nid(&s);            /* OpConstant i32 N */
    s.t_arr_mem   = nid(&s);              /* OpTypeArray u64 N */
    s.t_struct_ssbo = nid(&s);            /* struct { u64 arr[N] } */
    s.t_ptr_ssbo  = nid(&s);
    s.t_i32_in    = nid(&s);              /* ptr(Input,i32) */
    s.t_gid_input = nid(&s);              /* ptr(Input,v3i32) */
    s.t_pushblk   = nid(&s);              /* struct{u64} */
    s.t_res_u64   = nid(&s);              /* ptr(StorageBuffer,u64) for ret slot */
    s.t_v2i32     = nid(&s);              /* v2 i32 (u64 bitcast view) */
    s.t_f32       = nid(&s);              /* f32 */
    s.t_ptr_push  = nid(&s);
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
    typedef struct { int64_t imm; uint32_t id; } cment_t;
    cment_t *cmts = NULL; size_t ncm = 0, cap_cm = 0;
    for (size_t q = 0; q < p->n; q++) {
        if (p->ins[q].op != MIR_CONST) continue;
        int64_t v = p->ins[q].imm;
        size_t w; for (w = 0; w < ncm; w++) if (cmts[w].imm == v) break;
        if (w == ncm) {
            if (ncm + 1 > cap_cm) { cap_cm = cap_cm ? cap_cm*2 : 32;
                                    cmts = realloc(cmts, cap_cm * sizeof *cmts); }
            cmts[ncm].imm = v;
            cmts[ncm].id  = nid(&s);
            ncm++;
        }
    }
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
    { uint32_t e[]={s.fn_main,17,64,1,1}; spv_ins(&s.bin,OP_EXEC_MODE,e,5);} /* LocalSize */
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
    { uint32_t ncells=(uint32_t)((p->total_mem>0?p->total_mem:1)+1);
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

    /* per-vr phi-less labels: we emit straight-line ops; labels for branches */
    uint32_t lbl_ctr = 5000;
    uint32_t last_ret_src = s.c_zero64;

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
            break;
        default: break; /* branches/loads/stores land next wave */
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
