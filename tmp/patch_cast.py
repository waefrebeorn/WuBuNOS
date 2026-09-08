import re

with open('holyd_mir_eval.c', 'r') as f:
    content = f.read()

# 1. Add mir_gen_cast function after mir_truncate_to_type
old_truncate = '''static wubu_vr_t mir_truncate_to_type(HDMirGen *g, wubu_vr_t val, const HDType *type) {
    if (!type) return val;
    switch (type->kind) {
    case HD_TYPE_I8:
        val = wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFF));
        return wubu_mir_unop(g->prog, MIR_SEXT8, val);
    case HD_TYPE_U8:
        return wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFF));
    case HD_TYPE_I16:
        val = wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFF));
        return wubu_mir_unop(g->prog, MIR_SEXT16, val);
    case HD_TYPE_U16:
        return wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFF));
    case HD_TYPE_I32:
        val = wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFFFFFF));
        return wubu_mir_unop(g->prog, MIR_SEXT32, val);
    case HD_TYPE_U32:
        return wubu_mir_binop(g->prog, MIR_AND, val, wubu_mir_const(g->prog, 0xFFFFFFFF));
    case HD_TYPE_F64:
        /* Convert integer to double */
        return wubu_mir_unop(g->prog, MIR_DITOF, val);
    case HD_TYPE_I64:
    case HD_TYPE_U64:
    default:
        return val;
    }
}'''

new_truncate = old_truncate + '''

/* Type cast: convert val from source_type to target_type.
 * Handles int<->float, sign extension, and truncation. */
static wubu_vr_t mir_gen_cast(HDMirGen *g, wubu_vr_t val, const HDType *target_type, const HDASTNode *src_node) {
    if (!target_type) return val;
    /* Determine source type from AST node or default to I32 */
    HDType *src_type = src_node ? (HDType *)src_node->type : NULL;
    HDTypeKind src_k = src_type ? src_type->kind : HD_TYPE_I32;
    HDTypeKind tgt_k = target_type->kind;
    
    /* Same type: no conversion needed */
    if (src_k == tgt_k) return val;
    
    /* Float -> int */
    if (src_k == HD_TYPE_F64 && (tgt_k == HD_TYPE_I8 || tgt_k == HD_TYPE_U8 ||
        tgt_k == HD_TYPE_I16 || tgt_k == HD_TYPE_U16 ||
        tgt_k == HD_TYPE_I32 || tgt_k == HD_TYPE_U32 ||
        tgt_k == HD_TYPE_I64 || tgt_k == HD_TYPE_U64)) {
        return wubu_mir_unop(g->prog, MIR_DTOI, val);
    }
    
    /* Int -> float */
    if ((src_k == HD_TYPE_I8 || src_k == HD_TYPE_U8 || src_k == HD_TYPE_I16 ||
         src_k == HD_TYPE_U16 || src_k == HD_TYPE_I32 || src_k == HD_TYPE_U32 ||
         src_k == HD_TYPE_I64 || src_k == HD_TYPE_U64) && tgt_k == HD_TYPE_F64) {
        /* Use unsigned conversion for unsigned sources */
        if (src_k == HD_TYPE_U64 || src_k == HD_TYPE_U32 || src_k == HD_TYPE_U16 || src_k == HD_TYPE_U8)
            return wubu_mir_unop(g->prog, MIR_DITOF_U, val);
        return wubu_mir_unop(g->prog, MIR_DITOF, val);
    }
    
    /* Integer truncation/sign extension */
    return mir_truncate_to_type(g, val, target_type);
}
'''

content = content.replace(old_truncate, new_truncate)

# 2. Add type conversion in struct initializer
old_init = '''                        ev = mir_gen_expr(g, elem);
                    }
                    wubu_vr_t elem_addr = wubu_mir_binop(g->prog, MIR_ADD, addr,
                                                          wubu_mir_const(g->prog, (int64_t)offset));
                    wubu_mir_store(g->prog, elem_addr, ev);'''

new_init = '''                        ev = mir_gen_expr(g, elem);
                        /* Convert initializer value to member type */
                        if (is_struct_var && n->type && n->type->kind == HD_TYPE_STRUCT
                            && e < (uint32_t)n->type->n_members) {
                            HDType *member_type = n->type->members[e].type;
                            if (member_type) {
                                ev = mir_gen_cast(g, ev, member_type, elem);
                            }
                        }
                    }
                    wubu_vr_t elem_addr = wubu_mir_binop(g->prog, MIR_ADD, addr,
                                                          wubu_mir_const(g->prog, (int64_t)offset));
                    wubu_mir_store(g->prog, elem_addr, ev);'''

content = content.replace(old_init, new_init)

with open('holyd_mir_eval.c', 'w') as f:
    f.write(content)

print("Done")
