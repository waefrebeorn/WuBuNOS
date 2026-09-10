with open('holyd_mir_eval.c', 'r') as f:
    content = f.read()

# Replace POST_INC/POST_DEC
old_post = '''    case HD_AST_POST_INC:
    case HD_AST_POST_DEC: {
        /* tmp = v; v = v +/- 1; return tmp */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t tmp = mir_new_vr(g);
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_mir_mov_to(g->prog, tmp, v);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HD_AST_POST_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, tmp, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, tmp, one);
                /* Truncate updated value to variable type width */
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].type &&
                        (g->vars[i].type->kind == HD_TYPE_I8 || g->vars[i].type->kind == HD_TYPE_U8 ||
                         g->vars[i].type->kind == HD_TYPE_I16 || g->vars[i].type->kind == HD_TYPE_U16 ||
                         g->vars[i].type->kind == HD_TYPE_I32 || g->vars[i].type->kind == HD_TYPE_U32 ||
                         g->vars[i].type->kind == HD_TYPE_I64 || g->vars[i].type->kind == HD_TYPE_U64)) {
                        upd = mir_truncate_to_type(g, upd, g->vars[i].type);
                        break;
                    }
                }
                wubu_mir_store(g->prog, addr, upd);
                return tmp;
            }
        }
        return mir_gen_expr(g, n->child);
    }'''

new_post = '''    case HD_AST_POST_INC:
    case HD_AST_POST_DEC: {
        /* tmp = v; v = v +/- 1; return tmp */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                /* Check if this is a float variable */
                int is_float_var = 0;
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].is_float) {
                        is_float_var = 1;
                        break;
                    }
                }
                wubu_vr_t tmp = mir_new_vr(g);
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_mir_mov_to(g->prog, tmp, v);
                wubu_vr_t upd;
                if (is_float_var) {
                    union { double d; int64_t i; } u;
                    u.d = 1.0;
                    wubu_vr_t one_d = wubu_mir_const(g->prog, u.i);
                    upd = (n->kind == HD_AST_POST_INC)
                        ? wubu_mir_binop(g->prog, MIR_DADD, tmp, one_d)
                        : wubu_mir_binop(g->prog, MIR_DSUB, tmp, one_d);
                } else {
                    wubu_vr_t one = wubu_mir_const(g->prog, 1);
                    upd = (n->kind == HD_AST_POST_INC)
                        ? wubu_mir_binop(g->prog, MIR_ADD, tmp, one)
                        : wubu_mir_binop(g->prog, MIR_SUB, tmp, one);
                    /* Truncate updated value to variable type width */
                    for (int i = 0; i < g->n_vars; i++) {
                        if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].type &&
                            (g->vars[i].type->kind == HD_TYPE_I8 || g->vars[i].type->kind == HD_TYPE_U8 ||
                             g->vars[i].type->kind == HD_TYPE_I16 || g->vars[i].type->kind == HD_TYPE_U16 ||
                             g->vars[i].type->kind == HD_TYPE_I32 || g->vars[i].type->kind == HD_TYPE_U32 ||
                             g->vars[i].type->kind == HD_TYPE_I64 || g->vars[i].type->kind == HD_TYPE_U64)) {
                            upd = mir_truncate_to_type(g, upd, g->vars[i].type);
                            break;
                        }
                    }
                }
                wubu_mir_store(g->prog, addr, upd);
                return tmp;
            }
        }
        return mir_gen_expr(g, n->child);
    }'''

content = content.replace(old_post, new_post)

# Replace PRE_INC/PRE_DEC
old_pre = '''    case HD_AST_PRE_INC:
    case HD_AST_PRE_DEC: {
        /* v = v +/- 1; return v */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_vr_t one = wubu_mir_const(g->prog, 1);
                wubu_vr_t upd = (n->kind == HD_AST_PRE_INC)
                    ? wubu_mir_binop(g->prog, MIR_ADD, v, one)
                    : wubu_mir_binop(g->prog, MIR_SUB, v, one);
                /* Truncate updated value to variable type width */
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].type &&
                        (g->vars[i].type->kind == HD_TYPE_I8 || g->vars[i].type->kind == HD_TYPE_U8 ||
                         g->vars[i].type->kind == HD_TYPE_I16 || g->vars[i].type->kind == HD_TYPE_U16 ||
                         g->vars[i].type->kind == HD_TYPE_I32 || g->vars[i].type->kind == HD_TYPE_U32 ||
                         g->vars[i].type->kind == HD_TYPE_I64 || g->vars[i].type->kind == HD_TYPE_U64)) {
                        upd = mir_truncate_to_type(g, upd, g->vars[i].type);
                        break;
                    }
                }
                wubu_mir_store(g->prog, addr, upd);
                return upd;
            }
        }
        return mir_gen_expr(g, n->child);
    }'''

new_pre = '''    case HD_AST_PRE_INC:
    case HD_AST_PRE_DEC: {
        /* v = v +/- 1; return v */
        if (n->child && n->child->kind == HD_AST_IDENT) {
            wubu_vr_t addr = mir_find_var_addr(g, n->child->ident);
            if (addr) {
                /* Check if this is a float variable */
                int is_float_var = 0;
                for (int i = 0; i < g->n_vars; i++) {
                    if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].is_float) {
                        is_float_var = 1;
                        break;
                    }
                }
                wubu_vr_t v = wubu_mir_load(g->prog, addr);
                wubu_vr_t upd;
                if (is_float_var) {
                    union { double d; int64_t i; } u;
                    u.d = 1.0;
                    wubu_vr_t one_d = wubu_mir_const(g->prog, u.i);
                    upd = (n->kind == HD_AST_PRE_INC)
                        ? wubu_mir_binop(g->prog, MIR_DADD, v, one_d)
                        : wubu_mir_binop(g->prog, MIR_DSUB, v, one_d);
                } else {
                    wubu_vr_t one = wubu_mir_const(g->prog, 1);
                    upd = (n->kind == HD_AST_PRE_INC)
                        ? wubu_mir_binop(g->prog, MIR_ADD, v, one)
                        : wubu_mir_binop(g->prog, MIR_SUB, v, one);
                    /* Truncate updated value to variable type width */
                    for (int i = 0; i < g->n_vars; i++) {
                        if (strcmp(g->vars[i].name, n->child->ident) == 0 && g->vars[i].type &&
                            (g->vars[i].type->kind == HD_TYPE_I8 || g->vars[i].type->kind == HD_TYPE_U8 ||
                             g->vars[i].type->kind == HD_TYPE_I16 || g->vars[i].type->kind == HD_TYPE_U16 ||
                             g->vars[i].type->kind == HD_TYPE_I32 || g->vars[i].type->kind == HD_TYPE_U32 ||
                             g->vars[i].type->kind == HD_TYPE_I64 || g->vars[i].type->kind == HD_TYPE_U64)) {
                            upd = mir_truncate_to_type(g, upd, g->vars[i].type);
                            break;
                        }
                    }
                }
                wubu_mir_store(g->prog, addr, upd);
                return upd;
            }
        }
        return mir_gen_expr(g, n->child);
    }'''

content = content.replace(old_pre, new_pre)

with open('holyd_mir_eval.c', 'w') as f:
    f.write(content)

print("Done")
