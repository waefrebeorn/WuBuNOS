with open('wubu_isa_x86_64.c', 'r') as f:
    content = f.read()

old_code = '''            case MIR_DITOF_U: {
                /* unsigned int64 to double: if sign bit set, split into (val>>1)*2 + (val&1) */
                int sc = VR_ENC_SAFE(in->a);
                if (sc >= 0) emit_mov_rax_from_vr(&e, sc);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                /* test rax, rax; jns .positive */
                e8(&e, 0x48); e8(&e, 0x85); e8(&e, 0xC0);
                e8(&e, 0x0F); e8(&e, 0x89);
                int jns_rel_idx = e.n;
                e32(&e, 0); /* placeholder */
                /* Negative path: mov rcx, rax; shr rcx, 1; cvtsi2sd xmm0, rcx */
                e8(&e, 0x48); e8(&e, 0x89); e8(&e, 0xC1);
                e8(&e, 0x48); e8(&e, 0xD1); e8(&e, 0xE9);
                e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC1); /* cvtsi2sd xmm0, rcx */
                /* Load 2.0 into xmm2: mov rcx, 0x4000000000000000; movq xmm2, rcx */
                e8(&e, 0x48); e8(&e, 0xB9);
                e64(&e, 0x4000000000000000ULL); /* 2.0 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xD1); /* movq xmm2, rcx */
                /* mulsd xmm0, xmm2 */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x59); e8(&e, 0xC2);
                /* and rax, 1; cvtsi2sd xmm1, rax; addsd xmm0, xmm1 */
                e8(&e, 0x48); e8(&e, 0x83); e8(&e, 0xE0); e8(&e, 0x01);
                e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC8);
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x58); e8(&e, 0xC1); /* addsd xmm0, xmm1 */
                /* jmp .done */
                e8(&e, 0xE9);
                int jmp2_idx = e.n;
                e32(&e, 0); /* placeholder */
                /* .positive: cvtsi2sd xmm0, rax */
                int pos_off = e.n;
                *(int32_t*)(&e.code[jns_rel_idx]) = (int32_t)(pos_off - (jns_rel_idx + 4));
                e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC0);
                /* .done: movq rax, xmm0 */
                int done_off = e.n;
                *(int32_t*)(&e.code[jmp2_idx]) = (int32_t)(done_off - (jmp2_idx + 4));
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }'''

new_code = '''            case MIR_DITOF_U: {
                /* unsigned int64 to double: split into high/low 32-bit parts
                 * result = (double)(uint32_t)(val >> 32) * 4294967296.0 + (double)(uint32_t)val
                 * This is exact for all 64-bit unsigned values. */
                int sc = VR_ENC_SAFE(in->a);
                if (sc >= 0) emit_mov_rax_from_vr(&e, sc);
                else emit_load_rbp(&e, 0, spill_off(assign, assign_count, &e, in->a));
                /* Save original val to rcx: mov rcx, rax */
                e8(&e, 0x48); e8(&e, 0x89); e8(&e, 0xC1);
                /* High part: shr rcx, 32; cvtsi2sd xmm0, rcx (zero-extended to 64-bit) */
                e8(&e, 0x48); e8(&e, 0xC1); e8(&e, 0xE9); e8(&e, 0x20);
                /* mov edx, ecx (zero-extend high 32 bits to 64-bit for cvtsi2sd) */
                e8(&e, 0x89); e8(&e, 0xCA);  /* mov edx, ecx */
                e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xC2); /* cvtsi2sd xmm0, rdx */
                /* Load 4294967296.0 (2^32) into xmm2: mov rcx, 0x41F0000000000000; movq xmm2, rcx */
                e8(&e, 0x48); e8(&e, 0xB9);
                e64(&e, 0x41F0000000000000ULL); /* 4294967296.0 = 2^32 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x6E); e8(&e, 0xD1); /* movq xmm2, rcx */
                /* mulsd xmm0, xmm2 (high part * 2^32) */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x59); e8(&e, 0xC2);
                /* Low part: mov ecx, eax (original val); cvtsi2sd xmm1, rcx */
                e8(&e, 0x89); e8(&e, 0xC1);  /* mov ecx, eax */
                e8(&e, 0x31); e8(&e, 0xD2);  /* xor edx, edx (zero extend) */
                e8(&e, 0xF2); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x2A); e8(&e, 0xCA); /* cvtsi2sd xmm1, rdx */
                /* addsd xmm0, xmm1 (high*2^32 + low) */
                e8(&e, 0x66); e8(&e, 0x0F); e8(&e, 0x58); e8(&e, 0xC1);
                /* movq rax, xmm0 */
                e8(&e, 0x66); e8(&e, 0x48); e8(&e, 0x0F); e8(&e, 0x7E); e8(&e, 0xC0);
                break;
            }'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("Replaced MIR_DITOF_U code")
else:
    print("ERROR: Could not find old code")

with open('wubu_isa_x86_64.c', 'w') as f:
    f.write(content)
