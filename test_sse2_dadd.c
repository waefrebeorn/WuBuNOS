/* test_sse2_dadd.c -- goal 175 evidence: the f64 arithmetic path must encode
 * SSE2 (F2 0F 5x) instructions, not x87 (D8-DF).
 *
 * Builds a minimal MIR program (CONST, CONST, DADD, RET), compiles it with the
 * x86-64 backend and inspects the emitted bytes. Both the presence of the SSE2
 * opcode and the absence of an x87 arithmetic opcode are required. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_mir.h"
#include "x86_decode.h"

extern int wubu_isa_x86_64_compile(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_size);

static int contains(const uint8_t *code, size_t n, const uint8_t *pat, size_t len) {
    if (len == 0 || n < len) return 0;
    for (size_t i = 0; i + len <= n; i++)
        if (memcmp(code + i, pat, len) == 0) return 1;
    return 0;
}

/* Walk the real instruction stream: an x87 arithmetic instruction has primary
 * opcode D8-DF. A byte scan would false-positive on immediates, so decode. */
static int has_x87(const uint8_t *code, size_t n) {
    size_t off = 0;
    x86_insn_t ins;
    while (off < n) {
        if (x86_decode_one(code, n, off, &ins) != 0) return 0; /* stop at unknown */
        if (ins.opcode >= 0xD8 && ins.opcode <= 0xDF) return 1;
        if (ins.len == 0) return 0;
        off += ins.len;
    }
    return 0;
}

int main(void) {
    int pass = 0, fail = 0;

    struct { wubu_mir_op_t op; const char *name; int cmp; } ops[] = {
        { MIR_DADD, "addsd", 0 },
        { MIR_DSUB, "subsd", 0 },
        { MIR_DMUL, "mulsd", 0 },
        { MIR_DDIV, "divsd", 0 },
        { MIR_DGT,  "ucomisd", 1 },
    };
    /* F2 0F 58/5C/59/5E C1 : <op> xmm0, xmm1
     * 66 0F 2E C1          : ucomisd xmm0, xmm1 (comparison) */
    const uint8_t opc[] = { 0x58, 0x5C, 0x59, 0x5E, 0x2E };

    for (int k = 0; k < 5; k++) {
        wubu_mir_prog_t prog;
        memset(&prog, 0, sizeof prog);
        wubu_mir_instr_t ins[4];
        memset(ins, 0, sizeof ins);
        ins[0].op = MIR_CONST; ins[0].dst = 0; ins[0].imm = 0x3FF0000000000000LL; /* 1.0 */
        ins[1].op = MIR_CONST; ins[1].dst = 1; ins[1].imm = 0x4000000000000000LL; /* 2.0 */
        ins[2].op = ops[k].op; ins[2].dst = 2; ins[2].a = 0; ins[2].b = 1;
        ins[3].op = MIR_RET;   ins[3].a = 2;
        prog.ins = ins; prog.n = 4; prog.n_funcs = 0;

        uint8_t *code = NULL; size_t n = 0;
        int rc = wubu_isa_x86_64_compile(&prog, &code, &n);
        int ok = 0;
        if (rc == 0 && code && n > 0) {
            uint8_t want[4] = { ops[k].cmp ? 0x66 : 0xF2, 0x0F, opc[k], 0xC1 };
            ok = contains(code, n, want, sizeof want);
        }
        if (ok) {
            pass++;
            printf("  PASS  %s encodes %s %02X %02X C1\n", ops[k].name,
                   ops[k].cmp ? "66 0F" : "F2 0F", ops[k].cmp ? 0x0F : 0x0F, opc[k]);
        } else {
            fail++;
            printf("  FAIL  %s did not encode SSE2 F2 0F %02X C1\n", ops[k].name, opc[k]);
        }
        if (code && has_x87(code, n)) {
            fail++;
            printf("  FAIL  %s path emitted an x87 opcode (D8-DF)\n", ops[k].name);
        } else if (code) {
            pass++;
            printf("  PASS  %s path uses no x87 opcode\n", ops[k].name);
        }
        free(code);
    }

    printf("SSE2: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
