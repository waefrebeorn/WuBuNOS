/*
 * test_holyc_ptx.c -- HolyC PTX backend behaviour gate.
 *
 * The target referenced this file but it did not exist, so `make test_holyc_ptx`
 * could never build.  The backend is pure code GENERATION (no GPU needed to emit
 * PTX text), so the gate asserts the emitted assembly -- version/target header,
 * kernel entry + params, the m16n8k16 tensor-core MMA, and the epilogue --
 * plus the compile/exec contracts.
 *
 * C11, hosted.  Exit 0 = all gates passed.
 */
#include "holyd_ptx.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else      { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while (0)

#define HAS(hay, needle) (strstr((hay), (needle)) != NULL)

static void test_generator(void) {
    printf("\n[PTX Generator]\n");
    PTXGen gen;
    ptx_gen_init(&gen);
    CHECK(gen.code == NULL && gen.code_size == 0 && gen.code_cap == 0,
          "ptx_gen_init zeroes the generator");
    CHECK(gen.target_arch == HD_TARGET_PTX,
          "ptx_gen_init selects the PTX target architecture");

    ptx_emit(&gen, ".version 7.8\n");
    CHECK(gen.code != NULL &&
          gen.code_size == strlen(".version 7.8\n"),
          "ptx_emit appends and tracks the size");
    CHECK(gen.code[gen.code_size] == '\0', "ptx_emit NUL-terminates the buffer");

    /* Growth: emit past the initial 4096 cap and confirm no overflow. */
    ptx_gen_init(&gen);
    int ok = 1;
    for (int i = 0; i < 2000; i++) {
        ptx_emit(&gen, "    /* padding line %d */\n", i);
        if (gen.code[gen.code_size] != '\0') { ok = 0; break; }
    }
    CHECK(ok && gen.code_cap > 4096,
          "ptx_emit grows the buffer past the initial capacity");
    free(gen.code);
}

static void test_emitters(void) {
    printf("\n[PTX Emitters]\n");
    PTXGen gen;

    ptx_gen_init(&gen);
    ptx_emit_header(&gen);
    CHECK(HAS(gen.code, ".version 7.8"), "header emits the PTX version");
    CHECK(HAS(gen.code, ".target sm_80"), "header targets sm_80 (Ampere)");
    CHECK(HAS(gen.code, ".address_size 64"), "header declares 64-bit addressing");
    free(gen.code);

    ptx_gen_init(&gen);
    ptx_emit_matmul_kernel_prologue(&gen, "gpu_matmul");
    CHECK(HAS(gen.code, ".visible .entry gpu_matmul("),
          "prologue emits a visible entry point with the kernel name");
    CHECK(HAS(gen.code, ".param .b64 A_ptr") &&
          HAS(gen.code, ".param .u32 K"),
          "prologue declares the A/B/C pointer and M/N/K params");
    free(gen.code);

    ptx_gen_init(&gen);
    ptx_emit_epilogue(&gen);
    CHECK(HAS(gen.code, "}\n"), "epilogue closes the kernel body");
    free(gen.code);

    ptx_gen_init(&gen);
    ptx_emit_mma_f16_m16n8k16(&gen, "Ra", "Rb", "Rc");
    CHECK(HAS(gen.code, "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"),
          "mma emitter emits the m16n8k16 FP16 tensor-core instruction");
    CHECK(HAS(gen.code, "{Ra0, Ra1, Ra2, Ra3}"),
          "mma emitter names the accumulator registers");
    free(gen.code);
}

static void test_compile(void) {
    printf("\n[Holyd -> PTX Compile]\n");
    CHECK(hd_compile_ptx(NULL) == NULL, "NULL source is rejected");

    char *ptx = hd_compile_ptx("return 42;");
    CHECK(ptx != NULL, "hd_compile_ptx returns generated PTX text");
    if (ptx) {
        CHECK(HAS(ptx, ".version 7.8"), "compiled PTX carries the version header");
        CHECK(HAS(ptx, ".target sm_80"), "compiled PTX carries the target");
        CHECK(HAS(ptx, ".entry gpu_matmul("), "compiled PTX defines the matmul kernel");
        CHECK(HAS(ptx, "mma.sync.aligned.m16n8k16"),
              "compiled PTX contains the tensor-core MMA");
        size_t n = strlen(ptx);
        CHECK(n > 0 && ptx[n] == '\0', "compiled PTX is NUL-terminated");
        free(ptx);
    }
}

static void test_runtime_contract(void) {
    printf("\n[PTX Runtime Contract]\n");
    /* No CUDA on this host: exec must fail cleanly, not crash. */
    CHECK(hd_exec_ptx(NULL, NULL, 0) == -1, "NULL PTX/args is rejected");
    void *args[6] = {0};
    CHECK(hd_exec_ptx("// stub\n", args, 6) == -1,
          "exec without a CUDA driver returns -1");
    CHECK(hd_builtin_gpu_matmul(0, 0, 0, 0, 0, 0) == 0,
          "GpuMatMul built-in returns 0");
}

int main(void) {
    printf("=== HolyC PTX backend behaviour gate ===\n");
    test_generator();
    test_emitters();
    test_compile();
    test_runtime_contract();
    printf("\n=== holyc_ptx: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
