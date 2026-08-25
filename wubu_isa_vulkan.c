/*
 * wubu_isa_vulkan.c -- the Vulkan/SPIR-V driver object (the borg leg).
 *
 * Wraps wubu_isa_spirv.c (hand-encoded MIR->SPIR-V) into the standard
 * ISA driver vtable. Execution shells out to tools/vk_run (libvulkan),
 * which dispatches on ANY Vulkan device chosen by WUBU_VK_DEVICE
 * (default 0). One driver = every card: NVIDIA dGPU, AMD APU iGPU,
 * old recycled hardware, llvmpipe CPU fallback.
 *
 * ABI with vk_run:
 *   compile() writes SPIR-V to /tmp/wubu_kernel.spv (plus returns bytes)
 *   run()     invokes vk_run <dev> /tmp/wubu_kernel.spv <arg> <cells>
 *             and parses the decimal result.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int wubu_spirv_emit(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_n);

static uint32_t g_cells = 17;

static int vulkan_compile(const wubu_mir_prog_t *p,
                          uint8_t **out_code, size_t *out_size)
{
    if (!p || !out_code || !out_size) return -1;
    uint8_t *code = NULL;
    size_t n = 0;
    int rc_ = wubu_spirv_emit(p, &code, &n);
    if (getenv("DBG_VK")) fprintf(stderr, "[vk] emit rc=%d size=%zu\n", rc_, n);
    if (rc_ != 0) return -1;
    /* remember the module's mem-cell count for run()'s buffer sizing */
    { int has_tg=0; for (unsigned q=0;q<p->n;q++) if (p->ins[q].op==MIR_T_GEMM) has_tg=1;
      g_cells = (uint32_t)((p->total_mem > 0 ? p->total_mem : 1) + 1 + (has_tg?8:0)); }

    /* persist for the runner */
    FILE *f = fopen("/tmp/wubu_kernel.spv", "wb");
    if (!f) { free(code); return -1; }
    fwrite(code, 1, n, f);
    fclose(f);

    *out_code = code;
    *out_size = n;
    return 0;
}

/* mem cells needed: total_mem + result slot; keep a floor so tiny programs
 * still have an arg cell. Must match emitter's array size exactly. */
static int64_t vulkan_run(const uint8_t *code, size_t size, int64_t arg)
{
    (void)code; (void)size;
    uint32_t cells = g_cells;
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "/tmp/vk_run %s /tmp/wubu_kernel.spv %lld %u 2>/dev/null",
             getenv("WUBU_VK_DEVICE") ? getenv("WUBU_VK_DEVICE") : "0",
             (long long)arg, cells);
    FILE *f = popen(cmd, "r");
    if (!f) return 0;
    long long r = 0;
    if (fscanf(f, "%lld", &r) != 1) r = 0;
    pclose(f);
    return (int64_t)r;
}

static void vulkan_describe(void)
{
    printf("Vulkan/SPIR-V ISA driver (the borg leg)\n");
    printf("  Family:        gpu\n");
    printf("  Target:        ANY Vulkan device (WUBU_VK_DEVICE selects)\n");
    printf("  Exec model:    native (vk_run -> libvulkan -> D3D12/native ICD)\n");
    printf("  Compile:       MIR -> hand-encoded SPIR-V (no shader compiler)\n");
    printf("  Run:           spirv module -> compute dispatch -> SSBO cell 0\n");
}

const wubu_isa_driver_t wubu_isa_vulkan = {
    .name     = "vulkan",
    .family   = "gpu",
    .exec     = WUBU_ISA_NATIVE,
    .compile  = vulkan_compile,
    .run      = vulkan_run,
    .describe = vulkan_describe,
};
