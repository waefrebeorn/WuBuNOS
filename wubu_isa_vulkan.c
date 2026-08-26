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
static uint32_t g_result_cell = 0;  /* SSBO cell holding the return value */

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
      /* Multi-WG: per-lane scratch offsets scale by WUBU_VK_GROUPS; remember the
       * result cell (last C cell) for run() — cell 0 races across WGs. */
      unsigned gx_ = 1;
      { const char *ge = getenv("WUBU_VK_GROUPS"); if (ge) gx_=(unsigned)atoi(ge); if (gx_<1) gx_=1; }
      g_cells = (uint32_t)((p->total_mem > 0 ? p->total_mem : 1) + 1 + (has_tg?gx_*64*4+1:0));
      g_result_cell = has_tg ? (uint32_t)p->total_mem : 0; }

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
             "/tmp/vk_run %s /tmp/wubu_kernel.spv %lld %u",
             getenv("WUBU_VK_DEVICE") ? getenv("WUBU_VK_DEVICE") : "0",
             (long long)arg, cells);
    if (getenv("DBG_VK")) fprintf(stderr, "[vk] cmd: %s\n", cmd);
    FILE *f = NULL;
    unsigned gx_run = 1;
    { const char *ge = getenv("WUBU_VK_GROUPS"); if (ge) gx_run=(unsigned)atoi(ge); if (gx_run<1) gx_run=1; }
    if (gx_run > 1 && g_result_cell > 0) {
        /* multi-WG: cell 0 races (every WG stores it). Read the LAST C cell via
         * vk_run's WUBU_VK_DUMP instead — it prints "cell[i] = v" lines. */
        char cmd2[640];
        snprintf(cmd2, sizeof(cmd2),
                 "WUBU_VK_DUMP=%u /tmp/vk_run %s /tmp/wubu_kernel.spv %lld %u 2>&1 >/dev/null",
                 g_result_cell + 1,
                 getenv("WUBU_VK_DEVICE") ? getenv("WUBU_VK_DEVICE") : "0",
                 (long long)arg, cells);
        if (getenv("DBG_VK")) fprintf(stderr, "[vk] cmd2: %s\n", cmd2);
        f = popen(cmd2, "r");
        if (!f) return 0;
        char line[128]; long long rr = -1;
        while (fgets(line, sizeof line, f)) {
            long idx; long long v;
            if (sscanf(line, "cell[%ld] = %lld", &idx, &v) == 2 &&
                (unsigned long)idx == g_result_cell) { rr = v; break; }
        }
        int rc2 = pclose(f);
        if (rc2 != 0 || rr < 0) return -1;
        return rr;
    }
    f = popen(cmd, "r");
    if (!f) return 0;
    long long r = 0;
    /* dzn driver emits WARNING lines to stdout before the result number; skip non-numeric */
    while (!feof(f) && !ferror(f)) {
        int c2 = fgetc(f);
        if (c2=='-' || c2=='+' || (c2>='0' && c2<='9')) { ungetc(c2, f); break; }
    }
    if (fscanf(f, "%lld", &r) != 1) {
        /* Surface vk_run/device errors instead of swallowing them as 0. */
        char *line = NULL; size_t sz = 0;
        if (getline(&line, &sz, f) > 0) r = -1;  /* -1 = execution failure */
        free(line);
    }
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
