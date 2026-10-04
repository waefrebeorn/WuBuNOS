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
 *   compile() writes SPIR-V to a per-pid, per-module path (and returns bytes)
 *   run()     invokes vk_run <dev> <that same path> <arg> <cells>
 *             and parses the decimal result.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

int wubu_spirv_emit(const wubu_mir_prog_t *p, uint8_t **out, size_t *out_n);

static uint32_t g_cells = 17;
static uint32_t g_result_cell = 0;  /* SSBO cell holding the return value */

/* Same simple content hash the PTX backend uses, so a module change is
 * detected and two concurrent compiles never share a path. */
static unsigned vk_hash(const uint8_t *b, size_t n)
{
    unsigned h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 16777619u; }
    return h;
}

/* Set by vulkan_compile, read by vulkan_run: the exact SPIR-V file this
 * module was just written to. */
static char *g_spvpath = NULL;

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
      /* The SSBO element count must match what the emitter reserved.
       * NOTE: MIR load/store addresses are computed at RUNTIME through
       * MIR_LOAD chains off a high VR, not by a MIR_CONST the backend can
       * see, so the true high-water mark is not statically available here.
       * total_mem (a cell count) is the only conservative source we have. */
      unsigned gx_ = 1;
      { const char *ge = getenv("WUBU_VK_GROUPS"); if (ge) gx_=(unsigned)atoi(ge); if (gx_<1) gx_=1; }
      g_cells = (uint32_t)((p->total_mem > 0 ? p->total_mem : 1) + 1 + (has_tg?gx_*64*4+1:0));
      g_result_cell = has_tg ? (uint32_t)p->total_mem : 0; }

    /* Persist for the runner. The path must be unique per process AND per
     * module: a fixed "/tmp/wubu_kernel.spv" let two concurrent Vulkan
     * compiles overwrite each other's SPIR-V, so one backend would execute
     * the OTHER one's shader. (Same defect class as the PTX cubin, fixed in
     * b5f7c0a.) vulkan_run() re-derives the identical path from the same
     * pid + module bytes. */
    char spvpath[128];
    snprintf(spvpath, sizeof spvpath, "/tmp/wubu_kernel_%d_%08x.spv",
             (int)getpid(), (unsigned)vk_hash(code, n));
    g_spvpath = strdup(spvpath);
    FILE *f = fopen(spvpath, "wb");
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
    /* arg IS the memory pointer (see the ISA run contract). The out-of-process
     * runner receives it only as a decimal, so the cells it points at have to
     * travel separately -- without this the SSBO is zeroed and every MIR_LOAD
     * returns 0. */
    char imgpath[128];
    snprintf(imgpath, sizeof imgpath, "/tmp/wubu_vk_mem_%d.bin", (int)getpid());
    int have_img = 0;
    if (arg) {
        /* Clamp hard: the caller owns a buffer sized for the program's own
         * cells, which can be SMALLER than the scratch-extended g_cells.
         * Reading g_cells*8 would run off the end of it. */
        size_t nbytes = 0;
        for (unsigned q = 0; q < cells && q < 64u; q++) nbytes += 8;
        FILE *mf = fopen(imgpath, "wb");
        if (mf) {
            fwrite((const void *)(uintptr_t)arg, 1, nbytes, mf);
            fclose(mf);
            have_img = 1;
        }
    }
    /* vulkan_run() must use the SAME per-process path vulkan_compile wrote.
     * Running a module that was never compiled used to mean running whatever
     * another process had left in the shared /tmp/wubu_kernel.spv. */
    if (!g_spvpath) {
        fprintf(stderr, "[vulkan] run() before compile(); build with make vk_run\n");
        return -1;
    }
    snprintf(cmd, sizeof(cmd),
             "/tmp/vk_run %s %s %lld %u %s",
             getenv("WUBU_VK_DEVICE") ? getenv("WUBU_VK_DEVICE") : "0",
             g_spvpath, (long long)arg, cells, have_img ? imgpath : "none");
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
    /* A missing runner used to return 0 here, which is indistinguishable
     * from "the shader computed 0" -- that silence is what hid this bug for
     * as long as the Vulkan path was broken. Fail loudly instead. Build the
     * runner with `make vk_run`. */
    if (!f) {
        fprintf(stderr, "[vk] cannot exec: %s\n"
                        "     build the runner with `make vk_run`\n", cmd);
        return -1;  /* -1 = execution failure, distinct from a 0 result */
    }
    long long r = 0;
    /* dzn driver emits WARNING lines to stdout before the result number; skip non-numeric */
    while (!feof(f) && !ferror(f)) {
        int c2 = fgetc(f);
        if (c2=='-' || c2=='+' || (c2>='0' && c2<='9')) { ungetc(c2, f); break; }
    }
    if (fscanf(f, "%lld", &r) != 1) {
        /* No number came back. That means the runner failed to start (it is
         * spawned through a shell, so popen() itself succeeds and only the
         * child reports "not found") or the device errored. Either way this
         * is an EXECUTION FAILURE, not a zero result -- returning 0 here is
         * what made every broken Vulkan run look like "the shader answered
         * 0". Report -1 and say so. */
        fprintf(stderr, "[vk] no result from runner: %s\n"
                        "     build it with `make vk_run`\n", cmd);
        r = -1;
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
