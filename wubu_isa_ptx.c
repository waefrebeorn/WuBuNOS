/*
 * wubu_isa_ptx.c -- the NVIDIA PTX ISA driver (the GPU leg).
 *
 * The eighth driver in the ISA driver space. Consumes wubu_mir_t,
 * emits PTX assembly, compiles to cubin via ptxas, and executes on
 * the GPU via a pre-compiled CUDA host stub.
 *
 * Strategy:
 *   compile(): walk the MIR program, emit PTX text, write to /tmp,
 *              call ptxas to produce cubin, read cubin into memory.
 *   run():     write cubin to /tmp, invoke the pre-compiled host stub
 *              via system(), read its exit code as the int64 result.
 *   describe(): print driver info.
 *
 * The host stub (gpu_host_stub.cu) is compiled once at startup:
 *   nvcc -arch=sm_89 -o /tmp/gpu_host_stub gpu_host_stub.cu
 * It loads a cubin, launches wubu_kernel, prints the result as a
 * decimal int64 to stdout, and the driver reads it back.
 *
 * PTX kernel ABI:
 *   .visible .entry wubu_kernel(.param .b64 result, .param .b64 arg)
 *   — result is a pointer to one int64, arg is the input value.
 *
 * C11, self-contained.
 */
#include "wubu_isa_ptx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <errno.h>

/* If the CUDA host stub cannot reach a GPU (no device in this environment),
 * it blocks forever. Cap each launch so a missing GPU fails fast instead of
 * hanging the whole gauntlet. */
#define PTX_RUN_TIMEOUT_SEC 8

/* ---- PTX assembly emitter ---- */

typedef struct {
    char *text;
    size_t n, cap;
    uint32_t n_vregs;     /* highest virtual register used + 1 */
    uint32_t n_pred;      /* next predicate number to allocate */
} ptx_emitter_t;

static void ptx_emit(ptx_emitter_t *e, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (len <= 0) return;

    if (e->n + (size_t)len + 1 >= e->cap) {
        e->cap = e->cap ? e->cap * 2 : 8192;
        e->text = realloc(e->text, e->cap);
    }
    memcpy(e->text + e->n, buf, (size_t)len);
    e->n += (size_t)len;
    e->text[e->n] = '\0';
}

/* Map a MIR virtual register to a PTX register name like %r42.
 * We pre-declare all registers up front, so here we just track count. */
static uint32_t ptx_vr(ptx_emitter_t *e, wubu_vr_t vr)
{
    if (vr + 1 > e->n_vregs)
        e->n_vregs = vr + 1;
    return vr;
}

/* ---- MIR -> PTX translation ---- */

static void emit_kernel_body(ptx_emitter_t *e, const wubu_mir_prog_t *p)
{
    /* We need to track label positions. PTX labels are just identifiers
     * followed by ':'. Since we emit linearly, labels are emitted at the
     * right place. For forward branches we record the label id and emit
     * it when we hit a MIR_LABEL. */

    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *ins = &p->ins[i];
        uint32_t rd;

        switch (ins->op) {
        case MIR_CONST:
            rd = ptx_vr(e, ins->dst);
            /* mov.b64 %rd, imm — PTX allows signed immediates */
            ptx_emit(e, "    mov.b64 %%r%lld, %lld;\n", (long long)rd, (long long)ins->imm);
            break;

        case MIR_MOV:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    mov.b64 %%r%d, %%r%d;\n", (int)rd, (int)ptx_vr(e, ins->a));
            break;

        case MIR_ADD:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    add.s64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_SUB:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    sub.s64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_MUL:
            rd = ptx_vr(e, ins->dst);
            /* mul.lo.s64 gives the low 64 bits of the product */
            ptx_emit(e, "    mul.lo.s64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_DIV:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    div.s64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_MOD:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    rem.s64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_AND:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    and.b64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_OR:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    or.b64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_XOR:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    xor.b64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_SHL:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    shl.b64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_SHR:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    shr.u64 %%r%d, %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            break;

        case MIR_NEG:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    neg.s64 %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a));
            break;

        case MIR_NOT:
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    not.b64 %%r%d, %%r%d;\n",
                     (int)rd, (int)ptx_vr(e, ins->a));
            break;

        case MIR_LABEL:
            /* Label id becomes a PTX label */
            ptx_emit(e, "L_%u:\n", ins->label);
            break;

        case MIR_JMP:
            ptx_emit(e, "    bra L_%u;\n", ins->label);
            break;

        case MIR_JZ: {
            /* if (vr == 0) jump — use a predicate from the predicate pool */
            uint32_t predicate = e->n_pred++;
            ptx_emit(e, "    setp.eq.s64 p%u, %%r%d, 0;\n",
                     predicate, (int)ptx_vr(e, ins->a));
            ptx_emit(e, "    @p%u bra L_%u;\n", predicate, ins->label);
            break;
        }

        case MIR_JNZ: {
            /* if (vr != 0) jump */
            uint32_t predicate = e->n_pred++;
            ptx_emit(e, "    setp.ne.s64 p%u, %%r%d, 0;\n",
                     predicate, (int)ptx_vr(e, ins->a));
            ptx_emit(e, "    @p%u bra L_%u;\n", predicate, ins->label);
            break;
        }

        case MIR_ALLOC:
            /* base address already reserved via wubu_mir_alloc; the home
             * address vr is a MIR_CONST holding the cell index. Nothing to
             * emit — the const carries the address. */
            break;

        case MIR_LOAD: {
            /* dst = mem[a]; a is the cell index (a MIR_CONST vr). */
            rd = ptx_vr(e, ins->dst);
            ptx_emit(e, "    cvt.u32.s64 %rc1, %r%d;\n", (int)ptx_vr(e, ins->a));
            ptx_emit(e, "    mul.lo.u32 %rc1, %rc1, 8;\n");
            ptx_emit(e, "    cvt.u64.u32 %ra1, %rc1;\n");
            ptx_emit(e, "    add.u64 %ra1, %ra1, %ra0;\n");
            ptx_emit(e, "    ld.global.s64 %r%d, [%ra1];\n", (int)rd);
            break;
        }

        case MIR_STORE: {
            /* mem[a] = b; a is the cell index, b is the value. */
            ptx_emit(e, "    cvt.u32.s64 %rc1, %r%d;\n", (int)ptx_vr(e, ins->a));
            ptx_emit(e, "    mul.lo.u32 %rc1, %rc1, 8;\n");
            ptx_emit(e, "    cvt.u64.u32 %ra1, %rc1;\n");
            ptx_emit(e, "    add.u64 %ra1, %ra1, %ra0;\n");
            ptx_emit(e, "    st.global.s64 [%ra1], %r%d;\n", (int)ptx_vr(e, ins->b));
            break;
        }

        case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE:
        case MIR_GT: case MIR_GE:
        case MIR_ULT: case MIR_ULE: case MIR_UGT: case MIR_UGE: {
            int rd2 = ptx_vr(e, ins->dst);
            uint32_t pred = e->n_pred++;
            const char *cmp;
            switch (ins->op) {
                case MIR_EQ:  cmp = "eq.s64";  break;
                case MIR_NE:  cmp = "ne.s64";  break;
                case MIR_LT:  cmp = "lt.s64";  break;
                case MIR_LE:  cmp = "le.s64";  break;
                case MIR_GT:  cmp = "gt.s64";  break;
                case MIR_GE:  cmp = "ge.s64";  break;
                case MIR_ULT: cmp = "lt.u64";  break;
                case MIR_ULE: cmp = "le.u64";  break;
                case MIR_UGT: cmp = "gt.u64";  break;
                case MIR_UGE: cmp = "ge.u64";  break;
                default:      cmp = "eq.s64";  break;
            }
            ptx_emit(e, "    setp.%s p%u, %%r%d, %%r%d;\n",
                     cmp, pred, (int)ptx_vr(e, ins->a), (int)ptx_vr(e, ins->b));
            ptx_emit(e, "    selp.b64 %%r%d, 1, 0, p%u;\n", rd2, pred);
            break;
        }

        case MIR_RET: {
            /* Store the RET source vr to the result parameter */
            ptx_emit(e, "    ld.param.b64 %%rd_result, [result];\n");
            ptx_emit(e, "    st.global.s64 [%%rd_result], %%r%d;\n", (int)ptx_vr(e, ins->a));
            ptx_emit(e, "    ret;\n");
            break;
        }

        default:
            /* Unknown op — emit a comment so it's visible */
            ptx_emit(e, "    /* MIR op %d — not yet implemented */\n", (int)ins->op);
            break;
        }
    }
}

/* ---- Write the full PTX file from a MIR program ---- */

/* First pass: count how many virtual registers and predicates we need.
 * Each comparison (EQ/NE/LT/LE/GT/GE) needs one predicate for setp+selp.
 * Each JZ needs one predicate for its conditional branch test. */
static void count_regs(const wubu_mir_prog_t *p, uint32_t *out_vregs, uint32_t *out_preds)
{
    uint32_t max_vr = 0;
    uint32_t n_preds = 0;

    for (size_t i = 0; i < p->n; i++) {
        const wubu_mir_instr_t *ins = &p->ins[i];
        uint32_t local_max = 0;

        switch (ins->op) {
        case MIR_CONST:
        case MIR_MOV:
        case MIR_ADD: case MIR_SUB: case MIR_MUL: case MIR_DIV: case MIR_MOD:
        case MIR_AND: case MIR_OR: case MIR_XOR:
        case MIR_SHL: case MIR_SHR:
        case MIR_NEG: case MIR_NOT:
        case MIR_LOAD: case MIR_STORE:   /* LOAD/STORE: dst + a (+ b for STORE) */
            local_max = ins->dst + 1;
            if (ins->a + 1 > local_max) local_max = ins->a + 1;
            if (ins->b + 1 > local_max) local_max = ins->b + 1;
            break;
        case MIR_EQ: case MIR_NE: case MIR_LT: case MIR_LE: case MIR_GT: case MIR_GE:
            local_max = ins->dst + 1;
            n_preds++;  /* one predicate for setp */
            break;
        case MIR_JZ:
            local_max = ins->a + 1;
            n_preds++;  /* one predicate for the zero-test */
            break;
        default:
            break;
        }
        if (local_max > max_vr) max_vr = local_max;
    }

    *out_vregs = max_vr + 3;  /* +3: +1 zero-based safety, +2 scratch for mem[] addr math */
    *out_preds = n_preds + 2; /* small safety margin */
}

static char *emit_ptx(const wubu_mir_prog_t *p)
{
    ptx_emitter_t e;
    memset(&e, 0, sizeof(e));

    /* Count registers first so the header declaration is correct */
    uint32_t n_vregs, n_preds;
    count_regs(p, &n_vregs, &n_preds);

    /* PTX header */
    ptx_emit(&e, ".version 8.0\n");
    ptx_emit(&e, ".target sm_89\n");
    ptx_emit(&e, ".address_size 64\n\n");

    /* Kernel signature */
    ptx_emit(&e, ".visible .entry wubu_kernel(\n");
    ptx_emit(&e, "    .param .b64 result,\n");
    ptx_emit(&e, "    .param .b64 arg\n");
    ptx_emit(&e, ") {\n");

    /* Variable memory model: MIR programs address a flat int64[] array
     * (cell 0 reserved as null). Mirror the interpreter with a global
     * mem[] so MIR_LOAD/MIR_STORE work on the GPU. */
    size_t n_cells = (p->total_mem > 0) ? (size_t)(p->total_mem + 1) : 1;
    ptx_emit(&e, "    .global .b64 mem[%zu];\n", n_cells);

    /* Declare registers: %r0..%r{n_vregs-1} plus rd_result */
    ptx_emit(&e, "    .reg .b64 %r<%u>;\n", n_vregs + 1);
    ptx_emit(&e, "    .reg .b64 %rd_result;\n");
    ptx_emit(&e, "    .reg .pred p<%u>;\n", n_preds);
    ptx_emit(&e, "    .reg .u64 %ra<2>;\n");  /* address math for mem[] */
    ptx_emit(&e, "    .reg .u32 %rc<2>;\n\n"); /* 32-bit temp for offset */

    /* Base address of mem[] into %ra0 (a .u64 pointer register). */
    ptx_emit(&e, "    cvta.global.u64 %ra0, mem;\n\n");

    /* Load the arg parameter into vr 0 (the "value" register) */
    ptx_emit(&e, "    ld.param.b64 %r0, [arg];\n\n");

    /* Emit the kernel body */
    emit_kernel_body(&e, p);

    ptx_emit(&e, "}\n");
    return e.text;
}

/* ---- Host stub management ---- */

static int stub_compiled = 0;

static int ensure_stub(void)
{
    if (stub_compiled) return 0;

    /* Check if the stub binary already exists */
    if (access("/tmp/gpu_host_stub", X_OK) == 0) {
        stub_compiled = 1;
        return 0;
    }

    /* Compile the stub. The source lives alongside this .cu file.
     * On WSL2 the CUDA userspace (libcuda.so.1) is NOT in the default
     * linker path — it lives in /usr/lib/wsl/lib (the /dev/dxg GPU
     * route). Bare-metal Linux would have libcuda under the standard
     * path, so include BOTH so the stub links in either environment. */
    const char *stub_src = "/home/wubu/wubuos/src/compiler/gpu_host_stub.cu";
    int rc = system("nvcc -arch=sm_89 -O2 -o /tmp/gpu_host_stub "
                    " -Xptxas -v "  /* show ptxas info */
                    " /home/wubu/wubuos/src/compiler/gpu_host_stub.cu "
                    " -L/usr/lib/wsl/lib -L/usr/lib/x86_64-linux-gnu "
                    " -lcuda "
                    " 2>/tmp/gpu_stub_build.log");
    if (rc != 0) {
        fprintf(stderr, "[ptx] nvcc host stub compilation failed (rc=%d)\n", rc);
        fprintf(stderr, "[ptx] build log:\n");
        FILE *f = fopen("/tmp/gpu_stub_build.log", "r");
        if (f) {
            int c;
            while ((c = fgetc(f)) != EOF) fputc(c, stderr);
            fclose(f);
        }
        return -1;
    }
    stub_compiled = 1;
    return 0;
}

/* ---- Driver API: compile ----
 * Always available: emits PTX text into *out_code/out_size.
 * When WUBU_HOSTED is defined, also compiles to cubin via ptxas. */

static int ptx_compile(const wubu_mir_prog_t *p, uint8_t **out_code, size_t *out_size)
{
    if (!p || !out_code || !out_size) return -1;

    /* 1. Emit PTX assembly (always works, pure C11) */
    char *ptx = emit_ptx(p);
    if (!ptx) return -1;

#ifdef WUBU_HOSTED
    /* 2a. Hosted: compile PTX -> cubin via ptxas */
    const char *ptx_path = "/tmp/wubu_kernel.ptx";
    FILE *f = fopen(ptx_path, "w");
    if (!f) { free(ptx); return -1; }
    fputs(ptx, f);
    fclose(f);

    const char *cubin_path = "/tmp/wubu_kernel.cubin";
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ptxas -arch=sm_89 -O2 %s -o %s 2>/tmp/ptxas_build.log",
             ptx_path, cubin_path);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[ptx] ptxas compilation failed (rc=%d)\n", rc);
        free(ptx);
        return -1;
    }

    FILE *cf = fopen(cubin_path, "rb");
    if (!cf) { free(ptx); return -1; }
    fseek(cf, 0, SEEK_END);
    long cubin_size = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    uint8_t *cubin = malloc((size_t)cubin_size);
    if (!cubin) { fclose(cf); free(ptx); return -1; }
    fread(cubin, 1, (size_t)cubin_size, cf);
    fclose(cf);
    free(ptx);

    *out_code = cubin;
    *out_size = (size_t)cubin_size;
#else
    /* 2b. Self-hosted: return PTX text as-is. The caller can save it,
     *      compile later with ptxas, or use a future native PTX runtime. */
    *out_code = (uint8_t *)ptx;
    *out_size = strlen(ptx);
#endif

    return 0;
}

/* ---- Driver API: run ----
 * Hosted: launches on GPU via CUDA stub.
 * Self-hosted: prints the PTX and returns 0 (no GPU available). */

static int64_t ptx_run(const uint8_t *code, size_t size, int64_t arg)
{
    if (!code || size == 0) return 0;

#ifdef WUBU_HOSTED
    /* Ensure the host stub is compiled */
    if (ensure_stub() != 0) {
        fprintf(stderr, "[ptx] cannot run — host stub not available\n");
        return 0;
    }

    /* Write cubin to /tmp */
    const char *cubin_path = "/tmp/wubu_kernel.cubin";
    FILE *f = fopen(cubin_path, "wb");
    if (!f) return 0;
    fwrite(code, 1, size, f);
    fclose(f);

    /* Launch the host stub. The stub links libcuda.so.1, which on WSL2
     * lives in /usr/lib/wsl/lib; make sure that path is on LD_LIBRARY_PATH
     * so the dynamic loader finds it (matches the /dev/dxg GPU route). */
    char cmd[1024];
    const char *wsl = "/usr/lib/wsl/lib";
    char ldpath[1024];
    snprintf(ldpath, sizeof(ldpath), "%s:%s",
             wsl, getenv("LD_LIBRARY_PATH") ? getenv("LD_LIBRARY_PATH") : "");
    snprintf(cmd, sizeof(cmd),
             "LD_LIBRARY_PATH='%s' /tmp/gpu_host_stub %s %lld 2>/tmp/gpu_run.log",
             ldpath, cubin_path, (long long)arg);

    /* Launch the host stub with a hard timeout. The stub blocks forever if
     * no CUDA device is present, so the child arms SIGALRM; if it fires the
     * child dies and the parent reports "no result" (0). To avoid paying the
     * timeout on every one of the ~19k gauntlet tests, we probe ONCE: the
     * first call that fails (timeout or crash) marks the GPU permanently
     * unavailable and every later call returns 0 immediately. */
    char result_path[256];
    snprintf(result_path, sizeof(result_path), "/tmp/ptx_result_%d.txt", (int)getpid());
    remove(result_path);  /* clean slate — no stale reads */

    static int ptx_gpu_unavailable = -1;  /* -1 = not yet probed */
    if (ptx_gpu_unavailable == 1) return 0;  /* already known: no GPU */

    pid_t child = fork();
    if (child == 0) {
        /* child: run the stub; die if it stalls. Redirect stdout to the
         * result file so the parent can read the int64 back. */
        signal(SIGALRM, SIG_DFL);
        alarm((unsigned)PTX_RUN_TIMEOUT_SEC);
        FILE *rf = freopen(result_path, "w", stdout);
        (void)rf;
        int rc = system(cmd);
        (void)rc;
        _exit(0);
    }

    int64_t result = 0;
    int status = 0;
    time_t t0 = time(NULL);
    for (;;) {
        pid_t w = waitpid(child, &status, WNOHANG);
        if (w == child) break;
        if (w < 0 && errno != EINTR) break;
        if (difftime(time(NULL), t0) > (double)(PTX_RUN_TIMEOUT_SEC + 5)) {
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            ptx_gpu_unavailable = 1;
            fprintf(stderr, "[ptx] host stub unresponsive (no GPU?) — "
                            "disabling ptx for remaining tests\n");
            remove(result_path);
            return 0;
        }
        usleep(100000);  /* 100ms */
    }

    /* A child killed by signal (e.g. our SIGALRM) means the GPU is not
     * reachable — disable ptx for the rest of this run. */
    if (WIFSIGNALED(status)) {
        ptx_gpu_unavailable = 1;
        fprintf(stderr, "[ptx] host stub died (signal %d) — "
                        "disabling ptx for remaining tests\n",
                WTERMSIG(status));
        remove(result_path);
        return 0;
    }

    ptx_gpu_unavailable = 0;  /* a launch completed cleanly → GPU reachable */
    FILE *rf = fopen(result_path, "r");
    if (rf) {
        if (fscanf(rf, "%lld", (long long *)&result) != 1)
            result = 0;
        fclose(rf);
    }
    remove(result_path);
    return result;
#else
    /* Self-hosted: no GPU runtime available. Print PTX for offline compilation. */
    printf("[ptx] Generated PTX assembly (%zu bytes):\n", size);
    printf("--- BEGIN PTX ---\n");
    fwrite(code, 1, size, stdout);
    printf("--- END PTX ---\n");
    printf("[ptx] Compile with: ptxas -arch=sm_89 -o kernel.cubin kernel.ptx\n");
    (void)arg;
    return 0;
#endif
}

/* ---- Driver API: describe ---- */

static void ptx_describe(void)
{
    printf("PTX (NVIDIA GPU) ISA driver\n");
    printf("  Family:        gpu\n");
    printf("  Target:        sm_89 (Ada Lovelace / RTX 40-series)\n");
    printf("  PTX version:   8.0\n");
    printf("  Exec model:    native (CUDA driver API via host stub)\n");
    printf("  Compile:       MIR -> PTX -> cubin (ptxas)\n");
    printf("  Run:           cubin -> GPU launch -> result\n");
    printf("  MIR ops:       ADD SUB MUL DIV MOD AND OR XOR SHL SHR\n");
    printf("                 NEG NOT EQ NE LT LE GT GE MOV JMP JZ RET\n");
    printf("  Registers:     unlimited virtual -> PTX .reg .b64\n");
}

/* ---- The driver object ---- */

const wubu_isa_driver_t wubu_isa_ptx = {
    .name     = "ptx",
    .family   = "gpu",
    .exec     = WUBU_ISA_NATIVE,  /* GPU runs natively on this machine */
    .compile  = ptx_compile,
    .run      = ptx_run,
    .describe = ptx_describe,
};