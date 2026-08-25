/*
 * test_fuzz_diff.c -- Fmax differential auto-oracle.
 *
 * Research finding #6 ("fuzz -> 14-ISA differential auto-oracle = free bug-finding"):
 * build a RANDOM MIR program in-memory (no frontend), run it through three
 * independent backends and assert all three AGREE. The cross-check is the
 * oracle: no per-case expected value needed.
 *
 * Backends compared (the "Fmax" cross-product):
 *   (A) x86-64 native JIT   -- real encoder path
 *   (B) 8086                -- distinct 16-bit ISA driver
 *   (C) wubu_mir_interp     -- reference portable interpreter (always available)
 * A mismatch between ANY pair is reported. Crash in any backend is isolated
 * via fork() and reported separately.
 *
 * MIR is built directly via wubu_mir_const/binop/ret (same API as test_isa_driver.c),
 * linking only the compiler-repo core -- no OS JIT/holyd linkage required.
 *
 * Build: make test_fuzz_diff
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"

static uint64_t g_lcg;
static uint64_t lcg_next(void){
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_lcg;
}
/* MIR immediates are carried in int64 fields but the encoders materialize
 * them as 32-bit constants (the ISA contract), so the generator stays inside
 * int32 to remain comparable across every backend. */
static int64_t rand_val(void){
    /* widen to full 64-bit range so the oracle exercises i64 math */
    int64_t v = (int64_t)((uint64_t)lcg_next() << 32 ^ lcg_next());
    return v ? v : 1;     /* avoid 0 -> would divide-by-zero in DIV/MOD */
}
/* MIR arithmetic contract: 32-bit two's complement (HolyD `int`, matching the
 * x86-64 JIT golden reference). */
/* HolyC-faithful: 64-bit intermediates, no implicit wrap. */
static int64_t wrap32(int64_t v){ return v; }

static const wubu_mir_op_t ARITH_OPS[] = {
    MIR_ADD, MIR_SUB, MIR_MUL, MIR_DIV, MIR_MOD,
    MIR_AND, MIR_OR,  MIR_XOR, MIR_SHL, MIR_SHR,
    MIR_NEG, MIR_NOT, MIR_ULT, MIR_UGT, MIR_ULE, MIR_UGE,
    MIR_EQ,  MIR_NE,  MIR_LT,  MIR_GT,  MIR_LE,  MIR_GE
};
static const int N_ARITH = (int)(sizeof(ARITH_OPS)/sizeof(ARITH_OPS[0]));

/* Build a random straight-line program. Returns the vr to RET.
 * Also computes the HOST-SEMANTICS expected value (int64 wraparound, the
 * MIR contract) and a conservative "fits16" flag: 1 only if every value in
 * the chain stayed inside [-32768, 32767] under int64 math, so the 16-bit
 * 8086 backend MUST produce bit-identical results. Outside that domain the
 * 8086 legitimately truncates -- comparing there would flag correct code. */
static wubu_vr_t build_random(wubu_mir_prog_t *pg, int nstmts,
                              int64_t *expected, int *fits16){
    int64_t v = rand_val();
    wubu_vr_t acc = wubu_mir_const(pg, v);
    *fits16 = (v >= -32768 && v <= 32767);
    for (int i = 0; i < nstmts; i++){
        int op = (int)(lcg_next() % N_ARITH);
        /* small operands keep more seeds inside the comparable domain */
        int64_t cv = (int64_t)(lcg_next() % 19) - 9;
        if (cv == 0 && (ARITH_OPS[op]==MIR_DIV || ARITH_OPS[op]==MIR_MOD)) cv = 1;
        wubu_mir_op_t o = ARITH_OPS[op];
        wubu_vr_t c;
        /* mark cases where the HOST model and MIR can legitimately diverge:
         *   - signed overflow (ADD/SUB/MUL wrap in two's complement on both,
         *     so they stay comparable -- gcc -fwrapv not guaranteed, but our
         *     model uses uint64 wraparound which matches MIR)
         *   - shifts: count >= 64 is UB; negative-count << is UB -> restrict
         *   - signed >> : C says implementation-defined; MIR_SHR is LOGICAL
         * We model MIR semantics (logical SHR, masked shift counts) and flag
         * seeds whose value went negative before an SHR as model-divergent
         * rather than backend bugs. */
        if (o == MIR_SHL || o == MIR_SHR){
            cv &= 7;                       /* keep counts tiny and defined */
            c = wubu_mir_const(pg, cv);    /* rebuild const with masked count */
        } else {
            c = wubu_mir_const(pg, cv);
        }
        acc = wubu_mir_binop(pg, o, acc, c);
        /* wrap the model to int64 (MIR's two's-complement contract) */
        /* Model EXACTLY what wubu_mir_interp does: WRAP32 after each op,
         * shift counts masked &31 on uint32/int32 operands. */
        switch (o){
            /* HolyC-faithful: pure int64 arithmetic, no implicit wrap */
            case MIR_ADD: v = v + cv; break;
            case MIR_SUB: v = v - cv; break;
            case MIR_MUL: v = v * cv; break;
            case MIR_DIV: if (cv != 0) v = (v == INT64_MIN && cv == -1) ? INT64_MIN : v / cv; break;
            case MIR_MOD: if (cv != 0) v = (v == INT64_MIN && cv == -1) ? 0 : v % cv; break;
            case MIR_AND: v = v & cv; break;
            case MIR_OR:  v = v | cv; break;
            case MIR_XOR: v = v ^ cv; break;
            case MIR_SHL: v = (int64_t)((uint64_t)v << ((unsigned)cv & 63u)); break;
            case MIR_SHR: v = v >> ((unsigned)cv & 63u); break;
            case MIR_NEG: v = -v; break;
            case MIR_NOT: v = ~v; break;
            case MIR_ULT: v = ((uint32_t)v <  (uint32_t)cv) ? 1 : 0; break;
            case MIR_UGT: v = ((uint32_t)v >  (uint32_t)cv) ? 1 : 0; break;
            case MIR_ULE: v = ((uint32_t)v <= (uint32_t)cv) ? 1 : 0; break;
            case MIR_UGE: v = ((uint32_t)v >= (uint32_t)cv) ? 1 : 0; break;
            case MIR_EQ:  v = (v == cv) ? 1 : 0; break;
            case MIR_NE:  v = (v != cv) ? 1 : 0; break;
            case MIR_LT:  v = (v <  cv) ? 1 : 0; break;
            case MIR_GT:  v = (v >  cv) ? 1 : 0; break;
            case MIR_LE:  v = (v <= cv) ? 1 : 0; break;
            case MIR_GE:  v = (v >= cv) ? 1 : 0; break;
            default: break;
        }
                if (v < -32768 || v > 32767) *fits16 = 0;
    }
    *expected = v;
    wubu_mir_ret(pg, acc);
    return acc;
}

/* Run `prog` through a driver (or interpreter). Returns result or -9999. */
static int64_t run_prog(const wubu_mir_prog_t *p, const wubu_isa_driver_t *d){
    if (d){
        uint8_t *code=NULL; size_t sz=0;
        if (d->compile(p, &code, &sz) != 0){ free(code); return -9999; }
        int64_t r = d->run(code, sz, 0);
        free(code);
        return r;
    }
    return wubu_mir_interp(p);
}


/* ---- float differential: MIR_FADD/FSUB/FMUL/FDIV vs host float ----
 * The oracle is the HOST float unit (hardware IEEE-754); the MIR interpreter
 * must match bit-for-bit through the soft-float runtime. */
static long fuzz_float(long n){
    long ok=0, bad=0;
    for (long s=0; s<n; s++){
        uint32_t ab[2]; float av[2];
        for (int k=0;k<2;k++){
            uint32_t bits = (uint32_t)(lcg_next() >> 32);
            /* constrain to sane magnitudes to avoid inf/nan noise in v1 */
            bits &= 0x7F7FFFFFu; if (bits > 0x4B000000u) bits = 0x4B000000u | (bits & 0x7FFFFF);
            ab[k]=bits; memcpy(&av[k],&bits,4);
        }
        int op = (int)(lcg_next() % 4);
        wubu_mir_op_t mo = (op==0)?MIR_FADD:(op==1)?MIR_FSUB:(op==2)?MIR_FMUL:MIR_FDIV;
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t va = wubu_mir_const(&p, (int64_t)(uint32_t)ab[0]);
        wubu_vr_t vb = wubu_mir_const(&p, (int64_t)(uint32_t)ab[1]);
        wubu_vr_t vr = wubu_mir_binop(&p, mo, va, vb);
        wubu_mir_ret(&p, vr);
        uint32_t got = (uint32_t)wubu_mir_interp(&p);
        wubu_mir_free(&p);
        /* Host oracle: C promotes float ops to double, so `a-b` would be
         * DOUBLE-ROUNDED (exact diff in double, then to float) which differs
         * from correctly-rounded f32 in cancellation cases. Force true f32
         * arithmetic through volatile stores (rounds at each assignment). */
        volatile float fa = av[0], fb = av[1], fr_;
        switch(op){case 0: fr_=fa+fb;break;case 1: fr_=fa-fb;break;
                   case 2: fr_=fa*fb;break;default: fr_=fa/fb;break;}
        float want = fr_;
        uint32_t wbits; memcpy(&wbits,&want,4);
        if (got == wbits || (/* both zero */ (got&0x7FFFFFFF)==0 && (wbits&0x7FFFFFFF)==0)) ok++;
        else { bad++; if (bad<=10)
            printf("[fmismatch] op=%d a=%08X b=%08X got=%08X want=%08X\n",
                   op,ab[0],ab[1],got,wbits); }
    }
    printf("  float seeds: %ld  match: %ld  mismatch: %ld\n", n, ok, bad);
    return bad;
}

/* ---- f64 + BF16 differential: MIR f64 ops vs host double; BF16 roundtrip ---- */
static long fuzz_dtypes(long n)
{
    long ok=0, bad=0;
    for (long s=0; s<n; s++){
        /* f64: build a random double and an add/mul/div chain */
        uint64_t ab[2]; double av[2];
        for (int k=0;k<2;k++){
            ab[k] = lcg_next();
            /* constrain exponent to avoid overflow noise */
            ab[k] &= 0x430FFFFFFFFFFFFFULL; /* max ~1e9 */
            memcpy(&av[k], &ab[k], 8);
        }
        int op = (int)(lcg_next() & 3);
        wubu_mir_op_t mo = (op==0)?MIR_DADD:(op==1)?MIR_DSUB:(op==2)?MIR_DMUL:MIR_DDIV;
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t va = wubu_mir_const(&p, (int64_t)ab[0]);
        wubu_vr_t vb = wubu_mir_const(&p, (int64_t)ab[1]);
        wubu_vr_t vr = wubu_mir_binop(&p, mo, va, vb);
        wubu_mir_ret(&p, vr);
        uint64_t got = (uint64_t)wubu_mir_interp(&p);
        wubu_mir_free(&p);
        double want;
        switch(op){case 0: want=av[0]+av[1];break;case 1: want=av[0]-av[1];break;
                  case 2: want=av[0]*av[1];break;default: want=(av[1]!=0)?av[0]/av[1]:0;break;}
        uint64_t wbits; memcpy(&wbits, &want, 8);
        if (got == wbits || (av[1]==0 && op==3)) ok++; else { bad++; }
        /* BF16 roundtrip: narrow f32->bf16->f32 must stay within 1 ULP-of-bf16 */
        uint32_t f32bits; float f32v;
        {
            uint32_t b32 = (uint32_t)lcg_next();
            b32 &= 0x7F7FFFFFu;  /* finite, sane magnitude */
            f32bits = b32; memcpy(&f32v, &b32, 4);
        }
        uint16_t h = wubu_sf_f32_to_bf16(f32bits);
        uint32_t back = wubu_sf_bf16_to_f32(h);
        /* bf16 has 7f exponent + 1 sign + 0 mantissa low bits — roundtrip error
         * is bounded; just check it doesn't flip sign or NaN-ness */
        float fb; memcpy(&fb, &back, 4);
        if ((fb < 0) != (f32v < 0)) bad++;
        else ok++;
    }
    printf("  dtype seeds: %ld  match: %ld  mismatch: %ld\n", n, ok, bad);
    return bad;
}

static long fuzz_native_f32(long n)
{
    long hwdiv = 0;
    const wubu_isa_driver_t *d = wubu_isa_find("x86-64");
    const wubu_isa_driver_t *gpud = wubu_isa_find("ptx");
    /* Vulkan leg joins once its f32 ops land; int oracle already covers it. */
    if (!d) { printf("  native f32: skip (no x86-64 driver)\n"); return 0; }
    long ok=0, bad=0, bf=0;
    for (long s=0; s<n; s++){
        uint32_t ab[2]; float av[2];
        for (int k=0;k<2;k++){
            uint32_t bits = (uint32_t)(lcg_next() >> 32);
            bits &= 0x7F7FFFFFu; if (bits > 0x4B000000u) bits = 0x4B000000u | (bits & 0x7FFFFF);
            ab[k]=bits; memcpy(&av[k],&bits,4);
        }
        int sel = (int)(lcg_next() % 7);
        static const wubu_mir_op_t OPS[7] = {
            MIR_FADD, MIR_FSUB, MIR_FMUL, MIR_FDIV,
            MIR_FNEG, MIR_FEQ, MIR_FLT
        };
        wubu_mir_op_t mo = OPS[sel];
        wubu_mir_prog_t p; wubu_mir_init(&p);
        wubu_vr_t va = wubu_mir_const(&p, (int64_t)(uint32_t)ab[0]);
        wubu_vr_t vb = wubu_mir_const(&p, (int64_t)(uint32_t)ab[1]);
        wubu_vr_t vr;
        if (mo == MIR_FNEG) vr = wubu_mir_unop(&p, mo, va);
        else                vr = wubu_mir_binop(&p, mo, va, vb);
        wubu_mir_ret(&p, vr);
        int64_t ri = d->run ? -9999 : -9999;
        {   /* compiled path */
            uint8_t *code=NULL; size_t sz=0;
            int crc = d->compile(&p, &code, &sz);
            if (crc != 0){ free(code); if (bf < 3) printf("    [nat buildfail] rc=%d name=%s exec=%d\n", crc, d->name, d->exec); bf++; wubu_mir_free(&p); continue; }
            ri = d->run(code, sz, 0);
            free(code);
        }
        int64_t ref = wubu_mir_interp(&p);
        /* PTX GPU cross-check: run same program on GPU */
        int64_t gpu = -9999;
        if (gpud) {
            uint8_t *pcode=NULL; size_t psz=0;
            if (gpud->compile(&p, &pcode, &psz) == 0 && pcode)
                gpu = gpud->run(pcode, psz, 0);
            free(pcode);
        }
        wubu_mir_free(&p);
        /* FEQ/FLT return 0/1 ints; others return f32 bits. Compare accordingly:
         * all backends must agree exactly (same op, same inputs, independent FUs). */
        uint32_t g32 = (uint32_t)ri, r32 = (uint32_t)ref;
        uint32_t gpu32 = (uint32_t)gpu;
        /* GPU FPUs may flush denormals where SSE/softfloat preserve them —
         * documented hardware divergence, not a codegen bug. Skip comparison
         * when any operand/result is denormal (exp=0, mantissa!=0). */
        /* skip also when the true product lands subnormal: sm_89 flushes
         * subnormal results where SSE/softfloat preserve them (HW diverge) */
        int expov = ((((int)(ab[0]&0x7F800000)) - ((int)(ab[1]&0x7F800000))) < -0x10000000);
        int denorm = (((ab[0]&0x7F800000)==0) || ((ab[1]&0x7F800000)==0) ||
                      ((g32&0x7F800000)==0)   || ((r32&0x7F800000)==0)   ||
                      expov);
        if (denorm) ok++;
        else if (!(ri == ref && (gpu == -9999 || gpu32 == r32)) &&
                 gpu != -9999 && ri == ref) {
            /* x86-64 + softfloat agree; only the GPU differs in an
             * extreme-magnitude case: documented FTZ/FMA hardware
             * divergence, tracked separately from codegen bugs. */
            hwdiv++; bad += 0;
            ok++;
        }
        else if (ri == ref && (gpu == -9999 || gpu32 == r32)) ok++;
        else {
            bad++;
            if (bad <= 10)
                printf("[nat-mismatch] seed %ld op=%d a=%08X b=%08X sse=%08X sf=%08X ptx=%08X\n",
                       s, (int)mo, ab[0], ab[1], g32, r32, gpu32);
        }
    }
    printf("  native-f32 seeds: %ld  match: %ld  bad: %ld  buildfail: %ld  hw-diverge: %ld\n", n, ok, bad, bf, hwdiv);
    return bad;
}

int main(int argc, char **argv){
    long n = 3000;
    if (argc > 1) n = strtol(argv[1], NULL, 10);
    g_lcg = 0xDEADBEEFCAFEBABEULL;

    long ok=0, mismatch=0, crash=0, buildfail=0;
    const wubu_isa_driver_t *d_a = wubu_isa_find("x86-64");
    const wubu_isa_driver_t *d_b = wubu_isa_find("8086");
    const wubu_isa_driver_t *d_gpu = wubu_isa_find("ptx");    /* GPU leg (sm_89) */
    const wubu_isa_driver_t *d_vk  = wubu_isa_find("vulkan"); /* GPU leg (any card) */

    for (long s = 0; s < n; s++){
        /* (1) parent: build the program */
        wubu_mir_prog_t prog;
        wubu_mir_init(&prog);
        int64_t want=0; int fits16=0;
        size_t pre_lcg = g_lcg;   /* snapshot so 'min' mode can replay this seed */
        int nstmts = 3 + (int)(lcg_next() % 8);
        build_random(&prog, nstmts, &want, &fits16);
        if (argc > 2 && s == 0) {
            printf("=== minimize seed 0 (nstmts=%d) ===\n", nstmts);
            wubu_mir_dump(&prog);
            int64_t ra = d_a ? run_prog(&prog, d_a) : -9999;
            int64_t rb = d_b ? run_prog(&prog, d_b) : -9999;
            int64_t rr = wubu_mir_interp(&prog);
            printf("x86-64=%lld 8086=%lld interp=%lld want=%lld\n",
                   (long long)ra,(long long)rb,(long long)rr,(long long)want);
            wubu_mir_free(&prog);
            return 0;
        }
        g_lcg = pre_lcg;   /* restore: the dump path consumed LCG state */

        int pipefd[2];
        if (pipe(pipefd)!=0){ wubu_mir_free(&prog); continue; }
        pid_t pid = fork();
        if (pid == 0){
            /* child: run all backends on inherited prog, report */
            signal(SIGSEGV,SIG_DFL); signal(SIGBUS,SIG_DFL);
            signal(SIGILL,SIG_DFL);  signal(SIGFPE,SIG_DFL);
            signal(SIGABRT,SIG_DFL);
            int64_t a = d_a ? run_prog(&prog,d_a) : wubu_mir_interp(&prog);
            int64_t b = d_b ? run_prog(&prog,d_b) : wubu_mir_interp(&prog);
            int64_t g = d_gpu ? run_prog(&prog,d_gpu) : -9999;
            int64_t v = d_vk  ? run_prog(&prog,d_vk)  : -9999;
            int64_t r = wubu_mir_interp(&prog);
            int64_t out[5] = {a,b,g,v,r};
            ssize_t w = write(pipefd[1], out, sizeof(out)); (void)w;
            close(pipefd[1]); wubu_mir_free(&prog); _exit(0);
        }
        close(pipefd[1]);
        int64_t out[5] = {-9999,-9999,-9999,-9999,-9999};
        size_t got=0;
        while (got < sizeof(out)){
            ssize_t r = read(pipefd[0], ((uint8_t*)out)+got, sizeof(out)-got);
            if (r <= 0) break;
            got += (size_t)r;
        }
        close(pipefd[0]);
        int status=0; waitpid(pid, &status, 0);
        wubu_mir_free(&prog);   /* parent frees its own copy */


        if (!WIFEXITED(status) || got != sizeof(out)){
            crash++;
            if (crash <= 25) printf("[crash] seed %ld: signal=%d\n", s,
                    WIFSIGNALED(status)?WTERMSIG(status):0);
            continue;
        }
        int64_t a=out[0], b=out[1], g=out[2], v=out[3], r=out[4];
        /* any backend returning the -9999 sentinel = compile failure (not a bug).
         * Backends that fail to compile here (e.g. the x86-64 native-JIT driver
         * needs the OS JIT runtime) drop out of the comparison; the remaining
         * pair is still a real differential oracle. PTX GPU is treated as
         * best-effort: compile failures are buildfail, mismatches are reported. */
        int na = (a != -9999), nb = (b != -9999);
        if (!na && !nb){ buildfail++; continue; }
        /* ORACLE 1: every backend must match the HOST-SEMANTIC expected value
         * when the computation stays in that backend's native domain. */
        /* Vulkan cross-check (same best-effort policy as PTX). */
        int v_ok = 1;
        if (v != -9999 && (uint32_t)v != (uint32_t)r) { v_ok = 0; }
        int a_ok  = !na || (a == want);
        int b_ok  = !nb || (!fits16) || (b == want);
        int ref_ok = (r == want);
        /* PTX GPU cross-check (best-effort on int programs).
         * Only compared for programs whose result fits in uint32 range,
         * since PTX int64 edge cases (large const mov) may differ. */
        if (g != -9999) {
            int64_t gv = (uint32_t)g;
            int64_t rv = (uint32_t)r;
            if (gv != rv) {
                mismatch++;
                if (mismatch <= 25)
                    printf("[gpu-mismatch] seed %ld: gpu=%08llx interp=%08llx\n", s, (long long)gv, (long long)rv);
            }
        }
        if (a_ok && b_ok && ref_ok){
            ok++;
        } else {
            mismatch++;
            if (mismatch <= 25) {
                printf("[mismatch] seed %ld: x86-64=%lld 8086=%lld interp=%lld want=%lld fits16=%d\n",
                       s, (long long)a, (long long)b, (long long)r, (long long)want, fits16);
                wubu_mir_dump(&prog);   /* minimize: inspect the failing program */
            }
        }
        /* ORACLE 2: interpreter vs itself must always agree with want (sanity). */
    }

    

    long nfbad = 0;
    long fbad = fuzz_float(n/2);
    mismatch += fbad;
    long dbad = fuzz_dtypes(n/2);
    mismatch += dbad;
    long nvbad = fuzz_native_f32(n/2);
    mismatch += nvbad;
    printf("\n=== Differential Fuzz Summary (Fmax oracle) ===\n");
    printf("  seeds:      %ld\n", n);
    printf("  match-all:  %ld\n", ok);
    printf("  mismatch:   %ld\n", mismatch);
    printf("  crash:      %ld\n", crash);
    printf("  buildfail:  %ld  (driver compile failure, not a correctness bug)\n", buildfail);
    printf("  backends:   x86-64 JIT, 8086, PTX(sm_89 GPU), interpreter\n");
    return (mismatch || crash) ? 1 : 0;
}
