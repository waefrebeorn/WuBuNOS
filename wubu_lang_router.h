/*
 * wubu_lang_router.h — Language routing and policy enforcement.
 *
 * WuBuOS accepts:
 *   - HolyD (native, all 14 ISAs, full optimization)
 *   - C18 (HolyD's C subset — C11 + bugfix patches, no higher)
 *   - Brainfuck (the meme, compiled for real)
 *
 * Everything else is "foreign" — routed to containerized compilers
 * or rejected with a helpful message.
 *
 * C18 policy: C11 + ISO/IEC 9899:2018 defect reports. No C23, no C2x,
 * no GNU extensions, no C++. Jens Gustedt calls C17/C18 a "bugfix release"
 * of C11 — no new features, just clarifications. That's our baseline.
 *
 * The "brainfuck" exception exists because:
 *   1. It's a funny meme
 *   2. It demonstrates our compiler's simplicity (even BF compiles)
 *   3. AGI-era "backdoor coding" — hard-to-read languages that
 *      humans can't easily audit but AGIs can generate
 *   4. The next era of programming may involve languages that
 *      look like noise but encode complex behavior
 *
 * We accept the meme. We prepare for the future.
 */

#ifndef WUBU_LANG_ROUTER_H
#define WUBU_LANG_ROUTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#error "C11 only. No C++."
#endif

/* ---- Language categories */
typedef enum {
    LANG_HOLYC = 0,     /* Native — full support, all targets */
    LANG_BRAINFUCK,     /* The meme — compiled for real */
    LANG_C,             /* Foreign — route to containerized GCC/LLVM */
    LANG_CPP,           /* Foreign — route to containerized G++/Clang++ */
    LANG_RUST,          /* Foreign — route to containerized rustc */
    LANG_GO,            /* Foreign — route to containerized go */
    LANG_PYTHON,        /* JIT template → WASM or native */
    LANG_JAVASCRIPT,    /* JIT template → WASM or native */
    LANG_JAVA,          /* Foreign — route to containerized javac */
    LANG_SWIFT,         /* Foreign — route to containerized swiftc */
    LANG_FORTRAN,       /* Foreign — route to containerized gfortran */
    LANG_ADA,           /* Foreign — route to containerized gnat */
    LANG_COBOL,         /* Foreign — route to containerized cobc */
    LANG_UNKNOWN,       /* Reject with helpful message */
} lang_t;

/* ---- Compilation policy flags */
typedef enum {
    POLICY_ACCEPT     = 0x01,   /* We compile this natively */
    POLICY_TEMPLATE   = 0x02,   /* JIT template available */
    POLICY_CONTAINER  = 0x04,   /* Route to containerized compiler */
    POLICY_REJECT     = 0x08,   /* Reject — not supported */
    POLICY_MEME       = 0x10,   /* Accepted for meme value */
    POLICY_AGI_READY  = 0x20,   /* AGI-era backdoor coding candidate */
    POLICY_WASM_FALLBACK = 0x40,/* Can fall back to WASM */
} policy_flag_t;

/* ---- Routing decision */
typedef struct {
    lang_t       language;
    uint32_t     policy_flags;
    const char  *description;
    const char  *action_message;
    const char  *container_image;   /* For POLICY_CONTAINER */
    const char  *template_name;     /* For POLICY_TEMPLATE */
} lang_routing_t;

/* ---- API */

/* Detect language from file extension and/or shebang */
lang_t lang_detect(const char *filename, const char *source, size_t len);

/* Get routing decision for a language */
const lang_routing_t *lang_route(lang_t lang);

/* Get human-readable description */
const char *lang_describe(lang_t lang);

/* Check if a policy flag is set */
int lang_has_policy(lang_t lang, policy_flag_t flag);

/* Route and compile (returns 0 on success) */
int lang_compile(const char *filename, const char *source, size_t len,
                 const char *output, const char *target_isa);

/* ---- Built-in descriptions */

#define HOLYC_DESC      "HolyD — the sacred tongue. Full native support on all targets."
#define BRAINFUCK_DESC  "Brainfuck — the meme. Compiled for real because we can."
#define C_DESC          "C (foreign) — routed to containerized GCC. We prefer HolyD."
#define CPP_DESC        "C++ (foreign) — routed to containerized Clang++. We prefer HolyD."
#define RUST_DESC       "Rust (foreign) — routed to containerized rustc. We prefer HolyD."
#define PYTHON_DESC     "Python — JIT template available. Compiles to WASM or native."
#define JS_DESC         "JavaScript — JIT template available. Compiles to WASM or native."
#define FOREIGN_DESC    "Foreign language — routed to containerized compiler."
#define UNKNOWN_DESC    "Unknown language — rejected. Use HolyD or Brainfuck."

#endif /* WUBU_LANG_ROUTER_H */
