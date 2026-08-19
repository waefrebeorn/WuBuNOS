/*
 * holyd_mir_eval.h  --  HolyD AST → MIR → ISA driver evaluation.
 *
 * Parses HolyD, emits MIR, compiles + runs via any ISA driver.
 * This is the cross-target bridge for the universal test gauntlet.
 *
 * C11, self-contained.
 */
#ifndef HOLYC_MIR_EVAL_H
#define HOLYC_MIR_EVAL_H

#include <stdint.h>
struct wubu_isa_driver;
typedef struct wubu_isa_driver wubu_isa_driver_t;

#include "wubu_mir.h"   /* for wubu_mir_prog_t */

/* Parse HolyD source, emit canonical optimized MIR. Returns 0 on success
 * (prog initialized; caller frees via wubu_mir_free), -1 on error. */
int hd_build_mir(const char *source, wubu_mir_prog_t *prog);

/* Like hd_build_mir, but on failure fills errbuf (errcap) with the parser's
 * first diagnostic. Used by the gauntlet to classify failure reasons.
 * Returns 0 on success, -1 on error (errbuf valid then). */
int hd_build_mir_ex(const char *source, wubu_mir_prog_t *prog,
                    char *errbuf, size_t errcap);

/* Compile + run an already-built MIR program via the given driver. Falls back
 * to the portable interpreter if driver is NULL or its encoder fails. */
int64_t hd_run_prog(const wubu_mir_prog_t *prog, const wubu_isa_driver_t *driver);

/* Parse HolyD source, emit MIR, compile + run via the given driver.
 * Returns the result (0 on error). */
int64_t hd_eval_mir(const char *source, const wubu_isa_driver_t *driver);

#endif /* HOLYC_MIR_EVAL_H */
