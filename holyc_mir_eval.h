/*
 * holyc_mir_eval.h  --  HolyC AST → MIR → ISA driver evaluation.
 *
 * Parses HolyC, emits MIR, compiles + runs via any ISA driver.
 * This is the cross-target bridge for the universal test gauntlet.
 *
 * C11, self-contained.
 */
#ifndef HOLYC_MIR_EVAL_H
#define HOLYC_MIR_EVAL_H

#include <stdint.h>
struct wubu_isa_driver;
typedef struct wubu_isa_driver wubu_isa_driver_t;

/* Parse HolyC source, emit MIR, compile + run via the given driver.
 * Returns the result (0 on error). */
int64_t hc_eval_mir(const char *source, const wubu_isa_driver_t *driver);

#endif /* HOLYC_MIR_EVAL_H */
