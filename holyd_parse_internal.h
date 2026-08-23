/* holyd_parse_internal.h -- Internal helpers shared by holyd_parse sub-modules.
 * Public API + types in holyd_parse.h. The AST construction helpers live in
 * holyd_parse_ast.c and are declared here so all submodules link the SAME
 * implementation (no double-coding).
 */

#ifndef HOLYC_PARSE_INTERNAL_H
#define HOLYC_PARSE_INTERNAL_H

#include "holyd.h"
#include <stdlib.h>

/* -- AST construction helpers (holyd_parse_ast.c) --------------- */
HDASTNode *hd_ast_new(HDASTKind kind);
void       hd_ast_free(HDASTNode *node);
void       hd_ast_add_stmt(HDASTNode *block, HDASTNode *stmt);
void       hd_ast_add_arg(HDASTNode *call, HDASTNode *arg);

#endif /* HOLYC_PARSE_INTERNAL_H */
