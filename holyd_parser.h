/*
 * holyd_parser.h  --  HolyD Parser
 * Parses HDToken stream into Abstract Syntax Tree (AST).
 * Self-contained, C11, minimal includes.
 */
#ifndef WUBUNOS_HOLYC_PARSER_H
#define WUBUNOS_HOLYC_PARSER_H

#include "holyd_types.h"

void hd_parse_init(HDParser *p, HDLexer *lex);
HDASTNode *hd_parse_expr(HDParser *p);
HDASTNode *hd_parse_stmt(HDParser *p);
HDASTNode *hd_parse_decl(HDParser *p);
HDASTNode *hd_parse_compilation_unit(HDParser *p);
size_t hd_type_size(const HDType *t);
HDASTNode *hd_parse_block(HDParser *p);
HDTokenType hd_parse_peek(HDParser *p);

#endif /* WUBUNOS_HOLYC_PARSER_H */