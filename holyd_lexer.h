/*
 * holyd_lexer.h  --  HolyD Lexer
 * Tokenizes HolyD source code into HDToken stream.
 * Self-contained, C11, minimal includes.
 */
#ifndef WUBUNOS_HOLYC_LEXER_H
#define WUBUNOS_HOLYC_LEXER_H

#include "holyd_types.h"

void hd_lex_init(HDLexer *lex, const char *source);
HDTokenType hd_lex_next(HDLexer *lex);
HDTokenType hd_lex_peek(HDLexer *lex);
int hd_lex_expect(HDLexer *lex, HDTokenType expected);

/* Width classification for integer literals: 64 when a long/long-long suffix
 * is present, else 32. (HolyD targets SysV x86-64, where long == 64 bits.) */
int hd_int_width(const HDToken *t);

#endif /* WUBUNOS_HOLYC_LEXER_H */