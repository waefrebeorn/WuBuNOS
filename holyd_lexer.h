/*
 * holyc_lexer.h  --  HolyC Lexer
 * Tokenizes HolyC source code into HCToken stream.
 * Self-contained, C11, minimal includes.
 */
#ifndef WUBUNOS_HOLYC_LEXER_H
#define WUBUNOS_HOLYC_LEXER_H

#include "holyc_types.h"

void hc_lex_init(HCLexer *lex, const char *source);
HCTokenType hc_lex_next(HCLexer *lex);
HCTokenType hc_lex_peek(HCLexer *lex);
int hc_lex_expect(HCLexer *lex, HCTokenType expected);

#endif /* WUBUNOS_HOLYC_LEXER_H */