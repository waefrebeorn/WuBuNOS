/*
 * holyd_lexer.c  --  HolyD Lexer
 * Tokenizes HolyD source text into a stream of tokens.
 * Self-contained, C11, minimal includes.
 */

#include "holyd_types.h"
#include <stdarg.h>

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* Forward declarations */
HDTokenType hd_lex_next(HDLexer *lex);
static void hd_skip_whitespace(HDLexer *lex);

/* -- Keyword Table ------------------------------------------------ */

typedef struct {
    const char  *name;
    HDTokenType  type;
} HDKeyword;

static const HDKeyword hd_keywords[] = {
    /* Control flow */
    {"if",       HD_KW_IF},
    {"else",     HD_KW_ELSE},
    {"while",    HD_KW_WHILE},
    {"for",      HD_KW_FOR},
    {"do",       HD_KW_DO},
    {"switch",   HD_KW_SWITCH},
    {"case",     HD_KW_CASE},
    {"default",  HD_KW_DEFAULT},
    {"break",    HD_KW_BREAK},
    {"continue", HD_KW_CONTINUE},
    {"return",   HD_KW_RETURN},
    {"goto",     HD_KW_GOTO},

    /* HolyD type keywords */
    {"I0",       HD_KW_I0},
    {"I8",       HD_KW_I8},
    {"I16",      HD_KW_I16},
    {"I32",      HD_KW_I32},
    {"I64",      HD_KW_I64},
    {"U0",       HD_KW_U0},
    {"U8",       HD_KW_U8},
    {"U16",      HD_KW_U16},
    {"U32",      HD_KW_U32},
    {"U64",      HD_KW_U64},
    {"F64",      HD_KW_F64},
    {"Bool",     HD_KW_BOOL},

    /* C-compatible type keywords (aliases) */
    {"void",     HD_KW_U0},
    {"char",     HD_KW_I8},
    {"signed",   HD_KW_I32},
    {"short",    HD_KW_I16},
    {"int",      HD_KW_I32},
    {"long",     HD_KW_I64},
    {"unsigned", HD_KW_U32},
    {"short",    HD_KW_I16},
    {"char",     HD_KW_U8},
    {"float",    HD_KW_F64},
    {"double",   HD_KW_F64},
    {"bool",     HD_KW_BOOL},
    {"auto",     HD_KW_AUTO},

    /* Struct/class */
    {"class",    HD_KW_CLASS},
    {"struct",   HD_KW_STRUCT},
    {"union",    HD_KW_UNION},
    {"typedef",  HD_KW_TYPEDEF},
    {"enum",     HD_KW_ENUM},

    /* Storage class */
    {"static",   HD_KW_STATIC},
    {"extern",   HD_KW_EXTERN},
    {"public",   HD_KW_PUBLIC},
    {"const",    HD_KW_CONST},
    {"volatile", HD_KW_VOLATILE},
    {"inline",   HD_KW_INLINE},
    {"sizeof",   HD_KW_SIZEOF},
};

#define N_KEYWORDS (sizeof(hd_keywords) / sizeof(hd_keywords[0]))

/* -- Lexer Helpers ------------------------------------------------ */

static char hd_peek(const HDLexer *lex) {
    return lex->src[lex->pos];
}

static char hd_advance(HDLexer *lex) {
    char c = lex->src[lex->pos];
    lex->pos++;
    if (c == '\n') {
        lex->line++;
        lex->col = 0;
    } else {
        lex->col++;
    }
    return c;
}

static bool hd_is_at_end(const HDLexer *lex) {
    return lex->src[lex->pos] == '\0';
}

static void hd_lex_error(HDLexer *lex, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(lex->error, sizeof(lex->error), fmt, ap);
    va_end(ap);
    lex->has_error = true;
}

static HDTokenType hd_make_token(HDLexer *lex, HDTokenType type) {
    lex->tok.type = type;
    return type;
}

static HDTokenType hd_make_ident_or_keyword(HDLexer *lex, const char *text) {
    for (size_t i = 0; i < N_KEYWORDS; i++) {
        if (strcmp(text, hd_keywords[i].name) == 0) {
            return hd_make_token(lex, hd_keywords[i].type);
        }
    }
    strncpy(lex->tok.text, text, HD_MAX_TOKEN_LEN - 1);
    lex->tok.text[HD_MAX_TOKEN_LEN - 1] = '\0';
    return hd_make_token(lex, HD_TOK_IDENT);
}

/* -- Public API --------------------------------------------------- */

void hd_lex_init(HDLexer *lex, const char *source) {
    memset(lex, 0, sizeof(*lex));
    lex->src = source;
    lex->pos = 0;
    lex->line = 1;
    lex->col = 1;
    lex->has_error = false;
    lex->tok.type = HD_TOK_EOF;
    lex->tok.text[0] = '\0';
    hd_lex_next(lex);  /* Prime the first token */
}

static void hd_skip_whitespace(HDLexer *lex) {
    while (!hd_is_at_end(lex)) {
        char c = hd_peek(lex);
        if (c == ' ' || c == '\t' || c == '\r') {
            hd_advance(lex);
        } else if (c == '\n') {
            hd_advance(lex);
        } else {
            break;
        }
    }
}

static HDTokenType hd_scan_number(HDLexer *lex) {
    char buf[HD_MAX_TOKEN_LEN];
    int i = 0;
    bool is_hex = false;
    bool is_bin = false;
    bool is_float = false;

    if (hd_peek(lex) == '0') {
        hd_advance(lex);
        char c = hd_peek(lex);
        if (c == 'x' || c == 'X') {
            is_hex = true;
            hd_advance(lex);
        } else if (c == 'b' || c == 'B') {
            is_bin = true;
            hd_advance(lex);
        }
    }

    while (!hd_is_at_end(lex) && i < HD_MAX_TOKEN_LEN - 1) {
        char c = hd_peek(lex);
        if (is_hex && isxdigit((unsigned char)c)) {
            buf[i++] = hd_advance(lex);
        } else if (is_bin && (c == '0' || c == '1')) {
            buf[i++] = hd_advance(lex);
        } else if (!is_hex && !is_bin && isdigit((unsigned char)c)) {
            buf[i++] = hd_advance(lex);
        } else if (!is_hex && !is_bin && c == '.' && !is_float) {
            /* Decimal point → floating point literal */
            is_float = true;
            buf[i++] = hd_advance(lex);
            /* Accept trailing digits after the dot */
            while (!hd_is_at_end(lex) && i < HD_MAX_TOKEN_LEN - 1
                   && isdigit((unsigned char)hd_peek(lex))) {
                buf[i++] = hd_advance(lex);
            }
        } else {
            break;
        }
    }
    buf[i] = '\0';

    if (is_float) {
        lex->tok.float_val = strtod(buf, NULL);
        return hd_make_token(lex, HD_TOK_FLOAT);
    } else if (is_hex) {
        lex->tok.int_val = strtoll(buf, NULL, 16);
    } else if (is_bin) {
        lex->tok.int_val = strtoll(buf, NULL, 2);
    } else {
        lex->tok.int_val = strtoll(buf, NULL, 10);
    }
    /* Skip integer literal suffixes: L, LL, U, UL, ULL, LU, LLU, etc. */
    while (!hd_is_at_end(lex)) {
        char c = hd_peek(lex);
        if (c == 'L' || c == 'l' || c == 'U' || c == 'u') {
            hd_advance(lex);
        } else {
            break;
        }
    }
    return hd_make_token(lex, HD_TOK_INT);
}

static HDTokenType hd_scan_string(HDLexer *lex) {
    /* The opening quote was already consumed by hd_lex_next (it advanced past
     * the first char and dispatched to this scanner). lex->tok.text[0] holds it. */
    char quote = lex->tok.text[0];
    int i = 0;
    while (!hd_is_at_end(lex) && i < HD_MAX_STRING_LEN - 1) {
        char c = hd_peek(lex);
        if (c == quote) {
            hd_advance(lex);
            break;
        }
        if (c == '\\') {
            hd_advance(lex);
            char esc = hd_peek(lex);
            switch (esc) {
                case 'n': lex->tok.str_val[i++] = '\n'; break;
                case 't': lex->tok.str_val[i++] = '\t'; break;
                case 'r': lex->tok.str_val[i++] = '\r'; break;
                case '\\': lex->tok.str_val[i++] = '\\'; break;
                case '"': lex->tok.str_val[i++] = '"'; break;
                case '\'': lex->tok.str_val[i++] = '\''; break;
                case '0': lex->tok.str_val[i++] = '\0'; break;
                default: lex->tok.str_val[i++] = esc; break;
            }
            hd_advance(lex);
        } else {
            lex->tok.str_val[i++] = hd_advance(lex);
        }
    }
    lex->tok.str_val[i] = '\0';
    return hd_make_token(lex, quote == '"' ? HD_TOK_STRING : HD_TOK_CHAR);
}

static HDTokenType hd_scan_identifier(HDLexer *lex) {
    char buf[HD_MAX_TOKEN_LEN];
    int i = 0;
    while (!hd_is_at_end(lex) && i < HD_MAX_TOKEN_LEN - 1) {
        char c = hd_peek(lex);
        if (isalnum((unsigned char)c) || c == '_') {
            buf[i++] = hd_advance(lex);
        } else {
            break;
        }
    }
    buf[i] = '\0';
    return hd_make_ident_or_keyword(lex, buf);
}

HDTokenType hd_lex_next(HDLexer *lex) {
    if (lex->has_error) return HD_TOK_EOF;

    hd_skip_whitespace(lex);

    if (hd_is_at_end(lex)) {
        return hd_make_token(lex, HD_TOK_EOF);
    }

    lex->tok.line = lex->line;
    lex->tok.col = lex->col;
    lex->tok.text[0] = '\0';

    char c = hd_advance(lex);
    lex->tok.text[0] = c;   /* remember the leading char (used by string/char scanner) */

    switch (c) {
        /* Single-char tokens */
        case '(': return hd_make_token(lex, HD_TOK_LPAREN);
        case ')': return hd_make_token(lex, HD_TOK_RPAREN);
        case '{': return hd_make_token(lex, HD_TOK_LBRACE);
        case '}': return hd_make_token(lex, HD_TOK_RBRACE);
        case '[': return hd_make_token(lex, HD_TOK_LBRACKET);
        case ']': return hd_make_token(lex, HD_TOK_RBRACKET);
        case ',': return hd_make_token(lex, HD_TOK_COMMA);
        case ';': return hd_make_token(lex, HD_TOK_SEMI);
        case ':': return hd_make_token(lex, HD_TOK_COLON);
        case '?': return hd_make_token(lex, HD_TOK_QUESTION);
        case '.': return hd_make_token(lex, HD_TOK_DOT);
        case '~': return hd_make_token(lex, HD_TOK_TILDE);

        /* Multi-char operators */
        case '+':
            if (hd_peek(lex) == '+') { hd_advance(lex); return hd_make_token(lex, HD_TOK_PLUS_PLUS); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_PLUS_ASSIGN); }
            return hd_make_token(lex, HD_TOK_PLUS);
        case '-':
            if (hd_peek(lex) == '-') { hd_advance(lex); return hd_make_token(lex, HD_TOK_MINUS_MINUS); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_MINUS_ASSIGN); }
            if (hd_peek(lex) == '>') { hd_advance(lex); return hd_make_token(lex, HD_TOK_ARROW); }
            return hd_make_token(lex, HD_TOK_MINUS);
        case '*':
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_STAR_ASSIGN); }
            return hd_make_token(lex, HD_TOK_STAR);
        case '/':
            if (hd_peek(lex) == '/') {
                /* Skip line comment */
                while (!hd_is_at_end(lex) && hd_peek(lex) != '\n') hd_advance(lex);
                return hd_lex_next(lex);
            }
            if (hd_peek(lex) == '*') {
                /* Skip block comment */
                hd_advance(lex);
                int depth = 1;
                while (!hd_is_at_end(lex) && depth > 0) {
                    if (hd_advance(lex) == '*' && hd_peek(lex) == '/') {
                        hd_advance(lex);
                        depth--;
                    } else if (hd_peek(lex) == '*' && hd_peek(lex) == '/') {
                        /* nested */
                    }
                }
                return hd_lex_next(lex);
            }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_SLASH_ASSIGN); }
            return hd_make_token(lex, HD_TOK_SLASH);
        case '%':
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_PERCENT_ASSIGN); }
            return hd_make_token(lex, HD_TOK_PERCENT);
        case '&':
            if (hd_peek(lex) == '&') { hd_advance(lex); return hd_make_token(lex, HD_TOK_AND); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_AMP_ASSIGN); }
            return hd_make_token(lex, HD_TOK_AMP);
        case '|':
            if (hd_peek(lex) == '|') { hd_advance(lex); return hd_make_token(lex, HD_TOK_OR); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_PIPE_ASSIGN); }
            return hd_make_token(lex, HD_TOK_PIPE);
        case '^':
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_CARET_ASSIGN); }
            return hd_make_token(lex, HD_TOK_CARET);
        case '!':
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_NE); }
            return hd_make_token(lex, HD_TOK_BANG);
        case '=':
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_EQ); }
            return hd_make_token(lex, HD_TOK_ASSIGN);
        case '<':
            if (hd_peek(lex) == '<') { hd_advance(lex); if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_SHL_ASSIGN); } return hd_make_token(lex, HD_TOK_SHL); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_LE); }
            return hd_make_token(lex, HD_TOK_LT);
        case '>':
            if (hd_peek(lex) == '>') { hd_advance(lex); if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_SHR_ASSIGN); } return hd_make_token(lex, HD_TOK_SHR); }
            if (hd_peek(lex) == '=') { hd_advance(lex); return hd_make_token(lex, HD_TOK_GE); }
            return hd_make_token(lex, HD_TOK_GT);

        /* Literals */
        case '"':
        case '\'':
            return hd_scan_string(lex);

        default:
            if (isdigit((unsigned char)c)) {
                lex->pos--; /* step back */
                lex->col--;
                return hd_scan_number(lex);
            }
            if (isalpha((unsigned char)c) || c == '_') {
                lex->pos--; /* step back */
                lex->col--;
                return hd_scan_identifier(lex);
            }
            hd_lex_error(lex, "Unexpected character: '%c'", c);
            return HD_TOK_EOF;
    }
}

HDTokenType hd_lex_peek(HDLexer *lex) {
    return lex->tok.type;
}

int hd_lex_expect(HDLexer *lex, HDTokenType expected) {
    if (lex->tok.type != expected) {
        hd_lex_error(lex, "Expected token %d, got %d", expected, lex->tok.type);
        return -1;
    }
    return 0;
}