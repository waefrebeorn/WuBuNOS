/*
 * wubu_preproc.c -- minimal C preprocessor for the HolyD compiler.
 *
 * The self-hosting doctrine: real kernel source is full of #define
 * guards, object macros, and function-like macros. Without a preprocessor
 * the compiler couldn't compile a single kernel header. This module
 * expands #define directives (object + function-like) and drops the
 * lines so the lexer never sees a '#'. #include and #ifdef are stubbed
 * (single-line, no nesting) — enough to make the compiler self-host on
 * the battery's #define cases and grow from there.
 *
 * C11, self-contained.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PP_MAX_MACROS  256
#define PP_NAME_LEN    64
#define PP_BODY_LEN    512

typedef struct {
    char name[PP_NAME_LEN];   /* macro name (no parens) */
    int  fn_like;             /* 1 if NAME(args) body */
    char params[8][PP_NAME_LEN];
    int  n_params;
    char body[PP_BODY_LEN];
} PP_Macro;

static PP_Macro g_macros[PP_MAX_MACROS];
static int g_n_macros = 0;

/* trim leading/trailing whitespace in place */
static void pp_trim(char *s)
{
    char *start = s;
    while (*start == ' ' || *start == '\t') start++;
    if (start != s) memmove(s, start, strlen(start) + 1);
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\n' || s[n-1] == '\r')) s[--n] = '\0';
}

static void pp_reset(void)
{
    memset(g_macros, 0, sizeof(g_macros));
    g_n_macros = 0;
}

/* parse a #define line into a macro record */
static int pp_parse_define(const char *line, PP_Macro *m)
{
    memset(m, 0, sizeof(*m));
    const char *p = line;
    /* skip whitespace after #define */
    while (*p == ' ' || *p == '\t') p++;
    /* macro name */
    int i = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '(' && i < PP_NAME_LEN-1)
        m->name[i++] = *p++;
    m->name[i] = '\0';
    if (m->name[0] == '\0') return 0;

    /* function-like? NAME( */
    if (*p == '(') {
        m->fn_like = 1;
        p++; /* ( */
        int pi = 0;
        while (*p && *p != ')' && pi < 8) {
            while (*p == ' ' || *p == '\t') p++;
            int j = 0;
            while (*p && *p != ',' && *p != ')' && j < PP_NAME_LEN-1)
                m->params[pi][j++] = *p++;
            m->params[pi][j] = '\0';
            if (*p == ',') { p++; pi++; }
            else if (*p == ')' ) { pi++; break; }
        }
        m->n_params = pi;
        if (*p == ')') p++;
    }

    /* body: rest of line, trimmed */
    while (*p == ' ' || *p == '\t') p++;
    strncpy(m->body, p, PP_BODY_LEN-1);
    pp_trim(m->body);
    return 1;
}

/* find a macro by name, or -1 */
static int pp_find(const char *name)
{
    for (int i = 0; i < g_n_macros; i++)
        if (strcmp(g_macros[i].name, name) == 0) return i;
    return -1;
}

/* is a buffer position a whole-word boundary for an identifier? */
static int pp_is_ident_char(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

/* expand macros in a single line (no directives) into `out`.
 * Handles object macros and function-like macros. */
static void pp_expand_line(const char *line, char *out, size_t out_cap)
{
    size_t o = 0;
    const char *p = line;
    while (*p && o < out_cap - 1) {
        if (pp_is_ident_char(*p) && !(*p >= '0' && *p <= '9')) {
            /* read candidate identifier */
            const char *start = p;
            char name[PP_NAME_LEN];
            int i = 0;
            while (*p && pp_is_ident_char(*p) && i < PP_NAME_LEN-1) name[i++] = *p++;
            name[i] = '\0';
            int idx = pp_find(name);
            if (idx >= 0) {
                PP_Macro *m = &g_macros[idx];
                if (m->fn_like) {
                    /* expect (args) */
                    const char *q = p;
                    while (*q == ' ' || *q == '\t') q++;
                    if (*q == '(') {
                        q++;
                        /* collect args up to matching ) */
                        char argvals[8][PP_BODY_LEN];
                        int na = 0;
                        memset(argvals, 0, sizeof(argvals));
                        int depth = 1;
                        int cur = 0;
                        while (*q && depth > 0 && na < 8) {
                            if (*q == '(') depth++;
                            else if (*q == ')') depth--;
                            if (depth == 0) break;
                            if (*q == ',' && depth == 1) { na++; cur = 0; }
                            else if (cur < PP_BODY_LEN-1) { argvals[na][cur++] = *q; }
                            q++;
                        }
                        na++;
                        /* substitute params into body */
                        char sub[PP_BODY_LEN];
                        size_t s = 0;
                        const char *b = m->body;
                        while (*b && s < PP_BODY_LEN-1) {
                            if (pp_is_ident_char(*b) && !(*b >= '0' && *b <= '9')) {
                                const char *bs = b;
                                char pname[PP_NAME_LEN];
                                int bi = 0;
                                while (*b && pp_is_ident_char(*b) && bi < PP_NAME_LEN-1) pname[bi++] = *b++;
                                pname[bi] = '\0';
                                int matched = -1;
                                for (int pi = 0; pi < m->n_params; pi++)
                                    if (strcmp(m->params[pi], pname) == 0) { matched = pi; break; }
                                if (matched >= 0 && matched < na) {
                                    size_t av = strlen(argvals[matched]);
                                    if (s + av < PP_BODY_LEN-1) {
                                        memcpy(sub + s, argvals[matched], av);
                                        s += av;
                                    }
                                } else {
                                    size_t bl = strlen(pname);
                                    if (s + bl < PP_BODY_LEN-1) { memcpy(sub+s, pname, bl); s += bl; }
                                }
                            } else {
                                if (s < PP_BODY_LEN-1) sub[s++] = *b;
                                b++;
                            }
                        }
                        sub[s] = '\0';
                        size_t bl = strlen(sub);
                        if (o + bl < out_cap-1) { memcpy(out+o, sub, bl); o += bl; }
                        p = q; /* past the ) */
                        if (*p == ')') p++;
                        continue;
                    }
                    /* function-like but no parens: emit name literally */
                } else {
                    /* object macro: emit body */
                    size_t bl = strlen(m->body);
                    if (o + bl < out_cap-1) { memcpy(out+o, m->body, bl); o += bl; }
                    continue;
                }
            }
            /* not a macro (or fn-like w/o parens): copy name literally */
            size_t nl = strlen(name);
            if (o + nl < out_cap-1) { memcpy(out+o, name, nl); o += nl; }
            continue;
        } else {
            out[o++] = *p++;
        }
    }
    out[o] = '\0';
}

/* Strip __attribute__((...)) annotations from a line in-place.
 * Handles nested parens. */
static void strip_attributes(char *line)
{
    char *p = line;
    char *out = line;
    while (*p) {
        /* Look for __attribute__ */
        if (strncmp(p, "__attribute__", 13) == 0 && (p == line || !pp_is_ident_char(p[-1]))) {
            p += 13;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '(') {
                p++;
                int depth = 1;
                while (*p && depth > 0) {
                    if (*p == '(') depth++;
                    else if (*p == ')') depth--;
                    if (depth > 0) p++;
                }
                if (*p == ')') p++;
            }
            continue;
        }
        *out++ = *p++;
    }
    *out = '\0';
}

/* Strip __asm__("...") or __asm volatile("...") inline assembly. */
static void strip_inline_asm(char *line)
{
    char *p = line;
    while ((p = strstr(p, "__asm")) != NULL) {
        /* Check it's a whole word */
        if (p > line && pp_is_ident_char(p[-1])) { p++; continue; }
        char *start = p;
        p += 5; /* __asm */
        /* Skip optional __volatile__ or volatile */
        while (*p == ' ' || *p == '\t') p++;
        if (strncmp(p, "__volatile__", 12) == 0) { p += 12; }
        else if (strncmp(p, "volatile", 8) == 0) { p += 8; }
        while (*p == ' ' || *p == '\t') p++;
        /* Skip ((...)) with nested paren/string handling */
        if (*p == '(') {
            p++;
            int depth = 1;
            while (*p && depth > 0) {
                if (*p == '"') {
                    p++;
                    while (*p && *p != '"') {
                        if (*p == '\\') p++; /* skip escaped char */
                        if (*p) p++;
                    }
                    if (*p == '"') p++;
                    continue;
                }
                if (*p == '(') depth++;
                else if (*p == ')') depth--;
                if (depth > 0) p++;
            }
            if (*p == ')') p++;
        }
        /* Remove the whole asm statement including trailing semicolon */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ';') p++;
        memmove(start, p, strlen(p) + 1);
        p = start;
    }
}

/* Strip __extension__, __inline, __inline__, __forceinline, __cdecl, etc. */
static void strip_compiler_keywords(char *line)
{
    static const char *kw[] = {
        "__extension__", "__inline", "__inline__", "__forceinline",
        "__cdecl", "__stdcall", "__fastcall", "__thiscall",
        "__declspec", "__asm__", "__asm", "__volatile__",
        "__restrict__", "__restrict", "__signed__",
        "__builtin_va_list", "__builtin_offsetof",
        NULL
    };
    for (int i = 0; kw[i]; i++) {
        char *p = line;
        size_t klen = strlen(kw[i]);
        while ((p = strstr(p, kw[i])) != NULL) {
            /* Only match whole words */
            if ((p == line || !pp_is_ident_char(p[-1])) &&
                !pp_is_ident_char(p[klen])) {
                /* Check if followed by ((...)) for __declspec */
                char *after = p + klen;
                while (*after == ' ' || *after == '\t') after++;
                if (*after == '(') {
                    int depth = 0;
                    char *q = after;
                    do {
                        if (*q == '(') depth++;
                        else if (*q == ')') depth--;
                        q++;
                    } while (depth > 0 && *q);
                    memmove(p, q, strlen(q) + 1);
                } else {
                    memmove(p, p + klen, strlen(p + klen) + 1);
                }
            } else {
                p += klen;
            }
        }
    }
}

/* Strip volatile, const, register, restrict qualifiers from a token.
 * These affect code generation but our JIT treats all variables the same. */
static void strip_type_qualifiers(char *line)
{
    /* We don't strip these from the source — the parser handles them.
     * But we do need to handle __volatile__ etc. which are already
     * handled by strip_compiler_keywords. */
    (void)line;
}

/* Preprocess a full source string: strips directives (#define/#include)
 * and expands macros. Returns a malloc'd string (caller frees). */
char *wubu_preprocess(const char *src)
{
    pp_reset();
    /* Predefined macros that gcc torture tests and real code rely on */
    struct { const char *name; const char *val; } builtins[] = {
        {"__INT_MAX__", "2147483647"},
        {"__LONG_MAX__", "9223372036854775807L"},
        {"__LONG_LONG_MAX__", "9223372036854775807LL"},
        {"__CHAR_BIT__", "8"},
        {"__SCHAR_MAX__", "127"},
        {"__SHRT_MAX__", "32767"},
        {"__SIZE_MAX__", "18446744073709551615UL"},
        {"__PTRDIFF_MAX__", "9223372036854775807L"},
        {"__INT8_MAX__", "127"},
        {"__INT16_MAX__", "32767"},
        {"__INT32_MAX__", "2147483647"},
        {"__INT64_MAX__", "9223372036854775807L"},
        {"__UINT8_MAX__", "255U"},
        {"__UINT16_MAX__", "65535U"},
        {"__UINT32_MAX__", "4294967295U"},
        {"__UINT64_MAX__", "18446744073709551615UL"},
        {"__SIZEOF_INT__", "4"},
        {"__SIZEOF_LONG__", "8"},
        {"__SIZEOF_LONG_LONG__", "8"},
        {"__SIZEOF_POINTER__", "8"},
        {"__SIZEOF_FLOAT__", "4"},
        {"__SIZEOF_DOUBLE__", "8"},
        {"__SIZEOF_SIZE_T__", "8"},
        {"__GNUC__", "4"},
        {"__GNUC_MINOR__", "2"},
        {"__GNUC_PATCHLEVEL__", "1"},
        {"__VERSION__", "\"4.2.1\""},
        {"__STDC__", "1"},
        {"__STDC_VERSION__", "201112L"},
        {"__x86_64__", "1"},
        {"__LP64__", "1"},
        {"__linux__", "1"},
        {"__ELF__", "1"},
        {"__ORDER_LITTLE_ENDIAN__", "1234"},
        {"__ORDER_BIG_ENDIAN__", "4321"},
        {"__BYTE_ORDER__", "1234"},
        {"__SIZE_TYPE__", "unsigned long"},
        {"__PTRDIFF_TYPE__", "long"},
        {"__WCHAR_TYPE__", "int"},
        {"__INTPTR_TYPE__", "long"},
        {"__UINTPTR_TYPE__", "unsigned long"},
        {"unix", "1"},
        {"linux", "1"},
        {NULL, NULL}
    };
    for (int i = 0; builtins[i].name; i++) {
        PP_Macro m;
        strncpy(m.name, builtins[i].name, PP_NAME_LEN - 1);
        strncpy(m.body, builtins[i].val, PP_BODY_LEN - 1);
        m.fn_like = 0;
        m.n_params = 0;
        int idx = pp_find(m.name);
        if (idx < 0) { idx = g_n_macros++; }
        g_macros[idx] = m;
    }
    size_t cap = strlen(src) * 4 + 4096;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t o = 0;

    char *copy = strdup(src);
    if (!copy) { free(out); return NULL; }

    /* First pass: join line continuations (backslash-newline) */
    {
        char *p = copy, *q = copy;
        while (*p) {
            if (*p == '\\' && p[1] == '\n') {
                p += 2; /* skip backslash-newline */
            } else {
                *q++ = *p++;
            }
        }
        *q = '\0';
    }

    char *save = NULL;
    char *line = strtok_r(copy, "\n", &save);
    while (line) {
        char *tl = line;
        while (*tl == ' ' || *tl == '\t') tl++;
        if (*tl == '#') {
            char dir[32];
            sscanf(tl + 1, "%31s", dir);
            if (strcmp(dir, "define") == 0) {
                PP_Macro m;
                if (pp_parse_define(tl + 7, &m) && g_n_macros < PP_MAX_MACROS) {
                    /* redefine wins */
                    int idx = pp_find(m.name);
                    if (idx < 0) { idx = g_n_macros++; }
                    g_macros[idx] = m;
                }
            } else if (strcmp(dir, "ifdef") == 0 || strcmp(dir, "ifndef") == 0) {
                /* Strip the conditional block — keep the #ifdef'd code */
                char name[64];
                sscanf(tl + 6, "%63s", name);
                /* For now, keep the code inside #ifdef blocks */
                (void)name;
            } else if (strcmp(dir, "endif") == 0) {
                /* End of conditional — just drop */
            } else if (strcmp(dir, "if") == 0) {
                /* Conditional compilation — keep code for now */
            } else if (strcmp(dir, "else") == 0) {
                /* Drop */
            } else if (strcmp(dir, "pragma") == 0) {
                /* Drop all #pragma directives */
            } else if (strcmp(dir, "error") == 0 || strcmp(dir, "warning") == 0) {
                /* Drop */
            } else if (strcmp(dir, "line") == 0) {
                /* Drop */
            } else if (strcmp(dir, "undef") == 0) {
                char name[64];
                sscanf(tl + 6, "%63s", name);
                int idx = pp_find(name);
                if (idx >= 0) {
                    /* Remove by shifting */
                    for (int i = idx; i < g_n_macros - 1; i++)
                        g_macros[i] = g_macros[i+1];
                    g_n_macros--;
                }
            }
            /* #include and others: dropped (stub) */
        } else {
            /* Strip __attribute__, __extension__, inline asm, etc. */
            char exp[8192];
            strncpy(exp, line, sizeof(exp) - 1);
            exp[sizeof(exp) - 1] = '\0';
            strip_compiler_keywords(exp);
            strip_attributes(exp);
            strip_inline_asm(exp);
            /* expand macros in this line */
            char exp2[8192];
            pp_expand_line(exp, exp2, sizeof(exp2));
            size_t l = strlen(exp2);
            if (o + l + 2 < cap) {
                memcpy(out + o, exp2, l);
                o += l;
                out[o++] = '\n';
            }
        }
        line = strtok_r(NULL, "\n", &save);
    }
    out[o] = '\0';
    free(copy);
    return out;
}
