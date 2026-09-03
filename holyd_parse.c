/*
 * holyd_parse.c  --  WuBuNOS HolyD Parser + AST Utilities
 *
 * Recursive descent parser: tokens → AST.
 * Ported from ZealOS/src/Compiler/ParseExp.ZC + ParseStatement.ZC
 *
 * Grammar (simplified):
 *   program     → decl*
 *   decl        → func_decl | var_decl | struct_decl
 *   func_decl   → type ident '(' params ')' block
 *   var_decl    → type ident ['=' expr] ';'
 *   block       → '{' stmt* '}'
 *   stmt        → if_stmt | while_stmt | for_stmt | return_stmt
 *               | expr_stmt | block | var_decl
 *   expr        → assign
 *   assign      → ternary ['=' assign]
 *   ternary     → logic_or ['?' expr ':' ternary]
 *   logic_or    → logic_and ('||' logic_and)*
 *   logic_and   → equality ('&&' equality)*
 *   equality    → comparison (('=='|'!=') comparison)*
 *   comparison  → addition (('<'|'>'|'<='|'>=') addition)*
 *   addition    → multiplication (('+'|'-') multiplication)*
 *   multiplication → unary (('*'|'/'|'%') unary)*
 *   unary       → ('-'|'!'|'~'|'*'|'&') unary | postfix
 *   postfix     → primary ('++'|'--'|'[' expr ']'|'.' ident|'(' args ')')*
 *   primary     → INT_LIT | FLOAT_LIT | STRING_LIT | IDENT | '(' expr ')'
 */

#include "holyd.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -- AST Utilities ------------------------------------------------ */


/* -- AST Print (debug) -------------------------------------------- */


/* -- Type Size ---------------------------------------------------- */


/* -- Parser State ------------------------------------------------- */

static void parse_error(HDParser *p, const char *msg) {
    if (p->n_errors < HD_MAX_ERRORS) {
        snprintf(p->errors[p->n_errors], 256, "line %d: %s",
                 p->lex->line, msg);
        p->n_errors++;
    }
    p->has_error = true;
}

static HDTokenType peek(HDParser *p) {
    return p->lex->tok.type;
}

static HDTokenType advance(HDParser *p) {
    HDTokenType t = p->lex->tok.type;
    hd_lex_next(p->lex);
    return t;
}

static bool match(HDParser *p, HDTokenType type) {
    if (peek(p) == type) { advance(p); return true; }
    return false;
}

static void expect(HDParser *p, HDTokenType type) {
    if (peek(p) == type) { advance(p); return; }
    char msg[128];
    snprintf(msg, sizeof(msg), "expected token %d, got %d", type, peek(p));
    parse_error(p, msg);
}

/* -- Forward Declarations ----------------------------------------- */

static HDASTNode *parse_expr(HDParser *p);
static HDASTNode *parse_comma(HDParser *p);
static HDASTNode *parse_assign(HDParser *p);
static HDASTNode *parse_stmt(HDParser *p);
static HDASTNode *parse_decl(HDParser *p);

/* -- Parse Type --------------------------------------------------- */

static HDType *parse_type(HDParser *p) {
    HDType *t = (HDType *)calloc(1, sizeof(HDType));
    t->kind = HD_TYPE_I64; /* HolyD default */

    switch (peek(p)) {
        case HD_TOK_IDENT: {
            /* A typedef'd name is a type: `typedef int MyInt; MyInt x;`.
             * Look it up in the parser's typedef registry. */
            const char *id = p->lex->tok.text;
            int found_typedef = -1;
            for (int i = 0; i < p->n_typedefs; i++) {
                if (strcmp(p->typedef_names[i], id) == 0) { found_typedef = i; break; }
            }
            if (found_typedef >= 0) {
                HDType *rt = p->typedef_types[found_typedef];
                advance(p);
                /* do NOT return rt early — fall through so the pointer-star
                 * loop below wraps it (`typedef struct{int x;}S; S* p` must
                 * produce a PTR type, not skip the `*`). */
                free(t);              /* discard the default I64 placeholder */
                t = rt;
                break;
            }
            break;
        }
        case HD_KW_U0:   t->kind = HD_TYPE_VOID; advance(p); break;
        case HD_KW_I8:   t->kind = HD_TYPE_I8;   advance(p); break;
        case HD_KW_I16:  t->kind = HD_TYPE_I16;  advance(p); break;
        case HD_KW_I32:  t->kind = HD_TYPE_I32;  advance(p); break;
        case HD_KW_I64:  t->kind = HD_TYPE_I64;  advance(p);
                         /* `long long` = two I64 tokens (both 64-bit on
                          * x86-64). Consume the optional second `long` so
                          * `long long x;` / `sizeof(long long)` parse. */
                         if (peek(p) == HD_KW_I64) advance(p);
                         /* `long int` == `long` — consume the redundant `int` */
                         else if (peek(p) == HD_KW_I32) advance(p);
                         /* `long double` == `double` (F64) */
                         else if (peek(p) == HD_KW_F64) { t->kind = HD_TYPE_F64; advance(p); }
                         break;
        case HD_KW_U8:   t->kind = HD_TYPE_U8;   advance(p); break;
        case HD_KW_U16:  t->kind = HD_TYPE_U16;  advance(p); break;
        case HD_KW_U32:  t->kind = HD_TYPE_U32;  advance(p);
                         /* `unsigned long` == `unsigned long int` == `unsigned long` */
                         if (peek(p) == HD_KW_I64) { advance(p); }
                         else if (peek(p) == HD_KW_I32) { advance(p); }
                         break;
        case HD_KW_U64:  t->kind = HD_TYPE_U64;  advance(p);
                         /* `unsigned long` == `unsigned long int` */
                         if (peek(p) == HD_KW_I64 || peek(p) == HD_KW_I32) advance(p);
                         break;
        case HD_KW_F64:  t->kind = HD_TYPE_F64;  advance(p); break;
        case HD_KW_BOOL:  t->kind = HD_TYPE_BOOL; advance(p); break;
        case HD_KW_ENUM: {
            /* enum [Name] { A, B=2, C } — parse enumerators, register each
             * as a constant (GREEN=1) so later idents resolve. */
            advance(p); /* enum */
            char tag[HD_MAX_IDENT_LEN] = {0};
            if (peek(p) == HD_TOK_IDENT) {
                strncpy(tag, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                strncpy(t->name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                advance(p);
            }
            if (peek(p) == HD_TOK_LBRACE) {
                advance(p); /* { */
                int64_t val = 0;
                while (peek(p) != HD_TOK_RBRACE && peek(p) != HD_TOK_EOF) {
                    if (peek(p) == HD_TOK_IDENT) {
                        char cname[HD_MAX_IDENT_LEN];
                        strncpy(cname, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                        advance(p);
                        if (match(p, HD_TOK_ASSIGN)) {
                            if (peek(p) == HD_TOK_INT) { val = p->lex->tok.int_val; advance(p); }
                        }
                        /* register enumerator constant */
                        if (p->n_enum_consts < 64) {
                            strncpy(p->enum_const_names[p->n_enum_consts], cname, HD_MAX_IDENT_LEN - 1);
                            p->enum_const_vals[p->n_enum_consts] = val;
                            p->n_enum_consts++;
                        }
                        val++;
                    }
                    if (peek(p) == HD_TOK_COMMA) advance(p);
                    else break;
                }
                expect(p, HD_TOK_RBRACE);
                t->kind = HD_TYPE_ENUM;
                t->size = 4;
                /* register the tag */
                if (tag[0] != '\0') {
                    for (int i = 0; i < p->n_named_types; i++)
                        if (strcmp(p->named_type_names[i], tag) == 0) { p->named_types[i] = t; goto struct_done; }
                    if (p->n_named_types < 64) {
                        strncpy(p->named_type_names[p->n_named_types], tag, HD_MAX_IDENT_LEN - 1);
                        p->named_types[p->n_named_types] = t;
                        p->n_named_types++;
                    }
                }
                goto struct_done;
            }
            t->kind = HD_TYPE_ENUM;
            t->size = 4;
            /* enum Color; reference */
            if (tag[0] != '\0') {
                for (int i = 0; i < p->n_named_types; i++)
                    if (strcmp(p->named_type_names[i], tag) == 0) { *t = *(p->named_types[i]); goto struct_done; }
            }
            break;
        }
        case HD_KW_STRUCT:
        case HD_KW_UNION: {
            /* Struct/union type: (struct|union) Name { ... } or Name.
             * Same member grammar; a union overlaps all members at offset 0. */
            HDTypeKind comp_kind = (peek(p) == HD_KW_STRUCT) ? HD_TYPE_STRUCT : HD_TYPE_UNION;
            advance(p); /* struct | union */
            char tag[HD_MAX_IDENT_LEN] = {0};
            if (peek(p) == HD_TOK_IDENT) {
                strncpy(tag, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                strncpy(t->name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                advance(p);
            }
            if (peek(p) == HD_TOK_LBRACE) {
                /* Definition with members */
                advance(p); /* { */
                int64_t max_size = 0;
                int max_align = 1;
                while (peek(p) != HD_TOK_RBRACE && peek(p) != HD_TOK_EOF) {
                    HDType *member_type = parse_type(p);
                    /* Function-pointer member: `int (*fn)(int,int);` — the
                     * member name is inside `(*...)`, so after parse_type
                     * (which read `int`) the next token is `(` not IDENT.
                     * Detect `(*` and build a pointer-to-function type. */
                    bool mem_is_fnp = false;
                    if (peek(p) == HD_TOK_LPAREN) {
                        int saved_pos = p->lex->pos;
                        advance(p); /* ( */
                        if (peek(p) == HD_TOK_STAR) {
                            advance(p); /* * */
                            mem_is_fnp = true;
                        } else {
                            /* `(name)` — not a fn ptr, restore */
                            p->lex->pos = saved_pos;
                            hd_lex_next(p->lex);
                        }
                    }
                    if (peek(p) != HD_TOK_IDENT) {
                        parse_error(p, "expected member name");
                        break;
                    }
                    if (t->n_members < HD_MAX_PARAMS) {
                        strncpy(t->members[t->n_members].name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                        advance(p); /* consume member name */
                        if (mem_is_fnp) {
                            /* `int (*fn)(int,int)` — consume `)` then `(params)`.
                             * Build PTR(FUNC(...)). */
                            expect(p, HD_TOK_RPAREN);
                            if (peek(p) == HD_TOK_LPAREN) {
                                advance(p);
                                HDType *fn = (HDType *)calloc(1, sizeof(HDType));
                                fn->kind = HD_TYPE_FUNC;
                                fn->param_types = (HDType **)calloc(HD_MAX_PARAMS, sizeof(HDType *));
                                int pi = 0;
                                if (peek(p) != HD_TOK_RPAREN) {
                                    fn->param_types[pi] = parse_type(p);
                                    if (peek(p) == HD_TOK_IDENT) advance(p);
                                    pi++;
                                    while (match(p, HD_TOK_COMMA) && pi < HD_MAX_PARAMS) {
                                        fn->param_types[pi] = parse_type(p);
                                        if (peek(p) == HD_TOK_IDENT) advance(p);
                                        pi++;
                                    }
                                }
                                fn->n_params = pi;
                                expect(p, HD_TOK_RPAREN);
                                HDType *fpt = (HDType *)calloc(1, sizeof(HDType));
                                fpt->kind = HD_TYPE_PTR;
                                fpt->base = fn;
                                fpt->size = 8;
                                member_type = fpt;
                            } else {
                                /* `int (*fn)` — plain pointer-to-int */
                                HDType *pt = (HDType *)calloc(1, sizeof(HDType));
                                pt->kind = HD_TYPE_PTR;
                                pt->base = member_type;
                                pt->size = 8;
                                member_type = pt;
                            }
                        }
                        /* member may itself be an array: name[N] */
                        if (peek(p) == HD_TOK_LBRACKET) {
                            advance(p);
                            int asz = 0;
                            if (peek(p) == HD_TOK_INT) { asz = (int)p->lex->tok.int_val; advance(p); }
                            expect(p, HD_TOK_RBRACKET);
                            HDType *ma = (HDType *)calloc(1, sizeof(HDType));
                            ma->kind = HD_TYPE_ARRAY;
                            ma->base = member_type;
                            ma->array_size = asz;
                            member_type = ma;
                        }
                        size_t msz = hd_type_size(member_type);
                        if (comp_kind == HD_TYPE_UNION) {
                            /* union: every member starts at offset 0; size = max */
                            t->members[t->n_members].offset = 0;
                            if ((int64_t)msz > max_size) max_size = (int64_t)msz;
                            if ((int64_t)msz > max_align) max_align = (int)msz;
                        } else {
                            t->members[t->n_members].offset = t->size;
                            t->size += (int)((msz + 7) / 8);  /* member size in int64 cells */
                            t->align = 1;
                        }
                        t->members[t->n_members].type = member_type;
                        t->n_members++;
                    }
                    /* Additional declarators: `int a, b, c;` — same type */
                    while (match(p, HD_TOK_COMMA)) {
                        if (peek(p) != HD_TOK_IDENT) {
                            parse_error(p, "expected member name after comma");
                            break;
                        }
                        if (t->n_members < HD_MAX_PARAMS) {
                            strncpy(t->members[t->n_members].name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                            advance(p);
                            if (comp_kind == HD_TYPE_UNION) {
                                t->members[t->n_members].offset = 0;
                            } else {
                                t->members[t->n_members].offset = t->size;
                                size_t _msz = hd_type_size(member_type);
                                t->size += (int)((_msz + 7) / 8);  /* member size in int64 cells */
                                t->align = 1;
                            }
                            t->members[t->n_members].type = member_type;
                            t->n_members++;
                        }
                    }
                    expect(p, HD_TOK_SEMI);
                }
                expect(p, HD_TOK_RBRACE);
                t->kind = comp_kind;
                if (comp_kind == HD_TYPE_UNION) {
                    t->size = 1;  /* union = 1 int64 cell (all members overlap) */
                    t->align = 1;
                } else {
                    /* Trailing padding: the struct's size must be a multiple
                     * of its alignment (gcc rounds `{int (*fn)(int,int); int
                     * n;}` up to 16 because the pointer member aligns to 8).
                     * Without this, sizeof underestimated structs whose last
                     * member is narrower than an earlier wide member. */
                    if (t->align > 0 && (t->size % t->align) != 0)
                        t->size += t->align - (t->size % t->align);
                }
                /* register the tag so later `struct S x;` reuses this layout */
                if (tag[0] != '\0') {
                    for (int i = 0; i < p->n_named_types; i++) {
                        if (strcmp(p->named_type_names[i], tag) == 0) {
                            p->named_types[i] = t;  /* redefinition wins */
                            goto struct_done;
                        }
                    }
                    if (p->n_named_types < 64) {
                        strncpy(p->named_type_names[p->n_named_types], tag, HD_MAX_IDENT_LEN - 1);
                        p->named_types[p->n_named_types] = t;
                        p->n_named_types++;
                    }
                }
                goto struct_done;
            }
            t->kind = comp_kind;
            /* Forward/reference: `struct S x;` where S was defined earlier.
             * Reuse the registered layout instead of a fresh empty struct. */
            if (tag[0] != '\0') {
                for (int i = 0; i < p->n_named_types; i++) {
                    if (strcmp(p->named_type_names[i], tag) == 0) {
                        HDType *reg = p->named_types[i];
                        /* shallow-copy the layout into the fresh node */
                        *t = *reg;
                        t->name[0] = '\0';
                        strncpy(t->name, tag, HD_MAX_IDENT_LEN - 1);
                        goto struct_done;
                    }
                }
            }
            break;
        }
        default: break; /* Keep default I64 */
    }

struct_done:
    /* Pointer types: type * */
    while (peek(p) == HD_TOK_STAR) {
        advance(p);
        HDType *ptr = (HDType *)calloc(1, sizeof(HDType));
        ptr->kind = HD_TYPE_PTR;
        ptr->base = t;
        t = ptr;
    }

    return t;
}

/* -- Parse Primary ------------------------------------------------ */

/* sizeof expr parses a cast; declared here so parse_primary can use it
 * before parse_cast's later forward-declaration. */
static HDASTNode *parse_cast(HDParser *p);

static HDASTNode *parse_primary(HDParser *p) {
    /* An enum constant ident is an int literal: `GREEN` → 1. */
    if (peek(p) == HD_TOK_IDENT) {
        for (int i = 0; i < p->n_enum_consts; i++) {
            if (strcmp(p->enum_const_names[i], p->lex->tok.text) == 0) {
                HDASTNode *n = hd_ast_new(HD_AST_INT_LIT);
                n->int_val = p->enum_const_vals[i];
                advance(p);
                return n;
            }
        }
    }
    switch (peek(p)) {
        case HD_TOK_INT: {
            HDASTNode *n = hd_ast_new(HD_AST_INT_LIT);
            n->int_val = p->lex->tok.int_val;
            advance(p);
            return n;
        }
        case HD_TOK_FLOAT: {
            HDASTNode *n = hd_ast_new(HD_AST_FLOAT_LIT);
            n->float_val = p->lex->tok.float_val;
            HDType *ft = (HDType *)calloc(1, sizeof(HDType));
            ft->kind = HD_TYPE_F64; ft->size = 8;
            n->type = ft;
            advance(p);
            return n;
        }
        case HD_TOK_STRING: {
            HDASTNode *n = hd_ast_new(HD_AST_STRING_LIT);
            /* use str_val (the decoded string incl. escape sequences),
             * NOT text (which is the raw source span and can be empty
             * after a previous token's trailing-quote advance). */
            strncpy(n->str_val, p->lex->tok.str_val, HD_MAX_STRING_LEN - 1);
            advance(p);
            return n;
        }
        case HD_TOK_CHAR: {
            HDASTNode *n = hd_ast_new(HD_AST_CHAR_LIT);
            /* the lexer puts the decoded char in str_val[0]; int_val
             * is only set for HD_TOK_INT (the scanner path doesn't
             * touch int_val for chars). Codegen reads str_val[0], so
             * populate both for safety. */
            n->str_val[0] = p->lex->tok.str_val[0];
            n->int_val = (int64_t)(uint8_t)p->lex->tok.str_val[0];
            advance(p);
            return n;
        }
        case HD_TOK_IDENT: {
            HDASTNode *n = hd_ast_new(HD_AST_IDENT);
            strncpy(n->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
            advance(p);
            return n;
        }
        case HD_KW_SIZEOF: {
            /* sizeof(type) or sizeof expr — emit the type size as a literal.
             * After '(': if it's a type keyword, parse the type; otherwise
             * parse the inner expression and expect ')'. No backtracking —
             * capturing lex->pos mid-token points past the token, so a
             * pos-restore lands on the wrong char. */
            advance(p); /* sizeof */
            HDASTNode *n = hd_ast_new(HD_AST_SIZEOF);
            if (peek(p) == HD_TOK_LPAREN) {
                advance(p); /* ( */
                int t = peek(p);
                int is_type_kw = (t >= HD_KW_I0 && t <= HD_KW_VOLATILE);
                if (is_type_kw) {
                    n->type = parse_type(p);
                    expect(p, HD_TOK_RPAREN);
                    return n;
                }
                n->child = parse_expr(p);     /* sizeof (expr) */
                expect(p, HD_TOK_RPAREN);
                return n;
            }
            n->child = parse_cast(p);   /* sizeof expr (no parens) */
            return n;
        }
        case HD_TOK_LPAREN: {
            advance(p); /* ( */
            HDASTNode *expr = parse_expr(p);
            expect(p, HD_TOK_RPAREN);
            return expr;
        }
        case HD_TOK_LBRACE: {
            /* Braced initializer: {expr, expr, ...} or designated:
             *   { .field = val, [index] = val, ... }
             * Returns a BRACE_INIT node containing the element expressions. */
            advance(p); /* consume { */
            HDASTNode *init = hd_ast_new(HD_AST_BRACE_INIT);
            if (!init) { p->has_error = true; return NULL; }
            while (peek(p) != HD_TOK_RBRACE && peek(p) != HD_TOK_EOF) {
                /* Check for designated initializer: .field = val or [index] = val */
                if (peek(p) == HD_TOK_DOT) {
                    /* .field = val */
                    advance(p); /* consume . */
                    if (peek(p) != HD_TOK_IDENT) {
                        parse_error(p, "expected field name after .");
                        break;
                    }
                    HDASTNode *desig = hd_ast_new(HD_AST_DESIG_INIT);
                    strncpy(desig->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                    advance(p); /* consume field name */
                    expect(p, HD_TOK_ASSIGN);
                    desig->child = parse_assign(p);
                    hd_ast_add_arg(init, desig);
                } else if (peek(p) == HD_TOK_LBRACKET) {
                    /* [index] = val */
                    advance(p); /* consume [ */
                    int64_t idx = 0;
                    if (peek(p) == HD_TOK_INT) { idx = p->lex->tok.int_val; advance(p); }
                    else { parse_error(p, "expected constant index in []"); break; }
                    expect(p, HD_TOK_RBRACKET);
                    expect(p, HD_TOK_ASSIGN);
                    HDASTNode *desig = hd_ast_new(HD_AST_DESIG_INIT);
                    desig->ident[0] = '@'; /* marker: array index designator */
                    desig->int_val = idx;
                    desig->child = parse_assign(p);
                    hd_ast_add_arg(init, desig);
                } else {
                    hd_ast_add_arg(init, parse_assign(p));
                }
                if (peek(p) == HD_TOK_COMMA) {
                    advance(p);
                    if (peek(p) == HD_TOK_RBRACE) break;
                } else {
                    break;
                }
            }
            expect(p, HD_TOK_RBRACE);
            return init;
        }
        default:
            parse_error(p, "expected expression");
            return NULL;
    }
}

static HDASTNode *parse_postfix(HDParser *p) {
    HDASTNode *expr = parse_primary(p);

    while (true) {
        if (peek(p) == HD_TOK_LPAREN) {
            /* Function call */
            advance(p); /* ( */
            HDASTNode *call = hd_ast_new(HD_AST_FUNC_CALL);
            call->callee = expr;
            if (peek(p) != HD_TOK_RPAREN) {
                hd_ast_add_arg(call, parse_assign(p));
                while (match(p, HD_TOK_COMMA))
                    hd_ast_add_arg(call, parse_assign(p));
            }
            expect(p, HD_TOK_RPAREN);
            expr = call;
        } else if (peek(p) == HD_TOK_LBRACKET) {
            /* Array index */
            advance(p); /* [ */
            HDASTNode *idx = hd_ast_new(HD_AST_INDEX);
            idx->left = expr;
            idx->right = parse_expr(p);
            expect(p, HD_TOK_RBRACKET);
            expr = idx;
        } else if (peek(p) == HD_TOK_DOT) {
            /* Member access */
            advance(p); /* . */
            HDASTNode *mem = hd_ast_new(HD_AST_MEMBER);
            mem->left = expr;
            if (peek(p) == HD_TOK_IDENT) {
                strncpy(mem->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                advance(p);
            }
            expr = mem;
        } else if (peek(p) == HD_TOK_ARROW) {
            /* Arrow access */
            advance(p); /* -> */
            HDASTNode *mem = hd_ast_new(HD_AST_ARROW);
            mem->left = expr;
            if (peek(p) == HD_TOK_IDENT) {
                strncpy(mem->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                advance(p);
            }
            expr = mem;
        } else if (peek(p) == HD_TOK_PLUS_PLUS) {
            advance(p);
            HDASTNode *inc = hd_ast_new(HD_AST_POST_INC);
            inc->child = expr;
            expr = inc;
        } else if (peek(p) == HD_TOK_MINUS_MINUS) {
            advance(p);
            HDASTNode *dec = hd_ast_new(HD_AST_POST_DEC);
            dec->child = expr;
            expr = dec;
        } else {
            break;
        }
    }

    return expr;
}

/* -- Parse Unary -------------------------------------------------- */

static HDASTNode *parse_unary(HDParser *p) {
    if (peek(p) == HD_TOK_MINUS) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_NEG);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_BANG) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_NOT);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_TILDE) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_BITNOT);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_STAR) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_DEREF);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_AMP) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_ADDR);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_PLUS_PLUS) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_PRE_INC);
        n->child = parse_unary(p);
        return n;
    }
    if (peek(p) == HD_TOK_MINUS_MINUS) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_PRE_DEC);
        n->child = parse_unary(p);
        return n;
    }
    return parse_postfix(p);
}

/* -- Parse Binary (precedence climbing) --------------------------- */

typedef struct { HDTokenType tok; HDASTKind ast; } BinOp;

static const BinOp mul_ops[] = {
    {HD_TOK_STAR,  HD_AST_MUL}, {HD_TOK_SLASH, HD_AST_DIV}, {HD_TOK_PERCENT, HD_AST_MOD},
    {HD_TOK_EOF,   0},
};
static const BinOp add_ops[] = {
    {HD_TOK_PLUS,  HD_AST_ADD}, {HD_TOK_MINUS, HD_AST_SUB},
    {HD_TOK_EOF,   0},
};
static const BinOp shift_ops[] = {
    {HD_TOK_SHL,   HD_AST_SHL}, {HD_TOK_SHR, HD_AST_SHR},
    {HD_TOK_EOF,   0},
};
static const BinOp cmp_ops[] = {
    {HD_TOK_LT,    HD_AST_LT}, {HD_TOK_GT, HD_AST_GT},
    {HD_TOK_LE,    HD_AST_LE}, {HD_TOK_GE, HD_AST_GE},
    {HD_TOK_EOF,   0},
};
static const BinOp eq_ops[] = {
    {HD_TOK_EQ,    HD_AST_EQ}, {HD_TOK_NE, HD_AST_NE},
    {HD_TOK_EOF,   0},
};
static const BinOp bitand_ops[] = {{HD_TOK_AMP, HD_AST_BITAND}, {HD_TOK_EOF, 0}};
static const BinOp bitxor_ops[] = {{HD_TOK_CARET, HD_AST_BITXOR}, {HD_TOK_EOF, 0}};
static const BinOp bitor_ops[]  = {{HD_TOK_PIPE, HD_AST_BITOR}, {HD_TOK_EOF, 0}};

static HDASTNode *parse_binop(HDParser *p, HDASTNode *(*higher)(HDParser*), const BinOp *ops) {
    HDASTNode *left = higher(p);
    while (true) {
        bool found = false;
        for (int i = 0; ops[i].tok != HD_TOK_EOF; i++) {
            if (peek(p) == ops[i].tok) {
                advance(p);
                HDASTNode *n = hd_ast_new(ops[i].ast);
                n->left = left;
                n->right = higher(p);
                left = n;
                found = true;
                break;
            }
        }
        if (!found) break;
    }
    return left;
}

/* -- Parse Cast ----------------------------------------------------- */
static HDASTNode *parse_cast(HDParser *p);

static HDASTNode *parse_mul(HDParser *p)      { return parse_binop(p, parse_cast, mul_ops); }

/* -- Parse Cast ----------------------------------------------------- */
static HDASTNode *parse_cast(HDParser *p) {
    if (peek(p) == HD_TOK_LPAREN) {
        /* Look ahead to see if this is a cast: (type) expr */
        /* Save lexer state for backtracking */
        int saved_pos = p->lex->pos;
        int saved_line = p->lex->line;
        int saved_col = p->lex->col;
        advance(p); /* consume ( */
        
        /* Check if next token is a type keyword or identifier (typedef name) */
        bool is_type = false;
        HDTokenType tok = peek(p);
        if (tok == HD_KW_I8 || tok == HD_KW_I16 || tok == HD_KW_I32 ||
            tok == HD_KW_I64 || tok == HD_KW_U8 || tok == HD_KW_U16 ||
            tok == HD_KW_U32 || tok == HD_KW_U64 || tok == HD_KW_F64 ||
            tok == HD_KW_BOOL || tok == HD_KW_STRUCT || tok == HD_KW_UNION ||
            tok == HD_KW_ENUM || tok == HD_KW_TYPEDEF) {
            is_type = true;
        }
        /* An IDENT after `(` is only a cast if it's a typedef name or an enum
         * tag — otherwise it's a parenthesized expression like `(x)` or
         * `(x+1)`, NOT a cast `(Type)x`. Previously any leading IDENT was
         * treated as a type, so `(x)` was mis-tokenized as a cast and the
         * inner `x` was lost, causing "expected RPAREN, got IDENT". */
        if (tok == HD_TOK_IDENT) {
            const char *id = p->lex->tok.text;
            for (int i = 0; i < p->n_typedefs; i++)
                if (strcmp(p->typedef_names[i], id) == 0) { is_type = true; break; }
            if (!is_type)
                for (int i = 0; i < p->n_named_types; i++)
                    if (strcmp(p->named_type_names[i], id) == 0) { is_type = true; break; }
        }
        
        if (is_type) {
            /* This is a cast - parse the type */
            HDType *cast_type = parse_type(p);
            expect(p, HD_TOK_RPAREN);
            HDASTNode *expr = parse_cast(p);  /* right-associative for nested casts */
            HDASTNode *n = hd_ast_new(HD_AST_CAST);
            n->child = expr;
            n->type = cast_type;
            return n;
        }
        /* Not a cast: restore to the `(` itself and parse as a normal
         * parenthesized expression via parse_postfix. parse_postfix calls
         * parse_primary's LPAREN case (which parses `(expr)` and returns the
         * inner node) and then applies trailing postfix (.field, [idx], ->)
         * — so `(*p).a` keeps `.a`. The caller's binop loop then applies
         * `*4` to `(2+3)*4`. saved_pos points PAST the `(` (the lexer
         * already consumed it), so rewind one char to land on `(`. */
        p->lex->pos = saved_pos - 1;
        p->lex->line = saved_line;
        p->lex->col = saved_col;
        hd_lex_next(p->lex);
        return parse_postfix(p);
    }
    return parse_unary(p);
}
static HDASTNode *parse_add(HDParser *p)      { return parse_binop(p, parse_mul, add_ops); }
static HDASTNode *parse_shift(HDParser *p)    { return parse_binop(p, parse_add, shift_ops); }
static HDASTNode *parse_cmp(HDParser *p)      { return parse_binop(p, parse_shift, cmp_ops); }
static HDASTNode *parse_eq(HDParser *p)       { return parse_binop(p, parse_cmp, eq_ops); }
static HDASTNode *parse_bitand(HDParser *p)   { return parse_binop(p, parse_eq, bitand_ops); }
static HDASTNode *parse_bitxor(HDParser *p)   { return parse_binop(p, parse_bitand, bitxor_ops); }
static HDASTNode *parse_bitor(HDParser *p)    { return parse_binop(p, parse_bitxor, bitor_ops); }

/* logic_and, logic_or */
static HDASTNode *parse_logic_and(HDParser *p) {
    HDASTNode *left = parse_bitor(p);
    while (peek(p) == HD_TOK_AND) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_AND);
        n->left = left; n->right = parse_bitor(p);
        left = n;
    }
    return left;
}

static HDASTNode *parse_logic_or(HDParser *p) {
    HDASTNode *left = parse_logic_and(p);
    while (peek(p) == HD_TOK_OR) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_OR);
        n->left = left; n->right = parse_logic_and(p);
        left = n;
    }
    return left;
}

/* -- Parse Ternary ------------------------------------------------ */

static HDASTNode *parse_ternary(HDParser *p) {
    HDASTNode *expr = parse_logic_or(p);
    if (peek(p) == HD_TOK_QUESTION) {
        advance(p);
        HDASTNode *n = hd_ast_new(HD_AST_TERNARY);
        n->cond = expr;
        n->then_branch = parse_expr(p);
        expect(p, HD_TOK_COLON);
        n->else_branch = parse_ternary(p);
        return n;
    }
    return expr;
}

/* -- Parse Assignment --------------------------------------------- */

static HDASTNode *parse_assign(HDParser *p) {
    HDASTNode *left = parse_ternary(p);

    HDASTKind assign_kind = 0;
    switch (peek(p)) {
        case HD_TOK_ASSIGN:       assign_kind = HD_AST_ASSIGN; break;
        case HD_TOK_PLUS_ASSIGN:  assign_kind = HD_AST_ADD_ASSIGN; break;
        case HD_TOK_MINUS_ASSIGN: assign_kind = HD_AST_SUB_ASSIGN; break;
        case HD_TOK_STAR_ASSIGN:  assign_kind = HD_AST_MUL_ASSIGN; break;
        case HD_TOK_SLASH_ASSIGN: assign_kind = HD_AST_DIV_ASSIGN; break;
        case HD_TOK_PERCENT_ASSIGN: assign_kind = HD_AST_MOD_ASSIGN; break;
        case HD_TOK_SHL_ASSIGN:   assign_kind = HD_AST_SHL_ASSIGN; break;
        case HD_TOK_SHR_ASSIGN:   assign_kind = HD_AST_SHR_ASSIGN; break;
        case HD_TOK_AMP_ASSIGN:   assign_kind = HD_AST_AMP_ASSIGN; break;
        case HD_TOK_PIPE_ASSIGN:  assign_kind = HD_AST_PIPE_ASSIGN; break;
        case HD_TOK_CARET_ASSIGN: assign_kind = HD_AST_CARET_ASSIGN; break;
        default: return left;
    }

    advance(p);
    HDASTNode *n = hd_ast_new(assign_kind);
    n->left = left;
    n->right = parse_assign(p);
    return n;
}

/* -- Parse Expression --------------------------------------------- */

static HDASTNode *parse_expr(HDParser *p) {
    return parse_comma(p);
}

/* -- Parse Comma (lowest precedence binary) ------------------------ */

static HDASTNode *parse_comma(HDParser *p) {
    HDASTNode *left = parse_assign(p);
    while (peek(p) == HD_TOK_COMMA) {
        advance(p);
        HDASTNode *right = parse_assign(p);
        HDASTNode *n = hd_ast_new(HD_AST_COMMA);
        n->left = left;
        n->right = right;
        n->type = right->type; /* comma expr has type of right operand */
        left = n;
    }
    return left;
}

/* -- Parse Block -------------------------------------------------- */

HDASTNode *parse_block(HDParser *p) {
    expect(p, HD_TOK_LBRACE);
    HDASTNode *block = hd_ast_new(HD_AST_BLOCK);
    if (!block) { p->has_error = true; return NULL; }
    while (peek(p) != HD_TOK_RBRACE && peek(p) != HD_TOK_EOF) {
        int start_pos = p->lex->pos;
        /* Peek ahead: if the next non-statement token is '}', this is the
         * last statement in the block. Allow omitting the trailing semicolon
         * for expression statements (HolyD block-as-expression syntax). */
        hd_ast_add_stmt(block, parse_stmt(p));
        if (!p->has_error && p->lex->pos == start_pos) {
            parse_error(p, "unexpected token in block");
            break;
        }
        if (p->has_error) {
            /* If the error is a missing semicolon before '}', clear it —
             * the last statement in a block doesn't need one. */
            if (peek(p) == HD_TOK_RBRACE) {
                p->has_error = false;
                p->n_errors = 0;
            } else {
                break;
            }
        }
    }
    expect(p, HD_TOK_RBRACE);
    return block;
}

HDASTNode *hd_parse_block(HDParser *p) {
    return parse_block(p);
}

/* -- Parse Statement ---------------------------------------------- */

static HDASTNode *parse_stmt(HDParser *p) {
    /* If statement */
    if (match(p, HD_KW_IF)) {
        HDASTNode *n = hd_ast_new(HD_AST_IF);
        expect(p, HD_TOK_LPAREN);
        n->cond = parse_expr(p);
        expect(p, HD_TOK_RPAREN);
        n->then_branch = parse_stmt(p);
        if (match(p, HD_KW_ELSE))
            n->else_branch = parse_stmt(p);
        return n;
    }

    /* While statement */
    if (match(p, HD_KW_WHILE)) {
        HDASTNode *n = hd_ast_new(HD_AST_WHILE);
        expect(p, HD_TOK_LPAREN);
        n->cond = parse_expr(p);
        expect(p, HD_TOK_RPAREN);
        n->body = parse_stmt(p);
        return n;
    }

    /* Do-while statement: do body while(cond); */
    if (match(p, HD_KW_DO)) {
        HDASTNode *n = hd_ast_new(HD_AST_DO_WHILE);
        n->body = parse_stmt(p);
        if (match(p, HD_KW_WHILE)) {
            expect(p, HD_TOK_LPAREN);
            n->cond = parse_expr(p);
            expect(p, HD_TOK_RPAREN);
        } else {
            n->cond = NULL;  /* infinite loop if no while-clause */
        }
        match(p, HD_TOK_SEMI);
        return n;
    }

    /* Switch statement: switch(expr) { case VAL: ... default: ... } */
    if (match(p, HD_KW_SWITCH)) {
        HDASTNode *n = hd_ast_new(HD_AST_SWITCH);
        expect(p, HD_TOK_LPAREN);
        n->cond = parse_expr(p);
        expect(p, HD_TOK_RPAREN);
        /* Parse the body as a sequence of case/default labels + statements.
         * We collect case nodes (cond=value or NULL for default) in n->body's
         * stmts array. */
        HDASTNode *body = hd_ast_new(HD_AST_BLOCK);
        expect(p, HD_TOK_LBRACE);
        HDASTNode *current = NULL;   /* the case node being filled */
        while (peek(p) != HD_TOK_RBRACE && peek(p) != HD_TOK_EOF) {
            if (match(p, HD_KW_CASE)) {
                current = hd_ast_new(HD_AST_CASE);
                current->cond = parse_expr(p);
                expect(p, HD_TOK_COLON);
                current->body = hd_ast_new(HD_AST_BLOCK);
                hd_ast_add_stmt(body, current);
            } else if (match(p, HD_KW_DEFAULT)) {
                expect(p, HD_TOK_COLON);
                current = hd_ast_new(HD_AST_CASE);
                current->cond = NULL;   /* default */
                current->body = hd_ast_new(HD_AST_BLOCK);
                hd_ast_add_stmt(body, current);
            } else {
                if (!current) {
                    parse_error(p, "statement before first case in switch");
                    break;
                }
                hd_ast_add_stmt(current->body, parse_stmt(p));
            }
        }
        expect(p, HD_TOK_RBRACE);
        n->body = body;
        return n;
    }

    /* For statement */
    if (match(p, HD_KW_FOR)) {
        HDASTNode *n = hd_ast_new(HD_AST_FOR);
        expect(p, HD_TOK_LPAREN);
        /* C11 allows a DECLARATION in the for-init (`for(int i=0; ...)`)
         * which starts with a type keyword, not an expression. parse_expr
         * would fail on `int i=0` (HD_KW_I32 isn't an expr). Detect a
         * type-keyword start and route to hd_parse_decl instead. */
        HDTokenType itok = peek(p);
        bool type_start = (itok == HD_KW_I8 || itok == HD_KW_I16 ||
                           itok == HD_KW_I32 || itok == HD_KW_I64 ||
                           itok == HD_KW_U8  || itok == HD_KW_U16 ||
                           itok == HD_KW_U32 || itok == HD_KW_U64 ||
                           itok == HD_KW_F64 || itok == HD_KW_BOOL ||
                           itok == HD_KW_U0);
        if (type_start)
            n->init_expr = hd_parse_decl(p);   /* consumes decl + trailing ; */
        else if (peek(p) == HD_TOK_SEMI) {
            n->init_expr = NULL;               /* empty init: for(; ...) */
            expect(p, HD_TOK_SEMI);            /* consume the ; */
        } else {
            n->init_expr = parse_expr(p);
            expect(p, HD_TOK_SEMI);
        }
        n->cond = parse_expr(p);
        expect(p, HD_TOK_SEMI);
        n->update = parse_expr(p);
        expect(p, HD_TOK_RPAREN);
        n->body = parse_stmt(p);
        return n;
    }

    /* Return statement */
    if (match(p, HD_KW_RETURN)) {
        HDASTNode *n = hd_ast_new(HD_AST_RETURN);
        if (peek(p) != HD_TOK_SEMI)
            n->child = parse_expr(p);
        expect(p, HD_TOK_SEMI);
        return n;
    }

    /* Break */
    if (match(p, HD_KW_BREAK)) {
        expect(p, HD_TOK_SEMI);
        return hd_ast_new(HD_AST_BREAK);
    }

    /* Continue */
    if (match(p, HD_KW_CONTINUE)) {
        expect(p, HD_TOK_SEMI);
        return hd_ast_new(HD_AST_CONTINUE);
    }

    /* goto label; */
    if (match(p, HD_KW_GOTO)) {
        HDASTNode *n = hd_ast_new(HD_AST_GOTO);
        if (peek(p) != HD_TOK_IDENT) {
            parse_error(p, "goto requires a label name");
            return n;
        }
        strncpy(n->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
        advance(p);
        expect(p, HD_TOK_SEMI);
        return n;
    }

    /* Label: `ident:` — a statement label. Detect by peeking the token
     * after the current IDENT. We save the whole token + lexer position,
     * advance, and restore by direct assignment (a pos-restore + re-lex
     * lands on the wrong char because lex->pos mid-token points past the
     * token — same trap as sizeof). */
    if (peek(p) == HD_TOK_IDENT) {
        HDToken saved_tok = p->lex->tok;
        int saved_pos = p->lex->pos;
        advance(p);              /* ident */
        bool is_label = (peek(p) == HD_TOK_COLON);
        if (is_label) {
            HDASTNode *n = hd_ast_new(HD_AST_LABEL);
            strncpy(n->ident, saved_tok.text, HD_MAX_IDENT_LEN - 1);
            advance(p);          /* : */
            return n;
        }
        /* not a label: restore token + position to before `ident` */
        p->lex->tok = saved_tok;
        p->lex->pos = saved_pos;
    }

    /* Block */
    if (peek(p) == HD_TOK_LBRACE) {
        return parse_block(p);
    }

    /* Variable declaration (type followed by ident) */
    {
        HDTokenType _t = peek(p);
        if (_t >= HD_KW_I0 && _t <= HD_KW_VOLATILE) {
            return hd_parse_decl(p);
        }
    }

    /* A typedef'd name IS a type — `Point p;` is a decl. */
    if (peek(p) == HD_TOK_IDENT) {
        for (int i = 0; i < p->n_typedefs; i++) {
            if (strcmp(p->typedef_names[i], p->lex->tok.text) == 0)
                return hd_parse_decl(p);
        }
        /* Also treat enum-constant idents as constants (PARSE as const decl
         * so the codegen records them as module-level globals). */
        for (int i = 0; i < p->n_enum_consts; i++) {
            if (strcmp(p->enum_const_names[i], p->lex->tok.text) == 0)
                return hd_parse_decl(p);
        }
    }

    /* Expression statement */
    HDASTNode *expr = parse_expr(p);
    expect(p, HD_TOK_SEMI);
    HDASTNode *n = hd_ast_new(HD_AST_EXPR_STMT);
    n->child = expr;
    return n;
}

/* -- Parse Declaration -------------------------------------------- */

HDASTNode *hd_parse_decl(HDParser *p) {
    /* Handle `typedef`: `typedef <type> <name>;` — parse the type, take the
     * name, register it in the typedef registry so later declarations use it
     * as a type. Returns a no-op decl node. */
    if (match(p, HD_KW_TYPEDEF)) {
        HDType *base = parse_type(p);
        if (peek(p) != HD_TOK_IDENT) {
            parse_error(p, "expected typedef name");
            return NULL;
        }
        const char *tdname = p->lex->tok.text;
        if (p->n_typedefs < 64) {
            strncpy(p->typedef_names[p->n_typedefs], tdname, HD_MAX_IDENT_LEN - 1);
            p->typedef_types[p->n_typedefs] = base;
            p->n_typedefs++;
        }
        advance(p); /* consume the typedef name */
        expect(p, HD_TOK_SEMI);
        HDASTNode *n = hd_ast_new(HD_AST_STRUCT_DECL); /* no-op */
        n->type = base;
        /* For typedef'd structs/unions with empty tag, set the ident
         * so the MIR generator can register the type by name */
        if (base->kind == HD_TYPE_STRUCT || base->kind == HD_TYPE_UNION) {
            if (base->name[0] == '\0') {
                strncpy(base->name, tdname, HD_MAX_IDENT_LEN - 1);
            }
            strncpy(n->ident, base->name, HD_MAX_IDENT_LEN - 1);
        }
        return n;
    }

    /* Handle `static` storage class: strip it and parse the rest as a normal
     * declaration (static only matters for linking, which this JIT doesn't). */
    if (match(p, HD_KW_STATIC)) {
        return hd_parse_decl(p);
    }

    /* Handle `const`/`volatile` qualifiers: strip and parse as normal decl.
     * These affect type checking which the JIT doesn't enforce. */
    if (match(p, HD_KW_CONST) || match(p, HD_KW_VOLATILE)) {
        return hd_parse_decl(p);
    }

    /* Handle extern declarations: extern "C" func(...) or extern type name; */
    if (match(p, HD_KW_EXTERN)) {
        /* Check for extern "C" function declaration */
        if (peek(p) == HD_TOK_STRING) {
        /* Expect "C" string literal */
        if (!match(p, HD_TOK_STRING)) {
            parse_error(p, "expected extern string literal (e.g., \"C\")");
            return NULL;
        }
        /* Verify it's "C" */
        if (strcmp(p->lex->tok.text, "\"C\"") != 0 && strcmp(p->lex->tok.text, "C") != 0) {
            parse_error(p, "only extern \"C\" is supported");
            return NULL;
        }

        /* Parse return type */
        HDType *ret_type = parse_type(p);
        if (!ret_type) {
            parse_error(p, "expected return type after extern \"C\"");
            return NULL;
        }

        /* Expect function name */
        if (peek(p) != HD_TOK_IDENT) {
            parse_error(p, "expected function name");
            return NULL;
        }
        char func_name[HD_MAX_IDENT_LEN];
        strncpy(func_name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
        advance(p);

        /* Expect ( for parameters */
        expect(p, HD_TOK_LPAREN);

        /* Create extern declaration AST node */
        HDASTNode *ext = hd_ast_new(HD_AST_EXTERN_DECL);
        ext->extern_ret_type = ret_type;
        strncpy(ext->extern_c_name, func_name, HD_MAX_IDENT_LEN - 1);
        ext->extern_n_params = 0;

        /* Parse parameters */
        int pi = 0;
        if (peek(p) != HD_TOK_RPAREN) {
            /* `void` alone means zero parameters */
            if (peek(p) == HD_KW_U0 && p->lex->pos < (int)strlen(p->lex->src)) {
                /* Look ahead: is the next non-void token ')'? */
                int saved_pos = p->lex->pos;
                advance(p); /* consume void */
                if (peek(p) == HD_TOK_RPAREN) {
                    advance(p); /* consume ) */
                    ext->extern_n_params = 0;
                    goto done_extern_params;
                }
                /* Not void ) — rewind */
                p->lex->pos = saved_pos;
            }
            ext->extern_param_types[pi] = parse_type(p);
            if (peek(p) == HD_TOK_IDENT) {
                /* Skip parameter name */
                advance(p);
            }
            pi++;
            while (match(p, HD_TOK_COMMA) && pi < HD_MAX_PARAMS) {
                ext->extern_param_types[pi] = parse_type(p);
                if (peek(p) == HD_TOK_IDENT) {
                    advance(p);
                }
                pi++;
            }
        }
        ext->extern_n_params = pi;

        expect(p, HD_TOK_RPAREN);

done_extern_params:

        /* Optional -> ret_type for explicit return type */
        if (match(p, HD_TOK_ARROW)) {
            HDType *explicit_ret = parse_type(p);
            if (explicit_ret) ext->extern_ret_type = explicit_ret;
        }

        expect(p, HD_TOK_SEMI);
        return ext;
        } /* end extern "C" */
        /* extern variable declaration: extern type name; */
        /* For JIT purposes, treat as a normal variable declaration */
        return hd_parse_decl(p);
    }

    HDType *type = parse_type(p);

    /* Check if this is a struct/union/enum type definition without a variable name */
    if (type->kind == HD_TYPE_STRUCT || type->kind == HD_TYPE_UNION || type->kind == HD_TYPE_ENUM) {
        if (peek(p) == HD_TOK_SEMI) {
            /* Type definition like "struct Point { ... };"
             * For an ENUM, also emit the enumerator constants as module-level
             * globals so `int c = GREEN;` resolves GREEN to 1. */
            advance(p);
            HDASTNode *n = hd_ast_new(type->kind == HD_TYPE_ENUM ? HD_AST_BLOCK : HD_AST_STRUCT_DECL);
            n->type = type;
            if (type->kind == HD_TYPE_ENUM && p->n_enum_consts > 0) {
                /* build a BLOCK: enum decl + one const VAR_DECL per enumerator */
                int nconsts = p->n_enum_consts;
                /* NOTE: do NOT clear n_enum_consts — later declarations may
                 * reference these enum constants (e.g. `enum Color c = GREEN;`).
                 * The constants remain available for lookup. */
                for (int c = 0; c < nconsts; c++) {
                    HDASTNode *vd = hd_ast_new(HD_AST_VAR_DECL);
                    strncpy(vd->ident, p->enum_const_names[c], HD_MAX_IDENT_LEN - 1);
                    HDASTNode *init = hd_ast_new(HD_AST_INT_LIT);
                    init->int_val = p->enum_const_vals[c];
                    vd->init = init;
                    vd->type = type;
                    hd_ast_add_stmt(n, vd);
                }
                HDASTNode *ed = hd_ast_new(HD_AST_STRUCT_DECL);
                ed->type = type;
                hd_ast_add_stmt(n, ed);
            }
            return n;
        }
    }

    if (peek(p) != HD_TOK_IDENT && peek(p) != HD_TOK_STAR && peek(p) != HD_TOK_LPAREN) {
        parse_error(p, "expected identifier");
        return NULL;
    }

    int is_func_ptr = 0;
    /* Declarator parsing — three forms we support:
     *   int *p           → is_ptr=1, plain pointer var
     *   int (*p)(a,b)    → is_func_ptr=1, pointer-to-function var
     *   int name(a,b)    → FUNC_DECL (handled below by peek==LPAREN check)
     *   int name         → plain var
     * Parenthesized `(name)` for plain function decls is uncommon and
     * treated as identical to the non-paren form. */
    int is_ptr = 0;
    char name[HD_MAX_IDENT_LEN];

    if (peek(p) == HD_TOK_STAR) {
        /* `int *p` — simple pointer */
        is_ptr = 1;
        advance(p); /* * */
        if (peek(p) != HD_TOK_IDENT) {
            parse_error(p, "expected ident after '*'");
            return NULL;
        }
        strncpy(name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
        name[HD_MAX_IDENT_LEN - 1] = '\0';
        advance(p); /* name */
    } else if (peek(p) == HD_TOK_LPAREN) {
        /* Could be `(*name)(params)` (function pointer) or `(name)` (fn decl).
         * Lookahead: the token after '(' tells us which. */
        int saved_pos = p->lex->pos;
        advance(p); /* ( */
        if (peek(p) == HD_TOK_STAR) {
            /* `(*name)` — function pointer declarator */
            advance(p); /* * */
            if (peek(p) != HD_TOK_IDENT) {
                parse_error(p, "expected ident after '(*'");
                return NULL;
            }
            strncpy(name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
            name[HD_MAX_IDENT_LEN - 1] = '\0';
            advance(p); /* name */
            expect(p, HD_TOK_RPAREN);
            if (peek(p) == HD_TOK_LPAREN) {
                is_func_ptr = 1;
            } else {
                /* `(*p)` with no params — treat as plain pointer */
                is_ptr = 1;
            }
        } else {
            /* `(name)` — backtrack, treat as plain name (func decl or var) */
            p->lex->pos = saved_pos;
            p->lex->line = 0;
            hd_lex_next(p->lex);
            if (peek(p) != HD_TOK_IDENT) {
                parse_error(p, "expected identifier");
                return NULL;
            }
            strncpy(name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
            name[HD_MAX_IDENT_LEN - 1] = '\0';
            advance(p); /* name */
        }
    } else {
        /* Plain: `int name` or `int name(params)` */
        if (peek(p) != HD_TOK_IDENT) {
            parse_error(p, "expected identifier");
            return NULL;
        }
        strncpy(name, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
        name[HD_MAX_IDENT_LEN - 1] = '\0';
        advance(p); /* name */

        /* Function definition: `int name(params) { body }` — NOT a function
         * pointer variable (`int (*name)(params)` is handled by is_func_ptr).
         * Codegen already emits HD_AST_FUNC_DECL bodies, so parsing them makes
         * real C programs (and the gauntlet's `int main(){...}` TUs) build. */
        if (peek(p) == HD_TOK_LPAREN && !is_func_ptr) {
            HDASTNode *fn = hd_ast_new(HD_AST_FUNC_DECL);
            fn->type = type;
            strncpy(fn->ident, name, HD_MAX_IDENT_LEN - 1);
            fn->n_params = 0;
            int fn_params_void_consumed = 0;  /* set when (void) consumed ')' */
            advance(p); /* ( */
            /* `void` alone in the param list means zero parameters: `int foo(void)`.
             * Without this, `void` is parsed as a parameter *type* (HD_TYPE_VOID)
             * and the function gets a spurious first parameter that corrupts the
             * argument registers and memory allocation. */
            if (peek(p) == HD_KW_U0) {
                advance(p); /* consume void */
                if (peek(p) == HD_TOK_RPAREN) {
                    advance(p); /* consume ) */
                    fn->n_params = 0;
                    fn_params_void_consumed = 1;
                    goto done_params;
                }
                /* Not `void )` — it's a parameter named or typed with void.
                 * Put the token back by re-parsing from the saved position. */
                /* We already consumed `void`; fall through to normal parsing
                 * which will treat it as a parameter type. */
            }
            if (peek(p) != HD_TOK_RPAREN) {
                HDType *pt = parse_type(p);
                char pname[HD_MAX_IDENT_LEN] = {0};
                /* Detect function pointer parameter by scanning source for '(*'
                 * pattern. This avoids backtracking issues with the lexer. */
                if (peek(p) == HD_TOK_LPAREN && p->lex->pos < (int)strlen(p->lex->src) && p->lex->src[p->lex->pos] == '*') {
                    /* Function pointer: type(*name)(params) — skip the whole thing */
                    /* Skip (* */
                    advance(p); /* ( */
                    advance(p); /* * */
                    if (peek(p) == HD_TOK_IDENT) {
                        strncpy(pname, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                        advance(p);
                    }
                    expect(p, HD_TOK_RPAREN);
                    /* Skip parameter list (params) */
                    if (peek(p) == HD_TOK_LPAREN) {
                        advance(p);
                        int depth = 1;
                        while (depth > 0 && peek(p) != HD_TOK_EOF) {
                            if (peek(p) == HD_TOK_LPAREN) depth++;
                            else if (peek(p) == HD_TOK_RPAREN) depth--;
                            if (depth > 0) advance(p);
                        }
                        if (peek(p) == HD_TOK_RPAREN) advance(p);
                    }
                    /* Build function pointer type */
                    HDType *fn_type = (HDType *)calloc(1, sizeof(HDType));
                    fn_type->kind = HD_TYPE_FUNC;
                    fn_type->param_types = (HDType **)calloc(HD_MAX_PARAMS, sizeof(HDType *));
                    fn_type->n_params = 0;
                    HDType *fpt = (HDType *)calloc(1, sizeof(HDType));
                    fpt->kind = HD_TYPE_PTR;
                    fpt->base = fn_type;
                    fpt->size = 8;
                    pt = fpt;
                } else if (peek(p) == HD_TOK_IDENT) {
                    strncpy(pname, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                    advance(p);
                }
                fn->param_types[fn->n_params] = pt;
                strncpy(fn->param_names[fn->n_params], pname, HD_MAX_IDENT_LEN - 1);
                fn->n_params++;
                while (match(p, HD_TOK_COMMA) && fn->n_params < HD_MAX_PARAMS) {
                    pt = parse_type(p);
                    pname[0] = '\0';
                    /* Check for function pointer in subsequent params */
                    if (peek(p) == HD_TOK_LPAREN && p->lex->pos < (int)strlen(p->lex->src) && p->lex->src[p->lex->pos] == '*') {
                        advance(p);
                        advance(p);
                        if (peek(p) == HD_TOK_IDENT) advance(p);
                        expect(p, HD_TOK_RPAREN);
                        if (peek(p) == HD_TOK_LPAREN) {
                            advance(p);
                            int depth2 = 1;
                            while (depth2 > 0 && peek(p) != HD_TOK_EOF) {
                                if (peek(p) == HD_TOK_LPAREN) depth2++;
                                else if (peek(p) == HD_TOK_RPAREN) depth2--;
                                if (depth2 > 0) advance(p);
                            }
                            if (peek(p) == HD_TOK_RPAREN) advance(p);
                        }
                        HDType *fn2 = (HDType *)calloc(1, sizeof(HDType));
                        fn2->kind = HD_TYPE_FUNC;
                        fn2->param_types = (HDType **)calloc(HD_MAX_PARAMS, sizeof(HDType *));
                        fn2->n_params = 0;
                        HDType *fpt2 = (HDType *)calloc(1, sizeof(HDType));
                        fpt2->kind = HD_TYPE_PTR;
                        fpt2->base = fn2;
                        fpt2->size = 8;
                        pt = fpt2;
                    } else if (peek(p) == HD_TOK_IDENT) {
                        strncpy(pname, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                        advance(p);
                    }
                    fn->param_types[fn->n_params] = pt;
                    strncpy(fn->param_names[fn->n_params], pname, HD_MAX_IDENT_LEN - 1);
                    fn->n_params++;
                }
            }
done_params:
        /* The void) shortcut (goto done_params) already consumed ')'.
         * For all other paths we still need to consume ')'. */
        if (!fn_params_void_consumed) {
            expect(p, HD_TOK_RPAREN);  /* consume the closing paren */
        }
            if (peek(p) == HD_TOK_LBRACE) {
                fn->body = parse_block(p);
            } else {
                /* declaration without body: `int foo(int);` — consume ; */
                expect(p, HD_TOK_SEMI);
            }
            return fn;
        }
    }

    /* Function pointer variable: `int (*op)(params) [= init] ;`
     * Build a pointer-to-function type; the RHS `add` will later emit the
     * function's address via the function-table lookup in gen_expr. */
    if (is_func_ptr) {
        HDType *fn = (HDType *)calloc(1, sizeof(HDType));
        fn->kind = HD_TYPE_FUNC;
        fn->param_types = (HDType **)calloc(HD_MAX_PARAMS, sizeof(HDType *));
        /* parse params list */
        advance(p); /* ( */
        int pi = 0;
        if (peek(p) != HD_TOK_RPAREN) {
            fn->param_types[pi] = parse_type(p);
            if (peek(p) == HD_TOK_IDENT) advance(p);
            pi++;
            while (match(p, HD_TOK_COMMA) && pi < HD_MAX_PARAMS) {
                fn->param_types[pi] = parse_type(p);
                if (peek(p) == HD_TOK_IDENT) advance(p);
                pi++;
            }
        }
        fn->n_params = pi;
        expect(p, HD_TOK_RPAREN);
        HDType *pt = (HDType *)calloc(1, sizeof(HDType));
        pt->kind = HD_TYPE_PTR; pt->base = fn; pt->size = 8;
        type = pt;
        HDASTNode *var = hd_ast_new(HD_AST_VAR_DECL);
        strncpy(var->ident, name, HD_MAX_IDENT_LEN - 1);
        var->type = type;
        if (match(p, HD_TOK_ASSIGN))
            var->init = parse_expr(p);
        expect(p, HD_TOK_SEMI);
        return var;
    }

    /* Simple pointer variable: `int *p = expr ;` — wrap base type in PTR. */
    if (is_ptr) {
        HDType *pt = (HDType *)calloc(1, sizeof(HDType));
        pt->kind = HD_TYPE_PTR; pt->base = type; pt->size = 8;
        type = pt;
    }

    /* Function declaration: type name(params) { body } */
    if (peek(p) == HD_TOK_LPAREN) {
        advance(p); /* ( */
        HDASTNode *func = hd_ast_new(HD_AST_FUNC_DECL);
        strncpy(func->ident, name, HD_MAX_IDENT_LEN - 1);
        func->type = type;

        /* Parse parameters */
        int pi = 0;
        if (peek(p) != HD_TOK_RPAREN) {
            func->param_types[pi] = parse_type(p);
            if (peek(p) == HD_TOK_IDENT) {
                strncpy(func->param_names[pi], p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                advance(p);
            }
            pi++;
            while (match(p, HD_TOK_COMMA) && pi < HD_MAX_PARAMS) {
                func->param_types[pi] = parse_type(p);
                if (peek(p) == HD_TOK_IDENT) {
                    strncpy(func->param_names[pi], p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
                    advance(p);
                }
                pi++;
            }
        }
        func->n_params = pi;
        expect(p, HD_TOK_RPAREN);

        /* Parse body */
        func->body = parse_block(p);
        return func;
    }

    /* Variable declaration: type name [= expr] ;  (also type name[N] for
     * arrays). Also handles multiple declarators: type a=0,b=1,c=2 ; */
    /* First declarator */
    HDASTNode *first_var = hd_ast_new(HD_AST_VAR_DECL);
    strncpy(first_var->ident, name, HD_MAX_IDENT_LEN - 1);
    int dbg_decl = 1;

    /* Array declarator: name[N][M]... */
    int dims[8], n_dims = 0;
    while (peek(p) == HD_TOK_LBRACKET) {
        advance(p); /* [ */
        int arr_size = 0;
        if (peek(p) != HD_TOK_RBRACKET) {
            HDTokenType st = peek(p);
            if (st == HD_TOK_INT) { arr_size = (int)p->lex->tok.int_val; advance(p); }
        }
        expect(p, HD_TOK_RBRACKET);
        if (n_dims < 8) dims[n_dims++] = arr_size;
    }
    if (n_dims > 0) {
        HDType *base_type = type;
        for (int d = n_dims - 1; d >= 0; d--) {
            HDType *arr = (HDType *)calloc(1, sizeof(HDType));
            arr->kind = HD_TYPE_ARRAY;
            arr->base = base_type;
            arr->array_size = dims[d];
            arr->size = (dims[d] > 0) ? hd_type_size(base_type) * dims[d]
                                      : hd_type_size(base_type);
            base_type = arr;
        }
        type = base_type;
    }
    first_var->type = type;

    if (match(p, HD_TOK_ASSIGN)) {
        first_var->init = parse_assign(p);
    }

    /* Additional declarators: , name [= expr] */
    HDASTNode *block = NULL;
    while (match(p, HD_TOK_COMMA)) {
        HDASTNode *var = hd_ast_new(HD_AST_VAR_DECL);
        if (peek(p) != HD_TOK_IDENT) { parse_error(p, "expected ident after comma"); break; }
        strncpy(var->ident, p->lex->tok.text, HD_MAX_IDENT_LEN - 1);
        advance(p);
        int d2[8], n2 = 0;
        while (peek(p) == HD_TOK_LBRACKET) {
            advance(p);
            int asz = 0;
            if (peek(p) != HD_TOK_RBRACKET) {
                if (peek(p) == HD_TOK_INT) { asz = (int)p->lex->tok.int_val; advance(p); }
            }
            expect(p, HD_TOK_RBRACKET);
            if (n2 < 8) d2[n2++] = asz;
        }
        HDType *vtype = type;
        if (n2 > 0) {
            HDType *bt = vtype;
            for (int d = n2 - 1; d >= 0; d--) {
                HDType *arr = (HDType *)calloc(1, sizeof(HDType));
                arr->kind = HD_TYPE_ARRAY; arr->base = bt;
                arr->array_size = d2[d];
                arr->size = (d2[d] > 0) ? hd_type_size(bt) * d2[d] : hd_type_size(bt);
                bt = arr;
            }
            vtype = bt;
        }
        var->type = vtype;
        if (match(p, HD_TOK_ASSIGN)) {
            var->init = parse_assign(p);
        }
        if (!block) {
            block = hd_ast_new(HD_AST_BLOCK);
            block->no_scope_pop = 1;  /* don't push scope — vars are in parent scope */
            hd_ast_add_stmt(block, first_var);
        }
        hd_ast_add_stmt(block, var);
    }

    expect(p, HD_TOK_SEMI);
    return block ? block : first_var;
}

/* -- Parse Compilation Unit --------------------------------------- */

void hd_parse_init(HDParser *p, HDLexer *lex) {
    memset(p, 0, sizeof(*p));
    p->lex = lex;
}

HDASTNode *hd_parse_compilation_unit(HDParser *p) {
    HDASTNode *unit = hd_ast_new(HD_AST_BLOCK);

    while (peek(p) != HD_TOK_EOF && !p->has_error) {
        HDASTNode *decl = hd_parse_decl(p);
        if (decl) hd_ast_add_stmt(unit, decl);
        else break;
    }

    return unit;
}

HDASTNode *hd_parse_expr(HDParser *p) {
    return parse_expr(p);
}

HDASTNode *hd_parse_stmt(HDParser *p) {
    return parse_stmt(p);
}

HDTokenType hd_parse_peek(HDParser *p) {
    return peek(p);
}


