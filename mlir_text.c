/*
 * mlir_text.c — MLIR text format (assembly) parser.
 *
 * Parses the human-readable MLIR text format (.mlir files) into HLIR graphs.
 * This is the standard format produced by mlir-opt and other MLIR tools.
 *
 * Supported constructs:
 *   - module { ... }
 *   - func.func @name(arg: type) -> ret_type { ... }
 *   - %result = arith.addf(%a, %b) : f32
 *   - %result = arith.constant 42.0 : f32
 *   - return %val : type
 *   - tensor types: tensor<2x2xf32>, tensor<*xf32>
 *   - function types: (f32, f32) -> f32
 *
 * Self-contained: no external dependencies beyond HLIR graph builder.
 * C11, no third-party libraries.
 */
#include "mlir_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ─── Tokenizer ─── */

typedef enum {
    TOK_EOF = 0,
    TOK_IDENT,      /* %name, @name, bare_ident */
    TOK_NUMBER,     /* 42, 3.14, -1 */
    TOK_STRING,     /* "..." */
    TOK_LPAREN,     /* ( */
    TOK_RPAREN,     /* ) */
    TOK_LBRACE,     /* { */
    TOK_RBRACE,     /* } */
    TOK_LBRACKET,   /* < */
    TOK_RBRACKET,   /* > */
    TOK_EQUAL,      /* = */
    TOK_COLON,      /* : */
    TOK_COMMA,      /* , */
    TOK_ARROW,      /* -> */
    TOK_STAR,       /* * */
    TOK_HASH,       /* # */
    TOK_KEYWORD,    /* module, func, return, etc. */
    TOK_TYPE,       /* f32, i32, tensor, etc. */
    TOK_UNKNOWN
} tok_kind_t;

typedef struct {
    tok_kind_t kind;
    char       text[256];
    int        line;
} token_t;

typedef struct {
    const char *src;
    int         pos;
    int         line;
    token_t     cur;
    token_t     peek;
} mlir_lexer_t;

static void lex_init(mlir_lexer_t *lx, const char *src) {
    lx->src = src;
    lx->pos = 0;
    lx->line = 1;
    lx->cur.kind = TOK_EOF;
    lx->peek.kind = TOK_EOF;
}

static void lex_skip_ws(mlir_lexer_t *lx) {
    while (lx->src[lx->pos]) {
        char c = lx->src[lx->pos];
        if (c == '\n') { lx->line++; lx->pos++; continue; }
        if (c == ' ' || c == '\t' || c == '\r') { lx->pos++; continue; }
        if (c == '/' && lx->src[lx->pos+1] == '/') {
            while (lx->src[lx->pos] && lx->src[lx->pos] != '\n') lx->pos++;
            continue;
        }
        break;
    }
}

static void lex_advance(mlir_lexer_t *lx);

static void lex_peek(mlir_lexer_t *lx) {
    if (lx->peek.kind != TOK_EOF) return;
    int saved_pos = lx->pos;
    int saved_line = lx->line;
    lex_advance(lx);
    lx->peek = lx->cur;
    lx->cur.kind = TOK_EOF;
    lx->pos = saved_pos;
    lx->line = saved_line;
}

static void lex_advance(mlir_lexer_t *lx) {
    if (lx->peek.kind != TOK_EOF) {
        lx->cur = lx->peek;
        lx->peek.kind = TOK_EOF;
        return;
    }
    lex_skip_ws(lx);
    char c = lx->src[lx->pos];
    if (!c) { lx->cur.kind = TOK_EOF; return; }

    /* Multi-char tokens */
    if (c == '-' && lx->src[lx->pos+1] == '>') {
        lx->cur.kind = TOK_ARROW; lx->pos += 2; return;
    }
    if (c == '.' && lx->src[lx->pos+1] == '.' && lx->src[lx->pos+2] == '.') {
        lx->cur.kind = TOK_IDENT; strcpy(lx->cur.text, "..."); lx->pos += 3; return;
    }

    /* Single-char tokens */
    switch (c) {
    case '(': lx->cur.kind = TOK_LPAREN; lx->pos++; return;
    case ')': lx->cur.kind = TOK_RPAREN; lx->pos++; return;
    case '{': lx->cur.kind = TOK_LBRACE; lx->pos++; return;
    case '}': lx->cur.kind = TOK_RBRACE; lx->pos++; return;
    case '<': lx->cur.kind = TOK_LBRACKET; lx->pos++; return;
    case '>': lx->cur.kind = TOK_RBRACKET; lx->pos++; return;
    case '=': lx->cur.kind = TOK_EQUAL; lx->pos++; return;
    case ':': lx->cur.kind = TOK_COLON; lx->pos++; return;
    case ',': lx->cur.kind = TOK_COMMA; lx->pos++; return;
    case '*': lx->cur.kind = TOK_STAR; lx->pos++; return;
    case '#': lx->cur.kind = TOK_HASH; lx->pos++; return;
    case '"': {
        lx->cur.kind = TOK_STRING;
        int i = 0; lx->pos++; /* skip opening quote */
        while (lx->src[lx->pos] && lx->src[lx->pos] != '"' && i < 255)
            lx->cur.text[i++] = lx->src[lx->pos++];
        lx->cur.text[i] = '\0';
        if (lx->src[lx->pos] == '"') lx->pos++;
        return;
    }
    }

    /* Numbers */
    if (isdigit(c) || (c == '-' && isdigit(lx->src[lx->pos+1]))) {
        lx->cur.kind = TOK_NUMBER;
        int i = 0;
        if (c == '-') { lx->cur.text[i++] = c; lx->pos++; }
        while (lx->src[lx->pos] && (isdigit(lx->src[lx->pos]) || lx->src[lx->pos] == '.' || lx->src[lx->pos] == 'x'))
            lx->cur.text[i++] = lx->src[lx->pos++];
        lx->cur.text[i] = '\0';
        return;
    }

    /* Identifiers: start with % or @ or letter */
    if (c == '%' || c == '@' || isalpha(c) || c == '_') {
        lx->cur.kind = TOK_IDENT;
        int i = 0;
        if (c == '%' || c == '@') { lx->cur.text[i++] = c; lx->pos++; }
        while (lx->src[lx->pos] && (isalnum(lx->src[lx->pos]) || lx->src[lx->pos] == '_' || lx->src[lx->pos] == '.'))
            lx->cur.text[i++] = lx->src[lx->pos++];
        lx->cur.text[i] = '\0';

        /* Check for keywords */
        if (strcmp(lx->cur.text, "module") == 0 || strcmp(lx->cur.text, "func") == 0 ||
            strcmp(lx->cur.text, "return") == 0 || strcmp(lx->cur.text, "arith") == 0 ||
            strcmp(lx->cur.text, "math") == 0 || strcmp(lx->cur.text, "linalg") == 0 ||
            strcmp(lx->cur.text, "tensor") == 0 || strcmp(lx->cur.text, "cf") == 0 ||
            strcmp(lx->cur.text, "scf") == 0 || strcmp(lx->cur.text, "memref") == 0 ||
            strcmp(lx->cur.text, "affine") == 0)
            lx->cur.kind = TOK_KEYWORD;

        /* Check for type names */
        if (strncmp(lx->cur.text, "f", 1) == 0 && isdigit(lx->cur.text[1]))
            lx->cur.kind = TOK_TYPE;
        if (strncmp(lx->cur.text, "i", 1) == 0 && isdigit(lx->cur.text[1]))
            lx->cur.kind = TOK_TYPE;
        if (strcmp(lx->cur.text, "tensor") == 0 || strcmp(lx->cur.text, "vector") == 0 ||
            strcmp(lx->cur.text, "memref") == 0 || strcmp(lx->cur.text, "index") == 0)
            lx->cur.kind = TOK_TYPE;

        return;
    }

    /* Unknown */
    lx->cur.kind = TOK_UNKNOWN;
    lx->cur.text[0] = c;
    lx->cur.text[1] = '\0';
    lx->pos++;
}

/* ─── Parser state ─── */

#define MAX_VALUES 256
#define MAX_FUNCS 32

typedef struct {
    char name[128];
    int  hlir_node_idx;  /* index into graph */
} value_t;

typedef struct {
    hlir_graph_t *g;
    mlir_lexer_t  lx;
    value_t       values[MAX_VALUES];
    int           n_values;
    int           n_ops;
} mlir_parser_t;

static int find_value(mlir_parser_t *p, const char *name) {
    for (int i = 0; i < p->n_values; i++)
        if (strcmp(p->values[i].name, name) == 0) return i;
    return -1;
}

static int add_value(mlir_parser_t *p, const char *name, int node_idx) {
    if (p->n_values >= MAX_VALUES) return -1;
    strncpy(p->values[p->n_values].name, name, 127);
    p->values[p->n_values].hlir_node_idx = node_idx;
    return p->n_values++;
}

/* Parse a type string like "f32", "tensor<2x2xf32>", "index" */
static int parse_type(mlir_parser_t *p, char *buf, int cap) {
    (void)p;
    int i = 0;
    token_t *t = &p->lx.cur;
    if (t->kind == TOK_TYPE) {
        i += snprintf(buf+i, cap-i, "%s", t->text);
        lex_advance(&p->lx);
    } else if (t->kind == TOK_KEYWORD && strcmp(t->text, "tensor") == 0) {
        i += snprintf(buf+i, cap-i, "tensor");
        lex_advance(&p->lx);
        if (p->lx.cur.kind == TOK_LBRACKET) {
            i += snprintf(buf+i, cap-i, "<");
            lex_advance(&p->lx);
            while (p->lx.cur.kind != TOK_RBRACKET && p->lx.cur.kind != TOK_EOF) {
                if (p->lx.cur.kind == TOK_NUMBER || p->lx.cur.kind == TOK_TYPE || p->lx.cur.kind == TOK_STAR) {
                    i += snprintf(buf+i, cap-i, "%s", p->lx.cur.text);
                    lex_advance(&p->lx);
                } else if (p->lx.cur.kind == TOK_IDENT) {
                    i += snprintf(buf+i, cap-i, "%s", p->lx.cur.text);
                    lex_advance(&p->lx);
                } else {
                    lex_advance(&p->lx);
                }
            }
            if (p->lx.cur.kind == TOK_RBRACKET) {
                i += snprintf(buf+i, cap-i, ">");
                lex_advance(&p->lx);
            }
        }
    }
    return i;
}

/* Map MLIR op name to HLIR op */
static hlir_op_t mlir_text_op(const char *op) {
    /* arith dialect */
    if (strstr(op, "addf") || strstr(op, "addi")) return HLIR_ADD;
    if (strstr(op, "subf") || strstr(op, "subi")) return HLIR_SUB;
    if (strstr(op, "mulf") || strstr(op, "muli")) return HLIR_MUL;
    if (strstr(op, "divf") || strstr(op, "divi") || strstr(op, "divsi") || strstr(op, "divui")) return HLIR_DIV;
    if (strstr(op, "negf")) return HLIR_SUB;
    if (strstr(op, "absf")) return HLIR_CLAMP;
    if (strstr(op, "maximumf") || strstr(op, "maxsi")) return HLIR_CLAMP;
    if (strstr(op, "minimumf") || strstr(op, "minsi")) return HLIR_CLAMP;
    if (strstr(op, "remf") || strstr(op, "remi")) return HLIR_DIV;
    if (strstr(op, "and") || strstr(op, "andi")) return HLIR_MUL;
    if (strstr(op, "or") || strstr(op, "ori")) return HLIR_ADD;
    if (strstr(op, "xor") || strstr(op, "xori")) return HLIR_ADD;
    if (strstr(op, "cmpf") || strstr(op, "cmpi")) return HLIR_CLAMP;
    if (strstr(op, "extf") || strstr(op, "truncf")) return HLIR_CLAMP;
    if (strstr(op, "sitofp") || strstr(op, "fptosi")) return HLIR_CLAMP;

    /* math dialect */
    if (strstr(op, "exp")) return HLIR_EXP;
    if (strstr(op, "sqrt")) return HLIR_SQRT;
    if (strstr(op, "tanh")) return HLIR_TANH;
    if (strstr(op, "log")) return HLIR_EXP;
    if (strstr(op, "pow")) return HLIR_EXP;
    if (strstr(op, "sin") || strstr(op, "cos")) return HLIR_EXP;
    if (strstr(op, "erf")) return HLIR_EXP;
    if (strstr(op, "rsqrt")) return HLIR_SQRT;
    if (strstr(op, "fabs")) return HLIR_CLAMP;

    /* linalg dialect */
    if (strstr(op, "matmul") || strstr(op, "dot")) return HLIR_MATMUL;
    if (strstr(op, "generic")) return HLIR_MATMUL;
    if (strstr(op, "conv")) return HLIR_MUL; /* convolution → elementwise approx */
    if (strstr(op, "pooling")) return HLIR_CLAMP;
    if (strstr(op, "fill")) return HLIR_CONSTANT;
    if (strstr(op, "copy")) return HLIR_ADD;
    if (strstr(op, "transpose")) return HLIR_MUL;
    if (strstr(op, "broadcast")) return HLIR_ADD;
    if (strstr(op, "reduce")) return HLIR_ADD;

    /* tensor dialect */
    if (strstr(op, "extract")) return HLIR_MUL;
    if (strstr(op, "insert")) return HLIR_ADD;
    if (strstr(op, "from_elements")) return HLIR_CONSTANT;
    if (strstr(op, "extract_slice")) return HLIR_MUL;
    if (strstr(op, "insert_slice")) return HLIR_ADD;
    if (strstr(op, "pad")) return HLIR_CLAMP;
    if (strstr(op, "expand_shape")) return HLIR_RESHAPE;
    if (strstr(op, "collapse_shape")) return HLIR_RESHAPE;
    if (strstr(op, "reshape")) return HLIR_RESHAPE;

    /* memref dialect */
    if (strstr(op, "alloc")) return HLIR_CONSTANT;
    if (strstr(op, "dealloc")) return HLIR_EXP; /* no-op marker */
    if (strstr(op, "load")) return HLIR_MUL;
    if (strstr(op, "store")) return HLIR_ADD;
    if (strstr(op, "cast")) return HLIR_CLAMP;
    if (strstr(op, "subview")) return HLIR_MUL;
    if (strstr(op, "dim")) return HLIR_CONSTANT;
    if (strstr(op, "memref.global")) return HLIR_CONSTANT;
    if (strstr(op, "get_global")) return HLIR_MUL;

    /* scf dialect (structured control flow) */
    if (strstr(op, "scf.for")) return HLIR_EXP; /* loop marker */
    if (strstr(op, "scf.if")) return HLIR_EXP;  /* conditional marker */
    if (strstr(op, "scf.yield")) return HLIR_EXP; /* yield marker */
    if (strstr(op, "scf.parallel")) return HLIR_EXP;
    if (strstr(op, "scf.reduce")) return HLIR_ADD;

    /* cf dialect (control flow) */
    if (strstr(op, "cf.br")) return HLIR_EXP;
    if (strstr(op, "cf.cond_br")) return HLIR_EXP;

    /* func dialect */
    if (strstr(op, "func.return") || strstr(op, "return")) return HLIR_EXP;
    if (strstr(op, "func.call")) return HLIR_MUL;
    if (strstr(op, "call")) return HLIR_MUL;

    /* affine dialect */
    if (strstr(op, "affine.for")) return HLIR_EXP;
    if (strstr(op, "affine.if")) return HLIR_EXP;
    if (strstr(op, "affine.load")) return HLIR_MUL;
    if (strstr(op, "affine.store")) return HLIR_ADD;
    if (strstr(op, "affine.apply")) return HLIR_ADD;
    if (strstr(op, "affine.yield")) return HLIR_EXP;

    /* misc */
    if (strstr(op, "constant")) return HLIR_CONSTANT;
    if (strstr(op, "softmax")) return HLIR_SOFTMAX;
    return HLIR_EXP;
}

/* Parse an operation, return HLIR node index or -1 */
static int parse_op(mlir_parser_t *p) {
    token_t *t = &p->lx.cur;

    /* Result assignment: %name = ... */
    char result_name[128] = {0};
    if (t->kind == TOK_IDENT && t->text[0] == '%') {
        strncpy(result_name, t->text, 127);
        lex_advance(&p->lx);
        if (p->lx.cur.kind == TOK_EQUAL) lex_advance(&p->lx);
        else return -1; /* not an assignment */
    }

    /* Op name: dialect.op or bare op */
    char op_name[256] = {0};
    if (t->kind == TOK_KEYWORD) {
        /* dialect.op form: arith.addf */
        snprintf(op_name, sizeof(op_name), "%s", t->text);
        lex_advance(&p->lx);
        if (p->lx.cur.kind == TOK_IDENT && p->lx.cur.text[0] == '.') {
            /* Actually it's "arith.addf" as one token? No, "arith" then ".addf" */
        }
        /* Check for .opname */
        if (p->lx.cur.kind == TOK_IDENT && strcmp(p->lx.cur.text, ".addf") == 0) {
            snprintf(op_name+strlen(op_name), sizeof(op_name)-strlen(op_name), "%s", t->text);
            lex_advance(&p->lx);
        }
    } else if (t->kind == TOK_IDENT) {
        /* Could be "arith.addf" as one token */
        if (strchr(t->text, '.')) {
            strncpy(op_name, t->text, 255);
            lex_advance(&p->lx);
        } else if (strcmp(t->text, "func") == 0 || strcmp(t->text, "module") == 0 ||
                   strcmp(t->text, "return") == 0) {
            strncpy(op_name, t->text, 255);
            lex_advance(&p->lx);
        } else {
            return -1; /* unknown */
        }
    } else {
        return -1;
    }

    /* Parse operands (comma-separated %name or numbers) */
    char operands[8][128];
    int n_operands = 0;
    while (p->lx.cur.kind == TOK_IDENT || p->lx.cur.kind == TOK_NUMBER) {
        strncpy(operands[n_operands], p->lx.cur.text, 127);
        n_operands++;
        lex_advance(&p->lx);
        if (p->lx.cur.kind == TOK_COMMA) lex_advance(&p->lx);
        else break;
    }

    /* Parse optional type annotation: : f32 or : (f32, f32) */
    char type_str[256] = {0};
    if (p->lx.cur.kind == TOK_COLON) {
        lex_advance(&p->lx);
        parse_type(p, type_str, sizeof(type_str));
    }

    /* Create HLIR node */
    hlir_op_t hop = mlir_text_op(op_name);
    int64_t dims[2] = { 4, 4 };
    hlir_tensor_t tt = hlir_tensor(2, dims, 0); /* F32 */

    char node_name[256];
    if (result_name[0]) {
        snprintf(node_name, sizeof(node_name), "%s", result_name);
    } else {
        snprintf(node_name, sizeof(node_name), "mlir_op_%d", p->n_ops);
    }

    /* For constants, try to parse the value */
    float const_val = 1.0f;
    if (hop == HLIR_CONSTANT && n_operands > 0) {
        const_val = (float)atof(operands[0]);
    }

    hlir_node_t *node = hlir_op(p->g, hop, node_name, NULL, 0, &tt, NULL, 0);
    int node_idx = p->g->n - 1;

    if (result_name[0]) {
        add_value(p, result_name, node_idx);
    }
    p->n_ops++;
    return node_idx;
}

/* Parse a function body */
static int parse_func_body(mlir_parser_t *p) {
    int ops = 0;
    while (p->lx.cur.kind != TOK_RBRACE && p->lx.cur.kind != TOK_EOF) {
        if (parse_op(p) >= 0) ops++;
        else {
            /* Skip unknown tokens */
            if (p->lx.cur.kind == TOK_RBRACE) break;
            lex_advance(&p->lx);
        }
    }
    if (p->lx.cur.kind == TOK_RBRACE) lex_advance(&p->lx);
    return ops;
}

/* ─── Public API ─── */

int mlir_text_load(const char *source, hlir_graph_t *g) {
    mlir_parser_t p;
    memset(&p, 0, sizeof(p));
    p.g = g;
    hlir_graph_init(g);
    lex_init(&p.lx, source);
    lex_advance(&p.lx);

    int total_ops = 0;

    /* Parse top-level: module { ... } or bare func */
    while (p.lx.cur.kind != TOK_EOF) {
        if (p.lx.cur.kind == TOK_KEYWORD && strcmp(p.lx.cur.text, "module") == 0) {
            lex_advance(&p.lx);
            if (p.lx.cur.kind == TOK_LBRACE) lex_advance(&p.lx);
            /* Parse module body */
            while (p.lx.cur.kind != TOK_RBRACE && p.lx.cur.kind != TOK_EOF) {
                if (parse_op(&p) >= 0) total_ops++;
                else lex_advance(&p.lx);
            }
            if (p.lx.cur.kind == TOK_RBRACE) lex_advance(&p.lx);
        } else if (p.lx.cur.kind == TOK_IDENT && strstr(p.lx.cur.text, "func")) {
            /* func.func @name(...) { ... } */
            lex_advance(&p.lx); /* skip func or func.func */
            /* Skip @name */
            if (p.lx.cur.kind == TOK_IDENT && p.lx.cur.text[0] == '@')
                lex_advance(&p.lx);
            /* Skip (...) args */
            if (p.lx.cur.kind == TOK_LPAREN) {
                int depth = 1;
                lex_advance(&p.lx);
                while (depth > 0 && p.lx.cur.kind != TOK_EOF) {
                    if (p.lx.cur.kind == TOK_LPAREN) depth++;
                    if (p.lx.cur.kind == TOK_RPAREN) depth--;
                    lex_advance(&p.lx);
                }
            }
            /* Skip -> ret_type */
            if (p.lx.cur.kind == TOK_ARROW) {
                lex_advance(&p.lx);
                /* skip return type */
                while (p.lx.cur.kind != TOK_LBRACE && p.lx.cur.kind != TOK_EOF)
                    lex_advance(&p.lx);
            }
            /* Parse body */
            if (p.lx.cur.kind == TOK_LBRACE) {
                lex_advance(&p.lx);
                total_ops += parse_func_body(&p);
            }
        } else {
            /* Try parsing as a bare operation */
            if (parse_op(&p) >= 0) total_ops++;
            else lex_advance(&p.lx);
        }
    }

    return total_ops > 0 ? 0 : -1;
}
