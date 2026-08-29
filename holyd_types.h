/*
 * holyd_types.h  --  HolyD Compiler Core Types
 * Opaque struct forward declarations and shared type definitions.
 * C11, no external dependencies.
 */
#ifndef WUBUNOS_HOLYC_TYPES_H
#define WUBUNOS_HOLYC_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* -- Limits ------------------------------------------------------- */

#define HD_MAX_TOKEN_LEN    256
#define HD_MAX_IDENT_LEN    64
#define HD_MAX_AST_DEPTH    128
#define HD_MAX_LOCALS       64
#define HD_MAX_PARAMS       16
#define HD_MAX_STRING_LEN   4096
#define HD_MAX_ERRORS       32
#define HD_MAX_FUNCTIONS    64

/* -- Token Types -------------------------------------------------- */

typedef enum {
    HD_TOK_EOF = 0,

    /* Literals */
    HD_TOK_INT,
    HD_TOK_FLOAT,
    HD_TOK_CHAR,
    HD_TOK_STRING,

    HD_TOK_IDENT,

    /* Keywords */
    HD_KW_IF,
    HD_KW_ELSE,
    HD_KW_WHILE,
    HD_KW_FOR,
    HD_KW_DO,
    HD_KW_SWITCH,
    HD_KW_CASE,
    HD_KW_DEFAULT,
    HD_KW_BREAK,
    HD_KW_CONTINUE,
    HD_KW_RETURN,
    HD_KW_GOTO,

    HD_KW_I0, HD_KW_I8, HD_KW_I16, HD_KW_I32, HD_KW_I64,
    HD_KW_U0, HD_KW_U8, HD_KW_U16, HD_KW_U32, HD_KW_U64,
    HD_KW_F64,
    HD_KW_BOOL,
    /* Struct/class */
    HD_KW_CLASS,
    HD_KW_STRUCT,
    HD_KW_UNION,
    HD_KW_TYPEDEF,
    HD_KW_ENUM,
    HD_KW_STATIC,
    HD_KW_EXTERN,
    HD_KW_PUBLIC,
    HD_KW_CONST,
    HD_KW_VOLATILE,
    HD_KW_INLINE,
    HD_KW_SIZEOF,  /* dummy to keep enum open (was HD_KW_UNUSED) */

    /* Operators */
    HD_TOK_PLUS,
    HD_TOK_MINUS,
    HD_TOK_STAR,
    HD_TOK_SLASH,
    HD_TOK_PERCENT,
    HD_TOK_AMP,
    HD_TOK_PIPE,
    HD_TOK_CARET,
    HD_TOK_TILDE,
    HD_TOK_BANG,
    HD_TOK_ASSIGN,
    HD_TOK_LT,
    HD_TOK_GT,

    HD_TOK_LE,
    HD_TOK_GE,
    HD_TOK_EQ,
    HD_TOK_NE,
    HD_TOK_AND,
    HD_TOK_OR,
    HD_TOK_LSHIFT,
    HD_TOK_RSHIFT,
    HD_TOK_SHL,
    HD_TOK_SHR,
    HD_TOK_INC,
    HD_TOK_DEC,
    HD_TOK_ARROW,
    HD_TOK_DOT,
    HD_TOK_LPAREN,
    HD_TOK_RPAREN,
    HD_TOK_LBRACE,
    HD_TOK_RBRACE,
    HD_TOK_LBRACKET,
    HD_TOK_RBRACKET,
    HD_TOK_COMMA,
    HD_TOK_SEMI,
    HD_TOK_COLON,
    HD_TOK_QUESTION,
    HD_TOK_ARROW_RET,
    HD_TOK_PLUS_PLUS,
    HD_TOK_MINUS_MINUS,

    HD_TOK_PLUS_ASSIGN,
    HD_TOK_MINUS_ASSIGN,
    HD_TOK_STAR_ASSIGN,
    HD_TOK_SLASH_ASSIGN,
    HD_TOK_PERCENT_ASSIGN,
    HD_TOK_AMP_ASSIGN,
    HD_TOK_PIPE_ASSIGN,
    HD_TOK_CARET_ASSIGN,
    HD_TOK_LSHIFT_ASSIGN,
    HD_TOK_RSHIFT_ASSIGN,
    HD_TOK_SHL_ASSIGN,
    HD_TOK_SHR_ASSIGN,

    HD_TOK_ELLIPSIS,
} HDTokenType;

/* -- Token -------------------------------------------------------- */

typedef struct {
    HDTokenType type;
    char text[HD_MAX_TOKEN_LEN];
    int64_t int_val;
    double float_val;
    char str_val[HD_MAX_STRING_LEN];
    int line;
    int col;
} HDToken;

/* -- Type System -------------------------------------------------- */

typedef enum {
    HD_TYPE_VOID = 0,
    HD_TYPE_I8,
    HD_TYPE_I16,
    HD_TYPE_I32,
    HD_TYPE_I64,
    HD_TYPE_U8,
    HD_TYPE_U16,
    HD_TYPE_U32,
    HD_TYPE_U64,
    HD_TYPE_F64,
    HD_TYPE_BOOL,
    HD_TYPE_PTR,
    HD_TYPE_ARRAY,
    HD_TYPE_STRUCT,
    HD_TYPE_UNION,
    HD_TYPE_ENUM,
    HD_TYPE_FUNC,
} HDTypeKind;

typedef struct HDType HDType;

struct HDType {
    HDTypeKind kind;
    HDType *base;              /* for pointers/arrays */
    int64_t size;              /* size in bytes */
    int64_t align;             /* alignment requirement */
    int n_members;             /* for structs/unions */
    char name[HD_MAX_IDENT_LEN]; /* for struct/enum/typedef name */
    struct {
        char name[HD_MAX_IDENT_LEN];
        HDType *type;
        int64_t offset;
    } members[32];
    int array_size;            /* for arrays */
    /* for function types */
    HDType *ret_type;
    HDType **param_types;
    int n_params;
};

/* -- AST Node Kinds ----------------------------------------------- */

typedef enum {
    HD_AST_NONE = 0,
    HD_AST_INT_LIT,
    HD_AST_FLOAT_LIT,
    HD_AST_CHAR_LIT,
    HD_AST_STRING_LIT,
    HD_AST_BOOL_LIT,
    HD_AST_IDENT,
    HD_AST_ADD,
    HD_AST_SUB,
    HD_AST_MUL,
    HD_AST_DIV,
    HD_AST_MOD,
    HD_AST_AND,
    HD_AST_OR,
    HD_AST_BITAND,
    HD_AST_BITOR,
    HD_AST_BITXOR,
    HD_AST_SHL,
    HD_AST_SHR,
    HD_AST_EQ,
    HD_AST_NE,
    HD_AST_LT,
    HD_AST_LE,
    HD_AST_GT,
    HD_AST_GE,
    HD_AST_UNARY,
    HD_AST_CAST,
    HD_AST_SIZEOF,   /* sizeof(type) / sizeof expr — emits the type size as a literal */
    HD_AST_CALL,
    HD_AST_FUNC_CALL,
    HD_AST_INDEX,
    HD_AST_DOT,
    HD_AST_MEMBER,
    HD_AST_ARROW,
    HD_AST_TERNARY,
    HD_AST_ASSIGN,
    HD_AST_COMPOUND_ASSIGN,
    HD_AST_COMMA,         /* comma operator: (expr, expr) → eval both, return right */
    HD_AST_PRE_INC,
    HD_AST_PRE_DEC,
    HD_AST_POST_INC,
    HD_AST_POST_DEC,
    HD_AST_EXPR_STMT,
    HD_AST_RETURN,
    HD_AST_IF,
    HD_AST_WHILE,
    HD_AST_FOR,
    HD_AST_DO_WHILE,
    HD_AST_SWITCH,     /* switch(expr){case..} — cond=expr, body=block of CASE */
    HD_AST_CASE,       /* case VAL: — cond=value expr (NULL=default), body=stmts */
    HD_AST_GOTO,       /* goto label; — child=NULL, ident=label name */
    HD_AST_LABEL,      /* label: — ident=label name, emits a marker at current pos */
    HD_AST_BLOCK,
    HD_AST_VAR_DECL,
    HD_AST_FUNC_DECL,
    HD_AST_EXTERN_DECL,
    HD_AST_STRUCT_DECL,
    HD_AST_BREAK,
    HD_AST_CONTINUE,
    HD_AST_NEG,
    HD_AST_NOT,
    HD_AST_BITNOT,
    HD_AST_DEREF,
    HD_AST_ADDR,
    HD_AST_MOD_ASSIGN,
    HD_AST_SHL_ASSIGN,
    HD_AST_SHR_ASSIGN,
    HD_AST_AMP_ASSIGN,
    HD_AST_PIPE_ASSIGN,
    HD_AST_CARET_ASSIGN,
    HD_AST_ADD_ASSIGN,
    HD_AST_SUB_ASSIGN,
    HD_AST_MUL_ASSIGN,
    HD_AST_DIV_ASSIGN,
    HD_AST_BRACE_INIT,
} HDASTKind;

/* -- Forward declarations for opaque structs ---------------------------------- */

typedef struct HDLexer HDLexer;
typedef struct HDParser HDParser;
typedef struct HDGen HDGen;
typedef struct HDCompiler HDCompiler;
typedef struct HDASTNode HDASTNode;
typedef struct HDSymbol HDSymbol;
typedef struct HDSymTab HDSymTab;
typedef struct HDFunction HDFunction;

/* -- Symbol Table ------------------------------------------------------------- */

struct HDSymbol {
    char name[HD_MAX_IDENT_LEN];
    HDType *type;
    int stack_offset;
    bool is_global;
    bool is_param;
};

struct HDSymTab {
    HDSymbol locals[HD_MAX_LOCALS];
    int n_locals;
    int stack_size;
};

/* -- Function Table ----------------------------------------------------------- */

struct HDFunction {
    char name[HD_MAX_IDENT_LEN];
    void *func_ptr;
    int n_params;
    HDType *ret_type;   /* declared return type (struct → sret-capable) */
    size_t code_size;   /* size of function body code (for ELF emission) */
    /* Global RIP-relative fixups emitted INSIDE this function's body. Each
     * function is copied to its OWN exec buffer (separate from the main
     * code+data buffer), so a `mov [rip+disp32], rax` / `mov rax,[rip+disp32]`
     * that references a module-level global must be patched against the
     * FINAL data-section address — which is only known after the main exec
     * is allocated. Record (patch_pos within this body, global_offset) so
     * hd_eval can fix the exec copy up. */
    struct { size_t code_patch_pos; size_t global_offset; } global_patches[128];
    int n_global_patches;
};

/* -- Lexer struct (full definition needed by lexer.c) ------------------------ */

struct HDLexer {
    const char *src;
    int pos;
    int line;
    int col;
    HDToken tok;
    bool has_error;
    char error[256];
};

/* -- Parser struct (full definition needed by parser.c) ---------------------- */

struct HDParser {
    HDLexer *lex;
    HDASTNode *ast;
    bool has_error;
    char errors[HD_MAX_ERRORS][256];
    int n_errors;
    /* named-type registry: struct/union/enum tags defined with a body are
     * stored by name so a later `struct S x;` reference reuses the SAME
     * member layout (previously it built a fresh empty struct with no
     * members, so s.a resolved no offset and reads returned 0). */
    HDType *named_types[64];
    char named_type_names[64][HD_MAX_IDENT_LEN];
    int n_named_types;
    /* typedef registry: `typedef int MyInt;` makes MyInt a type name the
     * parser recognizes in later declarations. */
    HDType *typedef_types[64];
    char typedef_names[64][HD_MAX_IDENT_LEN];
    int n_typedefs;
    /* enum constants: `enum { RED, GREEN, BLUE }` gives GREEN value 1.
     * Recorded so `int c = GREEN;` resolves GREEN as a constant ident. */
    char enum_const_names[64][HD_MAX_IDENT_LEN];
    int64_t enum_const_vals[64];
    int n_enum_consts;
};

/* -- Code Generator struct (full definition needed by codegen.c) ------------- */

struct HDGen {
    uint8_t *code;
    size_t code_size;
    size_t code_cap;
    uint8_t *data;
    size_t data_size;
    size_t data_cap;
    HDSymTab symbols;
    int label_count;
    int loop_depth;
    int break_label;
    int continue_label;
    size_t break_patches[10][16];
    int n_break_patches[10];
    size_t continue_patches[10][16];
    int n_continue_patches[10];
    /* goto labels: a function-local registry of label names -> byte offset
     * in the emitted code. Forward gotos record a jump patch that is
     * resolved when the label is finally placed (or at function end). */
    struct { char name[HD_MAX_IDENT_LEN]; int offset; } labels[128];
    int n_labels;
    struct { size_t patch_pos; int label_idx; } label_patches[512];
    int n_label_patches;
    HDFunction functions[HD_MAX_FUNCTIONS];
    int n_functions;
    struct {
        char c_name[HD_MAX_IDENT_LEN];
        void *func_addr;
    } extern_funcs[32];
    int n_extern_funcs;
    struct {
        size_t code_patch_pos;
        size_t global_offset;
    } global_patches[128];
    int n_global_patches;
    bool has_error;
    char error[256];
    bool has_prologue;   /* set once emit_prologue() has built a stack frame */
    bool in_function;    /* true while emitting a function body (vs module-level) */
    bool module_scope;   /* true when top-level eval block: VAR_DECLs are globals
                          * (data section), so functions declared later can
                          * reference them and survive the FUNC_DECL keep-filter. */
    /* Tailslayer DRAM-refresh hedge: when true, every memory load the JIT
     * emits is preceded by a software prefetch (`prefetchnta`) so the DRAM
     * read is primed ahead of the actual load, hiding the periodic tREFI
     * refresh stall (~150-750ns) that would otherwise hit cold reads.
     * This is an IMPLICIT shim — it applies to all compiled code with no
     * source changes (the "for all code magically" Tailslayer port). */
    bool hedge_loads;
    /* self-recursion support: while compiling a function body, its own name
     * is recorded here so a call to itself can emit a rel32 placeholder that
     * is patched to the final exec address after the body is copied. Without
     * this, fib(n-1) inside fib() traps as unresolved (fib isn't registered
     * in gen->functions until the body is finished). */
    char current_function[HD_MAX_IDENT_LEN];
    size_t self_call_patches[32];
    int n_self_call_patches;
    /* Function call patches: when top-level code calls a JIT'd function,
     * we emit `call rel32` with a placeholder. After the main code is copied
     * to exec, these get patched: rel32 = fn_ptr - (exec + patch_pos + 4). */
    struct { size_t code_patch_pos; void *fn_ptr; } call_patches[32];
    int n_call_patches;
};

/* -- Compiler struct (full definition) --------------------------------------- */

struct HDCompiler {
    HDLexer lex;
    HDParser parse;
    HDGen gen;
    HDASTNode *ast;
    bool has_error;
    char error[256];
};

#endif /* WUBUNOS_HOLYC_TYPES_H */