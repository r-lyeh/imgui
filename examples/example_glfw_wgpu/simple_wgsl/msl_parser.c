/*
 * msl_parser.c - MSL (Metal Shading Language) Parser -> SSIR
 *
 * Parses MSL source code and produces an SSIR module directly using
 * the SSIR builder APIs. Designed to handle MSL output from ssir_to_msl.c
 * for round-trip testing, plus general MSL compute/vertex/fragment shaders.
 */

#include "simple_wgsl.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef MSL_MALLOC
#define MSL_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef MSL_REALLOC
#define MSL_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef MSL_FREE
#define MSL_FREE(p) free((p))
#endif

/* ============================================================================
 * String Utilities
 * ============================================================================ */

//s nonnull
static char *msl_strndup(const char *s, size_t n) {
    wgsl_compiler_assert(s != NULL, "msl_strndup: s is NULL");
    char *r = (char *)MSL_MALLOC(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

//s nullable
static char *msl_strdup(const char *s) {
    return s ? msl_strndup(s, strlen(s)) : NULL;
}

/* ============================================================================
 * Token Types
 * ============================================================================ */

typedef enum {
    MTK_EOF = 0,
    MTK_IDENT,
    MTK_INT_LIT,
    MTK_FLOAT_LIT,
    /* punctuation */
    MTK_SEMI, MTK_COMMA, MTK_COLON, MTK_DOT,
    MTK_LBRACE, MTK_RBRACE,
    MTK_LPAREN, MTK_RPAREN,
    MTK_LBRACKET, MTK_RBRACKET,
    MTK_LATTR, MTK_RATTR,       /* [[ and ]] */
    MTK_STAR, MTK_SLASH, MTK_PLUS, MTK_MINUS, MTK_PERCENT,
    MTK_AMP, MTK_PIPE, MTK_CARET, MTK_TILDE,
    MTK_EQ, MTK_BANG, MTK_LT, MTK_GT, MTK_QMARK,
    /* multi-char */
    MTK_EQEQ, MTK_NEQ, MTK_LE, MTK_GE,
    MTK_ANDAND, MTK_OROR,
    MTK_SHL, MTK_SHR,
    MTK_PLUSEQ, MTK_MINUSEQ, MTK_STAREQ, MTK_SLASHEQ, MTK_PERCENTEQ,
    MTK_AMPEQ, MTK_PIPEEQ, MTK_CARETEQ,
    MTK_COLONCOLON,
    MTK_PLUSPLUS, MTK_MINUSMINUS,
} MslTokType;

typedef struct {
    MslTokType type;
    const char *start;
    int length;
    int line, col;
    bool is_unsigned; /* for int literals ending with 'u' */
} MslToken;

/* ============================================================================
 * Lexer
 * ============================================================================ */

typedef struct {
    const char *src;
    size_t pos;
    int line, col;
} MslLexer;

//L nonnull
//src nonnull
static void mlx_init(MslLexer *L, const char *src) {
    wgsl_compiler_assert(L != NULL, "mlx_init: L is NULL");
    wgsl_compiler_assert(src != NULL, "mlx_init: src is NULL");
    L->src = src;
    L->pos = 0;
    L->line = 1;
    L->col = 1;
}

//L nonnull
static char mlx_peek(const MslLexer *L) { wgsl_compiler_assert(L != NULL, "mlx_peek: L is NULL"); return L->src[L->pos]; }
//L nonnull
static char mlx_peek2(const MslLexer *L) { wgsl_compiler_assert(L != NULL, "mlx_peek2: L is NULL"); return L->src[L->pos + 1]; }

//L nonnull
static void mlx_advance(MslLexer *L) {
    wgsl_compiler_assert(L != NULL, "mlx_advance: L is NULL");
    if (!L->src[L->pos]) return;
    if (L->src[L->pos] == '\n') { L->line++; L->col = 1; }
    else { L->col++; }
    L->pos++;
}

//L nonnull
static void mlx_skip_ws(MslLexer *L) {
    wgsl_compiler_assert(L != NULL, "mlx_skip_ws: L is NULL");
    for (;;) {
        char c = mlx_peek(L);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            mlx_advance(L);
            continue;
        }
        if (c == '/' && mlx_peek2(L) == '/') {
            while (mlx_peek(L) && mlx_peek(L) != '\n') mlx_advance(L);
            continue;
        }
        if (c == '/' && mlx_peek2(L) == '*') {
            mlx_advance(L); mlx_advance(L);
            while (mlx_peek(L)) {
                if (mlx_peek(L) == '*' && mlx_peek2(L) == '/') {
                    mlx_advance(L); mlx_advance(L);
                    break;
                }
                mlx_advance(L);
            }
            continue;
        }
        if (c == '#') {
            while (mlx_peek(L) && mlx_peek(L) != '\n') mlx_advance(L);
            continue;
        }
        break;
    }
}

//L nonnull
//s nonnull
static MslToken mlx_make(MslLexer *L, MslTokType t, const char *s, int len) {
    wgsl_compiler_assert(L != NULL, "mlx_make: L is NULL");
    wgsl_compiler_assert(s != NULL, "mlx_make: s is NULL");
    MslToken tok;
    tok.type = t;
    tok.start = s;
    tok.length = len;
    tok.line = L->line;
    tok.col = L->col;
    tok.is_unsigned = false;
    return tok;
}

//L nonnull
static MslToken mlx_next(MslLexer *L) {
    wgsl_compiler_assert(L != NULL, "mlx_next: L is NULL");
    mlx_skip_ws(L);
    const char *s = L->src + L->pos;
    int sl = L->line, sc = L->col;
    char c = mlx_peek(L);

    if (!c) return mlx_make(L, MTK_EOF, s, 0);

    /* Numbers */
    if (isdigit((unsigned char)c) || (c == '.' && isdigit((unsigned char)mlx_peek2(L)))) {
        bool is_float = false;
        const char *start = s;
        if (c == '0' && (mlx_peek2(L) == 'x' || mlx_peek2(L) == 'X')) {
            mlx_advance(L); mlx_advance(L);
            while (isxdigit((unsigned char)mlx_peek(L))) mlx_advance(L);
        } else {
            while (isdigit((unsigned char)mlx_peek(L))) mlx_advance(L);
            if (mlx_peek(L) == '.' && isdigit((unsigned char)L->src[L->pos + 1])) {
                is_float = true;
                mlx_advance(L);
                while (isdigit((unsigned char)mlx_peek(L))) mlx_advance(L);
            } else if (mlx_peek(L) == '.') {
                is_float = true;
                mlx_advance(L);
            }
            if (mlx_peek(L) == 'e' || mlx_peek(L) == 'E') {
                is_float = true;
                mlx_advance(L);
                if (mlx_peek(L) == '+' || mlx_peek(L) == '-') mlx_advance(L);
                while (isdigit((unsigned char)mlx_peek(L))) mlx_advance(L);
            }
        }
        if (is_float && (mlx_peek(L) == 'f' || mlx_peek(L) == 'h')) mlx_advance(L);
        MslToken tok = mlx_make(L, is_float ? MTK_FLOAT_LIT : MTK_INT_LIT, start, (int)(L->src + L->pos - start));
        tok.line = sl; tok.col = sc;
        if (!is_float && mlx_peek(L) == 'u') {
            tok.is_unsigned = true;
            mlx_advance(L);
            tok.length = (int)(L->src + L->pos - start);
        }
        return tok;
    }

    /* Identifiers/keywords */
    if (isalpha((unsigned char)c) || c == '_') {
        const char *start = s;
        while (isalnum((unsigned char)mlx_peek(L)) || mlx_peek(L) == '_') mlx_advance(L);
        MslToken tok = mlx_make(L, MTK_IDENT, start, (int)(L->src + L->pos - start));
        tok.line = sl; tok.col = sc;
        return tok;
    }

    /* Multi-char operators */
    mlx_advance(L);
    switch (c) {
    case ';': return mlx_make(L, MTK_SEMI, s, 1);
    case ',': return mlx_make(L, MTK_COMMA, s, 1);
    case '.': return mlx_make(L, MTK_DOT, s, 1);
    case '{': return mlx_make(L, MTK_LBRACE, s, 1);
    case '}': return mlx_make(L, MTK_RBRACE, s, 1);
    case '(': return mlx_make(L, MTK_LPAREN, s, 1);
    case ')': return mlx_make(L, MTK_RPAREN, s, 1);
    case '[':
        if (mlx_peek(L) == '[') { mlx_advance(L); return mlx_make(L, MTK_LATTR, s, 2); }
        return mlx_make(L, MTK_LBRACKET, s, 1);
    case ']':
        if (mlx_peek(L) == ']') { mlx_advance(L); return mlx_make(L, MTK_RATTR, s, 2); }
        return mlx_make(L, MTK_RBRACKET, s, 1);
    case '~': return mlx_make(L, MTK_TILDE, s, 1);
    case '?': return mlx_make(L, MTK_QMARK, s, 1);
    case '*':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_STAREQ, s, 2); }
        return mlx_make(L, MTK_STAR, s, 1);
    case '/':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_SLASHEQ, s, 2); }
        return mlx_make(L, MTK_SLASH, s, 1);
    case '%':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_PERCENTEQ, s, 2); }
        return mlx_make(L, MTK_PERCENT, s, 1);
    case '+':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_PLUSEQ, s, 2); }
        if (mlx_peek(L) == '+') { mlx_advance(L); return mlx_make(L, MTK_PLUSPLUS, s, 2); }
        return mlx_make(L, MTK_PLUS, s, 1);
    case '-':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_MINUSEQ, s, 2); }
        if (mlx_peek(L) == '-') { mlx_advance(L); return mlx_make(L, MTK_MINUSMINUS, s, 2); }
        return mlx_make(L, MTK_MINUS, s, 1);
    case '=':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_EQEQ, s, 2); }
        return mlx_make(L, MTK_EQ, s, 1);
    case '!':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_NEQ, s, 2); }
        return mlx_make(L, MTK_BANG, s, 1);
    case '<':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_LE, s, 2); }
        if (mlx_peek(L) == '<') { mlx_advance(L); return mlx_make(L, MTK_SHL, s, 2); }
        return mlx_make(L, MTK_LT, s, 1);
    case '>':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_GE, s, 2); }
        if (mlx_peek(L) == '>') { mlx_advance(L); return mlx_make(L, MTK_SHR, s, 2); }
        return mlx_make(L, MTK_GT, s, 1);
    case '&':
        if (mlx_peek(L) == '&') { mlx_advance(L); return mlx_make(L, MTK_ANDAND, s, 2); }
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_AMPEQ, s, 2); }
        return mlx_make(L, MTK_AMP, s, 1);
    case '|':
        if (mlx_peek(L) == '|') { mlx_advance(L); return mlx_make(L, MTK_OROR, s, 2); }
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_PIPEEQ, s, 2); }
        return mlx_make(L, MTK_PIPE, s, 1);
    case '^':
        if (mlx_peek(L) == '=') { mlx_advance(L); return mlx_make(L, MTK_CARETEQ, s, 2); }
        return mlx_make(L, MTK_CARET, s, 1);
    case ':':
        if (mlx_peek(L) == ':') { mlx_advance(L); return mlx_make(L, MTK_COLONCOLON, s, 2); }
        return mlx_make(L, MTK_COLON, s, 1);
    }

    return mlx_make(L, MTK_EOF, s, 1);
}

/* ============================================================================
 * Parser Structures
 * ============================================================================ */

/* Expression result: either a value or a pointer to a value */
typedef struct {
    uint32_t id;        /* SSIR value or pointer ID */
    uint32_t val_type;  /* SSIR type of the underlying value */
    bool is_ptr;        /* if true, id is a pointer; needs load for value */
} MslVal;

/* Local symbol (variable, parameter, or mapped global) */
typedef struct {
    char *name;
    uint32_t id;
    uint32_t type_id;   /* pointer type for locals/globals */
    uint32_t val_type;  /* value (pointee) type */
    bool is_out_var;    /* _out magic variable */
    bool is_in_var;     /* _in magic variable */
} MslSym;

#define MSL_SENTINEL_OUT  UINT32_MAX
#define MSL_SENTINEL_IN   (UINT32_MAX - 1)

/* Struct member info */
typedef struct {
    char *name;
    uint32_t ssir_type;
    SsirBuiltinVar builtin;
    bool has_location;
    uint32_t location;
    bool is_flat;
} MslStructMember;

/* Parsed struct definition */
typedef struct {
    char *name;
    MslStructMember *members;
    int member_count;
    uint32_t ssir_struct_type; /* 0 if not yet created */
    bool is_interface;         /* true if this is an entry-point I/O struct (has MSL attrs) */
} MslStructDef;

/* Main parser state */
typedef struct {
    MslLexer lex;
    MslToken cur;
    SsirModule *mod;
    MslToSsirOptions opts;

    /* Current function context */
    uint32_t func_id;
    uint32_t block_id;
    int is_entry;
    SsirStage entry_stage;
    uint32_t ep_index;

    /* Struct registry */
    MslStructDef *structs;
    int struct_count, struct_cap;

    /* Symbol table (per-function) */
    MslSym *syms;
    int sym_count, sym_cap;

    /* Entry point I/O globals */
    MslStructDef *out_struct;
    uint32_t *out_globals;    /* parallel to out_struct members */
    MslStructDef *in_struct;
    uint32_t *in_globals;     /* parallel to in_struct members */

    /* Interface global list for entry point */
    uint32_t *iface;
    int iface_count, iface_cap;

    /* Workgroup size (from attribute, if any) */
    uint32_t wg_size[3];

    /* Error */
    char error[512];
    int had_error;
} MslParser;

/* ============================================================================
 * Parser Helpers
 * ============================================================================ */

//p nonnull
//fmt nonnull
static void mp_error(MslParser *p, const char *fmt, ...) {
    wgsl_compiler_assert(p != NULL, "mp_error: p is NULL");
    wgsl_compiler_assert(fmt != NULL, "mp_error: fmt is NULL");
    if (p->had_error) return;
    p->had_error = 1;
    va_list a;
    va_start(a, fmt);
    vsnprintf(p->error, sizeof(p->error), fmt, a);
    va_end(a);
}

//p nonnull
static void mp_next(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_next: p is NULL");
    p->cur = mlx_next(&p->lex);
}

//p nonnull
static bool mp_check(MslParser *p, MslTokType t) {
    wgsl_compiler_assert(p != NULL, "mp_check: p is NULL");
    return p->cur.type == t;
}

//p nonnull
//kw nonnull
static bool mp_match_ident(MslParser *p, const char *kw) {
    wgsl_compiler_assert(p != NULL, "mp_match_ident: p is NULL");
    wgsl_compiler_assert(kw != NULL, "mp_match_ident: kw is NULL");
    return p->cur.type == MTK_IDENT &&
           p->cur.length == (int)strlen(kw) &&
           strncmp(p->cur.start, kw, p->cur.length) == 0;
}

//p nonnull
static char *mp_tok_str(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_tok_str: p is NULL");
    return msl_strndup(p->cur.start, p->cur.length);
}

//p nonnull
static bool mp_eat(MslParser *p, MslTokType t) {
    wgsl_compiler_assert(p != NULL, "mp_eat: p is NULL");
    if (p->cur.type != t) return false;
    mp_next(p);
    return true;
}

//p nonnull
static void mp_expect(MslParser *p, MslTokType t) {
    wgsl_compiler_assert(p != NULL, "mp_expect: p is NULL");
    if (!mp_eat(p, t))
        mp_error(p, "line %d: expected token type %d, got %d", p->cur.line, t, p->cur.type);
}

/* Add an interface global to current entry point */
//p nonnull
static void mp_add_iface(MslParser *p, uint32_t gid) {
    wgsl_compiler_assert(p != NULL, "mp_add_iface: p is NULL");
    if (p->iface_count >= p->iface_cap) {
        p->iface_cap = p->iface_cap ? p->iface_cap * 2 : 16;
        p->iface = (uint32_t *)MSL_REALLOC(p->iface, p->iface_cap * sizeof(uint32_t));
    }
    p->iface[p->iface_count++] = gid;
}

/* Add a symbol to the current function scope */
//p nonnull
//name nonnull
static void mp_add_sym(MslParser *p, const char *name, uint32_t id,
                        uint32_t type_id, uint32_t val_type,
                        bool is_out, bool is_in) {
    wgsl_compiler_assert(p != NULL, "mp_add_sym: p is NULL");
    wgsl_compiler_assert(name != NULL, "mp_add_sym: name is NULL");
    if (p->sym_count >= p->sym_cap) {
        p->sym_cap = p->sym_cap ? p->sym_cap * 2 : 32;
        p->syms = (MslSym *)MSL_REALLOC(p->syms, p->sym_cap * sizeof(MslSym));
    }
    MslSym *s = &p->syms[p->sym_count++];
    s->name = msl_strdup(name);
    s->id = id;
    s->type_id = type_id;
    s->val_type = val_type;
    s->is_out_var = is_out;
    s->is_in_var = is_in;
}

//p nonnull
//name nonnull
static MslSym *mp_find_sym(MslParser *p, const char *name) {
    wgsl_compiler_assert(p != NULL, "mp_find_sym: p is NULL");
    wgsl_compiler_assert(name != NULL, "mp_find_sym: name is NULL");
    for (int i = p->sym_count - 1; i >= 0; i--) {
        if (strcmp(p->syms[i].name, name) == 0) return &p->syms[i];
    }
    return NULL;
}

/* Clear per-function state */
//p nonnull
static void mp_clear_func(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_clear_func: p is NULL");
    for (int i = 0; i < p->sym_count; i++) MSL_FREE(p->syms[i].name);
    p->sym_count = 0;
    p->func_id = 0;
    p->block_id = 0;
    p->is_entry = 0;
    p->out_struct = NULL;
    p->in_struct = NULL;
    MSL_FREE(p->out_globals); p->out_globals = NULL;
    MSL_FREE(p->in_globals); p->in_globals = NULL;
    p->iface_count = 0;
}

/* ============================================================================
 * Struct Registry
 * ============================================================================ */

//p nonnull
//name nonnull
static MslStructDef *mp_find_struct(MslParser *p, const char *name) {
    wgsl_compiler_assert(p != NULL, "mp_find_struct: p is NULL");
    wgsl_compiler_assert(name != NULL, "mp_find_struct: name is NULL");
    for (int i = 0; i < p->struct_count; i++) {
        if (strcmp(p->structs[i].name, name) == 0) return &p->structs[i];
    }
    return NULL;
}

//p nonnull
//name nonnull
static MslStructDef *mp_add_struct(MslParser *p, const char *name) {
    wgsl_compiler_assert(p != NULL, "mp_add_struct: p is NULL");
    wgsl_compiler_assert(name != NULL, "mp_add_struct: name is NULL");
    if (p->struct_count >= p->struct_cap) {
        p->struct_cap = p->struct_cap ? p->struct_cap * 2 : 8;
        p->structs = (MslStructDef *)MSL_REALLOC(p->structs, p->struct_cap * sizeof(MslStructDef));
    }
    MslStructDef *sd = &p->structs[p->struct_count++];
    memset(sd, 0, sizeof(*sd));
    sd->name = msl_strdup(name);
    return sd;
}

//sd nonnull
//name nonnull
static void mp_struct_add_member(MslStructDef *sd, const char *name, uint32_t type,
                                  SsirBuiltinVar builtin, bool has_loc, uint32_t loc, bool flat) {
    wgsl_compiler_assert(sd != NULL, "mp_struct_add_member: sd is NULL");
    wgsl_compiler_assert(name != NULL, "mp_struct_add_member: name is NULL");
    int idx = sd->member_count++;
    sd->members = (MslStructMember *)MSL_REALLOC(sd->members, sd->member_count * sizeof(MslStructMember));
    MslStructMember *m = &sd->members[idx];
    m->name = msl_strdup(name);
    m->ssir_type = type;
    m->builtin = builtin;
    m->has_location = has_loc;
    m->location = loc;
    m->is_flat = flat;
}

/* ============================================================================
 * MSL Type Parsing
 * ============================================================================ */

/* Forward declaration */
static uint32_t mp_parse_type(MslParser *p);

/* Check if current identifier is a type name */
//p nonnull
static bool mp_is_type_name(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_is_type_name: p is NULL");
    if (p->cur.type != MTK_IDENT) return false;
    const char *s = p->cur.start;
    int n = p->cur.length;

    /* Scalar types */
    if ((n == 4 && strncmp(s, "void", 4) == 0) ||
        (n == 5 && strncmp(s, "float", 5) == 0) ||
        (n == 4 && strncmp(s, "half", 4) == 0) ||
        (n == 3 && strncmp(s, "int", 3) == 0) ||
        (n == 4 && strncmp(s, "uint", 4) == 0) ||
        (n == 4 && strncmp(s, "bool", 4) == 0))
        return true;

    /* Vector types: float2..4, half2..4, int2..4, uint2..4, bool2..4 */
    if (n >= 5 && n <= 6) {
        if ((strncmp(s, "float", 5) == 0 || strncmp(s, "uchar", 5) == 0) &&
            s[5] >= '2' && s[5] <= '4') return true;
    }
    if (n == 5 && strncmp(s, "half", 4) == 0 && s[4] >= '2' && s[4] <= '4') return true;
    if (n == 4 && strncmp(s, "int", 3) == 0 && s[3] >= '2' && s[3] <= '4') return true;
    if (n == 5 && strncmp(s, "uint", 4) == 0 && s[4] >= '2' && s[4] <= '4') return true;
    if (n == 5 && strncmp(s, "bool", 4) == 0 && s[4] >= '2' && s[4] <= '4') return true;

    /* Matrix types: float2x2..4x4, half2x2..4x4 */
    if (n == 9 && strncmp(s, "float", 5) == 0 && s[6] == 'x') return true;
    if (n == 8 && strncmp(s, "half", 4) == 0 && s[5] == 'x') return true;

    /* Texture/sampler types */
    if (n == 7 && strncmp(s, "sampler", 7) == 0) return true;
    if (n >= 9 && strncmp(s, "texture", 7) == 0) return true;
    if (n >= 7 && strncmp(s, "depth", 5) == 0) return true;

    /* array<...> */
    if (n == 5 && strncmp(s, "array", 5) == 0) return true;

    /* Known struct name */
    char *name = msl_strndup(s, n);
    MslStructDef *sd = mp_find_struct(p, name);
    MSL_FREE(name);
    if (sd) return true;

    return false;
}

/* Parse a base scalar/vector/matrix/struct type, returns SSIR type ID */
//p nonnull
static uint32_t mp_parse_type(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_type: p is NULL");
    if (p->cur.type != MTK_IDENT) {
        mp_error(p, "line %d: expected type name", p->cur.line);
        return 0;
    }

    char *name = mp_tok_str(p);
    mp_next(p);

    /* void */
    if (strcmp(name, "void") == 0) { MSL_FREE(name); return ssir_type_void(p->mod); }

    /* Scalars */
    if (strcmp(name, "float") == 0) { MSL_FREE(name); return ssir_type_f32(p->mod); }
    if (strcmp(name, "half") == 0) { MSL_FREE(name); return ssir_type_f16(p->mod); }
    if (strcmp(name, "int") == 0) { MSL_FREE(name); return ssir_type_i32(p->mod); }
    if (strcmp(name, "uint") == 0) { MSL_FREE(name); return ssir_type_u32(p->mod); }
    if (strcmp(name, "bool") == 0) { MSL_FREE(name); return ssir_type_bool(p->mod); }

    /* Vectors: float2..4, half2..4, int2..4, uint2..4 */
    if (strlen(name) >= 5 && strncmp(name, "float", 5) == 0 && name[5] >= '2' && name[5] <= '4' && name[6] == '\0') {
        int sz = name[5] - '0';
        MSL_FREE(name);
        return ssir_type_vec(p->mod, ssir_type_f32(p->mod), sz);
    }
    if (strlen(name) >= 5 && strncmp(name, "half", 4) == 0 && name[4] >= '2' && name[4] <= '4' && name[5] == '\0') {
        int sz = name[4] - '0';
        MSL_FREE(name);
        return ssir_type_vec(p->mod, ssir_type_f16(p->mod), sz);
    }
    if (strlen(name) >= 4 && strncmp(name, "int", 3) == 0 && name[3] >= '2' && name[3] <= '4' && name[4] == '\0') {
        int sz = name[3] - '0';
        MSL_FREE(name);
        return ssir_type_vec(p->mod, ssir_type_i32(p->mod), sz);
    }
    if (strlen(name) >= 5 && strncmp(name, "uint", 4) == 0 && name[4] >= '2' && name[4] <= '4' && name[5] == '\0') {
        int sz = name[4] - '0';
        MSL_FREE(name);
        return ssir_type_vec(p->mod, ssir_type_u32(p->mod), sz);
    }
    if (strlen(name) >= 5 && strncmp(name, "bool", 4) == 0 && name[4] >= '2' && name[4] <= '4' && name[5] == '\0') {
        int sz = name[4] - '0';
        MSL_FREE(name);
        return ssir_type_vec(p->mod, ssir_type_bool(p->mod), sz);
    }

    /* Matrices: float2x2..4x4, half2x2..4x4 */
    if (strlen(name) == 9 && strncmp(name, "float", 5) == 0 && name[6] == 'x') {
        int cols = name[5] - '0', rows = name[7] - '0';
        uint32_t col_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), rows);
        MSL_FREE(name);
        return ssir_type_mat(p->mod, col_type, cols, rows);
    }
    if (strlen(name) == 8 && strncmp(name, "half", 4) == 0 && name[5] == 'x') {
        int cols = name[4] - '0', rows = name[6] - '0';
        uint32_t col_type = ssir_type_vec(p->mod, ssir_type_f16(p->mod), rows);
        MSL_FREE(name);
        return ssir_type_mat(p->mod, col_type, cols, rows);
    }

    /* sampler */
    if (strcmp(name, "sampler") == 0) { MSL_FREE(name); return ssir_type_sampler(p->mod); }

    /* texture2d<T>, texture1d<T>, etc. */
    if (strncmp(name, "texture", 7) == 0 || strncmp(name, "depth", 5) == 0) {
        SsirTextureDim dim = SSIR_TEX_2D;
        bool is_depth = (name[0] == 'd');
        if (!is_depth) {
            if (strstr(name, "1d")) dim = SSIR_TEX_1D;
            else if (strstr(name, "3d")) dim = SSIR_TEX_3D;
            else if (strstr(name, "cube_array")) dim = SSIR_TEX_CUBE_ARRAY;
            else if (strstr(name, "cube")) dim = SSIR_TEX_CUBE;
            else if (strstr(name, "2d_array")) dim = SSIR_TEX_2D_ARRAY;
            else if (strstr(name, "2d_ms")) dim = SSIR_TEX_MULTISAMPLED_2D;
        } else {
            if (strstr(name, "cube")) dim = SSIR_TEX_CUBE;
            else if (strstr(name, "2d_array")) dim = SSIR_TEX_2D_ARRAY;
        }
        MSL_FREE(name);

        /* Parse template: <float> or <float, access::read> */
        uint32_t sampled_type = ssir_type_f32(p->mod);
        SsirAccessMode access = SSIR_ACCESS_READ;
        bool has_access = false;
        if (mp_eat(p, MTK_LT)) {
            sampled_type = mp_parse_type(p);
            if (mp_eat(p, MTK_COMMA)) {
                /* access::read/write/read_write */
                mp_next(p); /* skip 'access' */
                mp_eat(p, MTK_COLONCOLON);
                char *acc = mp_tok_str(p);
                if (strcmp(acc, "write") == 0) access = SSIR_ACCESS_WRITE;
                else if (strcmp(acc, "read_write") == 0) access = SSIR_ACCESS_READ_WRITE;
                MSL_FREE(acc);
                mp_next(p);
                has_access = true;
            }
            mp_eat(p, MTK_GT);
        }

        if (is_depth) return ssir_type_texture_depth(p->mod, dim);
        if (has_access) return ssir_type_texture_storage(p->mod, dim, 0, access);
        return ssir_type_texture(p->mod, dim, sampled_type);
    }

    /* array<T, N> */
    if (strcmp(name, "array") == 0) {
        MSL_FREE(name);
        mp_expect(p, MTK_LT);
        uint32_t elem = mp_parse_type(p);
        mp_expect(p, MTK_COMMA);
        char *len_str = mp_tok_str(p);
        uint32_t len = (uint32_t)atoi(len_str);
        MSL_FREE(len_str);
        mp_next(p);
        mp_eat(p, MTK_GT);
        return ssir_type_array(p->mod, elem, len);
    }

    /* Struct name */
    MslStructDef *sd = mp_find_struct(p, name);
    if (sd) {
        if (sd->is_interface) {
            /* Interface structs (with MSL attributes) are reconstructed from
             * entry-point globals by the MSL backend; don't create a redundant
             * SSIR struct type that would emit a duplicate definition. */
            MSL_FREE(name);
            return 0;
        }
        if (sd->ssir_struct_type == 0) {
            /* Create the SSIR struct type */
            uint32_t *mtypes = NULL;
            if (sd->member_count > 0) {
                mtypes = (uint32_t *)MSL_MALLOC(sd->member_count * sizeof(uint32_t));
                for (int i = 0; i < sd->member_count; i++) mtypes[i] = sd->members[i].ssir_type;
            }
            sd->ssir_struct_type = ssir_type_struct(p->mod, sd->name, mtypes, sd->member_count, NULL);
            MSL_FREE(mtypes);
        }
        MSL_FREE(name);
        return sd->ssir_struct_type;
    }

    /* Unknown type - treat as opaque */
    mp_error(p, "line %d: unknown type '%s'", p->cur.line, name);
    MSL_FREE(name);
    return ssir_type_f32(p->mod);
}

/* ============================================================================
 * Attribute Parsing: [[ attr ]]
 * ============================================================================ */

typedef struct {
    SsirBuiltinVar builtin;
    bool has_location;
    uint32_t location;
    bool has_binding;
    uint32_t binding;
    bool has_group;
    uint32_t group;
    bool is_stage_in;
    bool is_flat;
    bool has_color;
    uint32_t color;
} MslAttr;

//attr nonnull
static SsirBuiltinVar msl_attr_to_builtin(const char *attr) {
    wgsl_compiler_assert(attr != NULL, "msl_attr_to_builtin: attr is NULL");
    if (strcmp(attr, "position") == 0) return SSIR_BUILTIN_POSITION;
    if (strcmp(attr, "vertex_id") == 0) return SSIR_BUILTIN_VERTEX_INDEX;
    if (strcmp(attr, "instance_id") == 0) return SSIR_BUILTIN_INSTANCE_INDEX;
    if (strcmp(attr, "front_facing") == 0) return SSIR_BUILTIN_FRONT_FACING;
    if (strcmp(attr, "sample_id") == 0) return SSIR_BUILTIN_SAMPLE_INDEX;
    if (strcmp(attr, "sample_mask") == 0) return SSIR_BUILTIN_SAMPLE_MASK;
    if (strcmp(attr, "thread_position_in_grid") == 0) return SSIR_BUILTIN_GLOBAL_INVOCATION_ID;
    if (strcmp(attr, "thread_position_in_threadgroup") == 0) return SSIR_BUILTIN_LOCAL_INVOCATION_ID;
    if (strcmp(attr, "thread_index_in_threadgroup") == 0) return SSIR_BUILTIN_LOCAL_INVOCATION_INDEX;
    if (strcmp(attr, "threadgroup_position_in_grid") == 0) return SSIR_BUILTIN_WORKGROUP_ID;
    if (strcmp(attr, "threadgroups_per_grid") == 0) return SSIR_BUILTIN_NUM_WORKGROUPS;
    return SSIR_BUILTIN_NONE;
}

//p nonnull
static MslAttr mp_parse_attrs(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_attrs: p is NULL");
    MslAttr a;
    memset(&a, 0, sizeof(a));

    while (mp_check(p, MTK_LATTR)) {
        mp_next(p); /* skip [[ */
        char *attr_name = mp_tok_str(p);
        mp_next(p);

        if (strcmp(attr_name, "buffer") == 0 || strcmp(attr_name, "texture") == 0 ||
            strcmp(attr_name, "sampler") == 0) {
            bool is_buffer = (strcmp(attr_name, "buffer") == 0);
            bool is_texture = (strcmp(attr_name, "texture") == 0);
            (void)is_texture;
            mp_expect(p, MTK_LPAREN);
            char *num = mp_tok_str(p);
            uint32_t n = (uint32_t)atoi(num);
            MSL_FREE(num);
            mp_next(p);
            mp_expect(p, MTK_RPAREN);
            a.has_binding = true;
            a.binding = n;
            if (is_buffer) { a.has_group = true; a.group = 0; }
        } else if (strcmp(attr_name, "color") == 0) {
            mp_expect(p, MTK_LPAREN);
            char *num = mp_tok_str(p);
            a.has_color = true;
            a.color = (uint32_t)atoi(num);
            a.has_location = true;
            a.location = a.color;
            MSL_FREE(num);
            mp_next(p);
            mp_expect(p, MTK_RPAREN);
        } else if (strcmp(attr_name, "user") == 0) {
            mp_expect(p, MTK_LPAREN);
            char *loc_ident = mp_tok_str(p);
            /* loc_N format */
            if (strncmp(loc_ident, "loc_", 4) == 0) {
                a.has_location = true;
                a.location = (uint32_t)atoi(loc_ident + 4);
            }
            MSL_FREE(loc_ident);
            mp_next(p);
            mp_expect(p, MTK_RPAREN);
        } else if (strcmp(attr_name, "attribute") == 0) {
            mp_expect(p, MTK_LPAREN);
            char *num = mp_tok_str(p);
            a.has_location = true;
            a.location = (uint32_t)atoi(num);
            MSL_FREE(num);
            mp_next(p);
            mp_expect(p, MTK_RPAREN);
        } else if (strcmp(attr_name, "stage_in") == 0) {
            a.is_stage_in = true;
        } else if (strcmp(attr_name, "flat") == 0) {
            a.is_flat = true;
        } else if (strcmp(attr_name, "depth") == 0) {
            /* [[depth(any)]] */
            if (mp_eat(p, MTK_LPAREN)) {
                mp_next(p); /* skip 'any' or 'greater' etc */
                mp_expect(p, MTK_RPAREN);
            }
            a.builtin = SSIR_BUILTIN_FRAG_DEPTH;
        } else {
            a.builtin = msl_attr_to_builtin(attr_name);
        }
        MSL_FREE(attr_name);
        mp_expect(p, MTK_RATTR); /* ]] */
    }
    return a;
}

/* ============================================================================
 * Struct Definition Parsing
 * ============================================================================ */

//p nonnull
static void mp_parse_struct(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_struct: p is NULL");
    mp_next(p); /* skip 'struct' */
    char *name = mp_tok_str(p);
    mp_next(p);
    mp_expect(p, MTK_LBRACE);

    MslStructDef *sd = mp_add_struct(p, name);
    MSL_FREE(name);

    while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error) {
        uint32_t mtype = mp_parse_type(p);
        char *mname = mp_tok_str(p);
        mp_next(p);
        MslAttr attr = mp_parse_attrs(p);
        mp_expect(p, MTK_SEMI);

        /* If any member has a builtin or location attribute, this is an interface struct */
        if (attr.builtin != SSIR_BUILTIN_NONE || attr.has_location || attr.is_flat)
            sd->is_interface = true;

        mp_struct_add_member(sd, mname, mtype, attr.builtin,
                             attr.has_location, attr.location, attr.is_flat);
        MSL_FREE(mname);
    }
    mp_expect(p, MTK_RBRACE);
    mp_expect(p, MTK_SEMI);
}

/* ============================================================================
 * Value helpers
 * ============================================================================ */

/* Get the pointee type from a pointer SSIR type */
//p nonnull
static uint32_t mp_pointee_type(MslParser *p, uint32_t ptr_type_id) {
    wgsl_compiler_assert(p != NULL, "mp_pointee_type: p is NULL");
    SsirType *t = ssir_get_type(p->mod, ptr_type_id);
    if (t && t->kind == SSIR_TYPE_PTR) return t->ptr.pointee;
    return ptr_type_id;
}

/* Get the address space from a pointer SSIR type */
//p nonnull
static SsirAddressSpace mp_ptr_space(MslParser *p, uint32_t ptr_type_id) {
    wgsl_compiler_assert(p != NULL, "mp_ptr_space: p is NULL");
    SsirType *t = ssir_get_type(p->mod, ptr_type_id);
    if (t && t->kind == SSIR_TYPE_PTR) return t->ptr.space;
    return SSIR_ADDR_FUNCTION;
}

/* Ensure a value (load from pointer if needed) */
//p nonnull
static MslVal mp_ensure_val(MslParser *p, MslVal v) {
    wgsl_compiler_assert(p != NULL, "mp_ensure_val: p is NULL");
    if (v.is_ptr && p->func_id && p->block_id) {
        uint32_t loaded = ssir_build_load(p->mod, p->func_id, p->block_id, v.val_type, v.id);
        v.id = loaded;
        v.is_ptr = false;
    }
    return v;
}

/* Get the element type when indexing into a container type */
//p nonnull
static uint32_t mp_element_type(MslParser *p, uint32_t container_type) {
    wgsl_compiler_assert(p != NULL, "mp_element_type: p is NULL");
    SsirType *t = ssir_get_type(p->mod, container_type);
    if (!t) return container_type;
    switch (t->kind) {
    case SSIR_TYPE_RUNTIME_ARRAY: return t->runtime_array.elem;
    case SSIR_TYPE_ARRAY: return t->array.elem;
    case SSIR_TYPE_VEC: return t->vec.elem;
    default: return container_type;
    }
}

/* ============================================================================
 * Expression Parsing
 * ============================================================================ */

static MslVal mp_parse_expr(MslParser *p);
static MslVal mp_parse_assign_expr(MslParser *p);

/* Parse primary expression */
//p nonnull
static MslVal mp_parse_primary(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_primary: p is NULL");
    MslVal v = {0, 0, false};

    /* Parenthesized expression */
    if (mp_check(p, MTK_LPAREN)) {
        mp_next(p);
        v = mp_parse_expr(p);
        mp_expect(p, MTK_RPAREN);
        return v;
    }

    /* Numeric literals */
    if (mp_check(p, MTK_INT_LIT)) {
        char *s = mp_tok_str(p);
        bool is_u = p->cur.is_unsigned;
        mp_next(p);
        if (is_u) {
            /* Remove trailing 'u' for parsing */
            size_t len = strlen(s);
            if (len > 0 && s[len-1] == 'u') s[len-1] = '\0';
            uint32_t val = (uint32_t)strtoul(s, NULL, 0);
            v.id = ssir_const_u32(p->mod, val);
            v.val_type = ssir_type_u32(p->mod);
        } else {
            int32_t val = (int32_t)strtol(s, NULL, 0);
            v.id = ssir_const_i32(p->mod, val);
            v.val_type = ssir_type_i32(p->mod);
        }
        MSL_FREE(s);
        return v;
    }

    if (mp_check(p, MTK_FLOAT_LIT)) {
        char *s = mp_tok_str(p);
        mp_next(p);
        float val = (float)atof(s);
        v.id = ssir_const_f32(p->mod, val);
        v.val_type = ssir_type_f32(p->mod);
        MSL_FREE(s);
        return v;
    }

    /* Boolean literals */
    if (mp_match_ident(p, "true")) {
        mp_next(p);
        v.id = ssir_const_bool(p->mod, true);
        v.val_type = ssir_type_bool(p->mod);
        return v;
    }
    if (mp_match_ident(p, "false")) {
        mp_next(p);
        v.id = ssir_const_bool(p->mod, false);
        v.val_type = ssir_type_bool(p->mod);
        return v;
    }

    /* static_cast<T>(expr) */
    if (mp_match_ident(p, "static_cast")) {
        mp_next(p);
        mp_expect(p, MTK_LT);
        uint32_t target_type = mp_parse_type(p);
        mp_expect(p, MTK_GT);
        mp_expect(p, MTK_LPAREN);
        MslVal arg = mp_parse_expr(p);
        arg = mp_ensure_val(p, arg);
        mp_expect(p, MTK_RPAREN);
        v.id = ssir_build_convert(p->mod, p->func_id, p->block_id, target_type, arg.id);
        v.val_type = target_type;
        return v;
    }

    /* as_type<T>(expr) */
    if (mp_match_ident(p, "as_type")) {
        mp_next(p);
        mp_expect(p, MTK_LT);
        uint32_t target_type = mp_parse_type(p);
        mp_expect(p, MTK_GT);
        mp_expect(p, MTK_LPAREN);
        MslVal arg = mp_parse_expr(p);
        arg = mp_ensure_val(p, arg);
        mp_expect(p, MTK_RPAREN);
        v.id = ssir_build_bitcast(p->mod, p->func_id, p->block_id, target_type, arg.id);
        v.val_type = target_type;
        return v;
    }

    /* Type constructor or function call or identifier */
    if (mp_check(p, MTK_IDENT)) {
        /* Check if it's a type constructor: float4(...), int(...), etc. */
        if (mp_is_type_name(p)) {
            uint32_t ctor_type = mp_parse_type(p);
            if (mp_check(p, MTK_LPAREN)) {
                /* Type constructor: T(args...) */
                mp_next(p);
                uint32_t args[16];
                int argc = 0;
                while (!mp_check(p, MTK_RPAREN) && !mp_check(p, MTK_EOF) && !p->had_error) {
                    if (argc > 0) mp_expect(p, MTK_COMMA);
                    MslVal arg = mp_parse_assign_expr(p);
                    arg = mp_ensure_val(p, arg);
                    if (argc < 16) args[argc] = arg.id;
                    argc++;
                }
                mp_expect(p, MTK_RPAREN);
                v.id = ssir_build_construct(p->mod, p->func_id, p->block_id, ctor_type, args, argc);
                v.val_type = ctor_type;
                return v;
            }
            /* Type without parens - shouldn't happen in expression context */
            v.val_type = ctor_type;
            return v;
        }

        /* Identifier reference */
        char *name = mp_tok_str(p);
        mp_next(p);

        /* Look up in symbol table */
        MslSym *sym = mp_find_sym(p, name);
        if (sym) {
            v.id = sym->id;
            v.val_type = sym->val_type;
            v.is_ptr = !sym->is_out_var && !sym->is_in_var;
            if (sym->is_out_var || sym->is_in_var) {
                /* _out/_in are magic - they will be handled in postfix .member */
                v.is_ptr = false;
            }
            MSL_FREE(name);
            return v;
        }

        /* Check MSL builtin functions - return as special marker */
        /* These will be handled in the call expression parser */
        mp_error(p, "line %d: undefined identifier '%s'", p->cur.line, name);
        MSL_FREE(name);
        return v;
    }

    mp_error(p, "line %d: unexpected token in expression", p->cur.line);
    return v;
}

/* MSL builtin function name → SSIR builtin ID */
//name nonnull
//out nonnull
static int mp_msl_builtin_func(const char *name, SsirBuiltinId *out) {
    wgsl_compiler_assert(name != NULL, "mp_msl_builtin_func: name is NULL");
    wgsl_compiler_assert(out != NULL, "mp_msl_builtin_func: out is NULL");
    static const struct { const char *name; SsirBuiltinId id; } map[] = {
        {"sin", SSIR_BUILTIN_SIN}, {"cos", SSIR_BUILTIN_COS}, {"tan", SSIR_BUILTIN_TAN},
        {"asin", SSIR_BUILTIN_ASIN}, {"acos", SSIR_BUILTIN_ACOS}, {"atan", SSIR_BUILTIN_ATAN},
        {"atan2", SSIR_BUILTIN_ATAN2}, {"sinh", SSIR_BUILTIN_SINH}, {"cosh", SSIR_BUILTIN_COSH},
        {"tanh", SSIR_BUILTIN_TANH}, {"exp", SSIR_BUILTIN_EXP}, {"exp2", SSIR_BUILTIN_EXP2},
        {"log", SSIR_BUILTIN_LOG}, {"log2", SSIR_BUILTIN_LOG2}, {"pow", SSIR_BUILTIN_POW},
        {"sqrt", SSIR_BUILTIN_SQRT}, {"rsqrt", SSIR_BUILTIN_INVERSESQRT},
        {"abs", SSIR_BUILTIN_ABS}, {"sign", SSIR_BUILTIN_SIGN},
        {"floor", SSIR_BUILTIN_FLOOR}, {"ceil", SSIR_BUILTIN_CEIL},
        {"round", SSIR_BUILTIN_ROUND}, {"trunc", SSIR_BUILTIN_TRUNC},
        {"fract", SSIR_BUILTIN_FRACT}, {"min", SSIR_BUILTIN_MIN}, {"max", SSIR_BUILTIN_MAX},
        {"clamp", SSIR_BUILTIN_CLAMP}, {"saturate", SSIR_BUILTIN_SATURATE},
        {"mix", SSIR_BUILTIN_MIX}, {"step", SSIR_BUILTIN_STEP},
        {"smoothstep", SSIR_BUILTIN_SMOOTHSTEP}, {"dot", SSIR_BUILTIN_DOT},
        {"cross", SSIR_BUILTIN_CROSS}, {"length", SSIR_BUILTIN_LENGTH},
        {"distance", SSIR_BUILTIN_DISTANCE}, {"normalize", SSIR_BUILTIN_NORMALIZE},
        {"faceforward", SSIR_BUILTIN_FACEFORWARD}, {"reflect", SSIR_BUILTIN_REFLECT},
        {"refract", SSIR_BUILTIN_REFRACT}, {"all", SSIR_BUILTIN_ALL}, {"any", SSIR_BUILTIN_ANY},
        {"select", SSIR_BUILTIN_SELECT}, {"popcount", SSIR_BUILTIN_COUNTBITS},
        {"reverse_bits", SSIR_BUILTIN_REVERSEBITS}, {"clz", SSIR_BUILTIN_FIRSTLEADINGBIT},
        {"ctz", SSIR_BUILTIN_FIRSTTRAILINGBIT},
        {"dfdx", SSIR_BUILTIN_DPDX}, {"dfdy", SSIR_BUILTIN_DPDY}, {"fwidth", SSIR_BUILTIN_FWIDTH},
        {NULL, 0}
    };
    for (int i = 0; map[i].name; i++) {
        if (strcmp(name, map[i].name) == 0) { *out = map[i].id; return 1; }
    }
    return 0;
}

/* Swizzle char → index */
static int mp_swizzle_index(char c) {
    switch (c) {
    case 'x': case 'r': return 0;
    case 'y': case 'g': return 1;
    case 'z': case 'b': return 2;
    case 'w': case 'a': return 3;
    default: return -1;
    }
}

/* Parse postfix: .member, [index], (args) */
//p nonnull
static MslVal mp_parse_postfix(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_postfix: p is NULL");
    MslVal v = mp_parse_primary(p);
    if (p->had_error) return v;

    for (;;) {
        /* Member access: .name */
        if (mp_check(p, MTK_DOT)) {
            mp_next(p);
            char *member = mp_tok_str(p);
            mp_next(p);

            /* Check if v is _out/_in → map to output/input global via sentinel IDs */
            if (v.id == MSL_SENTINEL_OUT && p->out_struct) {
                /* PRE: out_globals allocated when out_struct set */
                wgsl_compiler_assert(p->out_globals != NULL, "out_globals NULL with out_struct set");
                for (int i = 0; i < p->out_struct->member_count; i++) {
                    /* PRE: member name not NULL */
                    wgsl_compiler_assert(p->out_struct->members[i].name != NULL, "out_struct member[%d].name NULL", i);
                    if (strcmp(p->out_struct->members[i].name, member) == 0) {
                        v.id = p->out_globals[i];
                        v.val_type = p->out_struct->members[i].ssir_type;
                        v.is_ptr = true;
                        break;
                    }
                }
                MSL_FREE(member);
                continue;
            }

            if (v.id == MSL_SENTINEL_IN && p->in_struct) {
                /* PRE: in_globals allocated when in_struct set */
                wgsl_compiler_assert(p->in_globals != NULL, "in_globals NULL with in_struct set");
                for (int i = 0; i < p->in_struct->member_count; i++) {
                    /* PRE: member name not NULL */
                    wgsl_compiler_assert(p->in_struct->members[i].name != NULL, "in_struct member[%d].name NULL", i);
                    if (strcmp(p->in_struct->members[i].name, member) == 0) {
                        v.id = p->in_globals[i];
                        v.val_type = p->in_struct->members[i].ssir_type;
                        v.is_ptr = true;
                        break;
                    }
                }
                MSL_FREE(member);
                continue;
            }

            /* Vector swizzle: .x, .y, .z, .w (single component) */
            SsirType *vt = ssir_get_type(p->mod, v.val_type);
            if (vt && vt->kind == SSIR_TYPE_VEC && strlen(member) == 1 &&
                mp_swizzle_index(member[0]) >= 0) {
                v = mp_ensure_val(p, v);
                int idx = mp_swizzle_index(member[0]);
                uint32_t elem_type = vt->vec.elem;
                v.id = ssir_build_extract(p->mod, p->func_id, p->block_id, elem_type, v.id, idx);
                v.val_type = elem_type;
                v.is_ptr = false;
                MSL_FREE(member);
                continue;
            }

            /* Multi-component swizzle: .xy, .xyz, etc. (max 4 components) */
            if (vt && vt->kind == SSIR_TYPE_VEC && strlen(member) > 1 &&
                strlen(member) <= 4 && mp_swizzle_index(member[0]) >= 0) {
                v = mp_ensure_val(p, v);
                int sz = (int)strlen(member);
                uint32_t indices[4];
                for (int i = 0; i < sz; i++) indices[i] = (uint32_t)mp_swizzle_index(member[i]);
                uint32_t result_type = ssir_type_vec(p->mod, vt->vec.elem, sz);
                /* Use shuffle with v, v and the swizzle indices */
                v.id = ssir_build_shuffle(p->mod, p->func_id, p->block_id, result_type, v.id, v.id, indices, sz);
                v.val_type = result_type;
                v.is_ptr = false;
                MSL_FREE(member);
                continue;
            }

            /* Struct member access */
            if (vt && vt->kind == SSIR_TYPE_STRUCT) {
                int member_idx = -1;
                if (strncmp(member, "member", 6) == 0 && isdigit((unsigned char)member[6])) {
                    member_idx = atoi(member + 6);
                } else {
                    /* Look up by name in struct registry */
                    for (int i = 0; i < p->struct_count; i++) {
                        if (p->structs[i].ssir_struct_type == v.val_type) {
                            for (int j = 0; j < p->structs[i].member_count; j++) {
                                if (strcmp(p->structs[i].members[j].name, member) == 0) {
                                    member_idx = j;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                if (member_idx >= 0 && (uint32_t)member_idx < vt->struc.member_count) {
                    /* PRE: member_idx in bounds */
                    wgsl_compiler_assert((uint32_t)member_idx < vt->struc.member_count, "member_idx %d out of bounds (count=%u)", member_idx, vt->struc.member_count);
                    uint32_t member_type = vt->struc.members[member_idx];
                    if (v.is_ptr) {
                        /* Access chain through pointer */
                        uint32_t idx_const = ssir_const_u32(p->mod, (uint32_t)member_idx);
                        SsirAddressSpace space = mp_ptr_space(p, v.val_type);
                        /* v.val_type is the struct type, but the pointer info is in the symbol */
                        /* Need to determine the pointer's address space */
                        /* For now, use FUNCTION as default */
                        uint32_t result_ptr = ssir_type_ptr(p->mod, member_type, space);
                        v.id = ssir_build_access(p->mod, p->func_id, p->block_id, result_ptr, v.id, &idx_const, 1);
                        v.val_type = member_type;
                        v.is_ptr = true;
                    } else {
                        v.id = ssir_build_extract(p->mod, p->func_id, p->block_id, member_type, v.id, member_idx);
                        v.val_type = member_type;
                    }
                }
                MSL_FREE(member);
                continue;
            }

            /* Texture member functions: .sample(), .read(), .write(), .get_width(), .get_height() */
            if (vt && (vt->kind == SSIR_TYPE_TEXTURE || vt->kind == SSIR_TYPE_TEXTURE_STORAGE ||
                       vt->kind == SSIR_TYPE_TEXTURE_DEPTH || vt->kind == SSIR_TYPE_SAMPLER)) {
                v = mp_ensure_val(p, v);
                MslVal tex = v;
                if (strcmp(member, "sample") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p);
                    MslVal samp = mp_parse_assign_expr(p); samp = mp_ensure_val(p, samp);
                    mp_expect(p, MTK_COMMA);
                    MslVal coord = mp_parse_assign_expr(p); coord = mp_ensure_val(p, coord);
                    /* Check for optional bias/level */
                    if (mp_eat(p, MTK_COMMA)) {
                        /* bias(...) or level(...) */
                        if (mp_match_ident(p, "bias")) {
                            mp_next(p); mp_expect(p, MTK_LPAREN);
                            MslVal bias = mp_parse_expr(p); bias = mp_ensure_val(p, bias);
                            mp_expect(p, MTK_RPAREN);
                            mp_expect(p, MTK_RPAREN);
                            uint32_t res_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), 4);
                            v.id = ssir_build_tex_sample_bias(p->mod, p->func_id, p->block_id,
                                                               res_type, tex.id, samp.id, coord.id, bias.id);
                            v.val_type = res_type;
                        } else if (mp_match_ident(p, "level")) {
                            mp_next(p); mp_expect(p, MTK_LPAREN);
                            MslVal lod = mp_parse_expr(p); lod = mp_ensure_val(p, lod);
                            mp_expect(p, MTK_RPAREN);
                            mp_expect(p, MTK_RPAREN);
                            uint32_t res_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), 4);
                            v.id = ssir_build_tex_sample_level(p->mod, p->func_id, p->block_id,
                                                                res_type, tex.id, samp.id, coord.id, lod.id);
                            v.val_type = res_type;
                        } else if (mp_match_ident(p, "gradient2d")) {
                            mp_next(p); mp_expect(p, MTK_LPAREN);
                            MslVal ddx = mp_parse_expr(p); ddx = mp_ensure_val(p, ddx);
                            mp_expect(p, MTK_COMMA);
                            MslVal ddy = mp_parse_expr(p); ddy = mp_ensure_val(p, ddy);
                            mp_expect(p, MTK_RPAREN);
                            mp_expect(p, MTK_RPAREN);
                            uint32_t res_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), 4);
                            v.id = ssir_build_tex_sample_grad(p->mod, p->func_id, p->block_id,
                                                               res_type, tex.id, samp.id, coord.id, ddx.id, ddy.id);
                            v.val_type = res_type;
                        } else {
                            mp_expect(p, MTK_RPAREN);
                        }
                    } else {
                        mp_expect(p, MTK_RPAREN);
                        uint32_t res_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), 4);
                        v.id = ssir_build_tex_sample(p->mod, p->func_id, p->block_id,
                                                      res_type, tex.id, samp.id, coord.id);
                        v.val_type = res_type;
                    }
                    v.is_ptr = false;
                } else if (strcmp(member, "sample_compare") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p);
                    MslVal samp = mp_parse_assign_expr(p); samp = mp_ensure_val(p, samp);
                    mp_expect(p, MTK_COMMA);
                    MslVal coord = mp_parse_assign_expr(p); coord = mp_ensure_val(p, coord);
                    mp_expect(p, MTK_COMMA);
                    MslVal ref = mp_parse_assign_expr(p); ref = mp_ensure_val(p, ref);
                    mp_expect(p, MTK_RPAREN);
                    uint32_t res_type = ssir_type_f32(p->mod);
                    v.id = ssir_build_tex_sample_cmp(p->mod, p->func_id, p->block_id,
                                                      res_type, tex.id, samp.id, coord.id, ref.id);
                    v.val_type = res_type;
                    v.is_ptr = false;
                } else if (strcmp(member, "read") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p);
                    MslVal coord = mp_parse_assign_expr(p); coord = mp_ensure_val(p, coord);
                    uint32_t level_id = 0;
                    if (mp_eat(p, MTK_COMMA)) {
                        MslVal lv = mp_parse_assign_expr(p); lv = mp_ensure_val(p, lv);
                        level_id = lv.id;
                    }
                    mp_expect(p, MTK_RPAREN);
                    uint32_t res_type = ssir_type_vec(p->mod, ssir_type_f32(p->mod), 4);
                    v.id = ssir_build_tex_load(p->mod, p->func_id, p->block_id,
                                                res_type, tex.id, coord.id, level_id);
                    v.val_type = res_type;
                    v.is_ptr = false;
                } else if (strcmp(member, "write") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p);
                    MslVal val = mp_parse_assign_expr(p); val = mp_ensure_val(p, val);
                    mp_expect(p, MTK_COMMA);
                    MslVal coord = mp_parse_assign_expr(p); coord = mp_ensure_val(p, coord);
                    mp_expect(p, MTK_RPAREN);
                    ssir_build_tex_store(p->mod, p->func_id, p->block_id, tex.id, coord.id, val.id);
                    v.id = 0;
                    v.val_type = ssir_type_void(p->mod);
                    v.is_ptr = false;
                } else if (strcmp(member, "get_width") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p); mp_expect(p, MTK_RPAREN);
                    uint32_t res_type = ssir_type_u32(p->mod);
                    v.id = ssir_build_tex_size(p->mod, p->func_id, p->block_id, res_type, tex.id, 0);
                    v.val_type = res_type;
                    v.is_ptr = false;
                } else if (strcmp(member, "get_height") == 0 && mp_check(p, MTK_LPAREN)) {
                    mp_next(p); mp_expect(p, MTK_RPAREN);
                    uint32_t res_type = ssir_type_u32(p->mod);
                    v.id = ssir_build_tex_size(p->mod, p->func_id, p->block_id, res_type, tex.id, 0);
                    v.val_type = res_type;
                    v.is_ptr = false;
                }
                MSL_FREE(member);
                continue;
            }

            MSL_FREE(member);
            continue;
        }

        /* Array index: [expr] */
        if (mp_check(p, MTK_LBRACKET)) {
            mp_next(p);
            MslVal idx = mp_parse_expr(p);
            idx = mp_ensure_val(p, idx);
            mp_expect(p, MTK_RBRACKET);

            uint32_t elem_type = mp_element_type(p, v.val_type);
            if (v.is_ptr) {
                /* Access chain through pointer */
                uint32_t result_ptr = ssir_type_ptr(p->mod, elem_type, SSIR_ADDR_FUNCTION);
                /* Try to recover the original address space */
                SsirGlobalVar *g = ssir_get_global(p->mod, v.id);
                if (g) {
                    SsirType *pt = ssir_get_type(p->mod, g->type);
                    if (pt && pt->kind == SSIR_TYPE_PTR)
                        result_ptr = ssir_type_ptr(p->mod, elem_type, pt->ptr.space);
                }
                v.id = ssir_build_access(p->mod, p->func_id, p->block_id, result_ptr, v.id, &idx.id, 1);
                v.val_type = elem_type;
                v.is_ptr = true;
            } else {
                v.id = ssir_build_extract_dyn(p->mod, p->func_id, p->block_id, elem_type, v.id, idx.id);
                v.val_type = elem_type;
            }
            continue;
        }

        /* Function call: expr(args...) - for non-method calls on identifiers */
        if (mp_check(p, MTK_LPAREN) && !v.is_ptr) {
            /* This shouldn't normally happen in postfix (handled in primary) */
            break;
        }

        break;
    }

    return v;
}

/* Parse unary expression */
//p nonnull
static MslVal mp_parse_unary(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_unary: p is NULL");
    if (mp_check(p, MTK_MINUS)) {
        mp_next(p);
        MslVal operand = mp_parse_unary(p);
        operand = mp_ensure_val(p, operand);
        MslVal v;
        v.id = ssir_build_neg(p->mod, p->func_id, p->block_id, operand.val_type, operand.id);
        v.val_type = operand.val_type;
        v.is_ptr = false;
        return v;
    }
    if (mp_check(p, MTK_BANG)) {
        mp_next(p);
        MslVal operand = mp_parse_unary(p);
        operand = mp_ensure_val(p, operand);
        MslVal v;
        v.id = ssir_build_not(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), operand.id);
        v.val_type = ssir_type_bool(p->mod);
        v.is_ptr = false;
        return v;
    }
    if (mp_check(p, MTK_TILDE)) {
        mp_next(p);
        MslVal operand = mp_parse_unary(p);
        operand = mp_ensure_val(p, operand);
        MslVal v;
        v.id = ssir_build_bit_not(p->mod, p->func_id, p->block_id, operand.val_type, operand.id);
        v.val_type = operand.val_type;
        v.is_ptr = false;
        return v;
    }
    return mp_parse_postfix(p);
}

/* Operator precedence */
static int mp_precedence(MslTokType t) {
    switch (t) {
    case MTK_OROR: return 1;
    case MTK_ANDAND: return 2;
    case MTK_PIPE: return 3;
    case MTK_CARET: return 4;
    case MTK_AMP: return 5;
    case MTK_EQEQ: case MTK_NEQ: return 6;
    case MTK_LT: case MTK_LE: case MTK_GT: case MTK_GE: return 7;
    case MTK_SHL: case MTK_SHR: return 8;
    case MTK_PLUS: case MTK_MINUS: return 9;
    case MTK_STAR: case MTK_SLASH: case MTK_PERCENT: return 10;
    default: return 0;
    }
}

/* Build binary instruction from operator token */
//p nonnull
static uint32_t mp_build_binop(MslParser *p, MslTokType op, uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(p != NULL, "mp_build_binop: p is NULL");
    switch (op) {
    case MTK_PLUS: return ssir_build_add(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_MINUS: return ssir_build_sub(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_STAR: return ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_SLASH: return ssir_build_div(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_PERCENT: return ssir_build_mod(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_EQEQ: return ssir_build_eq(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_NEQ: return ssir_build_ne(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_LT: return ssir_build_lt(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_LE: return ssir_build_le(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_GT: return ssir_build_gt(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_GE: return ssir_build_ge(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_ANDAND: return ssir_build_and(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_OROR: return ssir_build_or(p->mod, p->func_id, p->block_id, ssir_type_bool(p->mod), a, b);
    case MTK_AMP: return ssir_build_bit_and(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_PIPE: return ssir_build_bit_or(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_CARET: return ssir_build_bit_xor(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_SHL: return ssir_build_shl(p->mod, p->func_id, p->block_id, type, a, b);
    case MTK_SHR: return ssir_build_shr(p->mod, p->func_id, p->block_id, type, a, b);
    default: return a;
    }
}

/* Parse binary expression with precedence climbing */
//p nonnull
static MslVal mp_parse_binary(MslParser *p, int min_prec) {
    wgsl_compiler_assert(p != NULL, "mp_parse_binary: p is NULL");
    MslVal left = mp_parse_unary(p);
    if (p->had_error) return left;

    for (;;) {
        int prec = mp_precedence(p->cur.type);
        if (prec < min_prec || prec == 0) break;

        MslTokType op = p->cur.type;
        mp_next(p);

        MslVal right = mp_parse_binary(p, prec + 1);
        left = mp_ensure_val(p, left);
        right = mp_ensure_val(p, right);

        uint32_t result_type = left.val_type;
        if (op == MTK_EQEQ || op == MTK_NEQ || op == MTK_LT || op == MTK_LE ||
            op == MTK_GT || op == MTK_GE || op == MTK_ANDAND || op == MTK_OROR)
            result_type = ssir_type_bool(p->mod);

        left.id = mp_build_binop(p, op, result_type, left.id, right.id);
        left.val_type = result_type;
        left.is_ptr = false;
    }

    return left;
}

/* Parse full expression (binary with min precedence 1) */
//p nonnull
static MslVal mp_parse_expr(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_expr: p is NULL");
    return mp_parse_binary(p, 1);
}

/* Parse assignment expression (used in function args - same as expr for now) */
//p nonnull
static MslVal mp_parse_assign_expr(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_assign_expr: p is NULL");
    return mp_parse_expr(p);
}

/* ============================================================================
 * Statement Parsing
 * ============================================================================ */

static void mp_parse_block(MslParser *p);
static void mp_parse_stmt(MslParser *p);

/* Is current position the start of a type? (for var declarations) */
//p nonnull
static bool mp_at_var_decl(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_at_var_decl: p is NULL");
    if (mp_is_type_name(p)) return true;
    /* address space qualifiers before type */
    if (mp_match_ident(p, "device") || mp_match_ident(p, "constant") ||
        mp_match_ident(p, "threadgroup") || mp_match_ident(p, "thread"))
        return true;
    return false;
}

//p nonnull
static void mp_parse_stmt(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_stmt: p is NULL");
    if (p->had_error) return;

    /* Empty statement */
    if (mp_check(p, MTK_SEMI)) { mp_next(p); return; }

    /* Block */
    if (mp_check(p, MTK_LBRACE)) { mp_parse_block(p); return; }

    /* Return */
    if (mp_match_ident(p, "return")) {
        mp_next(p);
        if (mp_check(p, MTK_SEMI)) {
            mp_next(p);
            ssir_build_return_void(p->mod, p->func_id, p->block_id);
            return;
        }
        /* Check for 'return _out;' pattern */
        if (p->is_entry && p->out_struct && p->cur.type == MTK_IDENT) {
            char *rname = mp_tok_str(p);
            MslSym *sym = mp_find_sym(p, rname);
            MSL_FREE(rname);
            if (sym && sym->is_out_var) {
                /* Skip the expression and emit return_void (outputs via stores) */
                mp_next(p); /* skip _out */
                mp_expect(p, MTK_SEMI);
                ssir_build_return_void(p->mod, p->func_id, p->block_id);
                return;
            }
        }
        MslVal rv = mp_parse_expr(p);
        rv = mp_ensure_val(p, rv);
        mp_expect(p, MTK_SEMI);
        ssir_build_return(p->mod, p->func_id, p->block_id, rv.id);
        return;
    }

    /* If/else */
    if (mp_match_ident(p, "if")) {
        mp_next(p);
        mp_expect(p, MTK_LPAREN);
        MslVal cond = mp_parse_expr(p);
        cond = mp_ensure_val(p, cond);
        mp_expect(p, MTK_RPAREN);

        uint32_t true_blk = ssir_block_create(p->mod, p->func_id, "if_true");
        uint32_t false_blk = ssir_block_create(p->mod, p->func_id, "if_false");
        uint32_t merge_blk = ssir_block_create(p->mod, p->func_id, "if_merge");

        bool has_else = false;

        ssir_build_branch_cond_merge(p->mod, p->func_id, p->block_id,
                                      cond.id, true_blk, false_blk, merge_blk);

        /* True branch */
        p->block_id = true_blk;
        mp_expect(p, MTK_LBRACE);
        while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
            mp_parse_stmt(p);
        mp_expect(p, MTK_RBRACE);
        ssir_build_branch(p->mod, p->func_id, p->block_id, merge_blk);

        /* Else branch */
        if (mp_match_ident(p, "else")) {
            has_else = true;
            mp_next(p);
            p->block_id = false_blk;
            if (mp_match_ident(p, "if")) {
                mp_parse_stmt(p); /* recursive if-else */
            } else {
                mp_expect(p, MTK_LBRACE);
                while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
                    mp_parse_stmt(p);
                mp_expect(p, MTK_RBRACE);
            }
            ssir_build_branch(p->mod, p->func_id, p->block_id, merge_blk);
        }

        if (!has_else) {
            /* Empty false branch → just branch to merge */
            p->block_id = false_blk;
            ssir_build_branch(p->mod, p->func_id, p->block_id, merge_blk);
        }

        p->block_id = merge_blk;
        return;
    }

    /* For loop */
    if (mp_match_ident(p, "for")) {
        mp_next(p);
        mp_expect(p, MTK_LPAREN);

        /* Init */
        if (!mp_check(p, MTK_SEMI))
            mp_parse_stmt(p);
        else
            mp_next(p);

        uint32_t header = ssir_block_create(p->mod, p->func_id, "for_header");
        uint32_t body = ssir_block_create(p->mod, p->func_id, "for_body");
        uint32_t cont = ssir_block_create(p->mod, p->func_id, "for_cont");
        uint32_t merge = ssir_block_create(p->mod, p->func_id, "for_merge");

        ssir_build_branch(p->mod, p->func_id, p->block_id, header);
        p->block_id = header;

        /* Condition */
        ssir_build_loop_merge(p->mod, p->func_id, p->block_id, merge, cont);
        if (!mp_check(p, MTK_SEMI)) {
            MslVal cond = mp_parse_expr(p);
            cond = mp_ensure_val(p, cond);
            mp_expect(p, MTK_SEMI);
            ssir_build_branch_cond(p->mod, p->func_id, p->block_id, cond.id, body, merge);
        } else {
            mp_next(p);
            ssir_build_branch(p->mod, p->func_id, p->block_id, body);
        }

        /* Save continue expression for later */
        /* For now, parse and skip the increment */
        uint32_t saved_block = p->block_id;
        p->block_id = cont;
        if (!mp_check(p, MTK_RPAREN)) {
            MslVal inc_lhs = mp_parse_postfix(p);
            if (mp_check(p, MTK_EQ) || mp_check(p, MTK_PLUSEQ) || mp_check(p, MTK_MINUSEQ) ||
                mp_check(p, MTK_STAREQ) || mp_check(p, MTK_SLASHEQ)) {
                MslTokType aop = p->cur.type;
                mp_next(p);
                MslVal rhs = mp_parse_expr(p);
                rhs = mp_ensure_val(p, rhs);
                if (aop == MTK_EQ) {
                    ssir_build_store(p->mod, p->func_id, p->block_id, inc_lhs.id, rhs.id);
                } else {
                    MslVal loaded = mp_ensure_val(p, inc_lhs);
                    MslTokType bop = (aop == MTK_PLUSEQ) ? MTK_PLUS :
                                     (aop == MTK_MINUSEQ) ? MTK_MINUS :
                                     (aop == MTK_STAREQ) ? MTK_STAR : MTK_SLASH;
                    uint32_t res = mp_build_binop(p, bop, loaded.val_type, loaded.id, rhs.id);
                    ssir_build_store(p->mod, p->func_id, p->block_id, inc_lhs.id, res);
                }
            } else if (mp_check(p, MTK_PLUSPLUS)) {
                mp_next(p);
                MslVal loaded = mp_ensure_val(p, inc_lhs);
                uint32_t one = ssir_const_i32(p->mod, 1);
                uint32_t res = ssir_build_add(p->mod, p->func_id, p->block_id, loaded.val_type, loaded.id, one);
                ssir_build_store(p->mod, p->func_id, p->block_id, inc_lhs.id, res);
            } else if (mp_check(p, MTK_MINUSMINUS)) {
                mp_next(p);
                MslVal loaded = mp_ensure_val(p, inc_lhs);
                uint32_t one = ssir_const_i32(p->mod, 1);
                uint32_t res = ssir_build_sub(p->mod, p->func_id, p->block_id, loaded.val_type, loaded.id, one);
                ssir_build_store(p->mod, p->func_id, p->block_id, inc_lhs.id, res);
            }
        }
        ssir_build_branch(p->mod, p->func_id, p->block_id, header);

        mp_expect(p, MTK_RPAREN);

        /* Body */
        p->block_id = body;
        mp_expect(p, MTK_LBRACE);
        while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
            mp_parse_stmt(p);
        mp_expect(p, MTK_RBRACE);
        ssir_build_branch(p->mod, p->func_id, p->block_id, cont);

        p->block_id = merge;
        return;
    }

    /* While loop */
    if (mp_match_ident(p, "while")) {
        mp_next(p);
        uint32_t header = ssir_block_create(p->mod, p->func_id, "while_header");
        uint32_t body = ssir_block_create(p->mod, p->func_id, "while_body");
        uint32_t merge = ssir_block_create(p->mod, p->func_id, "while_merge");
        uint32_t cont = ssir_block_create(p->mod, p->func_id, "while_cont");

        ssir_build_branch(p->mod, p->func_id, p->block_id, header);
        p->block_id = header;

        ssir_build_loop_merge(p->mod, p->func_id, p->block_id, merge, cont);

        mp_expect(p, MTK_LPAREN);
        MslVal cond = mp_parse_expr(p);
        cond = mp_ensure_val(p, cond);
        mp_expect(p, MTK_RPAREN);
        ssir_build_branch_cond(p->mod, p->func_id, p->block_id, cond.id, body, merge);

        p->block_id = body;
        mp_expect(p, MTK_LBRACE);
        while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
            mp_parse_stmt(p);
        mp_expect(p, MTK_RBRACE);
        ssir_build_branch(p->mod, p->func_id, p->block_id, cont);

        /* Continue block just branches back to header */
        p->block_id = cont;
        ssir_build_branch(p->mod, p->func_id, p->block_id, header);

        p->block_id = merge;
        return;
    }

    /* Switch */
    if (mp_match_ident(p, "switch")) {
        mp_next(p);
        mp_expect(p, MTK_LPAREN);
        MslVal sel = mp_parse_expr(p);
        sel = mp_ensure_val(p, sel);
        mp_expect(p, MTK_RPAREN);
        mp_expect(p, MTK_LBRACE);
        /* Simplified: skip switch body for now */
        int depth = 1;
        while (depth > 0 && !mp_check(p, MTK_EOF)) {
            if (mp_check(p, MTK_LBRACE)) depth++;
            if (mp_check(p, MTK_RBRACE)) depth--;
            if (depth > 0) mp_next(p);
        }
        mp_expect(p, MTK_RBRACE);
        return;
    }

    /* Break/continue */
    if (mp_match_ident(p, "break")) { mp_next(p); mp_expect(p, MTK_SEMI); return; }
    if (mp_match_ident(p, "continue")) { mp_next(p); mp_expect(p, MTK_SEMI); return; }

    /* Discard */
    if (mp_match_ident(p, "discard_fragment")) {
        mp_next(p);
        mp_expect(p, MTK_LPAREN);
        mp_expect(p, MTK_RPAREN);
        mp_expect(p, MTK_SEMI);
        return;
    }

    /* threadgroup_barrier */
    if (mp_match_ident(p, "threadgroup_barrier")) {
        mp_next(p);
        mp_expect(p, MTK_LPAREN);
        /* mem_flags::mem_threadgroup or mem_flags::mem_device */
        mp_next(p); /* mem_flags */
        mp_expect(p, MTK_COLONCOLON);
        char *flag = mp_tok_str(p);
        SsirBarrierScope scope = SSIR_BARRIER_WORKGROUP;
        if (strcmp(flag, "mem_device") == 0) scope = SSIR_BARRIER_STORAGE;
        MSL_FREE(flag);
        mp_next(p);
        mp_expect(p, MTK_RPAREN);
        mp_expect(p, MTK_SEMI);
        ssir_build_barrier(p->mod, p->func_id, p->block_id, scope);
        return;
    }

    /* Variable declaration: TypeName varName (= expr)?; */
    /* Or: output struct init: TypeName _out = {}; */
    if (mp_at_var_decl(p)) {
        uint32_t vtype = mp_parse_type(p);
        char *vname = mp_tok_str(p);
        mp_next(p);

        /* Check for _out / _in struct initialization */
        if (p->is_entry && p->out_struct &&
            strcmp(vname, "_out") == 0 && mp_check(p, MTK_EQ)) {
            /* TypeName _out = {}; → skip, _out is magic */
            mp_next(p); /* = */
            mp_expect(p, MTK_LBRACE);
            mp_expect(p, MTK_RBRACE);
            mp_expect(p, MTK_SEMI);
            /* Add _out to symbol table as magic */
            mp_add_sym(p, "_out", MSL_SENTINEL_OUT, 0, vtype, true, false);
            MSL_FREE(vname);
            return;
        }

        /* Check if this is a temporary variable (_vNN pattern) with an initializer.
         * These are MSL backend temporaries from a previous round-trip. Map them
         * directly to the expression result to avoid name collisions with the
         * _vN naming scheme used by the MSL backend for SSIR instruction results. */
        bool is_temp = (vname[0] == '_' && vname[1] == 'v' && isdigit((unsigned char)vname[2]));
        if (is_temp && mp_check(p, MTK_EQ)) {
            mp_next(p); /* eat '=' */
            MslVal init = mp_parse_expr(p);
            init = mp_ensure_val(p, init);
            mp_add_sym(p, vname, init.id, 0, init.val_type, false, false);
            mp_expect(p, MTK_SEMI);
            MSL_FREE(vname);
            return;
        }

        /* Regular local variable */
        uint32_t ptr_type = ssir_type_ptr(p->mod, vtype, SSIR_ADDR_FUNCTION);
        uint32_t local_id = ssir_function_add_local(p->mod, p->func_id, vname, ptr_type);
        mp_add_sym(p, vname, local_id, ptr_type, vtype, false, false);

        if (mp_eat(p, MTK_EQ)) {
            MslVal init = mp_parse_expr(p);
            init = mp_ensure_val(p, init);
            ssir_build_store(p->mod, p->func_id, p->block_id, local_id, init.id);
        }
        mp_expect(p, MTK_SEMI);
        MSL_FREE(vname);
        return;
    }

    /* Expression statement or assignment */
    {
        MslVal lhs = mp_parse_postfix(p);

        /* Check for assignment operators */
        if (mp_check(p, MTK_EQ)) {
            mp_next(p);
            MslVal rhs = mp_parse_expr(p);
            rhs = mp_ensure_val(p, rhs);
            mp_expect(p, MTK_SEMI);
            if (lhs.is_ptr) {
                ssir_build_store(p->mod, p->func_id, p->block_id, lhs.id, rhs.id);
            }
            return;
        }

        /* Compound assignment: +=, -=, *=, /=, etc */
        MslTokType aop = p->cur.type;
        if (aop == MTK_PLUSEQ || aop == MTK_MINUSEQ || aop == MTK_STAREQ ||
            aop == MTK_SLASHEQ || aop == MTK_PERCENTEQ ||
            aop == MTK_AMPEQ || aop == MTK_PIPEEQ || aop == MTK_CARETEQ) {
            mp_next(p);
            MslVal rhs = mp_parse_expr(p);
            rhs = mp_ensure_val(p, rhs);
            mp_expect(p, MTK_SEMI);

            MslVal loaded = mp_ensure_val(p, lhs);
            MslTokType bop;
            switch (aop) {
            case MTK_PLUSEQ: bop = MTK_PLUS; break;
            case MTK_MINUSEQ: bop = MTK_MINUS; break;
            case MTK_STAREQ: bop = MTK_STAR; break;
            case MTK_SLASHEQ: bop = MTK_SLASH; break;
            case MTK_PERCENTEQ: bop = MTK_PERCENT; break;
            case MTK_AMPEQ: bop = MTK_AMP; break;
            case MTK_PIPEEQ: bop = MTK_PIPE; break;
            case MTK_CARETEQ: bop = MTK_CARET; break;
            default: bop = MTK_PLUS; break;
            }
            uint32_t res = mp_build_binop(p, bop, loaded.val_type, loaded.id, rhs.id);
            if (lhs.is_ptr)
                ssir_build_store(p->mod, p->func_id, p->block_id, lhs.id, res);
            return;
        }

        /* Prefix increment/decrement already applied; or just expression statement */
        /* Also handle postfix ++ / -- */
        if (mp_check(p, MTK_PLUSPLUS) || mp_check(p, MTK_MINUSMINUS)) {
            bool is_inc = mp_check(p, MTK_PLUSPLUS);
            mp_next(p);
            if (lhs.is_ptr) {
                MslVal loaded = mp_ensure_val(p, lhs);
                uint32_t one = ssir_const_i32(p->mod, 1);
                uint32_t res = is_inc ?
                    ssir_build_add(p->mod, p->func_id, p->block_id, loaded.val_type, loaded.id, one) :
                    ssir_build_sub(p->mod, p->func_id, p->block_id, loaded.val_type, loaded.id, one);
                ssir_build_store(p->mod, p->func_id, p->block_id, lhs.id, res);
            }
            mp_expect(p, MTK_SEMI);
            return;
        }

        /* Try to finish as remaining binary expression + semicolon */
        /* This handles cases where we parsed just the postfix part */
        mp_expect(p, MTK_SEMI);
    }
}

//p nonnull
static void mp_parse_block(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_block: p is NULL");
    mp_expect(p, MTK_LBRACE);
    while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
        mp_parse_stmt(p);
    mp_expect(p, MTK_RBRACE);
}

/* ============================================================================
 * Function Parsing
 * ============================================================================ */

/* Demangle MSL entry point name: "main0" -> "main" */
//name nonnull
static const char *mp_demangle_name(const char *name) {
    wgsl_compiler_assert(name != NULL, "mp_demangle_name: name is NULL");
    if (strcmp(name, "main0") == 0) return "main";
    return name;
}

//p nonnull
static void mp_parse_function(MslParser *p, SsirStage stage) {
    wgsl_compiler_assert(p != NULL, "mp_parse_function: p is NULL");
    mp_clear_func(p);

    /* Save return type name for interface struct lookup before mp_parse_type consumes it */
    char *ret_type_name = (p->cur.type == MTK_IDENT) ? mp_tok_str(p) : NULL;
    uint32_t ret_type = mp_parse_type(p);
    char *func_name = mp_tok_str(p);
    mp_next(p);

    p->is_entry = (stage == SSIR_STAGE_VERTEX || stage == SSIR_STAGE_FRAGMENT ||
                   stage == SSIR_STAGE_COMPUTE);
    p->entry_stage = stage;

    /* Check if return type is an interface struct (for vertex/fragment) */
    MslStructDef *ret_struct = NULL;
    if (p->is_entry && stage != SSIR_STAGE_COMPUTE && ret_type_name) {
        ret_struct = mp_find_struct(p, ret_type_name);
        if (ret_struct && !ret_struct->is_interface) ret_struct = NULL;
    }
    MSL_FREE(ret_type_name);

    /* Create SSIR function (entry points return void, outputs via globals) */
    uint32_t ssir_ret = (p->is_entry && ret_struct) ? ssir_type_void(p->mod) :
                         (p->is_entry && stage == SSIR_STAGE_COMPUTE) ? ssir_type_void(p->mod) :
                         ret_type;
    const char *demangled = mp_demangle_name(func_name);
    p->func_id = ssir_function_create(p->mod, demangled, ssir_ret);

    /* Create output globals from return struct */
    if (ret_struct && p->is_entry) {
        p->out_struct = ret_struct;
        p->out_globals = (uint32_t *)MSL_MALLOC(ret_struct->member_count * sizeof(uint32_t));
        /* PRE: allocation succeeded */
        wgsl_compiler_assert(p->out_globals != NULL, "out_globals allocation failed");
        for (int i = 0; i < ret_struct->member_count; i++) {
            uint32_t pt = ssir_type_ptr(p->mod, ret_struct->members[i].ssir_type, SSIR_ADDR_OUTPUT);
            uint32_t gid = ssir_global_var(p->mod, ret_struct->members[i].name, pt);
            if (ret_struct->members[i].builtin != SSIR_BUILTIN_NONE)
                ssir_global_set_builtin(p->mod, gid, ret_struct->members[i].builtin);
            if (ret_struct->members[i].has_location)
                ssir_global_set_location(p->mod, gid, ret_struct->members[i].location);
            p->out_globals[i] = gid;
            mp_add_iface(p, gid);
        }
    }

    /* Parse parameters */
    mp_expect(p, MTK_LPAREN);
    while (!mp_check(p, MTK_RPAREN) && !mp_check(p, MTK_EOF) && !p->had_error) {
        if (p->sym_count > 0 || p->iface_count > (p->out_struct ? p->out_struct->member_count : 0))
            mp_expect(p, MTK_COMMA);

        /* Check for address space qualifier */
        SsirAddressSpace addr_space = SSIR_ADDR_FUNCTION;
        bool has_addr_space = false;
        if (mp_match_ident(p, "device")) {
            addr_space = SSIR_ADDR_STORAGE; has_addr_space = true; mp_next(p);
        } else if (mp_match_ident(p, "constant")) {
            addr_space = SSIR_ADDR_UNIFORM; has_addr_space = true; mp_next(p);
        } else if (mp_match_ident(p, "threadgroup")) {
            addr_space = SSIR_ADDR_WORKGROUP; has_addr_space = true; mp_next(p);
        }

        /* Save type name for stage_in struct lookup */
        char *param_type_name = (p->cur.type == MTK_IDENT) ? mp_tok_str(p) : NULL;
        uint32_t param_type = mp_parse_type(p);

        /* Check for pointer (*) or reference (&) */
        bool is_ptr = false;
        bool is_ref = false;
        if (mp_check(p, MTK_STAR)) { is_ptr = true; mp_next(p); }
        else if (mp_check(p, MTK_AMP)) { is_ref = true; mp_next(p); }

        /* Parameter name */
        char *pname = mp_tok_str(p);
        mp_next(p);

        /* Attributes */
        MslAttr attr = mp_parse_attrs(p);

        if (p->is_entry) {
            /* Handle stage_in for fragment shaders */
            if (attr.is_stage_in) {
                /* Find the struct type by name and create input globals for each member */
                MslStructDef *in_struct = param_type_name ? mp_find_struct(p, param_type_name) : NULL;
                if (in_struct) {
                    p->in_struct = in_struct;
                    p->in_globals = (uint32_t *)MSL_MALLOC(in_struct->member_count * sizeof(uint32_t));
                    /* PRE: allocation succeeded */
                    wgsl_compiler_assert(p->in_globals != NULL, "in_globals allocation failed");
                    for (int i = 0; i < in_struct->member_count; i++) {
                        uint32_t pt = ssir_type_ptr(p->mod, in_struct->members[i].ssir_type, SSIR_ADDR_INPUT);
                        uint32_t gid = ssir_global_var(p->mod, in_struct->members[i].name, pt);
                        if (in_struct->members[i].has_location)
                            ssir_global_set_location(p->mod, gid, in_struct->members[i].location);
                        if (in_struct->members[i].builtin != SSIR_BUILTIN_NONE)
                            ssir_global_set_builtin(p->mod, gid, in_struct->members[i].builtin);
                        if (in_struct->members[i].is_flat)
                            ssir_global_set_interpolation(p->mod, gid, SSIR_INTERP_FLAT);
                        p->in_globals[i] = gid;
                        mp_add_iface(p, gid);
                    }
                    /* Add _in to symbol table */
                    mp_add_sym(p, pname, MSL_SENTINEL_IN, 0, param_type, false, true);
                }
            } else if (has_addr_space && (addr_space == SSIR_ADDR_STORAGE || addr_space == SSIR_ADDR_UNIFORM)) {
                /* Buffer parameter → SSIR global */
                uint32_t pointee_type = param_type;
                if (is_ptr) {
                    /* device float* → runtime_array<f32> */
                    pointee_type = ssir_type_runtime_array(p->mod, param_type);
                }
                uint32_t ptr_t = ssir_type_ptr(p->mod, pointee_type, addr_space);
                uint32_t gid = ssir_global_var(p->mod, pname, ptr_t);
                if (attr.has_binding) {
                    ssir_global_set_binding(p->mod, gid, attr.binding);
                    ssir_global_set_group(p->mod, gid, attr.has_group ? attr.group : 0);
                }
                mp_add_sym(p, pname, gid, ptr_t, pointee_type, false, false);
                mp_add_iface(p, gid);
            } else if (has_addr_space && addr_space == SSIR_ADDR_WORKGROUP) {
                /* Threadgroup parameter */
                uint32_t ptr_t = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_WORKGROUP);
                uint32_t gid = ssir_global_var(p->mod, pname, ptr_t);
                mp_add_sym(p, pname, gid, ptr_t, param_type, false, false);
                mp_add_iface(p, gid);
            } else if (attr.builtin != SSIR_BUILTIN_NONE) {
                /* Builtin parameter → input global */
                uint32_t ptr_t = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_INPUT);
                uint32_t gid = ssir_global_var(p->mod, pname, ptr_t);
                ssir_global_set_builtin(p->mod, gid, attr.builtin);
                mp_add_sym(p, pname, gid, ptr_t, param_type, false, false);
                mp_add_iface(p, gid);
            } else if (attr.has_binding) {
                /* Texture or sampler parameter */
                uint32_t ptr_t = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_UNIFORM_CONSTANT);
                uint32_t gid = ssir_global_var(p->mod, pname, ptr_t);
                ssir_global_set_binding(p->mod, gid, attr.binding);
                ssir_global_set_group(p->mod, gid, 0);
                mp_add_sym(p, pname, gid, ptr_t, param_type, false, false);
                mp_add_iface(p, gid);
            } else if (attr.has_location) {
                /* Input attribute */
                uint32_t ptr_t = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_INPUT);
                uint32_t gid = ssir_global_var(p->mod, pname, ptr_t);
                ssir_global_set_location(p->mod, gid, attr.location);
                mp_add_sym(p, pname, gid, ptr_t, param_type, false, false);
                mp_add_iface(p, gid);
            } else {
                /* Plain parameter (shouldn't happen for entry points) */
                uint32_t pid = ssir_function_add_param(p->mod, p->func_id, pname, param_type);
                mp_add_sym(p, pname, pid, param_type, param_type, false, false);
            }
        } else {
            /* Non-entry function parameter */
            uint32_t pid = ssir_function_add_param(p->mod, p->func_id, pname, param_type);
            mp_add_sym(p, pname, pid, param_type, param_type, false, false);
        }

        MSL_FREE(pname);
        MSL_FREE(param_type_name);
    }
    mp_expect(p, MTK_RPAREN);

    /* Create entry point */
    if (p->is_entry) {
        p->ep_index = ssir_entry_point_create(p->mod, stage, p->func_id, demangled);
        for (int i = 0; i < p->iface_count; i++)
            ssir_entry_point_add_interface(p->mod, p->ep_index, p->iface[i]);
        if (stage == SSIR_STAGE_COMPUTE)
            ssir_entry_point_set_workgroup_size(p->mod, p->ep_index,
                                                 p->wg_size[0] ? p->wg_size[0] : 1,
                                                 p->wg_size[1] ? p->wg_size[1] : 1,
                                                 p->wg_size[2] ? p->wg_size[2] : 1);
    }

    /* Parse function body */
    p->block_id = ssir_block_create(p->mod, p->func_id, "entry");
    mp_expect(p, MTK_LBRACE);
    while (!mp_check(p, MTK_RBRACE) && !mp_check(p, MTK_EOF) && !p->had_error)
        mp_parse_stmt(p);
    mp_expect(p, MTK_RBRACE);

    /* Ensure block has a terminator */
    {
        SsirBlock *blk = ssir_get_block(p->mod, p->func_id, p->block_id);
        if (blk && (blk->inst_count == 0 ||
            (blk->insts[blk->inst_count - 1].op != SSIR_OP_RETURN &&
             blk->insts[blk->inst_count - 1].op != SSIR_OP_RETURN_VOID &&
             blk->insts[blk->inst_count - 1].op != SSIR_OP_BRANCH &&
             blk->insts[blk->inst_count - 1].op != SSIR_OP_BRANCH_COND &&
             blk->insts[blk->inst_count - 1].op != SSIR_OP_UNREACHABLE))) {
            ssir_build_return_void(p->mod, p->func_id, p->block_id);
        }
    }

    MSL_FREE(func_name);
}

/* ============================================================================
 * Top-Level Parsing
 * ============================================================================ */

//p nonnull
static void mp_parse_toplevel(MslParser *p) {
    wgsl_compiler_assert(p != NULL, "mp_parse_toplevel: p is NULL");
    while (!mp_check(p, MTK_EOF) && !p->had_error) {
        /* 'using namespace metal;' */
        if (mp_match_ident(p, "using")) {
            while (!mp_check(p, MTK_SEMI) && !mp_check(p, MTK_EOF)) mp_next(p);
            mp_eat(p, MTK_SEMI);
            continue;
        }

        /* struct */
        if (mp_match_ident(p, "struct")) {
            mp_parse_struct(p);
            continue;
        }

        /* Entry point qualifiers */
        SsirStage stage = SSIR_STAGE_COMPUTE;
        bool is_entry = false;
        if (mp_match_ident(p, "kernel")) {
            stage = SSIR_STAGE_COMPUTE; is_entry = true; mp_next(p);
        } else if (mp_match_ident(p, "vertex")) {
            stage = SSIR_STAGE_VERTEX; is_entry = true; mp_next(p);
        } else if (mp_match_ident(p, "fragment")) {
            stage = SSIR_STAGE_FRAGMENT; is_entry = true; mp_next(p);
        }

        if (is_entry) {
            mp_parse_function(p, stage);
            continue;
        }

        /* Non-entry function or global constant */
        if (mp_is_type_name(p)) {
            /* Could be a function or a global constant */
            /* Peek ahead: type name ( → function */
            /* For now, try as function */
            mp_parse_function(p, SSIR_STAGE_COMPUTE); /* stage unused for non-entry */
            continue;
        }

        /* Skip unknown tokens */
        mp_next(p);
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

//msl_source nonnull
//opts nullable
//out_module nonnull
//out_error nullable
MslToSsirResult msl_to_ssir(const char *msl_source, const MslToSsirOptions *opts,
                              SsirModule **out_module, char **out_error) {
    if (!msl_source || !out_module) {
        if (out_error) *out_error = msl_strdup("Invalid input: null source or output pointer");
        return MSL_TO_SSIR_PARSE_ERROR;
    }

    MslParser parser;
    memset(&parser, 0, sizeof(parser));
    mlx_init(&parser.lex, msl_source);
    if (opts) parser.opts = *opts;

    parser.mod = ssir_module_create();
    if (!parser.mod) {
        if (out_error) *out_error = msl_strdup("Out of memory");
        return MSL_TO_SSIR_PARSE_ERROR;
    }

    /* Read first token */
    mp_next(&parser);

    /* Parse */
    mp_parse_toplevel(&parser);

    if (parser.had_error) {
        if (out_error) *out_error = msl_strdup(parser.error);
        ssir_module_destroy(parser.mod);
        /* Cleanup */
        for (int i = 0; i < parser.sym_count; i++) MSL_FREE(parser.syms[i].name);
        MSL_FREE(parser.syms);
        for (int i = 0; i < parser.struct_count; i++) {
            MSL_FREE(parser.structs[i].name);
            for (int j = 0; j < parser.structs[i].member_count; j++)
                MSL_FREE(parser.structs[i].members[j].name);
            MSL_FREE(parser.structs[i].members);
        }
        MSL_FREE(parser.structs);
        MSL_FREE(parser.iface);
        MSL_FREE(parser.out_globals);
        MSL_FREE(parser.in_globals);
        return MSL_TO_SSIR_PARSE_ERROR;
    }

    *out_module = parser.mod;

    /* Cleanup parser (but not the module!) */
    for (int i = 0; i < parser.sym_count; i++) MSL_FREE(parser.syms[i].name);
    MSL_FREE(parser.syms);
    for (int i = 0; i < parser.struct_count; i++) {
        MSL_FREE(parser.structs[i].name);
        for (int j = 0; j < parser.structs[i].member_count; j++)
            MSL_FREE(parser.structs[i].members[j].name);
        MSL_FREE(parser.structs[i].members);
    }
    MSL_FREE(parser.structs);
    MSL_FREE(parser.iface);
    MSL_FREE(parser.out_globals);
    MSL_FREE(parser.in_globals);

    return MSL_TO_SSIR_OK;
}

//str nullable
void msl_to_ssir_free(char *str) {
    MSL_FREE(str);
}

const char *msl_to_ssir_result_string(MslToSsirResult r) {
    switch (r) {
    case MSL_TO_SSIR_OK: return "Success";
    case MSL_TO_SSIR_PARSE_ERROR: return "Parse error";
    case MSL_TO_SSIR_TYPE_ERROR: return "Type error";
    case MSL_TO_SSIR_UNSUPPORTED: return "Unsupported feature";
    default: return "Unknown error";
    }
}
