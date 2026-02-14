/*
 * glsl_parser.c - GLSL 450 (Vulkan) GpLexer and GpParser
 *
 * Produces the same WgslAstNode AST as wgsl_parser.c.
 * Maps GLSL constructs to WGSL-compatible AST nodes:
 *   - layout(set=S, binding=B) → @group(S) @binding(B) attributes
 *   - layout(location=N) → @location(N) attribute
 *   - in/out/uniform/buffer/shared → address_space field
 *   - GLSL types (vec3, mat4) → TypeNode with GLSL name
 *   - Interface blocks → struct + global var
 */

#include "simple_wgsl.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ============================================================================
 * String Utilities
 * ============================================================================ */

//s nonnull
static char *glsl_strndup(const char *s, size_t n) {
    wgsl_compiler_assert(s != NULL, "glsl_strndup: s is NULL");
    char *r = (char *)NODE_MALLOC(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

//s allowed to be NULL
static char *glsl_strdup(const char *s) {
    return s ? glsl_strndup(s, strlen(s)) : NULL;
}

//cap nonnull
static void *gp_grow_ptr_array(void *p, int needed, int *cap, size_t elem) {
    wgsl_compiler_assert(cap != NULL, "gp_grow_ptr_array: cap is NULL");
    if (needed <= *cap) return p;
    int nc = (*cap == 0) ? 4 : (*cap * 2);
    while (nc < needed) nc *= 2;
    void *np = NODE_REALLOC(p, (size_t)nc * elem);
    if (!np) return p;
    *cap = nc;
    return np;
}

//arr nonnull, count nonnull, cap nonnull
static void gp_vec_push_node(WgslAstNode ***arr, int *count, int *cap,
                           WgslAstNode *v) {
    wgsl_compiler_assert(arr != NULL, "gp_vec_push_node: arr is NULL");
    wgsl_compiler_assert(count != NULL, "gp_vec_push_node: count is NULL");
    wgsl_compiler_assert(cap != NULL, "gp_vec_push_node: cap is NULL");
    *arr = (WgslAstNode **)gp_grow_ptr_array(*arr, *count + 1, cap,
                                           sizeof(WgslAstNode *));
    (*arr)[(*count)++] = v;
}

/* ============================================================================
 * GpLexer
 * ============================================================================ */

typedef enum GpTokenType {
    GP_TOK_EOF = 0,
    GP_TOK_IDENT,
    GP_TOK_NUMBER,
    /* punctuation */
    GP_TOK_COLON,
    GP_TOK_SEMI,
    GP_TOK_COMMA,
    GP_TOK_LBRACE,
    GP_TOK_RBRACE,
    GP_TOK_LPAREN,
    GP_TOK_RPAREN,
    GP_TOK_LT,
    GP_TOK_GT,
    GP_TOK_LBRACKET,
    GP_TOK_RBRACKET,
    GP_TOK_DOT,
    GP_TOK_STAR,
    GP_TOK_SLASH,
    GP_TOK_PLUS,
    GP_TOK_MINUS,
    GP_TOK_PERCENT,
    GP_TOK_EQ,
    GP_TOK_BANG,
    GP_TOK_TILDE,
    GP_TOK_QMARK,
    GP_TOK_AMP,
    GP_TOK_PIPE,
    GP_TOK_CARET,
    /* multi-char operators */
    GP_TOK_LE,
    GP_TOK_GE,
    GP_TOK_EQEQ,
    GP_TOK_NEQ,
    GP_TOK_ANDAND,
    GP_TOK_OROR,
    GP_TOK_PLUSPLUS,
    GP_TOK_MINUSMINUS,
    GP_TOK_SHL,
    GP_TOK_SHR,
    GP_TOK_PLUSEQ,
    GP_TOK_MINUSEQ,
    GP_TOK_STAREQ,
    GP_TOK_SLASHEQ,
    GP_TOK_PERCENTEQ,
    GP_TOK_AMPEQ,
    GP_TOK_PIPEEQ,
    GP_TOK_CARETEQ,
    GP_TOK_SHLEQ,
    GP_TOK_SHREQ,
    /* keywords */
    GP_TOK_STRUCT,
    GP_TOK_IF,
    GP_TOK_ELSE,
    GP_TOK_WHILE,
    GP_TOK_FOR,
    GP_TOK_DO,
    GP_TOK_SWITCH,
    GP_TOK_CASE,
    GP_TOK_DEFAULT,
    GP_TOK_BREAK,
    GP_TOK_CONTINUE,
    GP_TOK_RETURN,
    GP_TOK_DISCARD,
    GP_TOK_LAYOUT,
    GP_TOK_CONST,
    GP_TOK_PRECISION
} GpTokenType;

typedef struct GpToken {
    GpTokenType type;
    const char *start;
    int length;
    int line;
    int col;
    bool is_float;
} GpToken;

typedef struct GpLexer {
    const char *src;
    size_t src_len;
    size_t pos;
    int line;
    int col;
} GpLexer;

/* PRE: src is null-terminated, bounds checks rely on null byte */
//L nonnull
static bool gp_lx_peek2(const GpLexer *L, char a, char b) {
    wgsl_compiler_assert(L != NULL, "gp_lx_peek2: L is NULL");
    if (L->pos + 1 >= L->src_len) return false;
    return (L->src[L->pos] == a && L->src[L->pos + 1] == b);
}

//L nonnull
static void gp_lx_advance(GpLexer *L) {
    wgsl_compiler_assert(L != NULL, "gp_lx_advance: L is NULL");
    if (L->pos >= L->src_len) return;
    char c = L->src[L->pos];
    if (!c) return;
    L->pos++;
    if (c == '\n') { L->line++; L->col = 1; }
    else { L->col++; }
}

//L nonnull
static void gp_lx_skip_ws_comments(GpLexer *L) {
    wgsl_compiler_assert(L != NULL, "gp_lx_skip_ws_comments: L is NULL");
    for (;;) {
        if (L->pos >= L->src_len) break;
        char c = L->src[L->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            gp_lx_advance(L);
            continue;
        }
        /* PRE: check pos+1 bounds before access */
        if (c == '/' && L->pos + 1 < L->src_len && L->src[L->pos + 1] == '/') {
            while (L->src[L->pos] && L->src[L->pos] != '\n')
                gp_lx_advance(L);
            continue;
        }
        if (c == '/' && L->pos + 1 < L->src_len && L->src[L->pos + 1] == '*') {
            gp_lx_advance(L); gp_lx_advance(L);
            while (L->src[L->pos]) {
                if (L->pos + 1 < L->src_len && L->src[L->pos] == '*' && L->src[L->pos + 1] == '/') {
                    gp_lx_advance(L); gp_lx_advance(L);
                    break;
                }
                gp_lx_advance(L);
            }
            continue;
        }
        /* Skip preprocessor directives: # until end of line */
        if (c == '#') {
            while (L->src[L->pos] && L->src[L->pos] != '\n')
                gp_lx_advance(L);
            continue;
        }
        break;
    }
}

static bool gp_is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}
static bool gp_is_ident_part(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

//L nonnull, s nonnull
static GpToken gp_make_token(GpLexer *L, GpTokenType t, const char *s, int len, bool f) {
    wgsl_compiler_assert(L != NULL, "gp_make_token: L is NULL");
    wgsl_compiler_assert(s != NULL, "gp_make_token: s is NULL");
    GpToken tok;
    tok.type = t;
    tok.start = s;
    tok.length = len;
    tok.line = L->line;
    tok.col = L->col;
    tok.is_float = f;
    return tok;
}

static int gp_is_dec_digit_or_us(char c) { return isdigit((unsigned char)c) || c == '_'; }
static int gp_is_hex_digit_or_us(char c) { return isxdigit((unsigned char)c) || c == '_'; }

typedef struct { const char *word; int len; GpTokenType tok; } Keyword;

static const Keyword glsl_keywords[] = {
    {"struct",    6, GP_TOK_STRUCT},
    {"if",        2, GP_TOK_IF},
    {"else",      4, GP_TOK_ELSE},
    {"while",     5, GP_TOK_WHILE},
    {"for",       3, GP_TOK_FOR},
    {"do",        2, GP_TOK_DO},
    {"switch",    6, GP_TOK_SWITCH},
    {"case",      4, GP_TOK_CASE},
    {"default",   7, GP_TOK_DEFAULT},
    {"break",     5, GP_TOK_BREAK},
    {"continue",  8, GP_TOK_CONTINUE},
    {"return",    6, GP_TOK_RETURN},
    {"discard",   7, GP_TOK_DISCARD},
    {"layout",    6, GP_TOK_LAYOUT},
    {"const",     5, GP_TOK_CONST},
    {"precision", 9, GP_TOK_PRECISION},
    {NULL, 0, GP_TOK_EOF}
};

//L nonnull
static GpToken gp_lx_next(GpLexer *L) {
    wgsl_compiler_assert(L != NULL, "gp_lx_next: L is NULL");
    gp_lx_skip_ws_comments(L);
    const char *s = &L->src[L->pos];
    char c = *s;
    if (!c) return gp_make_token(L, GP_TOK_EOF, s, 0, false);

    /* multi-char operators (3-char first) */
    /* PRE: check pos+2 bounds before accessing 3-char operators */
    if (L->pos + 2 < L->src_len && L->src[L->pos] == '<' && L->src[L->pos+1] == '<' && L->src[L->pos+2] == '=') {
        gp_lx_advance(L); gp_lx_advance(L); gp_lx_advance(L);
        return gp_make_token(L, GP_TOK_SHLEQ, s, 3, false);
    }
    if (L->pos + 2 < L->src_len && L->src[L->pos] == '>' && L->src[L->pos+1] == '>' && L->src[L->pos+2] == '=') {
        gp_lx_advance(L); gp_lx_advance(L); gp_lx_advance(L);
        return gp_make_token(L, GP_TOK_SHREQ, s, 3, false);
    }

    /* 2-char operators */
    if (gp_lx_peek2(L, '<', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_LE, s, 2, false); }
    if (gp_lx_peek2(L, '<', '<')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_SHL, s, 2, false); }
    if (gp_lx_peek2(L, '>', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_GE, s, 2, false); }
    if (gp_lx_peek2(L, '>', '>')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_SHR, s, 2, false); }
    if (gp_lx_peek2(L, '=', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_EQEQ, s, 2, false); }
    if (gp_lx_peek2(L, '!', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_NEQ, s, 2, false); }
    if (gp_lx_peek2(L, '&', '&')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_ANDAND, s, 2, false); }
    if (gp_lx_peek2(L, '|', '|')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_OROR, s, 2, false); }
    if (gp_lx_peek2(L, '+', '+')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_PLUSPLUS, s, 2, false); }
    if (gp_lx_peek2(L, '-', '-')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_MINUSMINUS, s, 2, false); }
    if (gp_lx_peek2(L, '+', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_PLUSEQ, s, 2, false); }
    if (gp_lx_peek2(L, '-', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_MINUSEQ, s, 2, false); }
    if (gp_lx_peek2(L, '*', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_STAREQ, s, 2, false); }
    if (gp_lx_peek2(L, '/', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_SLASHEQ, s, 2, false); }
    if (gp_lx_peek2(L, '%', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_PERCENTEQ, s, 2, false); }
    if (gp_lx_peek2(L, '&', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_AMPEQ, s, 2, false); }
    if (gp_lx_peek2(L, '|', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_PIPEEQ, s, 2, false); }
    if (gp_lx_peek2(L, '^', '=')) { gp_lx_advance(L); gp_lx_advance(L); return gp_make_token(L, GP_TOK_CARETEQ, s, 2, false); }

    /* single-char punctuation */
    switch (c) {
    case ':': gp_lx_advance(L); return gp_make_token(L, GP_TOK_COLON, s, 1, false);
    case ';': gp_lx_advance(L); return gp_make_token(L, GP_TOK_SEMI, s, 1, false);
    case ',': gp_lx_advance(L); return gp_make_token(L, GP_TOK_COMMA, s, 1, false);
    case '{': gp_lx_advance(L); return gp_make_token(L, GP_TOK_LBRACE, s, 1, false);
    case '}': gp_lx_advance(L); return gp_make_token(L, GP_TOK_RBRACE, s, 1, false);
    case '(': gp_lx_advance(L); return gp_make_token(L, GP_TOK_LPAREN, s, 1, false);
    case ')': gp_lx_advance(L); return gp_make_token(L, GP_TOK_RPAREN, s, 1, false);
    case '<': gp_lx_advance(L); return gp_make_token(L, GP_TOK_LT, s, 1, false);
    case '>': gp_lx_advance(L); return gp_make_token(L, GP_TOK_GT, s, 1, false);
    case '[': gp_lx_advance(L); return gp_make_token(L, GP_TOK_LBRACKET, s, 1, false);
    case ']': gp_lx_advance(L); return gp_make_token(L, GP_TOK_RBRACKET, s, 1, false);
    case '.': gp_lx_advance(L); return gp_make_token(L, GP_TOK_DOT, s, 1, false);
    case '*': gp_lx_advance(L); return gp_make_token(L, GP_TOK_STAR, s, 1, false);
    case '/': gp_lx_advance(L); return gp_make_token(L, GP_TOK_SLASH, s, 1, false);
    case '+': gp_lx_advance(L); return gp_make_token(L, GP_TOK_PLUS, s, 1, false);
    case '-': gp_lx_advance(L); return gp_make_token(L, GP_TOK_MINUS, s, 1, false);
    case '%': gp_lx_advance(L); return gp_make_token(L, GP_TOK_PERCENT, s, 1, false);
    case '=': gp_lx_advance(L); return gp_make_token(L, GP_TOK_EQ, s, 1, false);
    case '!': gp_lx_advance(L); return gp_make_token(L, GP_TOK_BANG, s, 1, false);
    case '~': gp_lx_advance(L); return gp_make_token(L, GP_TOK_TILDE, s, 1, false);
    case '?': gp_lx_advance(L); return gp_make_token(L, GP_TOK_QMARK, s, 1, false);
    case '&': gp_lx_advance(L); return gp_make_token(L, GP_TOK_AMP, s, 1, false);
    case '|': gp_lx_advance(L); return gp_make_token(L, GP_TOK_PIPE, s, 1, false);
    case '^': gp_lx_advance(L); return gp_make_token(L, GP_TOK_CARET, s, 1, false);
    default: break;
    }

    /* identifiers and keywords */
    if (gp_is_ident_start(c)) {
        size_t p = L->pos;
        gp_lx_advance(L);
        /* PRE: pos < src_len for ident parsing */
        while (L->pos < L->src_len && gp_is_ident_part(L->src[L->pos]))
            gp_lx_advance(L);
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        for (const Keyword *kw = glsl_keywords; kw->word; kw++) {
            if (len == kw->len && strncmp(start, kw->word, (size_t)len) == 0)
                return gp_make_token(L, kw->tok, start, len, false);
        }
        return gp_make_token(L, GP_TOK_IDENT, start, len, false);
    }

    /* numbers */
    if (isdigit((unsigned char)c)) {
        size_t p = L->pos;
        bool is_float = false;
        /* PRE: check pos+1 bounds for hex prefix */
        if (c == '0' && L->pos + 1 < L->src_len && (L->src[L->pos + 1] == 'x' || L->src[L->pos + 1] == 'X')) {
            gp_lx_advance(L); gp_lx_advance(L);
            while (gp_is_hex_digit_or_us(L->src[L->pos]))
                gp_lx_advance(L);
            if (L->src[L->pos] == 'u' || L->src[L->pos] == 'U')
                gp_lx_advance(L);
            const char *start = &L->src[p];
            int len = (int)(&L->src[L->pos] - start);
            return gp_make_token(L, GP_TOK_NUMBER, start, len, false);
        }
        gp_lx_advance(L);
        while (gp_is_dec_digit_or_us(L->src[L->pos]))
            gp_lx_advance(L);
        /* PRE: check pos+1 bounds for decimal float detection */
        if (L->src[L->pos] == '.' && L->pos + 1 < L->src_len && isdigit((unsigned char)L->src[L->pos + 1])) {
            is_float = true;
            gp_lx_advance(L);
            while (gp_is_dec_digit_or_us(L->src[L->pos]))
                gp_lx_advance(L);
        } else if (L->src[L->pos] == '.' && (L->pos + 1 >= L->src_len || !gp_is_ident_start(L->src[L->pos + 1]))) {
            is_float = true;
            gp_lx_advance(L);
        }
        if (L->src[L->pos] == 'e' || L->src[L->pos] == 'E') {
            is_float = true;
            gp_lx_advance(L);
            if (L->src[L->pos] == '+' || L->src[L->pos] == '-')
                gp_lx_advance(L);
            while (gp_is_dec_digit_or_us(L->src[L->pos]))
                gp_lx_advance(L);
        }
        if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F') {
            is_float = true;
            gp_lx_advance(L);
        } else if (L->src[L->pos] == 'l' || L->src[L->pos] == 'L') {
            gp_lx_advance(L); /* long suffix */
            if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F') {
                is_float = true;
                gp_lx_advance(L);
            }
        } else if (!is_float && (L->src[L->pos] == 'u' || L->src[L->pos] == 'U')) {
            gp_lx_advance(L);
        }
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        return gp_make_token(L, GP_TOK_NUMBER, start, len, is_float);
    }

    /* . followed by digit is a float literal */
    /* PRE: check pos+1 bounds for float literal */
    if (c == '.' && L->pos + 1 < L->src_len && isdigit((unsigned char)L->src[L->pos + 1])) {
        size_t p = L->pos;
        gp_lx_advance(L); /* consume '.' */
        while (gp_is_dec_digit_or_us(L->src[L->pos]))
            gp_lx_advance(L);
        if (L->src[L->pos] == 'e' || L->src[L->pos] == 'E') {
            gp_lx_advance(L);
            if (L->src[L->pos] == '+' || L->src[L->pos] == '-')
                gp_lx_advance(L);
            while (gp_is_dec_digit_or_us(L->src[L->pos]))
                gp_lx_advance(L);
        }
        if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F')
            gp_lx_advance(L);
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        return gp_make_token(L, GP_TOK_NUMBER, start, len, true);
    }

    gp_lx_advance(L);
    return gp_make_token(L, GP_TOK_EOF, s, 0, false);
}

/* ============================================================================
 * GpParser
 * ============================================================================ */

/* Known struct names for declaration/expression disambiguation */
typedef struct KnownType {
    char *name;
} KnownType;

typedef struct GpParser {
    GpLexer L;
    GpToken cur;
    bool had_error;
    KnownType *known_types;
    int known_type_count;
    int known_type_cap;
} GpParser;

//P nonnull
static void gp_advance(GpParser *P) { wgsl_compiler_assert(P != NULL, "gp_advance: P is NULL"); P->cur = gp_lx_next(&P->L); }
//P nonnull
static bool gp_check(GpParser *P, GpTokenType t) { wgsl_compiler_assert(P != NULL, "gp_check: P is NULL"); return P->cur.type == t; }
//P nonnull
static bool gp_match(GpParser *P, GpTokenType t) {
    wgsl_compiler_assert(P != NULL, "gp_match: P is NULL");
    if (gp_check(P, t)) { gp_advance(P); return true; }
    return false;
}
//P nonnull, msg nonnull
static void gp_parse_error(GpParser *P, const char *msg) {
    wgsl_compiler_assert(P != NULL, "gp_parse_error: P is NULL");
    wgsl_compiler_assert(msg != NULL, "gp_parse_error: msg is NULL");
    fprintf(stderr, "[glsl-parser] error at %d:%d: %s\n",
            P->cur.line, P->cur.col, msg);
    P->had_error = true;
}
//P nonnull, msg nonnull
static void gp_expect(GpParser *P, GpTokenType t, const char *msg) {
    wgsl_compiler_assert(P != NULL, "gp_expect: P is NULL");
    wgsl_compiler_assert(msg != NULL, "gp_expect: msg is NULL");
    if (!gp_match(P, t)) gp_parse_error(P, msg);
}

//P nonnull
static WgslAstNode *gp_new_node(GpParser *P, WgslNodeType k) {
    wgsl_compiler_assert(P != NULL, "gp_new_node: P is NULL");
    WgslAstNode *n = NODE_ALLOC(WgslAstNode);
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->type = k;
    n->line = P->cur.line;
    n->col = P->cur.col;
    return n;
}

//t nonnull, s nonnull
static bool tok_eq(const GpToken *t, const char *s) {
    wgsl_compiler_assert(t != NULL, "tok_eq: t is NULL");
    wgsl_compiler_assert(s != NULL, "tok_eq: s is NULL");
    int len = (int)strlen(s);
    return t->length == len && strncmp(t->start, s, (size_t)len) == 0;
}

//P nonnull, t nonnull
static WgslAstNode *gp_new_ident(GpParser *P, const GpToken *t) {
    wgsl_compiler_assert(P != NULL, "gp_new_ident: P is NULL");
    wgsl_compiler_assert(t != NULL, "gp_new_ident: t is NULL");
    WgslAstNode *n = gp_new_node(P, WGSL_NODE_IDENT);
    n->ident.name = glsl_strndup(t->start, (size_t)t->length);
    return n;
}
//P nonnull, t nonnull
static WgslAstNode *gp_new_literal(GpParser *P, const GpToken *t) {
    wgsl_compiler_assert(P != NULL, "gp_new_literal: P is NULL");
    wgsl_compiler_assert(t != NULL, "gp_new_literal: t is NULL");
    WgslAstNode *n = gp_new_node(P, WGSL_NODE_LITERAL);
    n->literal.lexeme = glsl_strndup(t->start, (size_t)t->length);
    n->literal.kind = t->is_float ? WGSL_LIT_FLOAT : WGSL_LIT_INT;
    return n;
}
//P nonnull, name nonnull
static WgslAstNode *gp_new_type(GpParser *P, const char *name) {
    wgsl_compiler_assert(P != NULL, "gp_new_type: P is NULL");
    wgsl_compiler_assert(name != NULL, "gp_new_type: name is NULL");
    WgslAstNode *n = gp_new_node(P, WGSL_NODE_TYPE);
    n->type_node.name = glsl_strdup(name);
    return n;
}

/* Type keyword detection */
static const char *glsl_type_keywords[] = {
    "void", "bool", "int", "uint", "float", "double",
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "bvec2", "bvec3", "bvec4",
    "dvec2", "dvec3", "dvec4",
    "mat2", "mat3", "mat4",
    "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4",
    "mat4x2", "mat4x3", "mat4x4",
    "dmat2", "dmat3", "dmat4",
    "sampler2D", "sampler3D", "samplerCube",
    "sampler2DShadow", "samplerCubeShadow",
    "sampler2DArray", "sampler2DArrayShadow",
    "samplerBuffer", "sampler2DMS",
    "isampler2D", "isampler3D", "isamplerCube",
    "isampler2DArray", "isamplerBuffer", "isampler2DMS",
    "usampler2D", "usampler3D", "usamplerCube",
    "usampler2DArray", "usamplerBuffer", "usampler2DMS",
    "image2D", "image3D", "imageCube",
    "image2DArray", "imageBuffer",
    "iimage2D", "iimage3D", "uimage2D", "uimage3D",
    NULL
};

//name nonnull
static bool is_type_keyword_str(const char *name, int len) {
    wgsl_compiler_assert(name != NULL, "is_type_keyword_str: name is NULL");
    for (const char **kw = glsl_type_keywords; *kw; kw++) {
        if ((int)strlen(*kw) == len && strncmp(*kw, name, (size_t)len) == 0)
            return true;
    }
    return false;
}

//t nonnull
static bool is_type_keyword_tok(const GpToken *t) {
    wgsl_compiler_assert(t != NULL, "is_type_keyword_tok: t is NULL");
    if (t->type != GP_TOK_IDENT) return false;
    return is_type_keyword_str(t->start, t->length);
}

/* Known struct tracking */
//P nonnull, name nonnull
static void register_struct(GpParser *P, const char *name) {
    wgsl_compiler_assert(P != NULL, "register_struct: P is NULL");
    wgsl_compiler_assert(name != NULL, "register_struct: name is NULL");
    P->known_types = (KnownType *)gp_grow_ptr_array(
        P->known_types, P->known_type_count + 1,
        &P->known_type_cap, sizeof(KnownType));
    P->known_types[P->known_type_count].name = glsl_strdup(name);
    P->known_type_count++;
}

//P nonnull, t nonnull
static bool is_known_struct(const GpParser *P, const GpToken *t) {
    wgsl_compiler_assert(P != NULL, "is_known_struct: P is NULL");
    wgsl_compiler_assert(t != NULL, "is_known_struct: t is NULL");
    if (t->type != GP_TOK_IDENT) return false;
    for (int i = 0; i < P->known_type_count; i++) {
        if ((int)strlen(P->known_types[i].name) == t->length &&
            strncmp(P->known_types[i].name, t->start, (size_t)t->length) == 0)
            return true;
    }
    return false;
}

//P nonnull, t nonnull
static bool is_type_start(const GpParser *P, const GpToken *t) {
    wgsl_compiler_assert(P != NULL, "is_type_start: P is NULL");
    wgsl_compiler_assert(t != NULL, "is_type_start: t is NULL");
    return is_type_keyword_tok(t) || is_known_struct(P, t);
}

/* Storage qualifier detection */
//t nonnull
static bool is_storage_qualifier_tok(const GpToken *t) {
    wgsl_compiler_assert(t != NULL, "is_storage_qualifier_tok: t is NULL");
    if (t->type != GP_TOK_IDENT) return false;
    return tok_eq(t, "in") || tok_eq(t, "out") || tok_eq(t, "inout") ||
           tok_eq(t, "uniform") || tok_eq(t, "buffer") || tok_eq(t, "shared");
}

/* Interpolation qualifier detection */
//t nonnull
static bool is_interp_qualifier_tok(const GpToken *t) {
    wgsl_compiler_assert(t != NULL, "is_interp_qualifier_tok: t is NULL");
    if (t->type != GP_TOK_IDENT) return false;
    return tok_eq(t, "flat") || tok_eq(t, "smooth") || tok_eq(t, "noperspective");
}

/* Precision qualifier detection */
//t nonnull
static bool is_precision_qualifier_tok(const GpToken *t) {
    wgsl_compiler_assert(t != NULL, "is_precision_qualifier_tok: t is NULL");
    if (t->type != GP_TOK_IDENT) return false;
    return tok_eq(t, "highp") || tok_eq(t, "mediump") || tok_eq(t, "lowp");
}

/* Forward declarations */
static WgslAstNode *gp_parse_program(GpParser *P);
static WgslAstNode *gp_parse_block(GpParser *P);
static WgslAstNode *gp_parse_statement(GpParser *P);
static WgslAstNode *gp_parse_expr(GpParser *P);
static WgslAstNode *gp_parse_assignment(GpParser *P);
static WgslAstNode *gp_parse_conditional(GpParser *P);
static WgslAstNode *gp_parse_logical_or(GpParser *P);
static WgslAstNode *gp_parse_logical_and(GpParser *P);
static WgslAstNode *parse_bitwise_or(GpParser *P);
static WgslAstNode *parse_bitwise_xor(GpParser *P);
static WgslAstNode *parse_bitwise_and(GpParser *P);
static WgslAstNode *gp_parse_equality(GpParser *P);
static WgslAstNode *gp_parse_relational(GpParser *P);
static WgslAstNode *gp_parse_shift(GpParser *P);
static WgslAstNode *gp_parse_additive(GpParser *P);
static WgslAstNode *gp_parse_multiplicative(GpParser *P);
static WgslAstNode *gp_parse_unary(GpParser *P);
static WgslAstNode *gp_parse_postfix(GpParser *P);
static WgslAstNode *gp_parse_primary(GpParser *P);

/* ============================================================================
 * Layout Qualifier Parsing
 * ============================================================================ */

/*
 * Parse layout(...) and return a list of Attribute nodes.
 * Maps: set→group, binding→binding, location→location,
 *       local_size_x/y/z→workgroup_size
 */
//P nonnull, out_attrs nonnull
static int parse_layout_qualifier(GpParser *P, WgslAstNode ***out_attrs) {
    wgsl_compiler_assert(P != NULL, "parse_layout_qualifier: P is NULL");
    wgsl_compiler_assert(out_attrs != NULL, "parse_layout_qualifier: out_attrs is NULL");
    *out_attrs = NULL;
    if (!gp_check(P, GP_TOK_LAYOUT)) return 0;
    gp_advance(P); /* consume 'layout' */
    gp_expect(P, GP_TOK_LPAREN, "expected '(' after 'layout'");

    int cap = 0, count = 0;
    WgslAstNode **attrs = NULL;

    /* Track local_size components for workgroup_size attribute */
    int has_wg = 0;
    int wg_x = 1, wg_y = 1, wg_z = 1;

    while (!gp_check(P, GP_TOK_RPAREN) && !gp_check(P, GP_TOK_EOF)) {
        if (count > 0 || has_wg) {
            if (!gp_match(P, GP_TOK_COMMA)) break;
        }
        if (!gp_check(P, GP_TOK_IDENT)) { gp_parse_error(P, "expected layout qualifier name"); break; }
        GpToken name = P->cur;
        gp_advance(P);

        WgslAstNode *value = NULL;
        if (gp_match(P, GP_TOK_EQ)) {
            if (gp_check(P, GP_TOK_NUMBER)) {
                GpToken num = P->cur;
                gp_advance(P);
                value = gp_new_literal(P, &num);
            } else if (gp_check(P, GP_TOK_IDENT)) {
                GpToken id = P->cur;
                gp_advance(P);
                value = gp_new_ident(P, &id);
            } else {
                gp_parse_error(P, "expected value after '=' in layout qualifier");
            }
        }

        /* Map layout qualifiers to WGSL-style attributes */
        if (tok_eq(&name, "set") && value) {
            WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("group");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            gp_vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "binding") && value) {
            WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("binding");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            gp_vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "location") && value) {
            WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("location");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            gp_vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "local_size_x") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_x = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "local_size_y") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_y = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "local_size_z") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_z = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "push_constant")) {
            WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("push_constant");
            gp_vec_push_node(&attrs, &count, &cap, attr);
            if (value) wgsl_free_ast(value);
        } else {
            /* Unknown layout qualifier - store as-is */
            WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strndup(name.start, (size_t)name.length);
            if (value) {
                attr->attribute.arg_count = 1;
                attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
                attr->attribute.args[0] = value;
            }
            gp_vec_push_node(&attrs, &count, &cap, attr);
        }
    }
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");

    /* Emit workgroup_size attribute if local_size was found */
    if (has_wg) {
        WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
        attr->attribute.name = glsl_strdup("workgroup_size");

        char buf_x[32], buf_y[32], buf_z[32];
        snprintf(buf_x, sizeof(buf_x), "%d", wg_x);
        snprintf(buf_y, sizeof(buf_y), "%d", wg_y);
        snprintf(buf_z, sizeof(buf_z), "%d", wg_z);

        attr->attribute.arg_count = 3;
        attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *) * 3);
        WgslAstNode *ax = gp_new_node(P, WGSL_NODE_LITERAL);
        ax->literal.lexeme = glsl_strdup(buf_x);
        ax->literal.kind = WGSL_LIT_INT;
        WgslAstNode *ay = gp_new_node(P, WGSL_NODE_LITERAL);
        ay->literal.lexeme = glsl_strdup(buf_y);
        ay->literal.kind = WGSL_LIT_INT;
        WgslAstNode *az = gp_new_node(P, WGSL_NODE_LITERAL);
        az->literal.lexeme = glsl_strdup(buf_z);
        az->literal.kind = WGSL_LIT_INT;
        attr->attribute.args[0] = ax;
        attr->attribute.args[1] = ay;
        attr->attribute.args[2] = az;
        gp_vec_push_node(&attrs, &count, &cap, attr);
    }

    *out_attrs = attrs;
    return count;
}

/* ============================================================================
 * Type Parsing
 * ============================================================================ */

//name allowed to be NULL
static const char *normalize_glsl_type(const char *name) {
    if (!name) return name;
    /* scalars */
    if (strcmp(name, "float") == 0) return "f32";
    if (strcmp(name, "int") == 0) return "i32";
    if (strcmp(name, "uint") == 0) return "u32";
    if (strcmp(name, "double") == 0) return "f64";
    /* float vectors */
    if (strcmp(name, "vec2") == 0) return "vec2f";
    if (strcmp(name, "vec3") == 0) return "vec3f";
    if (strcmp(name, "vec4") == 0) return "vec4f";
    /* int vectors */
    if (strcmp(name, "ivec2") == 0) return "vec2i";
    if (strcmp(name, "ivec3") == 0) return "vec3i";
    if (strcmp(name, "ivec4") == 0) return "vec4i";
    /* uint vectors */
    if (strcmp(name, "uvec2") == 0) return "vec2u";
    if (strcmp(name, "uvec3") == 0) return "vec3u";
    if (strcmp(name, "uvec4") == 0) return "vec4u";
    /* bool vectors */
    if (strcmp(name, "bvec2") == 0) return "vec2<bool>";
    if (strcmp(name, "bvec3") == 0) return "vec3<bool>";
    if (strcmp(name, "bvec4") == 0) return "vec4<bool>";
    /* matrices (column-major, float only) */
    if (strcmp(name, "mat2") == 0) return "mat2x2f";
    if (strcmp(name, "mat3") == 0) return "mat3x3f";
    if (strcmp(name, "mat4") == 0) return "mat4x4f";
    if (strcmp(name, "mat2x2") == 0) return "mat2x2f";
    if (strcmp(name, "mat2x3") == 0) return "mat2x3f";
    if (strcmp(name, "mat2x4") == 0) return "mat2x4f";
    if (strcmp(name, "mat3x2") == 0) return "mat3x2f";
    if (strcmp(name, "mat3x3") == 0) return "mat3x3f";
    if (strcmp(name, "mat3x4") == 0) return "mat3x4f";
    if (strcmp(name, "mat4x2") == 0) return "mat4x2f";
    if (strcmp(name, "mat4x3") == 0) return "mat4x3f";
    if (strcmp(name, "mat4x4") == 0) return "mat4x4f";
    return name;
}

//P nonnull
static WgslAstNode *gp_parse_type_node(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_type_node: P is NULL");
    if (!gp_check(P, GP_TOK_IDENT) && !gp_check(P, GP_TOK_STRUCT)) {
        gp_parse_error(P, "expected type name");
        return NULL;
    }
    GpToken name = P->cur;
    gp_advance(P);
    char *tname = glsl_strndup(name.start, (size_t)name.length);
    const char *normalized = normalize_glsl_type(tname);
    WgslAstNode *T = gp_new_type(P, normalized);
    NODE_FREE(tname);
    return T;
}

/* Parse C-style array dimensions after a variable name: [N][M]... */
//P nonnull, base_type nonnull
static WgslAstNode *wrap_type_with_array_dims(GpParser *P, WgslAstNode *base_type) {
    wgsl_compiler_assert(P != NULL, "wrap_type_with_array_dims: P is NULL");
    wgsl_compiler_assert(base_type != NULL, "wrap_type_with_array_dims: base_type is NULL");
    while (gp_match(P, GP_TOK_LBRACKET)) {
        WgslAstNode *dim = NULL;
        if (!gp_check(P, GP_TOK_RBRACKET)) {
            if (gp_check(P, GP_TOK_NUMBER)) {
                GpToken num = P->cur;
                gp_advance(P);
                dim = gp_new_literal(P, &num);
            } else {
                dim = gp_parse_expr(P);
            }
        }
        gp_expect(P, GP_TOK_RBRACKET, "expected ']'");

        /* Wrap base_type in array<base_type, dim> */
        WgslAstNode *arr_type = gp_new_node(P, WGSL_NODE_TYPE);
        arr_type->type_node.name = glsl_strdup("array");
        arr_type->type_node.type_arg_count = 1;
        arr_type->type_node.type_args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
        arr_type->type_node.type_args[0] = base_type;
        if (dim) {
            arr_type->type_node.expr_arg_count = 1;
            arr_type->type_node.expr_args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            arr_type->type_node.expr_args[0] = dim;
        }
        base_type = arr_type;
    }
    return base_type;
}

/* ============================================================================
 * Declaration Parsing
 * ============================================================================ */

/* Parse a struct definition: struct Name { type field; ... }; */
//P nonnull
static WgslAstNode *parse_struct_def(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_struct_def: P is NULL");
    gp_expect(P, GP_TOK_STRUCT, "expected 'struct'");
    if (!gp_check(P, GP_TOK_IDENT)) {
        gp_parse_error(P, "expected struct name");
        return NULL;
    }
    GpToken name = P->cur;
    gp_advance(P);

    char *sname = glsl_strndup(name.start, (size_t)name.length);
    register_struct(P, sname);

    WgslAstNode *S = gp_new_node(P, WGSL_NODE_STRUCT);
    S->struct_decl.name = sname;

    gp_expect(P, GP_TOK_LBRACE, "expected '{'");
    int cap = 0, count = 0;
    WgslAstNode **fields = NULL;

    while (!gp_check(P, GP_TOK_RBRACE) && !gp_check(P, GP_TOK_EOF)) {
        /* Optional layout qualifier on struct members */
        WgslAstNode **field_attrs = NULL;
        int field_attr_count = 0;
        if (gp_check(P, GP_TOK_LAYOUT))
            field_attr_count = parse_layout_qualifier(P, &field_attrs);

        WgslAstNode *ftype = gp_parse_type_node(P);
        if (!gp_check(P, GP_TOK_IDENT)) {
            gp_parse_error(P, "expected field name");
            if (ftype) wgsl_free_ast(ftype);
            break;
        }
        GpToken fname = P->cur;
        gp_advance(P);

        /* C-style array dimensions on field */
        ftype = wrap_type_with_array_dims(P, ftype);

        gp_expect(P, GP_TOK_SEMI, "expected ';'");

        WgslAstNode *F = gp_new_node(P, WGSL_NODE_STRUCT_FIELD);
        F->struct_field.name = glsl_strndup(fname.start, (size_t)fname.length);
        F->struct_field.type = ftype;
        F->struct_field.attr_count = field_attr_count;
        F->struct_field.attrs = field_attrs;
        gp_vec_push_node(&fields, &count, &cap, F);
    }
    gp_expect(P, GP_TOK_RBRACE, "expected '}'");
    gp_match(P, GP_TOK_SEMI); /* optional semicolon after struct */

    S->struct_decl.field_count = count;
    S->struct_decl.fields = fields;
    return S;
}

/*
 * Parse an interface block:
 *   layout(...) storage_qual BlockName { members } instance_name;
 *
 * Decomposes into: struct BlockName + global var instance_name
 */
//P nonnull, out_struct nonnull, out_var nonnull
static void parse_interface_block(GpParser *P, WgslAstNode **out_struct,
                                  WgslAstNode **out_var,
                                  WgslAstNode **attrs, int attr_count,
                                  const char *address_space) {
    wgsl_compiler_assert(P != NULL, "parse_interface_block: P is NULL");
    wgsl_compiler_assert(out_struct != NULL, "parse_interface_block: out_struct is NULL");
    wgsl_compiler_assert(out_var != NULL, "parse_interface_block: out_var is NULL");
    *out_struct = NULL;
    *out_var = NULL;

    if (!gp_check(P, GP_TOK_IDENT)) {
        gp_parse_error(P, "expected interface block name");
        return;
    }
    GpToken block_name = P->cur;
    gp_advance(P);

    char *bname = glsl_strndup(block_name.start, (size_t)block_name.length);
    register_struct(P, bname);

    /* Parse block body as struct */
    WgslAstNode *S = gp_new_node(P, WGSL_NODE_STRUCT);
    S->struct_decl.name = glsl_strdup(bname);

    gp_expect(P, GP_TOK_LBRACE, "expected '{'");
    int cap = 0, count = 0;
    WgslAstNode **fields = NULL;

    while (!gp_check(P, GP_TOK_RBRACE) && !gp_check(P, GP_TOK_EOF)) {
        WgslAstNode **field_attrs = NULL;
        int field_attr_count = 0;
        if (gp_check(P, GP_TOK_LAYOUT))
            field_attr_count = parse_layout_qualifier(P, &field_attrs);

        WgslAstNode *ftype = gp_parse_type_node(P);
        if (!gp_check(P, GP_TOK_IDENT)) {
            gp_parse_error(P, "expected field name");
            if (ftype) wgsl_free_ast(ftype);
            break;
        }
        GpToken fname = P->cur;
        gp_advance(P);
        ftype = wrap_type_with_array_dims(P, ftype);
        gp_expect(P, GP_TOK_SEMI, "expected ';'");

        WgslAstNode *F = gp_new_node(P, WGSL_NODE_STRUCT_FIELD);
        F->struct_field.name = glsl_strndup(fname.start, (size_t)fname.length);
        F->struct_field.type = ftype;
        F->struct_field.attr_count = field_attr_count;
        F->struct_field.attrs = field_attrs;
        gp_vec_push_node(&fields, &count, &cap, F);
    }
    gp_expect(P, GP_TOK_RBRACE, "expected '}'");
    S->struct_decl.field_count = count;
    S->struct_decl.fields = fields;
    *out_struct = S;

    /* Instance name (optional) */
    char *inst_name = NULL;
    if (gp_check(P, GP_TOK_IDENT)) {
        GpToken iname = P->cur;
        gp_advance(P);
        inst_name = glsl_strndup(iname.start, (size_t)iname.length);
    } else {
        inst_name = glsl_strdup(bname);
    }
    gp_expect(P, GP_TOK_SEMI, "expected ';'");

    /* Create global var of block type */
    WgslAstNode *G = gp_new_node(P, WGSL_NODE_GLOBAL_VAR);
    G->global_var.name = inst_name;
    G->global_var.type = gp_new_type(P, bname);
    G->global_var.address_space = address_space ? glsl_strdup(address_space) : NULL;
    G->global_var.attr_count = attr_count;
    G->global_var.attrs = attrs;
    *out_var = G;

    NODE_FREE(bname);
}

/* Parse function parameters: (in/out/inout type name, ...) */
//P nonnull
static WgslAstNode *gp_parse_param(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_param: P is NULL");
    /* Optional parameter qualifier */
    char *param_qual = NULL;
    if (gp_check(P, GP_TOK_IDENT) && (tok_eq(&P->cur, "in") || tok_eq(&P->cur, "out") ||
                                 tok_eq(&P->cur, "inout"))) {
        param_qual = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        gp_advance(P);
    } else if (gp_check(P, GP_TOK_CONST)) {
        gp_advance(P); /* skip const on params */
    }

    WgslAstNode *type = gp_parse_type_node(P);
    if (!gp_check(P, GP_TOK_IDENT)) {
        gp_parse_error(P, "expected parameter name");
        if (param_qual) NODE_FREE(param_qual);
        if (type) wgsl_free_ast(type);
        return NULL;
    }
    GpToken name = P->cur;
    gp_advance(P);

    type = wrap_type_with_array_dims(P, type);

    WgslAstNode *Par = gp_new_node(P, WGSL_NODE_PARAM);
    Par->param.name = glsl_strndup(name.start, (size_t)name.length);
    Par->param.type = type;

    /* Store parameter qualifier as attribute if not "in" (default) */
    if (param_qual && (strcmp(param_qual, "out") == 0 || strcmp(param_qual, "inout") == 0)) {
        Par->param.attr_count = 1;
        Par->param.attrs = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
        WgslAstNode *attr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
        attr->attribute.name = param_qual;
        Par->param.attrs[0] = attr;
    } else {
        if (param_qual) NODE_FREE(param_qual);
    }
    return Par;
}

/* Parse function definition: type name(params) { body } */
//P nonnull, ret_type nonnull, name nonnull
static WgslAstNode *parse_function_def(GpParser *P, WgslAstNode *ret_type,
                                        const char *name,
                                        WgslAstNode **attrs, int attr_count) {
    wgsl_compiler_assert(P != NULL, "parse_function_def: P is NULL");
    wgsl_compiler_assert(ret_type != NULL, "parse_function_def: ret_type is NULL");
    wgsl_compiler_assert(name != NULL, "parse_function_def: name is NULL");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");
    int pcap = 0, pcount = 0;
    WgslAstNode **params = NULL;

    if (!gp_check(P, GP_TOK_RPAREN)) {
        /* Check for void parameter list: void f(void) */
        if (gp_check(P, GP_TOK_IDENT) && tok_eq(&P->cur, "void") && P->cur.length == 4) {
            /* peek ahead: if next is ')', this is void param list */
            GpLexer L2 = P->L;
            GpToken t2 = gp_lx_next(&L2);
            if (t2.type == GP_TOK_RPAREN) {
                gp_advance(P); /* consume "void" */
            } else {
                goto parse_params;
            }
        } else {
parse_params:;
            WgslAstNode *par = gp_parse_param(P);
            if (par) gp_vec_push_node(&params, &pcount, &pcap, par);
            while (gp_match(P, GP_TOK_COMMA)) {
                WgslAstNode *par2 = gp_parse_param(P);
                if (par2) gp_vec_push_node(&params, &pcount, &pcap, par2);
            }
        }
    }
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");

    WgslAstNode *body = gp_parse_block(P);

    WgslAstNode *F = gp_new_node(P, WGSL_NODE_FUNCTION);
    F->function.name = glsl_strdup(name);
    F->function.param_count = pcount;
    F->function.params = params;
    F->function.return_type = ret_type;
    F->function.body = body;
    F->function.attr_count = attr_count;
    F->function.attrs = attrs;
    return F;
}

/*
 * Parse a top-level external declaration.
 * Handles: layout? interp? storage? precision? type name → function or global var
 *          struct definition
 *          interface block
 *          precision declaration
 */
//P nonnull, extra_decls nonnull, extra_count nonnull, extra_cap nonnull
static WgslAstNode *parse_external_declaration(GpParser *P,
                                                WgslAstNode ***extra_decls,
                                                int *extra_count,
                                                int *extra_cap) {
    wgsl_compiler_assert(P != NULL, "parse_external_declaration: P is NULL");
    wgsl_compiler_assert(extra_decls != NULL, "parse_external_declaration: extra_decls is NULL");
    wgsl_compiler_assert(extra_count != NULL, "parse_external_declaration: extra_count is NULL");
    wgsl_compiler_assert(extra_cap != NULL, "parse_external_declaration: extra_cap is NULL");
    /* Layout qualifier */
    WgslAstNode **attrs = NULL;
    int attr_count = 0;
    if (gp_check(P, GP_TOK_LAYOUT))
        attr_count = parse_layout_qualifier(P, &attrs);

    /* Interpolation qualifier */
    char *interp = NULL;
    if (is_interp_qualifier_tok(&P->cur)) {
        interp = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        gp_advance(P);
    }

    /* Storage qualifier */
    char *storage = NULL;
    if (is_storage_qualifier_tok(&P->cur)) {
        storage = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        gp_advance(P);
    } else if (gp_check(P, GP_TOK_CONST)) {
        storage = glsl_strdup("const");
        gp_advance(P);
    }

    /* Precision qualifier (skip) */
    if (is_precision_qualifier_tok(&P->cur)) {
        gp_advance(P); /* skip precision qualifier */
    }

    /* Precision declaration: precision highp float; */
    if (gp_check(P, GP_TOK_PRECISION)) {
        gp_advance(P); /* consume 'precision' */
        if (is_precision_qualifier_tok(&P->cur))
            gp_advance(P);
        if (gp_check(P, GP_TOK_IDENT))
            gp_advance(P); /* type name */
        gp_expect(P, GP_TOK_SEMI, "expected ';' after precision declaration");
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        /* Free unused attrs */
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL; /* skip precision declarations */
    }

    /* Struct definition */
    if (gp_check(P, GP_TOK_STRUCT)) {
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return parse_struct_def(P);
    }

    /* At this point we need a type name.
     * If we have a storage qualifier and the next token is an identifier
     * that's NOT a type, it might be an interface block. */
    if (storage && gp_check(P, GP_TOK_IDENT) && !is_type_start(P, &P->cur)) {
        /* Could be an interface block: uniform BlockName { ... } instance; */
        /* Check if identifier is followed by '{' */
        GpLexer L2 = P->L;
        GpToken t2 = gp_lx_next(&L2);
        if (t2.type == GP_TOK_LBRACE) {
            /* Map storage qualifier to address space */
            const char *addr = NULL;
            if (strcmp(storage, "uniform") == 0) addr = "uniform";
            else if (strcmp(storage, "buffer") == 0) addr = "storage";
            else if (strcmp(storage, "shared") == 0) addr = "workgroup";
            else addr = storage;

            WgslAstNode *s_node = NULL, *v_node = NULL;
            parse_interface_block(P, &s_node, &v_node, attrs, attr_count, addr);
            NODE_FREE(storage);
            if (interp) NODE_FREE(interp);

            /* Add struct as extra declaration, return the global var */
            if (s_node)
                gp_vec_push_node(extra_decls, extra_count, extra_cap, s_node);
            return v_node;
        }
    }

    /* Handle standalone layout qualifier declaration: layout(...) in; */
    if (gp_check(P, GP_TOK_SEMI) && storage && attrs) {
        gp_advance(P); /* consume ';' */
        /* Create a global var node with the attributes (e.g. workgroup_size) */
        WgslAstNode *gv = gp_new_node(P, WGSL_NODE_GLOBAL_VAR);
        gv->global_var.name = glsl_strdup("__layout_decl");
        gv->global_var.address_space = storage; /* transfer ownership */
        gv->global_var.type = NULL;
        gv->global_var.attrs = attrs;
        gv->global_var.attr_count = attr_count;
        if (interp) NODE_FREE(interp);
        return gv;
    }

    /* Parse type */
    if (!gp_check(P, GP_TOK_IDENT)) {
        gp_parse_error(P, "expected type name or declaration");
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL;
    }
    WgslAstNode *type = gp_parse_type_node(P);

    /* Parse name */
    if (!gp_check(P, GP_TOK_IDENT)) {
        gp_parse_error(P, "expected declaration name");
        if (type) wgsl_free_ast(type);
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL;
    }
    GpToken name = P->cur;
    gp_advance(P);
    char *decl_name = glsl_strndup(name.start, (size_t)name.length);

    /* Function definition: type name(...) { ... } */
    if (gp_check(P, GP_TOK_LPAREN)) {
        /* Add interpolation as attribute if present */
        if (interp) {
            WgslAstNode *iattr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
            iattr->attribute.name = interp;
            gp_vec_push_node(&attrs, &attr_count, &(int){0}, iattr);
        }
        WgslAstNode *fn = parse_function_def(P, type, decl_name, attrs, attr_count);
        NODE_FREE(decl_name);
        if (storage) NODE_FREE(storage);
        return fn;
    }

    /* Global variable declaration */
    type = wrap_type_with_array_dims(P, type);

    /* Optional initializer */
    WgslAstNode *init = NULL;
    if (gp_match(P, GP_TOK_EQ)) {
        init = gp_parse_expr(P);
    }
    gp_expect(P, GP_TOK_SEMI, "expected ';'");

    /* If no storage qualifier and has init, treat as const-like (VarDecl) */
    if (!storage || strcmp(storage, "const") == 0) {
        if (storage) NODE_FREE(storage);
        if (interp) NODE_FREE(interp);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        WgslAstNode *V = gp_new_node(P, WGSL_NODE_VAR_DECL);
        V->var_decl.name = decl_name;
        V->var_decl.type = type;
        V->var_decl.init = init;
        return V;
    }

    /* Map storage qualifier to address space */
    char *addr_space = NULL;
    if (strcmp(storage, "in") == 0) addr_space = glsl_strdup("in");
    else if (strcmp(storage, "out") == 0) addr_space = glsl_strdup("out");
    else if (strcmp(storage, "uniform") == 0) addr_space = glsl_strdup("uniform");
    else if (strcmp(storage, "buffer") == 0) addr_space = glsl_strdup("storage");
    else if (strcmp(storage, "shared") == 0) addr_space = glsl_strdup("workgroup");
    else addr_space = glsl_strdup(storage);
    NODE_FREE(storage);

    /* Add interpolation as attribute */
    if (interp) {
        WgslAstNode *iattr = gp_new_node(P, WGSL_NODE_ATTRIBUTE);
        iattr->attribute.name = interp;
        int acap = attr_count;
        gp_vec_push_node(&attrs, &attr_count, &acap, iattr);
    }

    WgslAstNode *G = gp_new_node(P, WGSL_NODE_GLOBAL_VAR);
    G->global_var.name = decl_name;
    G->global_var.type = type;
    G->global_var.address_space = addr_space;
    G->global_var.attr_count = attr_count;
    G->global_var.attrs = attrs;

    if (init) wgsl_free_ast(init); /* discard init for qualified globals */

    return G;
}

//P nonnull
static WgslAstNode *gp_parse_block(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_block: P is NULL");
    gp_expect(P, GP_TOK_LBRACE, "expected '{'");
    WgslAstNode *B = gp_new_node(P, WGSL_NODE_BLOCK);
    int cap = 0, count = 0;
    WgslAstNode **stmts = NULL;
    while (!gp_check(P, GP_TOK_RBRACE) && !gp_check(P, GP_TOK_EOF)) {
        const char *before = P->cur.start;
        WgslAstNode *s = gp_parse_statement(P);
        if (s) gp_vec_push_node(&stmts, &count, &cap, s);
        if (P->cur.start == before) {
            gp_parse_error(P, "unexpected token, skipping");
            gp_advance(P);
        }
    }
    gp_expect(P, GP_TOK_RBRACE, "expected '}'");
    B->block.stmt_count = count;
    B->block.stmts = stmts;
    return B;
}

//P nonnull
static WgslAstNode *gp_parse_if_stmt(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_if_stmt: P is NULL");
    gp_expect(P, GP_TOK_IF, "expected 'if'");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");
    WgslAstNode *cond = gp_parse_expr(P);
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");

    WgslAstNode *then_b = NULL;
    if (gp_check(P, GP_TOK_LBRACE))
        then_b = gp_parse_block(P);
    else {
        /* single statement */
        then_b = gp_new_node(P, WGSL_NODE_BLOCK);
        WgslAstNode *s = gp_parse_statement(P);
        if (s) {
            then_b->block.stmt_count = 1;
            then_b->block.stmts = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            then_b->block.stmts[0] = s;
        }
    }

    WgslAstNode *else_b = NULL;
    if (gp_match(P, GP_TOK_ELSE)) {
        if (gp_check(P, GP_TOK_IF))
            else_b = gp_parse_if_stmt(P);
        else if (gp_check(P, GP_TOK_LBRACE))
            else_b = gp_parse_block(P);
        else {
            else_b = gp_new_node(P, WGSL_NODE_BLOCK);
            WgslAstNode *s = gp_parse_statement(P);
            if (s) {
                else_b->block.stmt_count = 1;
                else_b->block.stmts = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
                else_b->block.stmts[0] = s;
            }
        }
    }

    WgslAstNode *I = gp_new_node(P, WGSL_NODE_IF);
    I->if_stmt.cond = cond;
    I->if_stmt.then_branch = then_b;
    I->if_stmt.else_branch = else_b;
    return I;
}

//P nonnull
static WgslAstNode *gp_parse_while_stmt(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_while_stmt: P is NULL");
    gp_expect(P, GP_TOK_WHILE, "expected 'while'");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");
    WgslAstNode *cond = gp_parse_expr(P);
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");
    WgslAstNode *body = gp_check(P, GP_TOK_LBRACE) ? gp_parse_block(P) : gp_parse_statement(P);
    WgslAstNode *W = gp_new_node(P, WGSL_NODE_WHILE);
    W->while_stmt.cond = cond;
    W->while_stmt.body = body;
    return W;
}

//P nonnull
static WgslAstNode *parse_do_while_stmt(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_do_while_stmt: P is NULL");
    gp_expect(P, GP_TOK_DO, "expected 'do'");
    WgslAstNode *body = gp_parse_block(P);
    gp_expect(P, GP_TOK_WHILE, "expected 'while'");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");
    WgslAstNode *cond = gp_parse_expr(P);
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");
    gp_expect(P, GP_TOK_SEMI, "expected ';'");
    WgslAstNode *D = gp_new_node(P, WGSL_NODE_DO_WHILE);
    D->do_while_stmt.body = body;
    D->do_while_stmt.cond = cond;
    return D;
}

//P nonnull
static WgslAstNode *gp_parse_for_stmt(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_for_stmt: P is NULL");
    gp_expect(P, GP_TOK_FOR, "expected 'for'");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");

    /* init */
    WgslAstNode *init = NULL;
    if (gp_check(P, GP_TOK_SEMI)) {
        gp_advance(P);
    } else if (is_type_start(P, &P->cur)) {
        /* type-led declaration in for init */
        WgslAstNode *type = gp_parse_type_node(P);
        if (!gp_check(P, GP_TOK_IDENT)) {
            gp_parse_error(P, "expected variable name");
            if (type) wgsl_free_ast(type);
        } else {
            GpToken vname = P->cur;
            gp_advance(P);
            type = wrap_type_with_array_dims(P, type);
            WgslAstNode *vinit = NULL;
            if (gp_match(P, GP_TOK_EQ))
                vinit = gp_parse_expr(P);
            gp_expect(P, GP_TOK_SEMI, "expected ';'");
            init = gp_new_node(P, WGSL_NODE_VAR_DECL);
            init->var_decl.name = glsl_strndup(vname.start, (size_t)vname.length);
            init->var_decl.type = type;
            init->var_decl.init = vinit;
        }
    } else {
        WgslAstNode *e = gp_parse_expr(P);
        gp_expect(P, GP_TOK_SEMI, "expected ';'");
        init = gp_new_node(P, WGSL_NODE_EXPR_STMT);
        init->expr_stmt.expr = e;
    }

    /* cond */
    WgslAstNode *cond = NULL;
    if (!gp_check(P, GP_TOK_SEMI))
        cond = gp_parse_expr(P);
    gp_expect(P, GP_TOK_SEMI, "expected ';'");

    /* continue */
    WgslAstNode *cont = NULL;
    if (!gp_check(P, GP_TOK_RPAREN))
        cont = gp_parse_expr(P);
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");

    WgslAstNode *body = gp_check(P, GP_TOK_LBRACE) ? gp_parse_block(P) : gp_parse_statement(P);

    WgslAstNode *F = gp_new_node(P, WGSL_NODE_FOR);
    F->for_stmt.init = init;
    F->for_stmt.cond = cond;
    F->for_stmt.cont = cont;
    F->for_stmt.body = body;
    return F;
}

//P nonnull
static WgslAstNode *parse_switch_stmt(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_switch_stmt: P is NULL");
    gp_expect(P, GP_TOK_SWITCH, "expected 'switch'");
    gp_expect(P, GP_TOK_LPAREN, "expected '('");
    WgslAstNode *expr = gp_parse_expr(P);
    gp_expect(P, GP_TOK_RPAREN, "expected ')'");
    gp_expect(P, GP_TOK_LBRACE, "expected '{'");

    int cap = 0, count = 0;
    WgslAstNode **cases = NULL;

    while (!gp_check(P, GP_TOK_RBRACE) && !gp_check(P, GP_TOK_EOF)) {
        WgslAstNode *C = gp_new_node(P, WGSL_NODE_CASE);
        if (gp_match(P, GP_TOK_CASE)) {
            C->case_clause.expr = gp_parse_expr(P);
            gp_expect(P, GP_TOK_COLON, "expected ':'");
        } else if (gp_match(P, GP_TOK_DEFAULT)) {
            C->case_clause.expr = NULL;
            gp_expect(P, GP_TOK_COLON, "expected ':'");
        } else {
            gp_parse_error(P, "expected 'case' or 'default'");
            wgsl_free_ast(C);
            break;
        }

        int scap = 0, scount = 0;
        WgslAstNode **stmts = NULL;
        while (!gp_check(P, GP_TOK_CASE) && !gp_check(P, GP_TOK_DEFAULT) &&
               !gp_check(P, GP_TOK_RBRACE) && !gp_check(P, GP_TOK_EOF)) {
            const char *before = P->cur.start;
            WgslAstNode *s = gp_parse_statement(P);
            if (s) gp_vec_push_node(&stmts, &scount, &scap, s);
            else break;
            if (P->cur.start == before) {
                gp_parse_error(P, "unexpected token, skipping");
                gp_advance(P);
            }
        }
        C->case_clause.stmt_count = scount;
        C->case_clause.stmts = stmts;
        gp_vec_push_node(&cases, &count, &cap, C);
    }
    gp_expect(P, GP_TOK_RBRACE, "expected '}'");

    WgslAstNode *S = gp_new_node(P, WGSL_NODE_SWITCH);
    S->switch_stmt.expr = expr;
    S->switch_stmt.case_count = count;
    S->switch_stmt.cases = cases;
    return S;
}

/* Try to parse a local variable declaration (type-led) */
//P nonnull
static WgslAstNode *try_parse_local_var_decl(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "try_parse_local_var_decl: P is NULL");
    WgslAstNode *type = gp_parse_type_node(P);
    if (!gp_check(P, GP_TOK_IDENT)) {
        /* Not a declaration, backtrack not possible - error */
        gp_parse_error(P, "expected variable name after type");
        if (type) wgsl_free_ast(type);
        return NULL;
    }
    GpToken vname = P->cur;
    gp_advance(P);
    type = wrap_type_with_array_dims(P, type);

    WgslAstNode *init = NULL;
    if (gp_match(P, GP_TOK_EQ))
        init = gp_parse_expr(P);
    gp_expect(P, GP_TOK_SEMI, "expected ';'");

    WgslAstNode *V = gp_new_node(P, WGSL_NODE_VAR_DECL);
    V->var_decl.name = glsl_strndup(vname.start, (size_t)vname.length);
    V->var_decl.type = type;
    V->var_decl.init = init;
    return V;
}

//P nonnull
static WgslAstNode *gp_parse_statement(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_statement: P is NULL");
    /* Empty statement */
    if (gp_match(P, GP_TOK_SEMI))
        return NULL;

    /* Block */
    if (gp_check(P, GP_TOK_LBRACE))
        return gp_parse_block(P);

    /* Control flow */
    if (gp_check(P, GP_TOK_IF)) return gp_parse_if_stmt(P);
    if (gp_check(P, GP_TOK_WHILE)) return gp_parse_while_stmt(P);
    if (gp_check(P, GP_TOK_DO)) return parse_do_while_stmt(P);
    if (gp_check(P, GP_TOK_FOR)) return gp_parse_for_stmt(P);
    if (gp_check(P, GP_TOK_SWITCH)) return parse_switch_stmt(P);

    /* Jump statements */
    if (gp_match(P, GP_TOK_BREAK)) {
        gp_expect(P, GP_TOK_SEMI, "expected ';'");
        return gp_new_node(P, WGSL_NODE_BREAK);
    }
    if (gp_match(P, GP_TOK_CONTINUE)) {
        gp_expect(P, GP_TOK_SEMI, "expected ';'");
        return gp_new_node(P, WGSL_NODE_CONTINUE);
    }
    if (gp_match(P, GP_TOK_DISCARD)) {
        gp_expect(P, GP_TOK_SEMI, "expected ';'");
        return gp_new_node(P, WGSL_NODE_DISCARD);
    }
    if (gp_match(P, GP_TOK_RETURN)) {
        WgslAstNode *e = NULL;
        if (!gp_check(P, GP_TOK_SEMI))
            e = gp_parse_expr(P);
        gp_expect(P, GP_TOK_SEMI, "expected ';'");
        WgslAstNode *R = gp_new_node(P, WGSL_NODE_RETURN);
        R->return_stmt.expr = e;
        return R;
    }

    /* const declaration inside function */
    if (gp_check(P, GP_TOK_CONST)) {
        gp_advance(P);
        return try_parse_local_var_decl(P);
    }

    /* Type-led local variable declaration */
    if (is_type_keyword_tok(&P->cur)) {
        return try_parse_local_var_decl(P);
    }

    /* Check if IDENT is a known struct type followed by another IDENT */
    if (gp_check(P, GP_TOK_IDENT) && is_known_struct(P, &P->cur)) {
        GpLexer L2 = P->L;
        GpToken t2 = gp_lx_next(&L2);
        if (t2.type == GP_TOK_IDENT) {
            return try_parse_local_var_decl(P);
        }
    }

    /* Expression statement */
    WgslAstNode *e = gp_parse_expr(P);
    gp_expect(P, GP_TOK_SEMI, "expected ';'");
    WgslAstNode *ES = gp_new_node(P, WGSL_NODE_EXPR_STMT);
    ES->expr_stmt.expr = e;
    return ES;
}

/* ============================================================================
 * Expression Parsing (Precedence Climbing)
 * ============================================================================ */

//P nonnull
static WgslAstNode *gp_parse_expr(GpParser *P) { wgsl_compiler_assert(P != NULL, "gp_parse_expr: P is NULL"); return gp_parse_assignment(P); }

//P nonnull
static WgslAstNode *gp_parse_assignment(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_assignment: P is NULL");
    WgslAstNode *left = gp_parse_conditional(P);

    /* Simple assignment */
    if (gp_match(P, GP_TOK_EQ)) {
        WgslAstNode *right = gp_parse_assignment(P);
        WgslAstNode *A = gp_new_node(P, WGSL_NODE_ASSIGN);
        A->assign.op = glsl_strdup("=");
        A->assign.lhs = left;
        A->assign.rhs = right;
        return A;
    }

    /* Compound assignment operators */
    struct { GpTokenType tok; const char *op; } compounds[] = {
        {GP_TOK_PLUSEQ, "+="}, {GP_TOK_MINUSEQ, "-="}, {GP_TOK_STAREQ, "*="},
        {GP_TOK_SLASHEQ, "/="}, {GP_TOK_PERCENTEQ, "%="},
        {GP_TOK_AMPEQ, "&="}, {GP_TOK_PIPEEQ, "|="}, {GP_TOK_CARETEQ, "^="},
        {GP_TOK_SHLEQ, "<<="}, {GP_TOK_SHREQ, ">>="},
        {GP_TOK_EOF, NULL}
    };
    for (int i = 0; compounds[i].op; i++) {
        if (gp_match(P, compounds[i].tok)) {
            WgslAstNode *right = gp_parse_assignment(P);
            WgslAstNode *A = gp_new_node(P, WGSL_NODE_ASSIGN);
            A->assign.op = glsl_strdup(compounds[i].op);
            A->assign.lhs = left;
            A->assign.rhs = right;
            return A;
        }
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_conditional(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_conditional: P is NULL");
    WgslAstNode *c = gp_parse_logical_or(P);
    if (gp_match(P, GP_TOK_QMARK)) {
        WgslAstNode *t = gp_parse_assignment(P);
        gp_expect(P, GP_TOK_COLON, "expected ':'");
        WgslAstNode *e = gp_parse_assignment(P);
        WgslAstNode *T = gp_new_node(P, WGSL_NODE_TERNARY);
        T->ternary.cond = c;
        T->ternary.then_expr = t;
        T->ternary.else_expr = e;
        return T;
    }
    return c;
}

//P nonnull
static WgslAstNode *gp_parse_logical_or(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_logical_or: P is NULL");
    WgslAstNode *left = gp_parse_logical_and(P);
    while (gp_match(P, GP_TOK_OROR)) {
        WgslAstNode *right = gp_parse_logical_and(P);
        WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("||");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_logical_and(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_logical_and: P is NULL");
    WgslAstNode *left = parse_bitwise_or(P);
    while (gp_match(P, GP_TOK_ANDAND)) {
        WgslAstNode *right = parse_bitwise_or(P);
        WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("&&");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

//P nonnull
static WgslAstNode *parse_bitwise_or(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_bitwise_or: P is NULL");
    WgslAstNode *left = parse_bitwise_xor(P);
    while (gp_match(P, GP_TOK_PIPE)) {
        WgslAstNode *right = parse_bitwise_xor(P);
        WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("|");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

//P nonnull
static WgslAstNode *parse_bitwise_xor(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_bitwise_xor: P is NULL");
    WgslAstNode *left = parse_bitwise_and(P);
    while (gp_match(P, GP_TOK_CARET)) {
        WgslAstNode *right = parse_bitwise_and(P);
        WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("^");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

//P nonnull
static WgslAstNode *parse_bitwise_and(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "parse_bitwise_and: P is NULL");
    WgslAstNode *left = gp_parse_equality(P);
    while (gp_match(P, GP_TOK_AMP)) {
        WgslAstNode *right = gp_parse_equality(P);
        WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("&");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_equality(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_equality: P is NULL");
    WgslAstNode *left = gp_parse_relational(P);
    for (;;) {
        if (gp_match(P, GP_TOK_EQEQ)) {
            WgslAstNode *r = gp_parse_relational(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("==");
            B->binary.left = left;
            B->binary.right = r;
            left = B;
            continue;
        }
        if (gp_match(P, GP_TOK_NEQ)) {
            WgslAstNode *r = gp_parse_relational(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("!=");
            B->binary.left = left;
            B->binary.right = r;
            left = B;
            continue;
        }
        break;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_relational(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_relational: P is NULL");
    WgslAstNode *left = gp_parse_shift(P);
    for (;;) {
        if (gp_match(P, GP_TOK_LT)) {
            WgslAstNode *r = gp_parse_shift(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_GT)) {
            WgslAstNode *r = gp_parse_shift(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_LE)) {
            WgslAstNode *r = gp_parse_shift(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<=");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_GE)) {
            WgslAstNode *r = gp_parse_shift(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">=");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_shift(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_shift: P is NULL");
    WgslAstNode *left = gp_parse_additive(P);
    for (;;) {
        if (gp_match(P, GP_TOK_SHL)) {
            WgslAstNode *r = gp_parse_additive(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<<");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_SHR)) {
            WgslAstNode *r = gp_parse_additive(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">>");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_additive(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_additive: P is NULL");
    WgslAstNode *left = gp_parse_multiplicative(P);
    for (;;) {
        if (gp_match(P, GP_TOK_PLUS)) {
            WgslAstNode *r = gp_parse_multiplicative(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("+");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_MINUS)) {
            WgslAstNode *r = gp_parse_multiplicative(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("-");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_multiplicative(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_multiplicative: P is NULL");
    WgslAstNode *left = gp_parse_unary(P);
    for (;;) {
        if (gp_match(P, GP_TOK_STAR)) {
            WgslAstNode *r = gp_parse_unary(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("*");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_SLASH)) {
            WgslAstNode *r = gp_parse_unary(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("/");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (gp_match(P, GP_TOK_PERCENT)) {
            WgslAstNode *r = gp_parse_unary(P);
            WgslAstNode *B = gp_new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("%");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

//P nonnull
static WgslAstNode *gp_parse_unary(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_unary: P is NULL");
    if (gp_match(P, GP_TOK_PLUSPLUS)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("++"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (gp_match(P, GP_TOK_MINUSMINUS)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("--"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (gp_match(P, GP_TOK_PLUS)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("+"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (gp_match(P, GP_TOK_MINUS)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("-"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (gp_match(P, GP_TOK_BANG)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("!"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (gp_match(P, GP_TOK_TILDE)) {
        WgslAstNode *e = gp_parse_unary(P);
        WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("~"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    return gp_parse_postfix(P);
}

//P nonnull
static WgslAstNode *gp_parse_postfix(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_postfix: P is NULL");
    WgslAstNode *expr = gp_parse_primary(P);
    for (;;) {
        if (gp_match(P, GP_TOK_LPAREN)) {
            /* Function call / constructor call */
            int cap = 0, count = 0;
            WgslAstNode **args = NULL;
            if (!gp_check(P, GP_TOK_RPAREN)) {
                WgslAstNode *a = gp_parse_assignment(P);
                if (a) gp_vec_push_node(&args, &count, &cap, a);
                while (gp_match(P, GP_TOK_COMMA)) {
                    WgslAstNode *a2 = gp_parse_assignment(P);
                    if (a2) gp_vec_push_node(&args, &count, &cap, a2);
                }
            }
            gp_expect(P, GP_TOK_RPAREN, "expected ')'");
            WgslAstNode *C = gp_new_node(P, WGSL_NODE_CALL);
            C->call.callee = expr;
            C->call.arg_count = count;
            C->call.args = args;
            expr = C;
            continue;
        }
        if (gp_match(P, GP_TOK_LBRACKET)) {
            WgslAstNode *idx = gp_parse_expr(P);
            gp_expect(P, GP_TOK_RBRACKET, "expected ']'");
            WgslAstNode *I = gp_new_node(P, WGSL_NODE_INDEX);
            I->index.object = expr;
            I->index.index = idx;
            expr = I;
            continue;
        }
        if (gp_match(P, GP_TOK_DOT)) {
            if (!gp_check(P, GP_TOK_IDENT)) {
                gp_parse_error(P, "expected member name");
                break;
            }
            GpToken mem = P->cur;
            gp_advance(P);
            WgslAstNode *M = gp_new_node(P, WGSL_NODE_MEMBER);
            M->member.object = expr;
            M->member.member = glsl_strndup(mem.start, (size_t)mem.length);
            expr = M;
            continue;
        }
        if (gp_match(P, GP_TOK_PLUSPLUS)) {
            WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
            U->unary.op = glsl_strdup("++"); U->unary.is_postfix = 1; U->unary.expr = expr;
            expr = U;
            continue;
        }
        if (gp_match(P, GP_TOK_MINUSMINUS)) {
            WgslAstNode *U = gp_new_node(P, WGSL_NODE_UNARY);
            U->unary.op = glsl_strdup("--"); U->unary.is_postfix = 1; U->unary.expr = expr;
            expr = U;
            continue;
        }
        break;
    }
    return expr;
}

//P nonnull
static WgslAstNode *gp_parse_primary(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_primary: P is NULL");
    /* Type keyword as constructor: vec3(...), mat4(...), etc. */
    if (gp_check(P, GP_TOK_IDENT) && is_type_keyword_tok(&P->cur)) {
        GpToken t = P->cur;
        /* Look ahead: if followed by '(', treat as constructor */
        GpLexer L2 = P->L;
        GpToken t2 = gp_lx_next(&L2);
        if (t2.type == GP_TOK_LPAREN) {
            gp_advance(P); /* consume type keyword */
            char *raw = glsl_strndup(t.start, (size_t)t.length);
            const char *norm = normalize_glsl_type(raw);
            WgslAstNode *tn = gp_new_type(P, norm);
            NODE_FREE(raw);
            return tn;
        }
        /* Otherwise fall through to identifier */
    }

    /* Boolean literals */
    if (gp_check(P, GP_TOK_IDENT) && (tok_eq(&P->cur, "true") || tok_eq(&P->cur, "false"))) {
        GpToken t = P->cur;
        gp_advance(P);
        WgslAstNode *n = gp_new_node(P, WGSL_NODE_LITERAL);
        n->literal.lexeme = glsl_strndup(t.start, (size_t)t.length);
        n->literal.kind = WGSL_LIT_INT; /* treat booleans as int-like for now */
        return n;
    }

    /* Identifier */
    if (gp_check(P, GP_TOK_IDENT)) {
        GpToken name = P->cur;
        gp_advance(P);
        return gp_new_ident(P, &name);
    }

    /* Number */
    if (gp_check(P, GP_TOK_NUMBER)) {
        GpToken t = P->cur;
        gp_advance(P);
        return gp_new_literal(P, &t);
    }

    /* Parenthesized expression */
    if (gp_match(P, GP_TOK_LPAREN)) {
        WgslAstNode *e = gp_parse_expr(P);
        gp_expect(P, GP_TOK_RPAREN, "expected ')'");
        return e;
    }

    gp_parse_error(P, "expected expression");
    return gp_new_node(P, WGSL_NODE_LITERAL);
}

/* ============================================================================
 * Program (entry point)
 * ============================================================================ */

//P nonnull
static WgslAstNode *gp_parse_program(GpParser *P) {
    wgsl_compiler_assert(P != NULL, "gp_parse_program: P is NULL");
    WgslAstNode *root = gp_new_node(P, WGSL_NODE_PROGRAM);
    int cap = 0, count = 0;
    WgslAstNode **decls = NULL;

    while (!gp_check(P, GP_TOK_EOF)) {
        const char *before = P->cur.start;
        int ecap = 0, ecount = 0;
        WgslAstNode **extra = NULL;

        WgslAstNode *d = parse_external_declaration(P, &extra, &ecount, &ecap);

        /* Add any extra declarations (e.g., struct from interface block) first */
        for (int i = 0; i < ecount; i++)
            gp_vec_push_node(&decls, &count, &cap, extra[i]);
        if (extra) NODE_FREE(extra);

        if (d)
            gp_vec_push_node(&decls, &count, &cap, d);
        else if (P->had_error)
            break;

        if (P->cur.start == before) {
            gp_parse_error(P, "unexpected token, skipping");
            gp_advance(P);
        }
    }

    root->program.decl_count = count;
    root->program.decls = decls;
    return root;
}

/* ============================================================================
 * Built-in Variable Injection
 * ============================================================================ */

typedef struct {
    const char* glsl_name;
    const char* builtin_name;
    const char* wgsl_type;
    const char* addr_space; /* "in" or "out" */
    WgslStage stage;
} GlslBuiltinDef;

static const GlslBuiltinDef glsl_builtins[] = {
    {"gl_Position",            "position",              "vec4f", "out", WGSL_STAGE_VERTEX},
    {"gl_VertexIndex",         "vertex_index",          "u32",   "in",  WGSL_STAGE_VERTEX},
    {"gl_InstanceIndex",       "instance_index",        "u32",   "in",  WGSL_STAGE_VERTEX},
    {"gl_FragCoord",           "position",              "vec4f", "in",  WGSL_STAGE_FRAGMENT},
    {"gl_FrontFacing",         "front_facing",          "bool",  "in",  WGSL_STAGE_FRAGMENT},
    {"gl_FragDepth",           "frag_depth",            "f32",   "out", WGSL_STAGE_FRAGMENT},
    {"gl_GlobalInvocationID",  "global_invocation_id",  "vec3u", "in",  WGSL_STAGE_COMPUTE},
    {"gl_LocalInvocationID",   "local_invocation_id",   "vec3u", "in",  WGSL_STAGE_COMPUTE},
    {"gl_WorkGroupID",         "workgroup_id",          "vec3u", "in",  WGSL_STAGE_COMPUTE},
    {"gl_NumWorkGroups",       "num_workgroups",        "vec3u", "in",  WGSL_STAGE_COMPUTE},
    {"gl_LocalInvocationIndex","local_invocation_index", "u32",  "in",  WGSL_STAGE_COMPUTE},
    {NULL, NULL, NULL, NULL, WGSL_STAGE_UNKNOWN}
};

//name nonnull
static int ast_uses_ident(const WgslAstNode *n, const char *name) {
    wgsl_compiler_assert(name != NULL, "ast_uses_ident: name is NULL");
    if (!n) return 0;
    switch (n->type) {
    case WGSL_NODE_IDENT:
        return strcmp(n->ident.name, name) == 0;
    case WGSL_NODE_BINARY:
        return ast_uses_ident(n->binary.left, name) || ast_uses_ident(n->binary.right, name);
    case WGSL_NODE_ASSIGN:
        return ast_uses_ident(n->assign.lhs, name) || ast_uses_ident(n->assign.rhs, name);
    case WGSL_NODE_CALL:
        if (ast_uses_ident(n->call.callee, name)) return 1;
        for (int i = 0; i < n->call.arg_count; i++)
            if (ast_uses_ident(n->call.args[i], name)) return 1;
        return 0;
    case WGSL_NODE_MEMBER:
        return ast_uses_ident(n->member.object, name);
    case WGSL_NODE_INDEX:
        return ast_uses_ident(n->index.object, name) || ast_uses_ident(n->index.index, name);
    case WGSL_NODE_UNARY:
        return ast_uses_ident(n->unary.expr, name);
    case WGSL_NODE_TERNARY:
        return ast_uses_ident(n->ternary.cond, name) ||
               ast_uses_ident(n->ternary.then_expr, name) ||
               ast_uses_ident(n->ternary.else_expr, name);
    case WGSL_NODE_BLOCK:
        for (int i = 0; i < n->block.stmt_count; i++)
            if (ast_uses_ident(n->block.stmts[i], name)) return 1;
        return 0;
    case WGSL_NODE_VAR_DECL:
        return ast_uses_ident(n->var_decl.init, name);
    case WGSL_NODE_RETURN:
        return ast_uses_ident(n->return_stmt.expr, name);
    case WGSL_NODE_EXPR_STMT:
        return ast_uses_ident(n->expr_stmt.expr, name);
    case WGSL_NODE_IF:
        return ast_uses_ident(n->if_stmt.cond, name) ||
               ast_uses_ident(n->if_stmt.then_branch, name) ||
               ast_uses_ident(n->if_stmt.else_branch, name);
    case WGSL_NODE_WHILE:
        return ast_uses_ident(n->while_stmt.cond, name) || ast_uses_ident(n->while_stmt.body, name);
    case WGSL_NODE_FOR:
        return ast_uses_ident(n->for_stmt.init, name) || ast_uses_ident(n->for_stmt.cond, name) ||
               ast_uses_ident(n->for_stmt.cont, name) || ast_uses_ident(n->for_stmt.body, name);
    case WGSL_NODE_DO_WHILE:
        return ast_uses_ident(n->do_while_stmt.body, name) || ast_uses_ident(n->do_while_stmt.cond, name);
    case WGSL_NODE_SWITCH:
        if (ast_uses_ident(n->switch_stmt.expr, name)) return 1;
        for (int i = 0; i < n->switch_stmt.case_count; i++)
            if (ast_uses_ident(n->switch_stmt.cases[i], name)) return 1;
        return 0;
    case WGSL_NODE_CASE:
        if (ast_uses_ident(n->case_clause.expr, name)) return 1;
        for (int i = 0; i < n->case_clause.stmt_count; i++)
            if (ast_uses_ident(n->case_clause.stmts[i], name)) return 1;
        return 0;
    case WGSL_NODE_FUNCTION:
        return ast_uses_ident(n->function.body, name);
    default:
        return 0;
    }
}

//def nonnull
static WgslAstNode *make_builtin_global(const GlslBuiltinDef *def) {
    wgsl_compiler_assert(def != NULL, "make_builtin_global: def is NULL");
    WgslAstNode *gv = NODE_ALLOC(WgslAstNode);
    memset(gv, 0, sizeof(*gv));
    gv->type = WGSL_NODE_GLOBAL_VAR;
    gv->global_var.name = glsl_strdup(def->glsl_name);
    gv->global_var.address_space = glsl_strdup(def->addr_space);
    gv->global_var.type = NODE_ALLOC(WgslAstNode);
    memset(gv->global_var.type, 0, sizeof(WgslAstNode));
    gv->global_var.type->type = WGSL_NODE_TYPE;
    gv->global_var.type->type_node.name = glsl_strdup(def->wgsl_type);

    /* Create @builtin(name) attribute */
    WgslAstNode *attr = NODE_ALLOC(WgslAstNode);
    memset(attr, 0, sizeof(*attr));
    attr->type = WGSL_NODE_ATTRIBUTE;
    attr->attribute.name = glsl_strdup("builtin");
    attr->attribute.arg_count = 1;
    attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
    WgslAstNode *arg = NODE_ALLOC(WgslAstNode);
    memset(arg, 0, sizeof(*arg));
    arg->type = WGSL_NODE_IDENT;
    arg->ident.name = glsl_strdup(def->builtin_name);
    attr->attribute.args[0] = arg;

    gv->global_var.attr_count = 1;
    gv->global_var.attrs = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
    gv->global_var.attrs[0] = attr;

    return gv;
}

//ast allowed to be NULL
static void inject_glsl_builtins(WgslAstNode *ast, WgslStage stage) {
    if (!ast || ast->type != WGSL_NODE_PROGRAM || stage == WGSL_STAGE_UNKNOWN)
        return;

    int inject_count = 0;
    const GlslBuiltinDef *to_inject[32];

    for (const GlslBuiltinDef *def = glsl_builtins; def->glsl_name; def++) {
        if (def->stage != stage) continue;
        /* Scan AST for usage of this built-in */
        for (int i = 0; i < ast->program.decl_count; i++) {
            if (ast_uses_ident(ast->program.decls[i], def->glsl_name)) {
                /* PRE: inject_count < 32 */
                wgsl_compiler_assert(inject_count < 32, "inject_glsl_builtins: inject_count=%d >= 32", inject_count);
                if (inject_count < 32) to_inject[inject_count++] = def;
                break;
            }
        }
    }

    if (inject_count == 0) return;

    int old_count = ast->program.decl_count;
    int new_count = old_count + inject_count;
    WgslAstNode **new_decls = (WgslAstNode **)NODE_REALLOC(
        ast->program.decls, (size_t)new_count * sizeof(WgslAstNode *));
    if (!new_decls) return;

    /* Move existing decls to make room - builtins go first */
    memmove(new_decls + inject_count, new_decls, (size_t)old_count * sizeof(WgslAstNode *));
    for (int i = 0; i < inject_count; i++)
        new_decls[i] = make_builtin_global(to_inject[i]);

    ast->program.decls = new_decls;
    ast->program.decl_count = new_count;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

//ast allowed to be NULL
static void inject_stage_attr(WgslAstNode *ast, WgslStage stage) {
    if (!ast || ast->type != WGSL_NODE_PROGRAM || stage == WGSL_STAGE_UNKNOWN)
        return;
    const char *attr_name = NULL;
    if (stage == WGSL_STAGE_VERTEX) attr_name = "vertex";
    else if (stage == WGSL_STAGE_FRAGMENT) attr_name = "fragment";
    else if (stage == WGSL_STAGE_COMPUTE) attr_name = "compute";
    if (!attr_name) return;

    for (int i = 0; i < ast->program.decl_count; i++) {
        WgslAstNode *d = ast->program.decls[i];
        if (!d || d->type != WGSL_NODE_FUNCTION) continue;
        if (!d->function.name || strcmp(d->function.name, "main") != 0) continue;

        WgslAstNode *attr = NODE_ALLOC(WgslAstNode);
        memset(attr, 0, sizeof(*attr));
        attr->type = WGSL_NODE_ATTRIBUTE;
        attr->attribute.name = glsl_strdup(attr_name);

        int old_count = d->function.attr_count;
        int new_count = old_count + 1;
        WgslAstNode **new_attrs = (WgslAstNode **)NODE_REALLOC(
            d->function.attrs, (size_t)new_count * sizeof(WgslAstNode *));
        if (new_attrs) {
            new_attrs[old_count] = attr;
            d->function.attrs = new_attrs;
            d->function.attr_count = new_count;
        } else {
            NODE_FREE(attr->attribute.name);
            NODE_FREE(attr);
        }
        break;
    }
}

//source allowed to be NULL
WgslAstNode *glsl_parse(const char *source, WgslStage stage) {
    GpParser P;
    memset(&P, 0, sizeof(P));
    P.L.src = source ? source : "";
    P.L.src_len = strlen(P.L.src) + 1; /* include null terminator */
    P.L.pos = 0;
    P.L.line = 1;
    P.L.col = 1;
    gp_advance(&P);

    WgslAstNode *ast = gp_parse_program(&P);

    /* Inject synthetic built-in globals and stage attribute on main() */
    inject_glsl_builtins(ast, stage);
    inject_stage_attr(ast, stage);

    /* Free known type tracking */
    for (int i = 0; i < P.known_type_count; i++)
        NODE_FREE(P.known_types[i].name);
    NODE_FREE(P.known_types);

    return ast;
}
