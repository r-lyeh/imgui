// begin file wgsl_resolve.c
#include "simple_wgsl.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    int id;
} NameId;
typedef struct {
    NameId* items;
    int count;
    int cap;
} Scope;
typedef struct {
    const char* name;
    const WgslAstNode* node;
} NamedNode;
typedef struct {
    const WgslAstNode* ident;
    int id;
} IdentBind;

typedef struct {
    const WgslAstNode* fn;
    const char* name;
    int direct_syms_count, direct_syms_cap;
    int* direct_syms; /* symbol ids (1-based) */
    int calls_count, calls_cap;
    const char** calls; /* callee names */
    int is_entry;
    WgslStage stage;
} FnInfo;

struct WgslResolver {
    const WgslAstNode* program;

    WgslSymbolInfo* symbols;
    int sym_count, sym_cap;

    IdentBind* refmap;
    int ref_count, ref_cap;

    NamedNode* structs;
    int struct_count, struct_cap;

    NamedNode* functions;
    int fn_decl_count, fn_decl_cap;

    FnInfo* fn_infos;
    int fn_info_count, fn_info_cap;

    Scope* scopes;
    int scope_count, scope_cap;
};

/* basic utils */
static int str_eq(const char* a, const char* b) { return a && b && strcmp(a, b) == 0; }


//ptr nonnull
//cap nonnull
static void vec_grow(void** ptr, int* cap, size_t elsz) {
    wgsl_compiler_assert(ptr != NULL, "vec_grow: ptr is NULL");
    wgsl_compiler_assert(cap != NULL, "vec_grow: cap is NULL");
    if (*cap == 0) {
        *cap = 8;
        *ptr = NODE_MALLOC((*cap) * elsz);
    } else {
        *cap *= 2;
        *ptr = NODE_REALLOC(*ptr, (*cap) * elsz);
    }
}

/* scopes */
//r nonnull
static void scope_push(WgslResolver* r) {
    wgsl_compiler_assert(r != NULL, "scope_push: r is NULL");
    if (r->scope_count >= r->scope_cap)
        vec_grow((void**)&r->scopes, &r->scope_cap, sizeof(Scope));
    r->scopes[r->scope_count].items = NULL;
    r->scopes[r->scope_count].count = 0;
    r->scopes[r->scope_count].cap = 0;
    r->scope_count++;
}
//r nonnull
static void scope_pop(WgslResolver* r) {
    wgsl_compiler_assert(r != NULL, "scope_pop: r is NULL");
    if (r->scope_count == 0)
        return;
    Scope* s = &r->scopes[r->scope_count - 1];
    NODE_FREE(s->items);
    r->scope_count--;
}
//r nonnull
static void scope_put(WgslResolver* r, const char* name, int id) {
    wgsl_compiler_assert(r != NULL, "scope_put: r is NULL");
    /* PRE: scope_count > 0 for valid scope access */
    wgsl_compiler_assert(r->scope_count > 0, "scope_put: scope_count is 0");
    Scope* s = &r->scopes[r->scope_count - 1];
    if (s->count >= s->cap)
        vec_grow((void**)&s->items, &s->cap, sizeof(NameId));
    /* PRE: s->items != NULL after vec_grow */
    wgsl_compiler_assert(s->items != NULL, "scope_put: s->items NULL after grow");
    s->items[s->count].name = name;
    s->items[s->count].id = id;
    s->count++;
}
//r nonnull
static int scope_get(const WgslResolver* r, const char* name) {
    wgsl_compiler_assert(r != NULL, "scope_get: r is NULL");
    for (int si = r->scope_count - 1; si >= 0; si--) {
        const Scope* s = &r->scopes[si];
        /* PRE: s->items != NULL when s->count > 0 */
        wgsl_compiler_assert(s->count == 0 || s->items != NULL, "scope_get: s->items NULL with count > 0");
        for (int i = s->count - 1; i >= 0; i--)
            if (str_eq(s->items[i].name, name))
                return s->items[i].id;
    }
    return -1;
}

/* symbols */
//r nonnull
static void add_symbol(WgslResolver* r, WgslSymbolInfo s) {
    wgsl_compiler_assert(r != NULL, "add_symbol: r is NULL");
    if (r->sym_count >= r->sym_cap)
        vec_grow((void**)&r->symbols, &r->sym_cap, sizeof(WgslSymbolInfo));
    r->symbols[r->sym_count++] = s;
}
//r nonnull
static int next_id(const WgslResolver* r) { wgsl_compiler_assert(r != NULL, "next_id: r is NULL"); return r->sym_count + 1; } /* 1-based IDs */
//r nonnull
static void bind_ident(WgslResolver* r, const WgslAstNode* ident, int id) {
    wgsl_compiler_assert(r != NULL, "bind_ident: r is NULL");
    if (!ident || ident->type != WGSL_NODE_IDENT)
        return;
    if (r->ref_count >= r->ref_cap)
        vec_grow((void**)&r->refmap, &r->ref_cap, sizeof(IdentBind));
    r->refmap[r->ref_count].ident = ident;
    r->refmap[r->ref_count].id = id;
    r->ref_count++;
}

/* attrs */
static const WgslAstNode* get_attr_node(const WgslAstNode* any, const char* attr) {
    if (!any)
        return NULL;
    int n = 0;
    WgslAstNode* const* arr = NULL;
    switch (any->type) {
    case WGSL_NODE_FUNCTION:
        n = any->function.attr_count;
        arr = any->function.attrs;
        break;
    case WGSL_NODE_PARAM:
        n = any->param.attr_count;
        arr = any->param.attrs;
        break;
    case WGSL_NODE_GLOBAL_VAR:
        n = any->global_var.attr_count;
        arr = any->global_var.attrs;
        break;
    case WGSL_NODE_STRUCT_FIELD:
        n = any->struct_field.attr_count;
        arr = any->struct_field.attrs;
        break;
    default:
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        const WgslAstNode* a = arr[i];
        if (a && a->type == WGSL_NODE_ATTRIBUTE && a->attribute.name && str_eq(a->attribute.name, attr))
            return a;
    }
    return NULL;
}
static const char* attr_first_arg_string(const WgslAstNode* a) {
    if (!a || a->type != WGSL_NODE_ATTRIBUTE || a->attribute.arg_count <= 0)
        return NULL;
    /* PRE: args != NULL when arg_count > 0 */
    wgsl_compiler_assert(a->attribute.args != NULL, "attr_first_arg_string: args NULL with arg_count %d", a->attribute.arg_count);
    const WgslAstNode* x = a->attribute.args[0];
    if (!x)
        return NULL;
    if (x->type == WGSL_NODE_IDENT)
        return x->ident.name;
    if (x->type == WGSL_NODE_LITERAL)
        return x->literal.lexeme;
    return NULL;
}


static int parse_int_lexeme(const char* s) {
    if (!s)
        return -1;
    while (isspace((unsigned char)*s))
        s++;
    int sign = 1;
    if (*s == '-') {
        sign = -1;
        s++;
    }
    int v = 0;
    while (isdigit((unsigned char)*s)) {
        if (v > 214748364) return sign > 0 ? 2147483647 : -1;
        v = v * 10 + (*s - '0');
        s++;
    }
    return sign * v;
}
static int attr_first_arg_int(const WgslAstNode* a) {
    return parse_int_lexeme(attr_first_arg_string(a));
}
static const WgslAstNode* get_ret_attr_node(const WgslAstNode* fn, const char* attr) {
    if (!fn || fn->type != WGSL_NODE_FUNCTION)
        return NULL;
    for (int i = 0; i < fn->function.ret_attr_count; i++) {
        const WgslAstNode* a = fn->function.ret_attrs[i];
        if (a && a->type == WGSL_NODE_ATTRIBUTE && a->attribute.name && str_eq(a->attribute.name, attr))
            return a;
    }
    return NULL;
}
static int ret_attr_first_arg_int(const WgslAstNode* fn, const char* attr) {
    const WgslAstNode* a = get_ret_attr_node(fn, attr);
    if (!a)
        return -1;
    return attr_first_arg_int(a);
}
/* lookups */
static const WgslAstNode* find_function(const WgslAstNode* program, const char* name) {
    if (!program || program->type != WGSL_NODE_PROGRAM)
        return NULL;
    for (int i = 0; i < program->program.decl_count; i++) {
        const WgslAstNode* n = program->program.decls[i];
        if (n && n->type == WGSL_NODE_FUNCTION && n->function.name && str_eq(n->function.name, name))
            return n;
    }
    return NULL;
}
//r nonnull
static void add_struct(WgslResolver* r, const char* name, const WgslAstNode* node) {
    wgsl_compiler_assert(r != NULL, "add_struct: r is NULL");
    if (!name)
        return;
    if (r->struct_count >= r->struct_cap)
        vec_grow((void**)&r->structs, &r->struct_cap, sizeof(NamedNode));
    r->structs[r->struct_count].name = name;
    r->structs[r->struct_count].node = node;
    r->struct_count++;
}
//r nonnull
static const WgslAstNode* get_struct(const WgslResolver* r, const char* name) {
    wgsl_compiler_assert(r != NULL, "get_struct: r is NULL");
    for (int i = 0; i < r->struct_count; i++)
        if (str_eq(r->structs[i].name, name))
            return r->structs[i].node;
    return NULL;
}
//r nonnull
static void add_function_decl(WgslResolver* r, const char* name, const WgslAstNode* node) {
    wgsl_compiler_assert(r != NULL, "add_function_decl: r is NULL");
    if (!name)
        return;
    if (r->fn_decl_count >= r->fn_decl_cap)
        vec_grow((void**)&r->functions, &r->fn_decl_cap, sizeof(NamedNode));
    r->functions[r->fn_decl_count].name = name;
    r->functions[r->fn_decl_count].node = node;
    r->fn_decl_count++;
}
//r nonnull
//fn nonnull
static FnInfo* ensure_fn_info(WgslResolver* r, const WgslAstNode* fn) {
    wgsl_compiler_assert(r != NULL, "ensure_fn_info: r is NULL");
    wgsl_compiler_assert(fn != NULL, "ensure_fn_info: fn is NULL");
    for (int i = 0; i < r->fn_info_count; i++)
        if (r->fn_infos[i].fn == fn)
            return &r->fn_infos[i];
    if (r->fn_info_count >= r->fn_info_cap)
        vec_grow((void**)&r->fn_infos, &r->fn_info_cap, sizeof(FnInfo));
    FnInfo* fi = &r->fn_infos[r->fn_info_count++];
    memset(fi, 0, sizeof(*fi));
    fi->fn = fn;
    fi->name = fn->function.name;
    fi->stage = WGSL_STAGE_UNKNOWN;
    return fi;
}
//fi nonnull
static void record_fn_ref(FnInfo* fi, int sym_id) {
    wgsl_compiler_assert(fi != NULL, "record_fn_ref: fi is NULL");
    if (sym_id <= 0)
        return;
    for (int i = 0; i < fi->direct_syms_count; i++)
        if (fi->direct_syms[i] == sym_id)
            return;
    if (fi->direct_syms_count >= fi->direct_syms_cap)
        vec_grow((void**)&fi->direct_syms, &fi->direct_syms_cap, sizeof(int));
    fi->direct_syms[fi->direct_syms_count++] = sym_id;
}
//fi nonnull
static void record_fn_call(FnInfo* fi, const char* name) {
    wgsl_compiler_assert(fi != NULL, "record_fn_call: fi is NULL");
    if (!name)
        return;
    for (int i = 0; i < fi->calls_count; i++)
        if (str_eq(fi->calls[i], name))
            return;
    if (fi->calls_count >= fi->calls_cap)
        vec_grow((void**)&fi->calls, &fi->calls_cap, sizeof(char*));
    fi->calls[fi->calls_count++] = name;
}

/* numeric helpers from types */
static WgslNumericType elem_from_name(const char* n) {
    if (!n)
        return WGSL_NUM_UNKNOWN;
    if (str_eq(n, "f32"))
        return WGSL_NUM_F32;
    if (str_eq(n, "f16"))
        return WGSL_NUM_F16;
    if (str_eq(n, "i32"))
        return WGSL_NUM_I32;
    if (str_eq(n, "u32"))
        return WGSL_NUM_U32;
    if (str_eq(n, "bool"))
        return WGSL_NUM_BOOL;
    return WGSL_NUM_UNKNOWN;
}

//vecn nonnull
//elem nonnull
static void parse_vec_like_name(const char* name, int* vecn, WgslNumericType* elem) {
    wgsl_compiler_assert(vecn != NULL, "parse_vec_like_name: vecn is NULL");
    wgsl_compiler_assert(elem != NULL, "parse_vec_like_name: elem is NULL");
    *vecn = 0;
    *elem = WGSL_NUM_UNKNOWN;
    if (!name)
        return;
    if (strncmp(name, "vec", 3) != 0)
        return;
    /* PRE: strlen(name) >= 4 for name[3] access */
    wgsl_compiler_assert(strlen(name) >= 4, "parse_vec_like_name: strlen < 4 for '%s'", name);
    int n = name[3] - '0';
    if (n < 2 || n > 4)
        return;
    *vecn = n;
    size_t L = strlen(name);
    if (L >= 5) {
        char c = name[L - 1];
        if (c == 'f')
            *elem = WGSL_NUM_F32;
        else if (c == 'h')
            *elem = WGSL_NUM_F16;
        else if (c == 'i')
            *elem = WGSL_NUM_I32;
        else if (c == 'u')
            *elem = WGSL_NUM_U32;
    }
}

static int type_info(const WgslResolver* r, const WgslAstNode* t, int* components, WgslNumericType* num, int* bytes) {
    (void)r;
    if (!t || t->type != WGSL_NODE_TYPE)
        return 0;
    const char* n = t->type_node.name;
    int c = 0;
    WgslNumericType dt = WGSL_NUM_UNKNOWN;
    int sz = 0;
    if (str_eq(n, "f32") || str_eq(n, "i32") || str_eq(n, "u32") || str_eq(n, "bool")) {
        dt = elem_from_name(n);
        c = 1;
        sz = 4;
    } else if (strncmp(n, "vec", 3) == 0) {
        int fused_n = 0;
        WgslNumericType fused_dt = WGSL_NUM_UNKNOWN;
        parse_vec_like_name(n, &fused_n, &fused_dt);
        if (fused_n) {
            c = fused_n;
            dt = fused_dt == WGSL_NUM_UNKNOWN ? WGSL_NUM_F32 : fused_dt;
            sz = c * 4;
        } else {
            int lane = 0;
            if (n[3] >= '2' && n[3] <= '4')
                lane = n[3] - '0';
            /* PRE: expr_args != NULL when expr_arg_count > 0 */
            wgsl_compiler_assert(t->type_node.expr_arg_count == 0 || t->type_node.expr_args != NULL, "type_info: expr_args NULL with count %d", t->type_node.expr_arg_count);
            if (lane == 0 && t->type_node.expr_arg_count > 0 && t->type_node.expr_args[0] && t->type_node.expr_args[0]->type == WGSL_NODE_LITERAL)
                lane = parse_int_lexeme(t->type_node.expr_args[0]->literal.lexeme);
            if (lane < 2 || lane > 4)
                return 0;
            /* PRE: type_args != NULL when type_arg_count > 0 */
            wgsl_compiler_assert(t->type_node.type_arg_count == 0 || t->type_node.type_args != NULL, "type_info: type_args NULL with count %d", t->type_node.type_arg_count);
            if (t->type_node.type_arg_count > 0 && t->type_node.type_args[0] && t->type_node.type_args[0]->type == WGSL_NODE_TYPE)
                dt = elem_from_name(t->type_node.type_args[0]->type_node.name);
            else
                dt = WGSL_NUM_F32;
            c = lane;
            sz = c * 4;
        }
    } else {
        const WgslAstNode* sd = NULL;
        (void)sd;
        return 0;
    }
    if (components)
        *components = c;
    if (num)
        *num = dt;
    if (bytes)
        *bytes = sz;
    return 1;
}

/* --- New: rough static size calculator for buffer element types --- */
/* Returns 1 if size computed and writes to *out_bytes; returns 0 otherwise. */
static int type_min_size_bytes(const WgslResolver* r, const WgslAstNode* t, int* out_bytes);

static int type_depth_guard = 0;

//r nonnull
//out_bytes nonnull
static int struct_min_size_bytes(const WgslResolver* r, const WgslAstNode* sd, int* out_bytes) {
    wgsl_compiler_assert(r != NULL, "struct_min_size_bytes: r is NULL");
    wgsl_compiler_assert(out_bytes != NULL, "struct_min_size_bytes: out_bytes is NULL");
    if (!sd || sd->type != WGSL_NODE_STRUCT)
        return 0;
    if (type_depth_guard > 32)
        return 0;
    int total = 0;
    for (int i = 0; i < sd->struct_decl.field_count; i++) {
        const WgslAstNode* f = sd->struct_decl.fields[i];
        if (!f || f->type != WGSL_NODE_STRUCT_FIELD)
            continue;
        /* PRE: struct_field.type != NULL for valid field */
        wgsl_compiler_assert(f->struct_field.type != NULL, "struct_min_size_bytes: field %d type is NULL", i);
        int fb = 0;
        if (!type_min_size_bytes(r, f->struct_field.type, &fb))
            return 0;
        total += fb;
    }
    *out_bytes = total;
    return 1;
}

//r nonnull
//out_bytes nonnull
static int type_min_size_bytes(const WgslResolver* r, const WgslAstNode* t, int* out_bytes) {
    wgsl_compiler_assert(r != NULL, "type_min_size_bytes: r is NULL");
    wgsl_compiler_assert(out_bytes != NULL, "type_min_size_bytes: out_bytes is NULL");
    if (!t || t->type != WGSL_NODE_TYPE)
        return 0;

    /* Scalars and vectors */
    int comps = 0, bytes = 0;
    if (type_info(r, t, &comps, NULL, &bytes)) {
        *out_bytes = bytes;
        return 1;
    }

    /* PRE: type_node.name != NULL for valid type */
    wgsl_compiler_assert(t->type_node.name != NULL, "type_min_size_bytes: type_node.name NULL");
    const char* name = t->type_node.name;
    if (!name)
        return 0;

    /* array<T, N> */
    if (str_eq(name, "array")) {
        /* PRE: type_args != NULL when type_arg_count > 0 */
        wgsl_compiler_assert(t->type_node.type_arg_count == 0 || t->type_node.type_args != NULL, "type_min_size_bytes: type_args NULL with count > 0");
        if (t->type_node.type_arg_count <= 0 || !t->type_node.type_args[0])
            return 0;
        const WgslAstNode* elem_t = t->type_node.type_args[0];
        int elem_b = 0;
        if (!type_min_size_bytes(r, elem_t, &elem_b))
            return 0;

        /* Try count from expr args if present and literal. */
        int count = -1;
        /* PRE: expr_args != NULL when expr_arg_count > 0 */
        wgsl_compiler_assert(t->type_node.expr_arg_count == 0 || t->type_node.expr_args != NULL, "type_min_size_bytes: expr_args NULL with count > 0");
        if (t->type_node.expr_arg_count > 0 && t->type_node.expr_args[0]) {
            const WgslAstNode* n = t->type_node.expr_args[0];
            /* PRE: literal.lexeme != NULL for valid literal */
            wgsl_compiler_assert(n->type != WGSL_NODE_LITERAL || n->literal.lexeme != NULL, "type_min_size_bytes: literal.lexeme NULL");
            if (n->type == WGSL_NODE_LITERAL)
                count = parse_int_lexeme(n->literal.lexeme);
        }
        if (count <= 0)
            return 0; /* runtime-sized or unknown: cannot produce static min size */

        /* Ignore explicit @stride for now; assume tightly packed. */
        *out_bytes = elem_b * count;
        return 1;
    }

    /* struct reference */
    const WgslAstNode* sd = get_struct(r, name);
    if (sd) {
        type_depth_guard++;
        int ok = struct_min_size_bytes(r, sd, out_bytes);
        type_depth_guard--;
        return ok;
    }

    /* matrices, textures, samplers, unknown generics: not handled */
    return 0;
}

/* stages */
static int has_attr(const WgslAstNode* fn, const char* name) { return get_attr_node(fn, name) != NULL; }
static WgslStage detect_stage(const WgslAstNode* fn) {
    if (has_attr(fn, "vertex"))
        return WGSL_STAGE_VERTEX;
    if (has_attr(fn, "fragment"))
        return WGSL_STAGE_FRAGMENT;
    if (has_attr(fn, "compute"))
        return WGSL_STAGE_COMPUTE;
    return WGSL_STAGE_UNKNOWN;
}

/* walkers */
//r nonnull
static void walk_expr(WgslResolver* r, FnInfo* fi, const WgslAstNode* e) {
    wgsl_compiler_assert(r != NULL, "walk_expr: r is NULL");
    if (!e)
        return;
    switch (e->type) {
    case WGSL_NODE_IDENT: {
        int id = scope_get(r, e->ident.name);
        if (id >= 0) {
            bind_ident(r, e, id);
            if (fi && id > 0 && id <= r->sym_count && r->symbols[id - 1].kind == WGSL_SYM_GLOBAL)
                record_fn_ref(fi, id);
        }
    } break;
    case WGSL_NODE_CALL: {
        if (e->call.callee && e->call.callee->type == WGSL_NODE_IDENT) {
            /* PRE: ident.name != NULL for valid ident node */
            wgsl_compiler_assert(e->call.callee->ident.name != NULL, "walk_expr: call.callee->ident.name NULL");
            record_fn_call(fi, e->call.callee->ident.name);
        } else
            walk_expr(r, fi, e->call.callee);
        for (int i = 0; i < e->call.arg_count; i++)
            walk_expr(r, fi, e->call.args[i]);
    } break;
    case WGSL_NODE_BINARY:
        walk_expr(r, fi, e->binary.left);
        walk_expr(r, fi, e->binary.right);
        break;
    case WGSL_NODE_ASSIGN:
        walk_expr(r, fi, e->assign.lhs);
        walk_expr(r, fi, e->assign.rhs);
        break;
    case WGSL_NODE_MEMBER:
        walk_expr(r, fi, e->member.object);
        break;
    case WGSL_NODE_INDEX:
        walk_expr(r, fi, e->index.object);
        walk_expr(r, fi, e->index.index);
        break;
    case WGSL_NODE_UNARY:
        walk_expr(r, fi, e->unary.expr);
        break;
    case WGSL_NODE_TERNARY:
        walk_expr(r, fi, e->ternary.cond);
        walk_expr(r, fi, e->ternary.then_expr);
        walk_expr(r, fi, e->ternary.else_expr);
        break;
    default:
        break;
    }
}

//r nonnull
//v nonnull
static void declare_local(WgslResolver* r, const WgslAstNode* fn, const WgslAstNode* v) {
    wgsl_compiler_assert(r != NULL, "declare_local: r is NULL");
    wgsl_compiler_assert(v != NULL, "declare_local: v is NULL");
    if (!v->var_decl.name)
        return;
    WgslSymbolInfo s;
    memset(&s, 0, sizeof(s));
    s.id = next_id(r);
    s.kind = WGSL_SYM_LOCAL;
    s.name = v->var_decl.name;
    s.decl_node = v;
    s.function_node = fn;
    add_symbol(r, s);
    scope_put(r, s.name, s.id);
}

//r nonnull
static void walk_stmt(WgslResolver* r, const WgslAstNode* fn, FnInfo* fi, const WgslAstNode* s) {
    wgsl_compiler_assert(r != NULL, "walk_stmt: r is NULL");
    if (!s)
        return;
    switch (s->type) {
    case WGSL_NODE_BLOCK:
        scope_push(r);
        for (int i = 0; i < s->block.stmt_count; i++)
            walk_stmt(r, fn, fi, s->block.stmts[i]);
        scope_pop(r);
        break;
    case WGSL_NODE_VAR_DECL:
        declare_local(r, fn, s);
        walk_expr(r, fi, s->var_decl.init);
        break;
    case WGSL_NODE_RETURN:
        walk_expr(r, fi, s->return_stmt.expr);
        break;
    case WGSL_NODE_EXPR_STMT:
        walk_expr(r, fi, s->expr_stmt.expr);
        break;
    case WGSL_NODE_IF:
        walk_expr(r, fi, s->if_stmt.cond);
        walk_stmt(r, fn, fi, s->if_stmt.then_branch);
        walk_stmt(r, fn, fi, s->if_stmt.else_branch);
        break;
    case WGSL_NODE_WHILE:
        walk_expr(r, fi, s->while_stmt.cond);
        walk_stmt(r, fn, fi, s->while_stmt.body);
        break;
    case WGSL_NODE_FOR:
        scope_push(r);
        walk_stmt(r, fn, fi, s->for_stmt.init);
        walk_expr(r, fi, s->for_stmt.cond);
        walk_stmt(r, fn, fi, s->for_stmt.cont);
        walk_stmt(r, fn, fi, s->for_stmt.body);
        scope_pop(r);
        break;
    case WGSL_NODE_DO_WHILE:
        walk_stmt(r, fn, fi, s->do_while_stmt.body);
        walk_expr(r, fi, s->do_while_stmt.cond);
        break;
    case WGSL_NODE_SWITCH:
        walk_expr(r, fi, s->switch_stmt.expr);
        for (int i = 0; i < s->switch_stmt.case_count; i++)
            walk_stmt(r, fn, fi, s->switch_stmt.cases[i]);
        break;
    case WGSL_NODE_CASE:
        if (s->case_clause.expr) walk_expr(r, fi, s->case_clause.expr);
        for (int i = 0; i < s->case_clause.stmt_count; i++)
            walk_stmt(r, fn, fi, s->case_clause.stmts[i]);
        break;
    case WGSL_NODE_BREAK:
    case WGSL_NODE_CONTINUE:
    case WGSL_NODE_DISCARD:
        break;
    default:
        break;
    }
}

/* declarations */
//r nonnull
//gv nonnull
static void declare_global_from_globalvar(WgslResolver* r, const WgslAstNode* gv) {
    wgsl_compiler_assert(r != NULL, "declare_global_from_globalvar: r is NULL");
    wgsl_compiler_assert(gv != NULL, "declare_global_from_globalvar: gv is NULL");
    WgslSymbolInfo s;
    memset(&s, 0, sizeof(s));
    s.id = next_id(r);
    s.kind = WGSL_SYM_GLOBAL;
    s.name = gv->global_var.name;
    s.decl_node = gv;

    const WgslAstNode* g = get_attr_node(gv, "group");
    const WgslAstNode* b = get_attr_node(gv, "binding");
    if (g) {
        s.has_group = 1;
        s.group_index = attr_first_arg_int(g);
    }
    if (b) {
        s.has_binding = 1;
        s.binding_index = attr_first_arg_int(b);
    }

    /* New: compute minBindingSize for buffer bindings. */
    s.has_min_binding_size = 0;
    s.min_binding_size = 0;
    if (gv->global_var.address_space &&
        (str_eq(gv->global_var.address_space, "uniform") ||
         str_eq(gv->global_var.address_space, "storage"))) {
        int bytes = 0;
        if (type_min_size_bytes(r, gv->global_var.type, &bytes) && bytes > 0) {
            s.has_min_binding_size = 1;
            s.min_binding_size = bytes;
        }
    }

    add_symbol(r, s);
    scope_put(r, s.name, s.id);
}
//r nonnull
static void declare_global_from_vardecl(WgslResolver* r, const WgslAstNode* vd) {
    wgsl_compiler_assert(r != NULL, "declare_global_from_vardecl: r is NULL");
    if (!vd || vd->type != WGSL_NODE_VAR_DECL || !vd->var_decl.name)
        return;
    WgslSymbolInfo s;
    memset(&s, 0, sizeof(s));
    s.id = next_id(r);
    s.kind = WGSL_SYM_GLOBAL;
    s.name = vd->var_decl.name;
    s.decl_node = vd;
    add_symbol(r, s);
    scope_put(r, s.name, s.id);
}
//r nonnull
//param nonnull
static void declare_param(WgslResolver* r, const WgslAstNode* fn, const WgslAstNode* param) {
    wgsl_compiler_assert(r != NULL, "declare_param: r is NULL");
    wgsl_compiler_assert(param != NULL, "declare_param: param is NULL");
    if (!param->param.name)
        return;
    WgslSymbolInfo s;
    memset(&s, 0, sizeof(s));
    s.id = next_id(r);
    s.kind = WGSL_SYM_PARAM;
    s.name = param->param.name;
    s.decl_node = param;
    s.function_node = fn;
    add_symbol(r, s);
    scope_put(r, s.name, s.id);
}


/* stages */
static int has_attr(const WgslAstNode* fn, const char* name);
static WgslStage detect_stage(const WgslAstNode* fn);

/* build */
WgslResolver* wgsl_resolver_build(const WgslAstNode* program) {
    if (!program || program->type != WGSL_NODE_PROGRAM)
        return NULL;
    WgslResolver* r = (WgslResolver*)NODE_ALLOC(WgslResolver);
    r->program = program;
    scope_push(r);

    for (int i = 0; i < program->program.decl_count; i++) {
        const WgslAstNode* d = program->program.decls[i];
        if (!d)
            continue;
        if (d->type == WGSL_NODE_STRUCT)
            add_struct(r, d->struct_decl.name, d);
        if (d->type == WGSL_NODE_FUNCTION && d->function.name)
            add_function_decl(r, d->function.name, d);
    }
    for (int i = 0; i < program->program.decl_count; i++) {
        const WgslAstNode* d = program->program.decls[i];
        if (!d)
            continue;
        if (d->type == WGSL_NODE_GLOBAL_VAR && d->global_var.name)
            declare_global_from_globalvar(r, d);
        if (d->type == WGSL_NODE_VAR_DECL && d->var_decl.name)
            declare_global_from_vardecl(r, d);
    }
    for (int i = 0; i < program->program.decl_count; i++) {
        const WgslAstNode* d = program->program.decls[i];
        if (!d || d->type != WGSL_NODE_FUNCTION)
            continue;
        scope_push(r);
        for (int p = 0; p < d->function.param_count; p++) {
            const WgslAstNode* prm = d->function.params[p];
            if (prm && prm->type == WGSL_NODE_PARAM && prm->param.name)
                declare_param(r, d, prm);
        }
        FnInfo* fi = ensure_fn_info(r, d);
        fi->stage = detect_stage(d);
        fi->is_entry = (fi->stage != WGSL_STAGE_UNKNOWN);
        walk_stmt(r, d, fi, d->function.body);
        scope_pop(r);
    }
    return r;
}

/* free */
void wgsl_resolver_free(WgslResolver* r) {
    if (!r)
        return;
    for (int i = 0; i < r->scope_count; i++)
        NODE_FREE(r->scopes[i].items);
    NODE_FREE(r->scopes);
    NODE_FREE(r->symbols);
    NODE_FREE(r->refmap);
    NODE_FREE(r->structs);
    NODE_FREE(r->functions);
    if (r->fn_infos) {
        for (int i = 0; i < r->fn_info_count; i++) {
            NODE_FREE(r->fn_infos[i].direct_syms);
            NODE_FREE(r->fn_infos[i].calls);
        }
    }
    NODE_FREE(r->fn_infos);
    NODE_FREE(r);
}

/* copies */
static const WgslSymbolInfo* copy_symbols_subset(const WgslResolver* r, const int* ids, int id_count, int* out_count) {
    if (!r || id_count <= 0) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    /* PRE: r->symbols != NULL when r->sym_count > 0 */
    wgsl_compiler_assert(r->sym_count == 0 || r->symbols != NULL, "copy_symbols_subset: symbols NULL with count %d", r->sym_count);
    WgslSymbolInfo* arr = (WgslSymbolInfo*)NODE_MALLOC(sizeof(WgslSymbolInfo) * id_count);
    int k = 0;
    for (int i = 0; i < id_count; i++) {
        int id = ids[i];
        if (id <= 0 || id > r->sym_count)
            continue;
        arr[k++] = r->symbols[id - 1];
    }
    if (out_count)
        *out_count = k;
    return arr;
}
const WgslSymbolInfo* wgsl_resolver_all_symbols(const WgslResolver* r, int* out_count) {
    if (!r || r->sym_count == 0) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    WgslSymbolInfo* arr = (WgslSymbolInfo*)NODE_MALLOC(sizeof(WgslSymbolInfo) * r->sym_count);
    for (int i = 0; i < r->sym_count; i++)
        arr[i] = r->symbols[i];
    if (out_count)
        *out_count = r->sym_count;
    return arr;
}
const WgslSymbolInfo* wgsl_resolver_globals(const WgslResolver* r, int* out_count) {
    if (!r) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    int *ids = NULL, cap = 0, cnt = 0;
    for (int i = 0; i < r->sym_count; i++) {
        if (r->symbols[i].kind == WGSL_SYM_GLOBAL) {
            if (cnt >= cap)
                vec_grow((void**)&ids, &cap, sizeof(int));
            ids[cnt++] = r->symbols[i].id;
        }
    }
    const WgslSymbolInfo* res = copy_symbols_subset(r, ids, cnt, out_count);
    NODE_FREE(ids);
    return res;
}
const WgslSymbolInfo* wgsl_resolver_binding_vars(const WgslResolver* r, int* out_count) {
    if (!r) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    int *ids = NULL, cap = 0, cnt = 0;
    for (int i = 0; i < r->sym_count; i++) {
        const WgslSymbolInfo* s = &r->symbols[i];
        if (s->kind == WGSL_SYM_GLOBAL && s->has_group && s->has_binding) {
            if (cnt >= cap)
                vec_grow((void**)&ids, &cap, sizeof(int));
            ids[cnt++] = s->id;
        }
    }
    const WgslSymbolInfo* res = copy_symbols_subset(r, ids, cnt, out_count);
    NODE_FREE(ids);
    return res;
}

/* ident binding */
int wgsl_resolver_ident_symbol_id(const WgslResolver* r, const WgslAstNode* ident_node) {
    if (!r || !ident_node)
        return -1;
    for (int i = 0; i < r->ref_count; i++)
        if (r->refmap[i].ident == ident_node)
            return r->refmap[i].id;
    return -1;
}

int wgsl_resolver_fragment_outputs(const WgslResolver* r, const char* fragment_entry_name, WgslFragmentOutput** frag_outputs) {
    if (frag_outputs)
        *frag_outputs = NULL;
    if (!r || !fragment_entry_name || !frag_outputs)
        return 0;
    const WgslAstNode* fn = find_function(r->program, fragment_entry_name);
    if (!fn || detect_stage(fn) != WGSL_STAGE_FRAGMENT)
        return 0;

    int count = 0, cap = 0;
    WgslFragmentOutput* arr = NULL;

    int loc_ret = ret_attr_first_arg_int(fn, "location");
    if (loc_ret >= 0 && fn->function.return_type) {
        int comps = 0, bytes = 0;
        WgslNumericType nt = WGSL_NUM_UNKNOWN;
        if (type_info(r, fn->function.return_type, &comps, &nt, &bytes)) {
            cap = 1;
            arr = (WgslFragmentOutput*)NODE_MALLOC(sizeof(WgslFragmentOutput) * 1);
            arr[0].location = loc_ret;
            arr[0].component_count = comps;
            arr[0].numeric_type = nt;
            count = 1;
        }
    } else if (fn->function.return_type && fn->function.return_type->type == WGSL_NODE_TYPE) {
        /* PRE: type_node.name != NULL for valid return type */
        wgsl_compiler_assert(fn->function.return_type->type_node.name != NULL, "wgsl_resolver_fragment_outputs: return_type->type_node.name NULL");
        const char* tn = fn->function.return_type->type_node.name;
        const WgslAstNode* sd = get_struct(r, tn);
        if (sd) {
            for (int f = 0; f < sd->struct_decl.field_count; f++) {
                const WgslAstNode* fld = sd->struct_decl.fields[f];
                if (!fld || fld->type != WGSL_NODE_STRUCT_FIELD)
                    continue;
                const WgslAstNode* loca = get_attr_node(fld, "location");
                if (!loca)
                    continue;
                int loc = attr_first_arg_int(loca);
                int comps = 0, bytes = 0;
                WgslNumericType nt = WGSL_NUM_UNKNOWN;
                if (!type_info(r, fld->struct_field.type, &comps, &nt, &bytes))
                    continue;
                if (count >= cap) {
                    cap = cap ? cap * 2 : 4;
                    arr = (WgslFragmentOutput*)NODE_REALLOC(arr, sizeof(WgslFragmentOutput) * cap);
                }
                arr[count].location = loc;
                arr[count].component_count = comps;
                arr[count].numeric_type = nt;
                count++;
            }
        }
    }

    *frag_outputs = arr;
    return count;
}


/* vertex inputs */
int wgsl_resolver_vertex_inputs(const WgslResolver* r, const char* vertex_entry_name, WgslVertexSlot** out_slots) {
    if (out_slots)
        *out_slots = NULL;
    if (!r || !vertex_entry_name || !out_slots)
        return 0;
    const WgslAstNode* fn = find_function(r->program, vertex_entry_name);
    if (!fn || detect_stage(fn) != WGSL_STAGE_VERTEX)
        return 0;
    WgslVertexSlot* arr = NULL;
    int count = 0, cap = 0;
    for (int i = 0; i < fn->function.param_count; i++) {
        const WgslAstNode* p = fn->function.params[i];
        if (!p || p->type != WGSL_NODE_PARAM)
            continue;
        if (get_attr_node(p, "builtin"))
            continue;
        const WgslAstNode* a_loc = get_attr_node(p, "location");
        if (a_loc) {
            int loc = attr_first_arg_int(a_loc);
            int comps = 0, bytes = 0;
            WgslNumericType nt = WGSL_NUM_UNKNOWN;
            /* PRE: param.type != NULL for valid param with location */
            wgsl_compiler_assert(p->param.type != NULL, "wgsl_resolver_vertex_inputs: param %d type is NULL", i);
            type_info(r, p->param.type, &comps, &nt, &bytes);
            if (count >= cap) {
                cap = cap ? cap * 2 : 8;
                arr = (WgslVertexSlot*)NODE_REALLOC(arr, sizeof(WgslVertexSlot) * cap);
            }
            arr[count].location = loc;
            arr[count].component_count = comps;
            arr[count].numeric_type = nt;
            arr[count].byte_size = bytes;
            count++;
            continue;
        }
        if (p->param.type && p->param.type->type == WGSL_NODE_TYPE) {
            /* PRE: type_node.name != NULL for valid type */
            wgsl_compiler_assert(p->param.type->type_node.name != NULL, "wgsl_resolver_vertex_inputs: param.type->type_node.name NULL");
            const char* tn = p->param.type->type_node.name;
            const WgslAstNode* sd = get_struct(r, tn);
            if (sd) {
                for (int f = 0; f < sd->struct_decl.field_count; f++) {
                    const WgslAstNode* fld = sd->struct_decl.fields[f];
                    if (!fld || fld->type != WGSL_NODE_STRUCT_FIELD)
                        continue;
                    const WgslAstNode* loca = get_attr_node(fld, "location");
                    if (!loca)
                        continue;
                    int loc = attr_first_arg_int(loca);
                    int comps = 0, bytes = 0;
                    WgslNumericType nt = WGSL_NUM_UNKNOWN;
                    type_info(r, fld->struct_field.type, &comps, &nt, &bytes);
                    if (count >= cap) {
                        cap = cap ? cap * 2 : 8;
                        arr = (WgslVertexSlot*)NODE_REALLOC(arr, sizeof(WgslVertexSlot) * cap);
                    }
                    arr[count].location = loc;
                    arr[count].component_count = comps;
                    arr[count].numeric_type = nt;
                    arr[count].byte_size = bytes;
                    count++;
                }
            }
        }
    }
    *out_slots = arr;
    return count;
}

/* entrypoints */
const WgslResolverEntrypoint* wgsl_resolver_entrypoints(const WgslResolver* r, int* out_count) {
    if (!r) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    int n = 0;
    for (int i = 0; i < r->fn_info_count; i++)
        if (r->fn_infos[i].is_entry)
            n++;
    if (n == 0) {
        if (out_count)
            *out_count = 0;
        return NULL;
    }
    WgslResolverEntrypoint* eps = (WgslResolverEntrypoint*)NODE_MALLOC(sizeof(WgslResolverEntrypoint) * n);
    int k = 0;
    for (int i = 0; i < r->fn_info_count; i++)
        if (r->fn_infos[i].is_entry) {
            eps[k].name = r->fn_infos[i].name;
            eps[k].stage = r->fn_infos[i].stage;
            eps[k].function_node = r->fn_infos[i].fn;
            k++;
        }
    if (out_count)
        *out_count = k;
    return eps;
}

/* transitive refs */
//r nonnull
static const FnInfo* find_fninfo_by_name(const WgslResolver* r, const char* name) {
    wgsl_compiler_assert(r != NULL, "find_fninfo_by_name: r is NULL");
    for (int i = 0; i < r->fn_info_count; i++)
        if (str_eq(r->fn_infos[i].name, name))
            return &r->fn_infos[i];
    return NULL;
}
//r nonnull
//visited_fn nonnull
//keep_sym nonnull
static void dfs_collect(const WgslResolver* r, const FnInfo* fi, char* visited_fn, char* keep_sym) {
    wgsl_compiler_assert(r != NULL, "dfs_collect: r is NULL");
    wgsl_compiler_assert(visited_fn != NULL, "dfs_collect: visited_fn is NULL");
    wgsl_compiler_assert(keep_sym != NULL, "dfs_collect: keep_sym is NULL");
    if (!fi)
        return;
    int self_idx = -1;
    for (int j = 0; j < r->fn_info_count; j++)
        if (r->fn_infos[j].fn == fi->fn) {
            self_idx = j;
            break;
        }
    if (self_idx >= 0 && visited_fn[self_idx])
        return;
    if (self_idx >= 0)
        visited_fn[self_idx] = 1;

    for (int i = 0; i < fi->direct_syms_count; i++) {
        int id = fi->direct_syms[i];
        if (id > 0 && id <= r->sym_count && r->symbols[id - 1].kind == WGSL_SYM_GLOBAL)
            keep_sym[id] = 1;
    }
    for (int c = 0; c < fi->calls_count; c++) {
        const FnInfo* callee = find_fninfo_by_name(r, fi->calls[c]);
        if (callee)
            dfs_collect(r, callee, visited_fn, keep_sym);
    }
}
static const WgslSymbolInfo* entrypoint_syms(const WgslResolver* r, const char* entry_name, int only_binding, int* out_count) {
    if (out_count)
        *out_count = 0;
    if (!r || !entry_name)
        return NULL;
    const FnInfo* root = find_fninfo_by_name(r, entry_name);
    if (!root || !root->is_entry)
        return NULL;

    char* keep = (char*)NODE_MALLOC((size_t)(r->sym_count + 1));
    memset(keep, 0, (size_t)(r->sym_count + 1));
    char* visited = (char*)NODE_MALLOC((size_t)(r->fn_info_count + 1));
    memset(visited, 0, (size_t)(r->fn_info_count + 1));
    dfs_collect(r, root, visited, keep);

    int *ids = NULL, cap = 0, cnt = 0;
    for (int id = 1; id <= r->sym_count; id++) {
        if (!keep[id])
            continue;
        const WgslSymbolInfo* s = &r->symbols[id - 1];
        if (s->kind != WGSL_SYM_GLOBAL)
            continue;
        if (only_binding && !(s->has_group && s->has_binding))
            continue;
        if (cnt >= cap)
            vec_grow((void**)&ids, &cap, sizeof(int));
        ids[cnt++] = id;
    }
    NODE_FREE(keep);
    NODE_FREE(visited);
    const WgslSymbolInfo* res = copy_symbols_subset(r, ids, cnt, out_count);
    NODE_FREE(ids);
    return res;
}
const WgslSymbolInfo* wgsl_resolver_entrypoint_globals(const WgslResolver* r, const char* entry_name, int* out_count) { return entrypoint_syms(r, entry_name, 0, out_count); }
const WgslSymbolInfo* wgsl_resolver_entrypoint_binding_vars(const WgslResolver* r, const char* entry_name, int* out_count) { return entrypoint_syms(r, entry_name, 1, out_count); }

/* free helper */
void wgsl_resolve_free(void* p) {
    if (p)
        NODE_FREE(p);
}

// end file wgsl_resolve.c
