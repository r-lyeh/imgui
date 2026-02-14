/*
 * SSIR to MSL (Metal Shading Language) Converter
 *
 * Converts SSIR (Simple Shader IR) to MSL text.
 * Modeled after ssir_to_glsl.c with Metal-specific syntax and conventions.
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef STM_MALLOC
#define STM_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STM_REALLOC
#define STM_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STM_FREE
#define STM_FREE(p) free((p))
#endif

/* ============================================================================
 * String Buffer
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int indent;
} MslBuf;

static void mb_init(MslBuf *b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    b->indent = 0;
}

static void mb_free(MslBuf *b) {
    STM_FREE(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static int mb_reserve(MslBuf *b, size_t need) {
    if (b->len + need + 1 <= b->cap) return 1;
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->len + need + 1) nc *= 2;
    char *nd = (char *)STM_REALLOC(b->data, nc);
    if (!nd) return 0;
    b->data = nd;
    b->cap = nc;
    return 1;
}

static void mb_append(MslBuf *b, const char *s) {
    size_t sl = strlen(s);
    if (!mb_reserve(b, sl)) return;
    memcpy(b->data + b->len, s, sl);
    b->len += sl;
    b->data[b->len] = '\0';
}

static void mb_appendf(MslBuf *b, const char *fmt, ...) {
    char buf[1024];
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    if (n > 0) mb_append(b, buf);
}

static void mb_indent(MslBuf *b) {
    for (int i = 0; i < b->indent; i++) mb_append(b, "    ");
}

static void mb_nl(MslBuf *b) { mb_append(b, "\n"); }

/* ============================================================================
 * Interface Field Tracking (for entry point I/O)
 * ============================================================================ */

typedef struct {
    uint32_t global_id;
    uint32_t pointee_type;
    SsirBuiltinVar builtin;
    bool has_location;
    uint32_t location;
    SsirInterpolation interp;
    SsirInterpolationSampling interp_sampling;
    bool invariant;
    SsirAddressSpace space;
} MslInterfaceField;

/* ============================================================================
 * Converter Context
 * ============================================================================ */

typedef struct {
    const SsirModule *mod;
    SsirToMslOptions opts;
    MslBuf sb;

    char **id_names;
    uint32_t id_names_cap;

    const SsirFunction *current_func;
    SsirEntryPoint *active_ep;

    /* Entry point I/O tracking */
    MslInterfaceField *output_fields;
    uint32_t output_field_count;
    MslInterfaceField *input_fields;
    uint32_t input_field_count;
    bool in_entry_point;
    bool has_stage_in;

    uint32_t *use_counts;
    SsirInst **inst_map;
    uint32_t inst_map_cap;

    char last_error[256];
} MslCtx;

static void mctx_init(MslCtx *c, const SsirModule *m, const SsirToMslOptions *o) {
    memset(c, 0, sizeof(*c));
    c->mod = m;
    if (o) c->opts = *o;
    mb_init(&c->sb);
    c->id_names_cap = m->next_id;
    c->id_names = (char **)STM_MALLOC(c->id_names_cap * sizeof(char *));
}

static void mctx_free(MslCtx *c) {
    if (c->id_names) {
        for (uint32_t i = 0; i < c->id_names_cap; i++) STM_FREE(c->id_names[i]);
        STM_FREE(c->id_names);
    }
    STM_FREE(c->use_counts);
    STM_FREE(c->inst_map);
    STM_FREE(c->output_fields);
    STM_FREE(c->input_fields);
    mb_free(&c->sb);
}

static const char *get_name(MslCtx *c, uint32_t id) {
    if (id < c->id_names_cap && c->id_names[id]) return c->id_names[id];
    static char buf[32];
    snprintf(buf, sizeof(buf), "_v%u", id);
    return buf;
}

static int is_msl_reserved(const char *n) {
    static const char *kw[] = {
        "using", "namespace", "metal", "kernel", "vertex", "fragment",
        "device", "constant", "threadgroup", "thread", "texture",
        "sampler", "half", "float", "int", "uint", "bool", "char",
        "short", "ushort", "uchar", "void", "atomic", "packed_float3",
        "class", "struct", "union", "enum", "template", "this",
        "true", "false", "if", "else", "for", "while", "do",
        "switch", "case", "default", "break", "continue", "return",
        "discard_fragment", "as_type", "static_cast",
        NULL
    };
    for (const char **p = kw; *p; p++)
        if (strcmp(n, *p) == 0) return 1;
    return 0;
}

static void set_name(MslCtx *c, uint32_t id, const char *n) {
    if (id >= c->id_names_cap) return;
    STM_FREE(c->id_names[id]);
    if (n && is_msl_reserved(n)) {
        size_t len = strlen(n) + 2;
        char *tmp = (char *)STM_MALLOC(len);
        snprintf(tmp, len, "_%s", n);
        c->id_names[id] = tmp;
    } else {
        c->id_names[id] = n ? strdup(n) : NULL;
    }
}

/* ============================================================================
 * Output Field Helpers
 * ============================================================================ */

static const MslInterfaceField *find_output_field(MslCtx *c, uint32_t global_id) {
    for (uint32_t i = 0; i < c->output_field_count; i++) {
        if (c->output_fields[i].global_id == global_id) return &c->output_fields[i];
    }
    return NULL;
}

static const MslInterfaceField *find_input_field(MslCtx *c, uint32_t global_id) {
    for (uint32_t i = 0; i < c->input_field_count; i++) {
        if (c->input_fields[i].global_id == global_id) return &c->input_fields[i];
    }
    return NULL;
}

/* ============================================================================
 * MSL Builtin Variable Attribute Mappings
 * ============================================================================ */

static const char *builtin_to_msl_attr(SsirBuiltinVar bv, bool is_output) {
    switch (bv) {
    case SSIR_BUILTIN_VERTEX_INDEX:           return "vertex_id";
    case SSIR_BUILTIN_INSTANCE_INDEX:         return "instance_id";
    case SSIR_BUILTIN_POSITION:               return "position";
    case SSIR_BUILTIN_FRONT_FACING:           return "front_facing";
    case SSIR_BUILTIN_FRAG_DEPTH:             return "depth(any)";
    case SSIR_BUILTIN_SAMPLE_INDEX:           return "sample_id";
    case SSIR_BUILTIN_SAMPLE_MASK:            return "sample_mask";
    case SSIR_BUILTIN_LOCAL_INVOCATION_ID:    return "thread_position_in_threadgroup";
    case SSIR_BUILTIN_LOCAL_INVOCATION_INDEX: return "thread_index_in_threadgroup";
    case SSIR_BUILTIN_GLOBAL_INVOCATION_ID:   return "thread_position_in_grid";
    case SSIR_BUILTIN_WORKGROUP_ID:           return "threadgroup_position_in_grid";
    case SSIR_BUILTIN_NUM_WORKGROUPS:         return "threadgroups_per_grid";
    case SSIR_BUILTIN_POINT_SIZE:             return "point_size";
    case SSIR_BUILTIN_CLIP_DISTANCE:          return "clip_distance";
    case SSIR_BUILTIN_LAYER:                  return "render_target_array_index";
    case SSIR_BUILTIN_VIEWPORT_INDEX:         return "viewport_array_index";
    case SSIR_BUILTIN_FRAG_COORD:             return "position";
    case SSIR_BUILTIN_HELPER_INVOCATION:      return "helper_invocation";
    case SSIR_BUILTIN_BASE_VERTEX:            return "base_vertex";
    case SSIR_BUILTIN_BASE_INSTANCE:          return "base_instance";
    case SSIR_BUILTIN_SUBGROUP_SIZE:          return "thread_execution_width";
    case SSIR_BUILTIN_SUBGROUP_INVOCATION_ID: return "thread_index_in_simdgroup";
    case SSIR_BUILTIN_SUBGROUP_ID:            return "simdgroup_index_in_threadgroup";
    case SSIR_BUILTIN_NUM_SUBGROUPS:          return "simdgroups_per_threadgroup";
    default: return NULL;
    }
}

/* ============================================================================
 * MSL Builtin Function Mappings
 * ============================================================================ */

static const char *bfunc_to_msl(SsirBuiltinId id) {
    switch (id) {
    case SSIR_BUILTIN_SIN:           return "sin";
    case SSIR_BUILTIN_COS:           return "cos";
    case SSIR_BUILTIN_TAN:           return "tan";
    case SSIR_BUILTIN_ASIN:          return "asin";
    case SSIR_BUILTIN_ACOS:          return "acos";
    case SSIR_BUILTIN_ATAN:          return "atan";
    case SSIR_BUILTIN_ATAN2:         return "atan2";
    case SSIR_BUILTIN_SINH:          return "sinh";
    case SSIR_BUILTIN_COSH:          return "cosh";
    case SSIR_BUILTIN_TANH:          return "tanh";
    case SSIR_BUILTIN_ASINH:         return "asinh";
    case SSIR_BUILTIN_ACOSH:         return "acosh";
    case SSIR_BUILTIN_ATANH:         return "atanh";
    case SSIR_BUILTIN_EXP:           return "exp";
    case SSIR_BUILTIN_EXP2:          return "exp2";
    case SSIR_BUILTIN_LOG:           return "log";
    case SSIR_BUILTIN_LOG2:          return "log2";
    case SSIR_BUILTIN_POW:           return "pow";
    case SSIR_BUILTIN_SQRT:          return "sqrt";
    case SSIR_BUILTIN_INVERSESQRT:   return "rsqrt";
    case SSIR_BUILTIN_ABS:           return "abs";
    case SSIR_BUILTIN_SIGN:          return "sign";
    case SSIR_BUILTIN_FLOOR:         return "floor";
    case SSIR_BUILTIN_CEIL:          return "ceil";
    case SSIR_BUILTIN_ROUND:         return "round";
    case SSIR_BUILTIN_TRUNC:         return "trunc";
    case SSIR_BUILTIN_FRACT:         return "fract";
    case SSIR_BUILTIN_MIN:           return "min";
    case SSIR_BUILTIN_MAX:           return "max";
    case SSIR_BUILTIN_CLAMP:         return "clamp";
    case SSIR_BUILTIN_SATURATE:      return "saturate";
    case SSIR_BUILTIN_MIX:           return "mix";
    case SSIR_BUILTIN_STEP:          return "step";
    case SSIR_BUILTIN_SMOOTHSTEP:    return "smoothstep";
    case SSIR_BUILTIN_DOT:           return "dot";
    case SSIR_BUILTIN_CROSS:         return "cross";
    case SSIR_BUILTIN_LENGTH:        return "length";
    case SSIR_BUILTIN_DISTANCE:      return "distance";
    case SSIR_BUILTIN_NORMALIZE:     return "normalize";
    case SSIR_BUILTIN_FACEFORWARD:   return "faceforward";
    case SSIR_BUILTIN_REFLECT:       return "reflect";
    case SSIR_BUILTIN_REFRACT:       return "refract";
    case SSIR_BUILTIN_ALL:           return "all";
    case SSIR_BUILTIN_ANY:           return "any";
    case SSIR_BUILTIN_SELECT:        return "select";
    case SSIR_BUILTIN_COUNTBITS:     return "popcount";
    case SSIR_BUILTIN_REVERSEBITS:   return "reverse_bits";
    case SSIR_BUILTIN_FIRSTLEADINGBIT: return "clz";
    case SSIR_BUILTIN_FIRSTTRAILINGBIT: return "ctz";
    case SSIR_BUILTIN_EXTRACTBITS:   return "extract_bits";
    case SSIR_BUILTIN_INSERTBITS:    return "insert_bits";
    case SSIR_BUILTIN_DPDX:          return "dfdx";
    case SSIR_BUILTIN_DPDY:          return "dfdy";
    case SSIR_BUILTIN_FWIDTH:        return "fwidth";
    case SSIR_BUILTIN_DPDX_COARSE:   return "dfdx";
    case SSIR_BUILTIN_DPDY_COARSE:   return "dfdy";
    case SSIR_BUILTIN_DPDX_FINE:     return "dfdx";
    case SSIR_BUILTIN_DPDY_FINE:     return "dfdy";
    case SSIR_BUILTIN_FMA:           return "fma";
    case SSIR_BUILTIN_ISINF:         return "isinf";
    case SSIR_BUILTIN_ISNAN:         return "isnan";
    case SSIR_BUILTIN_DEGREES:       return NULL; /* special: x * 57.2957795131f */
    case SSIR_BUILTIN_RADIANS:       return NULL; /* special: x * 0.0174532925199f */
    case SSIR_BUILTIN_MODF:          return "modf";
    case SSIR_BUILTIN_FREXP:         return "frexp";
    case SSIR_BUILTIN_LDEXP:         return "ldexp";
    case SSIR_BUILTIN_DETERMINANT:   return "determinant";
    case SSIR_BUILTIN_TRANSPOSE:     return "transpose";
    case SSIR_BUILTIN_PACK4X8SNORM:  return NULL; /* special */
    case SSIR_BUILTIN_PACK4X8UNORM:  return NULL;
    case SSIR_BUILTIN_PACK2X16SNORM: return NULL;
    case SSIR_BUILTIN_PACK2X16UNORM: return NULL;
    case SSIR_BUILTIN_PACK2X16FLOAT: return NULL;
    case SSIR_BUILTIN_UNPACK4X8SNORM:  return NULL;
    case SSIR_BUILTIN_UNPACK4X8UNORM:  return NULL;
    case SSIR_BUILTIN_UNPACK2X16SNORM: return NULL;
    case SSIR_BUILTIN_UNPACK2X16UNORM: return NULL;
    case SSIR_BUILTIN_UNPACK2X16FLOAT: return NULL;
    case SSIR_BUILTIN_SUBGROUP_BALLOT: return "simd_ballot";
    case SSIR_BUILTIN_SUBGROUP_BROADCAST: return "simd_broadcast";
    case SSIR_BUILTIN_SUBGROUP_ADD: return "simd_sum";
    case SSIR_BUILTIN_SUBGROUP_MIN: return "simd_min";
    case SSIR_BUILTIN_SUBGROUP_MAX: return "simd_max";
    case SSIR_BUILTIN_SUBGROUP_ALL: return "simd_all";
    case SSIR_BUILTIN_SUBGROUP_ANY: return "simd_any";
    case SSIR_BUILTIN_SUBGROUP_SHUFFLE: return "simd_shuffle";
    case SSIR_BUILTIN_SUBGROUP_PREFIX_ADD: return "simd_prefix_exclusive_sum";
    default: return "unknown_builtin";
    }
}

/* ============================================================================
 * MSL Type Emission
 * ============================================================================ */

static void emit_type(MslCtx *c, uint32_t tid, MslBuf *b);

static void emit_type(MslCtx *c, uint32_t tid, MslBuf *b) {
    SsirType *t = ssir_get_type((SsirModule *)c->mod, tid);
    if (!t) { mb_appendf(b, "_type%u", tid); return; }

    switch (t->kind) {
    case SSIR_TYPE_VOID: mb_append(b, "void"); break;
    case SSIR_TYPE_BOOL: mb_append(b, "bool"); break;
    case SSIR_TYPE_I32:  mb_append(b, "int"); break;
    case SSIR_TYPE_U32:  mb_append(b, "uint"); break;
    case SSIR_TYPE_F32:  mb_append(b, "float"); break;
    case SSIR_TYPE_F16:  mb_append(b, "half"); break;
    case SSIR_TYPE_F64:  mb_append(b, "double"); break;
    case SSIR_TYPE_I8:   mb_append(b, "char"); break;
    case SSIR_TYPE_U8:   mb_append(b, "uchar"); break;
    case SSIR_TYPE_I16:  mb_append(b, "short"); break;
    case SSIR_TYPE_U16:  mb_append(b, "ushort"); break;
    case SSIR_TYPE_I64:  mb_append(b, "long"); break;
    case SSIR_TYPE_U64:  mb_append(b, "ulong"); break;

    case SSIR_TYPE_VEC: {
        SsirType *el = ssir_get_type((SsirModule *)c->mod, t->vec.elem);
        const char *prefix = "float";
        if (el) {
            switch (el->kind) {
            case SSIR_TYPE_I32:  prefix = "int"; break;
            case SSIR_TYPE_U32:  prefix = "uint"; break;
            case SSIR_TYPE_BOOL: prefix = "bool"; break;
            case SSIR_TYPE_F16:  prefix = "half"; break;
            default: break;
            }
        }
        mb_appendf(b, "%s%u", prefix, t->vec.size);
        break;
    }

    case SSIR_TYPE_MAT: {
        SsirType *col_t = ssir_get_type((SsirModule *)c->mod, t->mat.elem);
        const char *prefix = "float";
        if (col_t && col_t->kind == SSIR_TYPE_VEC) {
            SsirType *el = ssir_get_type((SsirModule *)c->mod, col_t->vec.elem);
            if (el && el->kind == SSIR_TYPE_F16) prefix = "half";
        }
        mb_appendf(b, "%s%ux%u", prefix, t->mat.cols, t->mat.rows);
        break;
    }

    case SSIR_TYPE_ARRAY:
        mb_append(b, "array<");
        emit_type(c, t->array.elem, b);
        mb_appendf(b, ", %u>", t->array.length);
        break;

    case SSIR_TYPE_RUNTIME_ARRAY:
        /* Runtime arrays in MSL are handled via pointer parameters */
        emit_type(c, t->runtime_array.elem, b);
        break;

    case SSIR_TYPE_STRUCT:
        mb_append(b, t->struc.name ? t->struc.name : "_Struct");
        break;

    case SSIR_TYPE_PTR:
        emit_type(c, t->ptr.pointee, b);
        break;

    case SSIR_TYPE_SAMPLER:
        mb_append(b, "sampler");
        break;

    case SSIR_TYPE_SAMPLER_COMPARISON:
        mb_append(b, "sampler");
        break;

    case SSIR_TYPE_TEXTURE: {
        SsirType *st = ssir_get_type((SsirModule *)c->mod, t->texture.sampled_type);
        const char *elem = "float";
        if (st) {
            if (st->kind == SSIR_TYPE_I32) elem = "int";
            else if (st->kind == SSIR_TYPE_U32) elem = "uint";
            else if (st->kind == SSIR_TYPE_F16) elem = "half";
        }
        switch (t->texture.dim) {
        case SSIR_TEX_1D:              mb_appendf(b, "texture1d<%s>", elem); break;
        case SSIR_TEX_2D:              mb_appendf(b, "texture2d<%s>", elem); break;
        case SSIR_TEX_3D:              mb_appendf(b, "texture3d<%s>", elem); break;
        case SSIR_TEX_CUBE:            mb_appendf(b, "texturecube<%s>", elem); break;
        case SSIR_TEX_2D_ARRAY:        mb_appendf(b, "texture2d_array<%s>", elem); break;
        case SSIR_TEX_CUBE_ARRAY:      mb_appendf(b, "texturecube_array<%s>", elem); break;
        case SSIR_TEX_MULTISAMPLED_2D: mb_appendf(b, "texture2d_ms<%s>", elem); break;
        case SSIR_TEX_1D_ARRAY:        mb_appendf(b, "texture1d_array<%s>", elem); break;
        case SSIR_TEX_BUFFER:          mb_appendf(b, "texture_buffer<%s>", elem); break;
        case SSIR_TEX_MULTISAMPLED_2D_ARRAY: mb_appendf(b, "texture2d_ms_array<%s>", elem); break;
        default:                       mb_appendf(b, "texture2d<%s>", elem); break;
        }
        break;
    }

    case SSIR_TYPE_TEXTURE_STORAGE: {
        const char *access = "access::write";
        switch (t->texture_storage.access) {
        case SSIR_ACCESS_READ:       access = "access::read"; break;
        case SSIR_ACCESS_WRITE:      access = "access::write"; break;
        case SSIR_ACCESS_READ_WRITE: access = "access::read_write"; break;
        }
        switch (t->texture_storage.dim) {
        case SSIR_TEX_1D: mb_appendf(b, "texture1d<float, %s>", access); break;
        case SSIR_TEX_2D: mb_appendf(b, "texture2d<float, %s>", access); break;
        case SSIR_TEX_3D: mb_appendf(b, "texture3d<float, %s>", access); break;
        default:          mb_appendf(b, "texture2d<float, %s>", access); break;
        }
        break;
    }

    case SSIR_TYPE_TEXTURE_DEPTH: {
        switch (t->texture_depth.dim) {
        case SSIR_TEX_2D:       mb_append(b, "depth2d<float>"); break;
        case SSIR_TEX_CUBE:     mb_append(b, "depthcube<float>"); break;
        case SSIR_TEX_2D_ARRAY: mb_append(b, "depth2d_array<float>"); break;
        default:                mb_append(b, "depth2d<float>"); break;
        }
        break;
    }

    default:
        mb_appendf(b, "_type%u", tid);
        break;
    }
}

/* Emit MSL declaration: type name */
static void emit_decl(MslCtx *c, uint32_t tid, const char *name, MslBuf *b) {
    SsirType *t = ssir_get_type((SsirModule *)c->mod, tid);
    if (t && t->kind == SSIR_TYPE_ARRAY) {
        mb_append(b, "array<");
        emit_type(c, t->array.elem, b);
        mb_appendf(b, ", %u> %s", t->array.length, name);
    } else {
        emit_type(c, tid, b);
        mb_appendf(b, " %s", name);
    }
}

/* ============================================================================
 * Constant Emission
 * ============================================================================ */

static void emit_constant(MslCtx *c, SsirConstant *k, MslBuf *b);

static void emit_constant(MslCtx *c, SsirConstant *k, MslBuf *b) {
    switch (k->kind) {
    case SSIR_CONST_BOOL:
        mb_append(b, k->bool_val ? "true" : "false");
        break;
    case SSIR_CONST_I32:
        mb_appendf(b, "%d", k->i32_val);
        break;
    case SSIR_CONST_U32:
        mb_appendf(b, "%uu", k->u32_val);
        break;
    case SSIR_CONST_F32: {
        float f = k->f32_val;
        if (floorf(f) == f && fabsf(f) < 1e6f)
            mb_appendf(b, "%.1f", (double)f);
        else
            mb_appendf(b, "%g", (double)f);
        break;
    }
    case SSIR_CONST_F16: {
        float fv = ssir_f16_to_f32(k->f16_val);
        if (floorf(fv) == fv && fabsf(fv) < 1e6f)
            mb_appendf(b, "half(%.1f)", (double)fv);
        else
            mb_appendf(b, "half(%g)", (double)fv);
        break;
    }
    case SSIR_CONST_F64: {
        double d = k->f64_val;
        if (floor(d) == d && fabs(d) < 1e15)
            mb_appendf(b, "%.1f", d);
        else
            mb_appendf(b, "%g", d);
        break;
    }
    case SSIR_CONST_I8:
        mb_appendf(b, "%d", (int)k->i8_val);
        break;
    case SSIR_CONST_U8:
        mb_appendf(b, "%uu", (unsigned)k->u8_val);
        break;
    case SSIR_CONST_I16:
        mb_appendf(b, "%d", (int)k->i16_val);
        break;
    case SSIR_CONST_U16:
        mb_appendf(b, "%uu", (unsigned)k->u16_val);
        break;
    case SSIR_CONST_I64:
        mb_appendf(b, "%lldL", (long long)k->i64_val);
        break;
    case SSIR_CONST_U64:
        mb_appendf(b, "%lluUL", (unsigned long long)k->u64_val);
        break;
    case SSIR_CONST_COMPOSITE: {
        emit_type(c, k->type, b);
        mb_append(b, "(");
        for (uint32_t i = 0; i < k->composite.count; i++) {
            if (i > 0) mb_append(b, ", ");
            SsirConstant *elem = ssir_get_constant((SsirModule *)c->mod, k->composite.components[i]);
            if (elem) emit_constant(c, elem, b);
            else mb_appendf(b, "_const%u", k->composite.components[i]);
        }
        mb_append(b, ")");
        break;
    }
    case SSIR_CONST_NULL:
        emit_type(c, k->type, b);
        mb_append(b, "(0)");
        break;
    default:
        mb_appendf(b, "_const%u", k->id);
        break;
    }
}

/* ============================================================================
 * Expression Emission
 * ============================================================================ */

static void emit_expr(MslCtx *c, uint32_t id, MslBuf *b);

static SsirInst *find_inst(MslCtx *c, uint32_t id) {
    if (c->inst_map && id < c->inst_map_cap) return c->inst_map[id];
    return NULL;
}

static SsirFunctionParam *find_param(MslCtx *c, uint32_t id) {
    if (!c->current_func) return NULL;
    for (uint32_t i = 0; i < c->current_func->param_count; i++) {
        if (c->current_func->params[i].id == id) return &c->current_func->params[i];
    }
    return NULL;
}

static SsirLocalVar *find_local(MslCtx *c, uint32_t id) {
    if (!c->current_func) return NULL;
    for (uint32_t i = 0; i < c->current_func->local_count; i++) {
        if (c->current_func->locals[i].id == id) return &c->current_func->locals[i];
    }
    return NULL;
}

static SsirTextureDim resolve_texture_dim(MslCtx *c, uint32_t id) {
    SsirModule *mod = (SsirModule *)c->mod;
    /* Check if it's a global variable */
    SsirGlobalVar *g = ssir_get_global(mod, id);
    if (g) {
        SsirType *pt = ssir_get_type(mod, g->type);
        if (pt && pt->kind == SSIR_TYPE_PTR) {
            SsirType *tt = ssir_get_type(mod, pt->ptr.pointee);
            if (tt) {
                if (tt->kind == SSIR_TYPE_TEXTURE) return tt->texture.dim;
                if (tt->kind == SSIR_TYPE_TEXTURE_STORAGE) return tt->texture_storage.dim;
                if (tt->kind == SSIR_TYPE_TEXTURE_DEPTH) return tt->texture_depth.dim;
            }
        }
    }
    /* Check if it's a LOAD instruction result - follow to the pointer source */
    SsirInst *inst = find_inst(c, id);
    if (inst && inst->op == SSIR_OP_LOAD)
        return resolve_texture_dim(c, inst->operands[0]);
    /* Check function parameters */
    SsirFunctionParam *p = find_param(c, id);
    if (p) {
        SsirType *tt = ssir_get_type(mod, p->type);
        if (tt) {
            if (tt->kind == SSIR_TYPE_TEXTURE) return tt->texture.dim;
            if (tt->kind == SSIR_TYPE_TEXTURE_STORAGE) return tt->texture_storage.dim;
            if (tt->kind == SSIR_TYPE_TEXTURE_DEPTH) return tt->texture_depth.dim;
        }
    }
    return SSIR_TEX_2D; /* fallback */
}

static void emit_binop(MslCtx *c, SsirInst *inst, const char *op, MslBuf *b) {
    mb_append(b, "(");
    emit_expr(c, inst->operands[0], b);
    mb_append(b, op);
    emit_expr(c, inst->operands[1], b);
    mb_append(b, ")");
}

static void emit_unop(MslCtx *c, SsirInst *inst, const char *op, MslBuf *b) {
    mb_append(b, "(");
    mb_append(b, op);
    emit_expr(c, inst->operands[0], b);
    mb_append(b, ")");
}

static void emit_expr(MslCtx *c, uint32_t id, MslBuf *b) {
    /* Constant? */
    SsirConstant *k = ssir_get_constant((SsirModule *)c->mod, id);
    if (k) { emit_constant(c, k, b); return; }

    /* Global? */
    SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, id);
    if (g) {
        /* Check if output field in entry point -> redirect to _out.name */
        if (c->in_entry_point) {
            const MslInterfaceField *of = find_output_field(c, id);
            if (of) {
                mb_appendf(b, "_out.%s", get_name(c, id));
                return;
            }
            /* Check if input field in fragment stage_in -> redirect to _in.name */
            if (c->has_stage_in) {
                const MslInterfaceField *inf = find_input_field(c, id);
                if (inf) {
                    mb_appendf(b, "_in.%s", get_name(c, id));
                    return;
                }
            }
        }
        mb_append(b, get_name(c, id));
        return;
    }

    /* Param? */
    if (find_param(c, id)) { mb_append(b, get_name(c, id)); return; }

    /* Local? */
    if (find_local(c, id)) { mb_append(b, get_name(c, id)); return; }

    /* Instruction? */
    SsirInst *inst = find_inst(c, id);
    if (!inst) { mb_append(b, get_name(c, id)); return; }

    /* If materialized as a temporary (multi-use), emit the name */
    if (id < c->id_names_cap && c->id_names[id]) {
        mb_append(b, c->id_names[id]);
        return;
    }

    switch (inst->op) {
    /* Arithmetic */
    case SSIR_OP_ADD: emit_binop(c, inst, " + ", b); break;
    case SSIR_OP_SUB: emit_binop(c, inst, " - ", b); break;
    case SSIR_OP_MUL: emit_binop(c, inst, " * ", b); break;
    case SSIR_OP_DIV: emit_binop(c, inst, " / ", b); break;
    case SSIR_OP_MOD: emit_binop(c, inst, " % ", b); break;
    case SSIR_OP_REM: {
        /* C-style remainder: fmod for float, % for int */
        SsirType *rem_t = ssir_get_type((SsirModule *)c->mod, inst->type);
        if (rem_t && ssir_type_is_float(rem_t)) {
            mb_append(b, "fmod(");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ", ");
            emit_expr(c, inst->operands[1], b);
            mb_append(b, ")");
        } else {
            emit_binop(c, inst, " % ", b);
        }
        break;
    }
    case SSIR_OP_NEG: emit_unop(c, inst, "-", b); break;

    /* Matrix */
    case SSIR_OP_MAT_MUL: emit_binop(c, inst, " * ", b); break;
    case SSIR_OP_MAT_TRANSPOSE:
        mb_append(b, "transpose(");
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ")");
        break;

    /* Bitwise */
    case SSIR_OP_BIT_AND: emit_binop(c, inst, " & ", b); break;
    case SSIR_OP_BIT_OR:  emit_binop(c, inst, " | ", b); break;
    case SSIR_OP_BIT_XOR: emit_binop(c, inst, " ^ ", b); break;
    case SSIR_OP_BIT_NOT: emit_unop(c, inst, "~", b); break;
    case SSIR_OP_SHL:     emit_binop(c, inst, " << ", b); break;
    case SSIR_OP_SHR:    emit_binop(c, inst, " >> ", b); break;
    case SSIR_OP_SHR_LOGICAL: {
        /* Logical shift right: must cast signed types to unsigned */
        SsirType *shr_type = ssir_get_type((SsirModule *)c->mod, inst->type);
        bool shr_signed = shr_type && (ssir_type_is_signed(shr_type) ||
            (shr_type->kind == SSIR_TYPE_VEC &&
             ssir_type_is_signed(ssir_get_type((SsirModule *)c->mod, shr_type->vec.elem))));
        if (shr_signed) {
            mb_append(b, "static_cast<");
            emit_type(c, inst->type, b);
            mb_append(b, ">(static_cast<");
            /* emit unsigned equivalent */
            if (shr_type->kind == SSIR_TYPE_VEC) {
                uint32_t uvec = ssir_type_vec((SsirModule *)c->mod,
                    ssir_type_u32((SsirModule *)c->mod), shr_type->vec.size);
                emit_type(c, uvec, b);
            } else {
                mb_append(b, "uint");
            }
            mb_append(b, ">(");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ") >> ");
            emit_expr(c, inst->operands[1], b);
            mb_append(b, ")");
        } else {
            emit_binop(c, inst, " >> ", b);
        }
        break;
    }

    /* Comparison */
    case SSIR_OP_EQ: emit_binop(c, inst, " == ", b); break;
    case SSIR_OP_NE: emit_binop(c, inst, " != ", b); break;
    case SSIR_OP_LT: emit_binop(c, inst, " < ", b); break;
    case SSIR_OP_LE: emit_binop(c, inst, " <= ", b); break;
    case SSIR_OP_GT: emit_binop(c, inst, " > ", b); break;
    case SSIR_OP_GE: emit_binop(c, inst, " >= ", b); break;

    /* Logical */
    case SSIR_OP_AND: emit_binop(c, inst, " && ", b); break;
    case SSIR_OP_OR:  emit_binop(c, inst, " || ", b); break;
    case SSIR_OP_NOT: emit_unop(c, inst, "!", b); break;

    /* Composite */
    case SSIR_OP_CONSTRUCT: {
        emit_type(c, inst->type, b);
        mb_append(b, "(");
        uint32_t cnt = inst->operand_count;
        const uint32_t *comps = inst->operands;
        if (inst->extra_count > 0) { cnt = inst->extra_count; comps = inst->extra; }
        for (uint32_t i = 0; i < cnt; i++) {
            if (i > 0) mb_append(b, ", ");
            emit_expr(c, comps[i], b);
        }
        mb_append(b, ")");
        break;
    }

    case SSIR_OP_EXTRACT: {
        emit_expr(c, inst->operands[0], b);
        uint32_t idx = inst->operands[1];
        if (idx < 4) {
            const char sw[] = "xyzw";
            mb_appendf(b, ".%c", sw[idx]);
        } else {
            mb_appendf(b, "[%u]", idx);
        }
        break;
    }

    case SSIR_OP_INSERT:
    case SSIR_OP_INSERT_DYN:
        mb_append(b, get_name(c, id));
        break;

    case SSIR_OP_SHUFFLE: {
        emit_type(c, inst->type, b);
        mb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) mb_append(b, ", ");
            uint32_t si = inst->extra[i];
            uint32_t v1_sz = 4;
            if (si < v1_sz) {
                emit_expr(c, inst->operands[0], b);
            } else {
                emit_expr(c, inst->operands[1], b);
                si -= v1_sz;
            }
            if (si < 4) {
                const char sw[] = "xyzw";
                mb_appendf(b, ".%c", sw[si]);
            }
        }
        mb_append(b, ")");
        break;
    }

    case SSIR_OP_SPLAT:
        emit_type(c, inst->type, b);
        mb_append(b, "(");
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_EXTRACT_DYN:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, "[");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, "]");
        break;

    /* Memory */
    case SSIR_OP_LOAD:
        emit_expr(c, inst->operands[0], b);
        break;

    case SSIR_OP_ACCESS: {
        SsirModule *amod = (SsirModule *)c->mod;
        emit_expr(c, inst->operands[0], b);
        /* Trace the struct type through the access chain for member names */
        uint32_t cur_type_id = 0;
        {
            SsirGlobalVar *ag = ssir_get_global(amod, inst->operands[0]);
            if (ag) {
                SsirType *pt = ssir_get_type(amod, ag->type);
                if (pt && pt->kind == SSIR_TYPE_PTR)
                    cur_type_id = pt->ptr.pointee;
            }
            if (!cur_type_id) {
                SsirInst *ai = find_inst(c, inst->operands[0]);
                if (ai && ai->op == SSIR_OP_LOAD) {
                    SsirGlobalVar *lg = ssir_get_global(amod, ai->operands[0]);
                    if (lg) {
                        SsirType *pt = ssir_get_type(amod, lg->type);
                        if (pt && pt->kind == SSIR_TYPE_PTR)
                            cur_type_id = pt->ptr.pointee;
                    }
                }
            }
        }
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            uint32_t idx = inst->extra[i];
            SsirConstant *ic = ssir_get_constant(amod, idx);
            SsirType *cur_st = cur_type_id ? ssir_get_type(amod, cur_type_id) : NULL;
            if (ic && (ic->kind == SSIR_CONST_U32 || ic->kind == SSIR_CONST_I32)) {
                uint32_t midx = (ic->kind == SSIR_CONST_U32) ? ic->u32_val : (uint32_t)ic->i32_val;
                const char *mname = NULL;
                if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT &&
                    cur_st->struc.member_names && midx < cur_st->struc.member_count)
                    mname = cur_st->struc.member_names[midx];
                if (mname)
                    mb_appendf(b, ".%s", mname);
                else
                    mb_appendf(b, ".member%u", midx);
                /* Advance type through struct member */
                if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT && midx < cur_st->struc.member_count)
                    cur_type_id = cur_st->struc.members[midx];
                else
                    cur_type_id = 0;
            } else {
                mb_append(b, "[");
                emit_expr(c, idx, b);
                mb_append(b, "]");
                /* Advance type through array element */
                if (cur_st && (cur_st->kind == SSIR_TYPE_ARRAY || cur_st->kind == SSIR_TYPE_RUNTIME_ARRAY))
                    cur_type_id = cur_st->array.elem;
                else
                    cur_type_id = 0;
            }
        }
        break;
    }

    case SSIR_OP_ARRAY_LEN:
        /* MSL doesn't have runtime array length; use buffer size / element size */
        mb_append(b, "/* arrayLength unsupported in MSL */0u");
        break;

    /* Call */
    case SSIR_OP_CALL: {
        uint32_t callee_id = inst->operands[0];
        SsirFunction *callee = ssir_get_function((SsirModule *)c->mod, callee_id);
        if (callee && callee->name)
            mb_append(b, callee->name);
        else
            mb_append(b, get_name(c, callee_id));
        mb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) mb_append(b, ", ");
            emit_expr(c, inst->extra[i], b);
        }
        mb_append(b, ")");
        break;
    }

    case SSIR_OP_BUILTIN: {
        SsirBuiltinId bid = (SsirBuiltinId)inst->operands[0];

        /* firstLeadingBit needs special handling: WGSL returns the bit index
         * of the most significant differing bit, while MSL clz() returns the
         * count of leading zeros. Relationship: firstLeadingBit(x) = 31 - clz(x)
         * for unsigned. For signed, operate on the absolute or inverted value. */
        if (bid == SSIR_BUILTIN_FIRSTLEADINGBIT && inst->extra_count > 0) {
            SsirType *rt = ssir_get_type((SsirModule *)c->mod, inst->type);
            bool is_signed = rt && (rt->kind == SSIR_TYPE_I32 ||
                (rt->kind == SSIR_TYPE_VEC && ssir_type_is_signed(
                    ssir_get_type((SsirModule *)c->mod, rt->vec.elem))));
            if (is_signed) {
                mb_append(b, "int(31u - clz(");
                emit_expr(c, inst->extra[0], b);
                mb_append(b, " >= 0 ? uint(");
                emit_expr(c, inst->extra[0], b);
                mb_append(b, ") : uint(~(");
                emit_expr(c, inst->extra[0], b);
                mb_append(b, "))))");
            } else {
                mb_append(b, "(31u - clz(");
                emit_expr(c, inst->extra[0], b);
                mb_append(b, "))");
            }
            break;
        }

        /* degrees/radians: MSL doesn't have these, use multiplication */
        if (bid == SSIR_BUILTIN_DEGREES && inst->extra_count > 0) {
            mb_append(b, "(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, " * 57.2957795131f)");
            break;
        }
        if (bid == SSIR_BUILTIN_RADIANS && inst->extra_count > 0) {
            mb_append(b, "(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, " * 0.0174532925199f)");
            break;
        }
        /* Pack/unpack builtins */
        if (bid == SSIR_BUILTIN_PACK4X8SNORM && inst->extra_count > 0) {
            mb_append(b, "pack_float_to_snorm4x8(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_PACK4X8UNORM && inst->extra_count > 0) {
            mb_append(b, "pack_float_to_unorm4x8(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_PACK2X16SNORM && inst->extra_count > 0) {
            mb_append(b, "pack_float_to_snorm2x16(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_PACK2X16UNORM && inst->extra_count > 0) {
            mb_append(b, "pack_float_to_unorm2x16(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_PACK2X16FLOAT && inst->extra_count > 0) {
            mb_append(b, "as_type<uint>(half2(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, "))"); break;
        }
        if (bid == SSIR_BUILTIN_UNPACK4X8SNORM && inst->extra_count > 0) {
            mb_append(b, "unpack_snorm4x8_to_float(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_UNPACK4X8UNORM && inst->extra_count > 0) {
            mb_append(b, "unpack_unorm4x8_to_float(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_UNPACK2X16SNORM && inst->extra_count > 0) {
            mb_append(b, "unpack_snorm2x16_to_float(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_UNPACK2X16UNORM && inst->extra_count > 0) {
            mb_append(b, "unpack_unorm2x16_to_float(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, ")"); break;
        }
        if (bid == SSIR_BUILTIN_UNPACK2X16FLOAT && inst->extra_count > 0) {
            mb_append(b, "float2(as_type<half2>(");
            emit_expr(c, inst->extra[0], b);
            mb_append(b, "))"); break;
        }

        const char *name = bfunc_to_msl(bid);
        mb_append(b, name ? name : "unknown_builtin");
        mb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) mb_append(b, ", ");
            emit_expr(c, inst->extra[i], b);
        }
        mb_append(b, ")");
        break;
    }

    /* Conversion */
    case SSIR_OP_CONVERT:
        mb_append(b, "static_cast<");
        emit_type(c, inst->type, b);
        mb_append(b, ">(");
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_BITCAST:
        mb_append(b, "as_type<");
        emit_type(c, inst->type, b);
        mb_append(b, ">(");
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ")");
        break;

    /* Texture - MSL uses member function syntax */
    case SSIR_OP_TEX_SAMPLE:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_BIAS:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", bias(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, "))");
        break;

    case SSIR_OP_TEX_SAMPLE_LEVEL:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", level(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, "))");
        break;

    case SSIR_OP_TEX_SAMPLE_GRAD:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", gradient2d(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, "))");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample_compare(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample_compare(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ", level(");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, "))");
        break;

    case SSIR_OP_TEX_SAMPLE_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", bias(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, "), ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", level(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, "), ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", gradient2d(");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, "), ");
        emit_expr(c, inst->operands[5], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".sample_compare(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_GATHER:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".gather(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", component::");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_GATHER_CMP:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".gather_compare(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_GATHER_OFFSET:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".gather(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[4], b);
        mb_append(b, ", component::");
        emit_expr(c, inst->operands[3], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_LOAD:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".read(");
        emit_expr(c, inst->operands[1], b);
        if (inst->operands[2] != 0) {
            mb_append(b, ", ");
            emit_expr(c, inst->operands[2], b);
        }
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_SIZE: {
        SsirTextureDim dim = resolve_texture_dim(c, inst->operands[0]);
        switch (dim) {
        case SSIR_TEX_1D:
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_width()");
            break;
        case SSIR_TEX_3D:
            mb_append(b, "uint3(");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_width(), ");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_height(), ");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_depth())");
            break;
        default: /* 2D, cube, 2D array, cube array, MS 2D */
            mb_append(b, "uint2(");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_width(), ");
            emit_expr(c, inst->operands[0], b);
            mb_append(b, ".get_height())");
            break;
        }
        break;
    }

    case SSIR_OP_TEX_QUERY_LOD:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".calculate_clamped_lod(");
        emit_expr(c, inst->operands[1], b);
        mb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        mb_append(b, ")");
        break;

    case SSIR_OP_TEX_QUERY_LEVELS:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".get_num_mip_levels()");
        break;

    case SSIR_OP_TEX_QUERY_SAMPLES:
        emit_expr(c, inst->operands[0], b);
        mb_append(b, ".get_num_samples()");
        break;

    case SSIR_OP_PHI:
        mb_append(b, get_name(c, id));
        break;

    default:
        mb_append(b, get_name(c, id));
        break;
    }
}

/* ============================================================================
 * Block and Statement Emission
 * ============================================================================ */

typedef struct {
    bool *emitted;
    uint32_t count;
} BlkState;

static void emit_block(MslCtx *c, SsirBlock *blk, SsirFunction *fn, BlkState *bs);

static void emit_stmt(MslCtx *c, SsirInst *inst, SsirBlock *blk,
                       SsirFunction *fn, BlkState *bs) {
    switch (inst->op) {
    case SSIR_OP_STORE:
        mb_indent(&c->sb);
        emit_expr(c, inst->operands[0], &c->sb);
        mb_append(&c->sb, " = ");
        emit_expr(c, inst->operands[1], &c->sb);
        mb_append(&c->sb, ";\n");
        break;

    case SSIR_OP_TEX_STORE:
        mb_indent(&c->sb);
        emit_expr(c, inst->operands[0], &c->sb);
        mb_append(&c->sb, ".write(");
        emit_expr(c, inst->operands[2], &c->sb);
        mb_append(&c->sb, ", ");
        emit_expr(c, inst->operands[1], &c->sb);
        mb_append(&c->sb, ");\n");
        break;

    case SSIR_OP_RETURN_VOID:
        mb_indent(&c->sb);
        if (c->in_entry_point && c->output_field_count > 0)
            mb_append(&c->sb, "return _out;\n");
        else
            mb_append(&c->sb, "return;\n");
        break;

    case SSIR_OP_RETURN:
        mb_indent(&c->sb);
        if (c->in_entry_point && c->output_field_count > 0) {
            mb_append(&c->sb, "return _out;\n");
        } else {
            mb_append(&c->sb, "return ");
            emit_expr(c, inst->operands[0], &c->sb);
            mb_append(&c->sb, ";\n");
        }
        break;

    case SSIR_OP_BRANCH: {
        uint32_t target = inst->operands[0];
        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == target && !bs->emitted[i]) {
                emit_block(c, &fn->blocks[i], fn, bs);
                break;
            }
        }
        break;
    }

    case SSIR_OP_BRANCH_COND: {
        uint32_t cond = inst->operands[0];
        uint32_t true_blk = inst->operands[1];
        uint32_t false_blk = inst->operands[2];
        uint32_t merge_blk = (inst->operand_count > 3) ? inst->operands[3] : 0;

        /* Temporarily mark the merge block as emitted so it doesn't get
           pulled in recursively via branch instructions inside the if body. */
        int merge_idx = -1;
        if (merge_blk) {
            for (uint32_t i = 0; i < fn->block_count; i++) {
                if (fn->blocks[i].id == merge_blk && !bs->emitted[i]) {
                    merge_idx = (int)i;
                    bs->emitted[i] = true;
                    break;
                }
            }
        }

        mb_indent(&c->sb);
        mb_append(&c->sb, "if (");
        emit_expr(c, cond, &c->sb);
        mb_append(&c->sb, ") {\n");
        c->sb.indent++;

        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == true_blk && !bs->emitted[i]) {
                emit_block(c, &fn->blocks[i], fn, bs);
                break;
            }
        }

        c->sb.indent--;

        /* Only emit else clause if false block differs from merge block */
        if (false_blk != merge_blk || merge_blk == 0) {
            mb_indent(&c->sb);
            mb_append(&c->sb, "} else {\n");
            c->sb.indent++;

            for (uint32_t i = 0; i < fn->block_count; i++) {
                if (fn->blocks[i].id == false_blk && !bs->emitted[i]) {
                    emit_block(c, &fn->blocks[i], fn, bs);
                    break;
                }
            }

            c->sb.indent--;
        }

        mb_indent(&c->sb);
        mb_append(&c->sb, "}\n");

        /* Now emit the merge block after the if-else */
        if (merge_idx >= 0) {
            bs->emitted[merge_idx] = false;  /* Un-mark so we can emit it */
            emit_block(c, &fn->blocks[merge_idx], fn, bs);
        }
        break;
    }

    case SSIR_OP_BARRIER:
        mb_indent(&c->sb);
        switch ((SsirBarrierScope)inst->operands[0]) {
        case SSIR_BARRIER_WORKGROUP: mb_append(&c->sb, "threadgroup_barrier(mem_flags::mem_threadgroup);\n"); break;
        case SSIR_BARRIER_STORAGE:   mb_append(&c->sb, "threadgroup_barrier(mem_flags::mem_device);\n"); break;
        case SSIR_BARRIER_SUBGROUP:  mb_append(&c->sb, "simdgroup_barrier(mem_flags::mem_threadgroup);\n"); break;
        case SSIR_BARRIER_IMAGE:     mb_append(&c->sb, "threadgroup_barrier(mem_flags::mem_texture);\n"); break;
        }
        break;

    case SSIR_OP_DISCARD:
        mb_indent(&c->sb);
        mb_append(&c->sb, "discard_fragment();\n");
        break;

    case SSIR_OP_INSERT_DYN:
        mb_indent(&c->sb);
        emit_decl(c, inst->type, get_name(c, inst->result), &c->sb);
        mb_append(&c->sb, " = ");
        emit_expr(c, inst->operands[0], &c->sb);
        mb_append(&c->sb, ";\n");
        mb_indent(&c->sb);
        mb_append(&c->sb, get_name(c, inst->result));
        mb_append(&c->sb, "[");
        emit_expr(c, inst->operands[2], &c->sb);
        mb_append(&c->sb, "] = ");
        emit_expr(c, inst->operands[1], &c->sb);
        mb_append(&c->sb, ";\n");
        break;

    case SSIR_OP_LOOP_MERGE:
    case SSIR_OP_SELECTION_MERGE:
    case SSIR_OP_UNREACHABLE:
        break;

    default:
        if (inst->result != 0) {
            SsirType *rt = ssir_get_type((SsirModule *)c->mod, inst->type);
            bool is_void = (rt && rt->kind == SSIR_TYPE_VOID);
            bool has_side_effects = (inst->op == SSIR_OP_CALL ||
                                     inst->op == SSIR_OP_ATOMIC);
            bool must_materialize = (inst->op == SSIR_OP_PHI || has_side_effects);
            uint32_t uses = (c->use_counts && inst->result < c->mod->next_id)
                            ? c->use_counts[inst->result] : 2;
            if (is_void && !has_side_effects) break;
            if (must_materialize || uses > 1) {
                mb_indent(&c->sb);
                if (!is_void) {
                    emit_decl(c, inst->type, get_name(c, inst->result), &c->sb);
                    mb_append(&c->sb, " = ");
                }
                emit_expr(c, inst->result, &c->sb);
                mb_append(&c->sb, ";\n");
                if (!is_void)
                    set_name(c, inst->result, get_name(c, inst->result));
            }
        }
        break;
    }
}

static void emit_block(MslCtx *c, SsirBlock *blk, SsirFunction *fn, BlkState *bs) {
    for (uint32_t i = 0; i < fn->block_count; i++) {
        if (fn->blocks[i].id == blk->id) {
            if (bs->emitted[i]) return;
            bs->emitted[i] = true;
            break;
        }
    }

    for (uint32_t i = 0; i < blk->inst_count; i++) {
        emit_stmt(c, &blk->insts[i], blk, fn, bs);
    }
}

/* ============================================================================
 * Struct Emission
 * ============================================================================ */

static bool type_is_buffer_pointee(MslCtx *c, uint32_t type_id) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];
        SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR) continue;
        if (pt->ptr.space != SSIR_ADDR_UNIFORM && pt->ptr.space != SSIR_ADDR_STORAGE)
            continue;
        if (pt->ptr.pointee == type_id) return true;
    }
    return false;
}

static void emit_struct_def(MslCtx *c, SsirType *t) {
    if (t->kind != SSIR_TYPE_STRUCT) return;

    mb_appendf(&c->sb, "struct %s {\n", t->struc.name ? t->struc.name : "_Struct");
    for (uint32_t i = 0; i < t->struc.member_count; i++) {
        mb_append(&c->sb, "    ");
        char mname_buf[32];
        const char *mname = (t->struc.member_names && t->struc.member_names[i])
            ? t->struc.member_names[i] : NULL;
        if (!mname) { snprintf(mname_buf, sizeof(mname_buf), "member%u", i); mname = mname_buf; }
        SsirType *mt = ssir_get_type((SsirModule *)c->mod, t->struc.members[i]);
        if (mt && mt->kind == SSIR_TYPE_RUNTIME_ARRAY) {
            /* Flexible array member */
            emit_type(c, mt->runtime_array.elem, &c->sb);
            mb_appendf(&c->sb, " %s[1]", mname);
        } else if (mt && mt->kind == SSIR_TYPE_VEC && mt->vec.size == 3 &&
                   (t->struc.layout_rule == SSIR_LAYOUT_STD430 || t->struc.layout_rule == SSIR_LAYOUT_SCALAR)) {
            /* Use packed types for vec3 in std430/scalar layout to match 12-byte stride */
            SsirType *elem = ssir_get_type((SsirModule *)c->mod, mt->vec.elem);
            const char *packed = "packed_float3";
            if (elem) {
                if (elem->kind == SSIR_TYPE_I32) packed = "packed_int3";
                else if (elem->kind == SSIR_TYPE_U32) packed = "packed_uint3";
                else if (elem->kind == SSIR_TYPE_F16) packed = "packed_half3";
            }
            mb_appendf(&c->sb, "%s %s", packed, mname);
        } else {
            emit_decl(c, t->struc.members[i], mname, &c->sb);
        }
        mb_append(&c->sb, ";\n");
    }
    mb_append(&c->sb, "};\n\n");
}

/* ============================================================================
 * Entry Point Name Mangling
 * ============================================================================ */

/* MSL forbids "main" as a function name for entry points */
static const char *msl_entry_name(const char *name) {
    if (name && strcmp(name, "main") == 0) return "main0";
    return name;
}

/* ============================================================================
 * Entry Point Interface Collection (with stage-aware filtering)
 * ============================================================================ */

/* Check if a global is used as an output by a vertex entry point */
static bool is_vertex_output(const SsirModule *mod, uint32_t global_id) {
    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        SsirEntryPoint *ep = &mod->entry_points[i];
        if (ep->stage != SSIR_STAGE_VERTEX) continue;
        for (uint32_t j = 0; j < ep->interface_count; j++) {
            if (ep->interface[j] == global_id) {
                SsirGlobalVar *g = ssir_get_global((SsirModule *)mod, global_id);
                if (!g) continue;
                SsirType *pt = ssir_get_type((SsirModule *)mod, g->type);
                if (pt && pt->kind == SSIR_TYPE_PTR && pt->ptr.space == SSIR_ADDR_OUTPUT)
                    return true;
            }
        }
    }
    return false;
}

/* Filter: is this input builtin valid for the given stage? */
static bool valid_input_for_stage(SsirStage stage, SsirBuiltinVar bv) {
    if (bv == SSIR_BUILTIN_NONE) return true;
    switch (stage) {
    case SSIR_STAGE_VERTEX:
        return bv == SSIR_BUILTIN_VERTEX_INDEX || bv == SSIR_BUILTIN_INSTANCE_INDEX;
    case SSIR_STAGE_FRAGMENT:
        return bv == SSIR_BUILTIN_POSITION || bv == SSIR_BUILTIN_FRONT_FACING ||
               bv == SSIR_BUILTIN_SAMPLE_INDEX || bv == SSIR_BUILTIN_SAMPLE_MASK;
    case SSIR_STAGE_COMPUTE:
        return bv == SSIR_BUILTIN_LOCAL_INVOCATION_ID || bv == SSIR_BUILTIN_LOCAL_INVOCATION_INDEX ||
               bv == SSIR_BUILTIN_GLOBAL_INVOCATION_ID || bv == SSIR_BUILTIN_WORKGROUP_ID ||
               bv == SSIR_BUILTIN_NUM_WORKGROUPS;
    case SSIR_STAGE_GEOMETRY:
    case SSIR_STAGE_TESS_CONTROL:
    case SSIR_STAGE_TESS_EVAL:
        return false;
    }
    return false;
}

/* Filter: is this output builtin/global valid for the given stage? */
static bool valid_output_for_stage(SsirStage stage, SsirBuiltinVar bv, const SsirModule *mod, uint32_t global_id) {
    switch (stage) {
    case SSIR_STAGE_VERTEX:
        return true; /* vertex can output position, varyings */
    case SSIR_STAGE_FRAGMENT:
        if (bv == SSIR_BUILTIN_POSITION) return false; /* position is vertex-only output */
        if (bv == SSIR_BUILTIN_FRAG_DEPTH || bv == SSIR_BUILTIN_SAMPLE_MASK) return true;
        if (bv == SSIR_BUILTIN_NONE) {
            /* Location-based: skip if this is actually a vertex varying */
            return !is_vertex_output(mod, global_id);
        }
        return false;
    case SSIR_STAGE_COMPUTE:
        return false; /* compute has no outputs */
    case SSIR_STAGE_GEOMETRY:
    case SSIR_STAGE_TESS_CONTROL:
    case SSIR_STAGE_TESS_EVAL:
        return false;
    }
    return false;
}

static void collect_interface_fields(MslCtx *c, SsirEntryPoint *ep) {
    STM_FREE(c->output_fields);
    STM_FREE(c->input_fields);
    c->output_fields = NULL;
    c->input_fields = NULL;
    c->output_field_count = 0;
    c->input_field_count = 0;

    if (!ep) return;

    /* First pass: count valid fields */
    uint32_t out_count = 0, in_count = 0;
    for (uint32_t i = 0; i < ep->interface_count; i++) {
        SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, ep->interface[i]);
        if (!g) continue;
        SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR) continue;
        if (pt->ptr.space == SSIR_ADDR_OUTPUT &&
            valid_output_for_stage(ep->stage, g->builtin, c->mod, g->id))
            out_count++;
        if (pt->ptr.space == SSIR_ADDR_INPUT &&
            valid_input_for_stage(ep->stage, g->builtin))
            in_count++;
    }

    if (out_count > 0) {
        c->output_fields = (MslInterfaceField *)STM_MALLOC(out_count * sizeof(MslInterfaceField));
    }
    if (in_count > 0) {
        c->input_fields = (MslInterfaceField *)STM_MALLOC(in_count * sizeof(MslInterfaceField));
    }

    /* Second pass: fill, applying stage filters */
    for (uint32_t i = 0; i < ep->interface_count; i++) {
        SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, ep->interface[i]);
        if (!g) continue;
        SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR) continue;

        MslInterfaceField f;
        f.global_id = g->id;
        f.pointee_type = pt->ptr.pointee;
        f.builtin = g->builtin;
        f.has_location = g->has_location;
        f.location = g->location;
        f.interp = g->interp;
        f.interp_sampling = g->interp_sampling;
        f.invariant = g->invariant;
        f.space = pt->ptr.space;

        if (pt->ptr.space == SSIR_ADDR_OUTPUT &&
            valid_output_for_stage(ep->stage, g->builtin, c->mod, g->id)) {
            c->output_fields[c->output_field_count++] = f;
        } else if (pt->ptr.space == SSIR_ADDR_INPUT &&
                   valid_input_for_stage(ep->stage, g->builtin)) {
            c->input_fields[c->input_field_count++] = f;
        }
    }
}

/* ============================================================================
 * Entry Point Output/Input Struct Emission
 * ============================================================================ */

static void emit_output_struct(MslCtx *c, SsirEntryPoint *ep) {
    if (c->output_field_count == 0) return;

    const char *fname = msl_entry_name(ep->name ? ep->name : "entry");
    mb_appendf(&c->sb, "struct %s_out {\n", fname);

    for (uint32_t i = 0; i < c->output_field_count; i++) {
        MslInterfaceField *f = &c->output_fields[i];
        mb_append(&c->sb, "    ");
        emit_decl(c, f->pointee_type, get_name(c, f->global_id), &c->sb);

        if (f->builtin != SSIR_BUILTIN_NONE) {
            const char *attr = builtin_to_msl_attr(f->builtin, true);
            if (attr) mb_appendf(&c->sb, " [[%s]]", attr);
        } else if (f->has_location) {
            /* Vertex output varying -> [[user(loc_N)]] */
            /* Fragment output -> [[color(N)]] */
            if (ep->stage == SSIR_STAGE_FRAGMENT) {
                mb_appendf(&c->sb, " [[color(%u)]]", f->location);
            } else {
                mb_appendf(&c->sb, " [[user(loc_%u)]]", f->location);
            }
        }
        mb_append(&c->sb, ";\n");
    }

    mb_appendf(&c->sb, "};\n\n");
}

static void emit_stage_in_struct(MslCtx *c, SsirEntryPoint *ep) {
    if (c->input_field_count == 0) return;
    if (ep->stage != SSIR_STAGE_FRAGMENT) return;

    const char *fname = msl_entry_name(ep->name ? ep->name : "entry");
    mb_appendf(&c->sb, "struct %s_in {\n", fname);

    for (uint32_t i = 0; i < c->input_field_count; i++) {
        MslInterfaceField *f = &c->input_fields[i];
        mb_append(&c->sb, "    ");
        emit_decl(c, f->pointee_type, get_name(c, f->global_id), &c->sb);

        if (f->builtin != SSIR_BUILTIN_NONE) {
            const char *attr = builtin_to_msl_attr(f->builtin, false);
            if (attr) mb_appendf(&c->sb, " [[%s]]", attr);
        } else if (f->has_location) {
            mb_appendf(&c->sb, " [[user(loc_%u)]]", f->location);
        }
        if (f->interp == SSIR_INTERP_FLAT) {
            mb_append(&c->sb, " [[flat]]");
        } else if (f->interp_sampling == SSIR_INTERP_SAMPLING_CENTROID) {
            if (f->interp == SSIR_INTERP_LINEAR)
                mb_append(&c->sb, " [[centroid_no_perspective]]");
            else
                mb_append(&c->sb, " [[centroid_perspective]]");
        } else if (f->interp_sampling == SSIR_INTERP_SAMPLING_SAMPLE) {
            if (f->interp == SSIR_INTERP_LINEAR)
                mb_append(&c->sb, " [[sample_no_perspective]]");
            else
                mb_append(&c->sb, " [[sample_perspective]]");
        }
        if (f->invariant) {
            mb_append(&c->sb, " [[invariant]]");
        }
        mb_append(&c->sb, ";\n");
    }

    mb_appendf(&c->sb, "};\n\n");
}

/* ============================================================================
 * Function Emission
 * ============================================================================ */

static SsirEntryPoint *find_ep(MslCtx *c, uint32_t func_id) {
    for (uint32_t i = 0; i < c->mod->entry_point_count; i++) {
        if (c->mod->entry_points[i].function == func_id)
            return &c->mod->entry_points[i];
    }
    return NULL;
}

static void emit_function(MslCtx *c, SsirFunction *fn) {
    SsirEntryPoint *ep = find_ep(c, fn->id);
    int is_entry = (ep != NULL);

    if (is_entry) {
        collect_interface_fields(c, ep);
        c->in_entry_point = true;

        /* Emit output struct if needed */
        emit_output_struct(c, ep);

        /* Emit stage_in struct for fragment shaders */
        c->has_stage_in = (ep->stage == SSIR_STAGE_FRAGMENT && c->input_field_count > 0);
        emit_stage_in_struct(c, ep);

        /* Early fragment tests attribute */
        if (ep->early_fragment_tests) {
            mb_append(&c->sb, "[[early_fragment_tests]]\n");
        }

        /* Entry point qualifier */
        switch (ep->stage) {
        case SSIR_STAGE_VERTEX:  mb_append(&c->sb, "vertex "); break;
        case SSIR_STAGE_FRAGMENT: mb_append(&c->sb, "fragment "); break;
        case SSIR_STAGE_COMPUTE: mb_append(&c->sb, "kernel "); break;
        case SSIR_STAGE_GEOMETRY: mb_append(&c->sb, "kernel "); break;
        case SSIR_STAGE_TESS_CONTROL: mb_append(&c->sb, "kernel "); break;
        case SSIR_STAGE_TESS_EVAL: mb_append(&c->sb, "vertex "); break;
        }

        /* Return type */
        const char *ename = msl_entry_name(ep->name ? ep->name : "entry");
        if (c->output_field_count > 0) {
            mb_appendf(&c->sb, "%s_out", ename);
        } else {
            mb_append(&c->sb, "void");
        }

        mb_appendf(&c->sb, " %s(\n", msl_entry_name(fn->name ? fn->name : get_name(c, fn->id)));

        /* Parameters */
        int first = 1;

        /* Bound resources as parameters */
        for (uint32_t i = 0; i < ep->interface_count; i++) {
            SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, ep->interface[i]);
            if (!g) continue;
            SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
            if (!pt || pt->kind != SSIR_TYPE_PTR) continue;

            SsirAddressSpace space = pt->ptr.space;
            uint32_t pointee_id = pt->ptr.pointee;
            SsirType *pointee = ssir_get_type((SsirModule *)c->mod, pointee_id);

            /* Skip input/output - handled separately */
            if (space == SSIR_ADDR_INPUT || space == SSIR_ADDR_OUTPUT) continue;

            if (!first) mb_append(&c->sb, ",\n");
            first = 0;
            mb_append(&c->sb, "    ");

            switch (space) {
            case SSIR_ADDR_UNIFORM:
                mb_append(&c->sb, "constant ");
                if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
                    emit_type(c, pointee_id, &c->sb);
                    mb_appendf(&c->sb, "& %s", get_name(c, g->id));
                } else {
                    emit_type(c, pointee_id, &c->sb);
                    mb_appendf(&c->sb, "& %s", get_name(c, g->id));
                }
                mb_appendf(&c->sb, " [[buffer(%u)]]", g->has_binding ? g->binding : 0);
                break;

            case SSIR_ADDR_STORAGE:
                mb_appendf(&c->sb, "%sdevice ", g->non_writable ? "const " : "");
                if (pointee && pointee->kind == SSIR_TYPE_RUNTIME_ARRAY) {
                    emit_type(c, pointee->runtime_array.elem, &c->sb);
                    mb_appendf(&c->sb, "* %s", get_name(c, g->id));
                } else if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
                    emit_type(c, pointee_id, &c->sb);
                    mb_appendf(&c->sb, "& %s", get_name(c, g->id));
                } else {
                    emit_type(c, pointee_id, &c->sb);
                    mb_appendf(&c->sb, "& %s", get_name(c, g->id));
                }
                mb_appendf(&c->sb, " [[buffer(%u)]]", g->has_binding ? g->binding : 0);
                break;

            case SSIR_ADDR_UNIFORM_CONSTANT:
                /* Textures and samplers */
                emit_type(c, pointee_id, &c->sb);
                mb_appendf(&c->sb, " %s", get_name(c, g->id));
                if (pointee && (pointee->kind == SSIR_TYPE_TEXTURE ||
                                pointee->kind == SSIR_TYPE_TEXTURE_STORAGE ||
                                pointee->kind == SSIR_TYPE_TEXTURE_DEPTH)) {
                    mb_appendf(&c->sb, " [[texture(%u)]]", g->has_binding ? g->binding : 0);
                } else if (pointee && (pointee->kind == SSIR_TYPE_SAMPLER ||
                                       pointee->kind == SSIR_TYPE_SAMPLER_COMPARISON)) {
                    mb_appendf(&c->sb, " [[sampler(%u)]]", g->has_binding ? g->binding : 0);
                }
                break;

            case SSIR_ADDR_WORKGROUP:
                /* Threadgroup parameters */
                mb_append(&c->sb, "threadgroup ");
                emit_type(c, pointee_id, &c->sb);
                mb_appendf(&c->sb, "* %s", get_name(c, g->id));
                break;

            default:
                emit_type(c, pointee_id, &c->sb);
                mb_appendf(&c->sb, " %s", get_name(c, g->id));
                break;
            }
        }

        /* Input builtins as direct parameters (for vertex/compute) */
        if (ep->stage != SSIR_STAGE_FRAGMENT) {
            for (uint32_t i = 0; i < c->input_field_count; i++) {
                MslInterfaceField *f = &c->input_fields[i];
                if (!first) mb_append(&c->sb, ",\n");
                first = 0;
                mb_append(&c->sb, "    ");

                emit_decl(c, f->pointee_type, get_name(c, f->global_id), &c->sb);

                if (f->builtin != SSIR_BUILTIN_NONE) {
                    const char *attr = builtin_to_msl_attr(f->builtin, false);
                    if (attr) mb_appendf(&c->sb, " [[%s]]", attr);
                } else if (f->has_location) {
                    mb_appendf(&c->sb, " [[attribute(%u)]]", f->location);
                }
            }
        }

        /* Fragment stage_in parameter */
        if (c->has_stage_in) {
            if (!first) mb_append(&c->sb, ",\n");
            first = 0;
            mb_appendf(&c->sb, "    %s_in _in [[stage_in]]", ename);
        }

        mb_append(&c->sb, "\n) {\n");
    } else {
        /* Non-entry function */
        c->in_entry_point = false;
        c->has_stage_in = false;
        c->output_field_count = 0;
        c->input_field_count = 0;

        SsirType *ret = ssir_get_type((SsirModule *)c->mod, fn->return_type);
        if (ret && ret->kind != SSIR_TYPE_VOID) {
            emit_type(c, fn->return_type, &c->sb);
        } else {
            mb_append(&c->sb, "void");
        }
        mb_appendf(&c->sb, " %s(", fn->name ? fn->name : get_name(c, fn->id));

        for (uint32_t i = 0; i < fn->param_count; i++) {
            if (i > 0) mb_append(&c->sb, ", ");
            emit_decl(c, fn->params[i].type, get_name(c, fn->params[i].id), &c->sb);
        }
        mb_append(&c->sb, ") {\n");
    }

    c->sb.indent++;
    c->current_func = fn;

    /* Compute use counts for inlining decisions */
    STM_FREE(c->use_counts);
    c->use_counts = (uint32_t *)STM_MALLOC(c->mod->next_id * sizeof(uint32_t));
    if (c->use_counts) {
        memset(c->use_counts, 0, c->mod->next_id * sizeof(uint32_t));
        ssir_count_uses((SsirFunction *)fn, c->use_counts, c->mod->next_id);
    }

    /* Build instruction lookup map */
    STM_FREE(c->inst_map);
    c->inst_map_cap = c->mod->next_id;
    c->inst_map = (SsirInst **)STM_MALLOC(c->inst_map_cap * sizeof(SsirInst *));
    if (c->inst_map) {
        memset(c->inst_map, 0, c->inst_map_cap * sizeof(SsirInst *));
        for (uint32_t bi = 0; bi < fn->block_count; bi++) {
            SsirBlock *blk = &fn->blocks[bi];
            for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                uint32_t rid = blk->insts[ii].result;
                if (rid && rid < c->inst_map_cap)
                    c->inst_map[rid] = &blk->insts[ii];
            }
        }
    }

    /* Declare output struct variable */
    if (is_entry && c->output_field_count > 0) {
        mb_indent(&c->sb);
        mb_appendf(&c->sb, "%s_out _out = {};\n",
                   msl_entry_name(ep->name ? ep->name : "entry"));
    }

    /* Local variables */
    for (uint32_t i = 0; i < fn->local_count; i++) {
        SsirLocalVar *l = &fn->locals[i];
        SsirType *lpt = ssir_get_type((SsirModule *)c->mod, l->type);
        uint32_t pointee = (lpt && lpt->kind == SSIR_TYPE_PTR) ? lpt->ptr.pointee : l->type;

        mb_indent(&c->sb);
        emit_decl(c, pointee, get_name(c, l->id), &c->sb);

        if (l->has_initializer) {
            mb_append(&c->sb, " = ");
            SsirConstant *init = ssir_get_constant((SsirModule *)c->mod, l->initializer);
            if (init) emit_constant(c, init, &c->sb);
        }
        mb_append(&c->sb, ";\n");
    }

    /* Function body */
    if (fn->block_count > 0) {
        BlkState bs;
        bs.count = fn->block_count;
        bs.emitted = (bool *)STM_MALLOC(fn->block_count * sizeof(bool));
        if (bs.emitted) {
            memset(bs.emitted, 0, fn->block_count * sizeof(bool));
            emit_block(c, &fn->blocks[0], fn, &bs);
            STM_FREE(bs.emitted);
        }
    }

    c->current_func = NULL;
    c->in_entry_point = false;
    c->has_stage_in = false;
    c->sb.indent--;
    mb_append(&c->sb, "}\n\n");
}

/* ============================================================================
 * Name Assignment
 * ============================================================================ */

static SsirAddressSpace global_addr_space(MslCtx *c, SsirGlobalVar *g) {
    SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
    if (pt && pt->kind == SSIR_TYPE_PTR) return pt->ptr.space;
    return SSIR_ADDR_FUNCTION;
}

static int global_name_taken(MslCtx *c, const char *name, uint32_t exclude_id) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];
        if (g->id == exclude_id || g->builtin != SSIR_BUILTIN_NONE) continue;
        if (g->id < c->id_names_cap && c->id_names[g->id] &&
            strcmp(c->id_names[g->id], name) == 0) return 1;
    }
    return 0;
}

static void assign_names(MslCtx *c) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];

        if (g->name && c->opts.preserve_names) {
            set_name(c, g->id, g->name);
            const char *assigned = c->id_names[g->id];
            if (assigned && global_name_taken(c, assigned, g->id)) {
                SsirAddressSpace sp = global_addr_space(c, g);
                const char *prefix = (sp == SSIR_ADDR_OUTPUT) ? "_out_" : "_in_";
                size_t len = strlen(prefix) + strlen(g->name) + 1;
                char *tmp = (char *)STM_MALLOC(len);
                snprintf(tmp, len, "%s%s", prefix, g->name);
                STM_FREE(c->id_names[g->id]);
                c->id_names[g->id] = tmp;
            }
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "_g%u", g->id);
            set_name(c, g->id, buf);
        }
    }

    for (uint32_t i = 0; i < c->mod->function_count; i++) {
        SsirFunction *fn = &c->mod->functions[i];
        if (fn->name && c->opts.preserve_names)
            set_name(c, fn->id, fn->name);

        for (uint32_t j = 0; j < fn->param_count; j++) {
            SsirFunctionParam *p = &fn->params[j];
            if (p->name && c->opts.preserve_names) {
                set_name(c, p->id, p->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_p%u", p->id);
                set_name(c, p->id, buf);
            }
        }

        for (uint32_t j = 0; j < fn->local_count; j++) {
            SsirLocalVar *l = &fn->locals[j];
            if (l->name && c->opts.preserve_names) {
                set_name(c, l->id, l->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_l%u", l->id);
                set_name(c, l->id, buf);
            }
        }
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

SsirToMslResult ssir_to_msl(const SsirModule *mod,
                              const SsirToMslOptions *opts,
                              char **out_msl,
                              char **out_error) {
    if (!mod || !out_msl) {
        if (out_error) *out_error = strdup("Invalid input: null module or output pointer");
        return SSIR_TO_MSL_ERR_INVALID_INPUT;
    }

    MslCtx ctx;
    mctx_init(&ctx, mod, opts);
    if (!ctx.id_names) {
        if (out_error) *out_error = strdup("Out of memory");
        return SSIR_TO_MSL_ERR_OOM;
    }

    /* Phase 1: Assign names */
    assign_names(&ctx);

    /* Phase 2: MSL header */
    mb_append(&ctx.sb, "#include <metal_stdlib>\n");
    mb_append(&ctx.sb, "using namespace metal;\n\n");

    /* Phase 3: Emit struct definitions */
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_STRUCT) {
            emit_struct_def(&ctx, &mod->types[i]);
        }
    }

    /* Phase 3b: Emit specialization constants */
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        SsirConstant *k = &mod->constants[i];
        if (!k->is_specialization) continue;
        mb_append(&ctx.sb, "constant ");
        emit_type(&ctx, k->type, &ctx.sb);
        mb_appendf(&ctx.sb, " %s", k->name ? k->name : "spec_const");
        mb_appendf(&ctx.sb, " [[function_constant(%u)]];\n", k->spec_id);
    }

    /* Phase 4: Emit non-entry functions first */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        if (!find_ep(&ctx, mod->functions[i].id)) {
            emit_function(&ctx, &mod->functions[i]);
        }
    }

    /* Phase 5: Emit entry point functions */
    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        SsirEntryPoint *ep = &mod->entry_points[i];
        SsirFunction *fn = ssir_get_function((SsirModule *)mod, ep->function);
        if (fn) {
            emit_function(&ctx, fn);
        }
    }

    *out_msl = ctx.sb.data ? strdup(ctx.sb.data) : strdup("");
    mctx_free(&ctx);
    return SSIR_TO_MSL_OK;
}

void ssir_to_msl_free(void *p) {
    STM_FREE(p);
}

const char *ssir_to_msl_result_string(SsirToMslResult result) {
    switch (result) {
    case SSIR_TO_MSL_OK:                return "Success";
    case SSIR_TO_MSL_ERR_INVALID_INPUT: return "Invalid input";
    case SSIR_TO_MSL_ERR_UNSUPPORTED:   return "Unsupported feature";
    case SSIR_TO_MSL_ERR_INTERNAL:      return "Internal error";
    case SSIR_TO_MSL_ERR_OOM:           return "Out of memory";
    default:                            return "Unknown error";
    }
}
