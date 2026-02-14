/*
 * SSIR to WGSL Converter - Implementation
 *
 * Converts SSIR (Simple Shader IR) directly to WGSL text.
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

#ifndef STW_MALLOC
#define STW_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STW_REALLOC
#define STW_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STW_FREE
#define STW_FREE(p) free((p))
#endif

/* ============================================================================
 * String Buffer
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int indent;
} StwStringBuffer;

static void stw_sb_init(StwStringBuffer *sb) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->indent = 0;
}

static void stw_sb_free(StwStringBuffer *sb) {
    STW_FREE(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}

static int stw_sb_reserve(StwStringBuffer *sb, size_t need) {
    if (sb->len + need + 1 <= sb->cap) return 1;
    size_t ncap = sb->cap ? sb->cap : 256;
    while (ncap < sb->len + need + 1) ncap *= 2;
    char *nd = (char *)STW_REALLOC(sb->data, ncap);
    if (!nd) return 0;
    sb->data = nd;
    sb->cap = ncap;
    return 1;
}

static void stw_sb_append(StwStringBuffer *sb, const char *s) {
    size_t slen = strlen(s);
    if (!stw_sb_reserve(sb, slen)) return;
    memcpy(sb->data + sb->len, s, slen);
    sb->len += slen;
    sb->data[sb->len] = '\0';
}

static void stw_sb_appendf(StwStringBuffer *sb, const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n > 0) {
        stw_sb_append(sb, buf);
    }
}

static void stw_sb_indent(StwStringBuffer *sb) {
    for (int i = 0; i < sb->indent; i++) {
        stw_sb_append(sb, "    ");
    }
}

static void stw_sb_newline(StwStringBuffer *sb) {
    stw_sb_append(sb, "\n");
}

/* ============================================================================
 * Converter Context
 * ============================================================================ */

typedef struct {
    const SsirModule *mod;
    SsirToWgslOptions opts;
    StwStringBuffer sb;

    /* ID-to-name mapping */
    char **id_names;
    uint32_t id_names_cap;

    /* Current function context for expression lookup */
    const SsirFunction *current_func;

    /* Use count tracking for inlining */
    uint32_t *use_counts;

    /* Instruction lookup map for O(1) find_instruction */
    SsirInst **inst_map;
    uint32_t inst_map_cap;

    /* Error state */
    char last_error[256];
} SsirToWgslContext;

static void ctx_init(SsirToWgslContext *ctx, const SsirModule *mod, const SsirToWgslOptions *opts) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->mod = mod;
    if (opts) ctx->opts = *opts;
    stw_sb_init(&ctx->sb);

    ctx->id_names_cap = mod->next_id;
    ctx->id_names = (char **)STW_MALLOC(ctx->id_names_cap * sizeof(char *));
}

static void ctx_free(SsirToWgslContext *ctx) {
    if (ctx->id_names) {
        for (uint32_t i = 0; i < ctx->id_names_cap; i++) {
            STW_FREE(ctx->id_names[i]);
        }
        STW_FREE(ctx->id_names);
    }
    STW_FREE(ctx->use_counts);
    STW_FREE(ctx->inst_map);
    stw_sb_free(&ctx->sb);
}

static const char *stw_get_id_name(SsirToWgslContext *ctx, uint32_t id) {
    if (id < ctx->id_names_cap && ctx->id_names[id]) {
        return ctx->id_names[id];
    }
    static char buf[32];
    snprintf(buf, sizeof(buf), "_v%u", id);
    return buf;
}

static void set_id_name(SsirToWgslContext *ctx, uint32_t id, const char *name) {
    if (id >= ctx->id_names_cap) return;
    STW_FREE(ctx->id_names[id]);
    ctx->id_names[id] = name ? strdup(name) : NULL;
}

/* ============================================================================
 * Builtin Mappings
 * ============================================================================ */

static const char *builtin_var_to_wgsl(SsirBuiltinVar bv) {
    switch (bv) {
    case SSIR_BUILTIN_NONE:                 return NULL;
    case SSIR_BUILTIN_VERTEX_INDEX:         return "vertex_index";
    case SSIR_BUILTIN_INSTANCE_INDEX:       return "instance_index";
    case SSIR_BUILTIN_POSITION:             return "position";
    case SSIR_BUILTIN_FRONT_FACING:         return "front_facing";
    case SSIR_BUILTIN_FRAG_DEPTH:           return "frag_depth";
    case SSIR_BUILTIN_SAMPLE_INDEX:         return "sample_index";
    case SSIR_BUILTIN_SAMPLE_MASK:          return "sample_mask";
    case SSIR_BUILTIN_LOCAL_INVOCATION_ID:  return "local_invocation_id";
    case SSIR_BUILTIN_LOCAL_INVOCATION_INDEX: return "local_invocation_index";
    case SSIR_BUILTIN_GLOBAL_INVOCATION_ID: return "global_invocation_id";
    case SSIR_BUILTIN_WORKGROUP_ID:         return "workgroup_id";
    case SSIR_BUILTIN_NUM_WORKGROUPS:       return "num_workgroups";
    case SSIR_BUILTIN_CLIP_DISTANCE:        return "clip_distances";
    case SSIR_BUILTIN_FRAG_COORD:           return "position";
    case SSIR_BUILTIN_PRIMITIVE_ID:         return "primitive_index";
    case SSIR_BUILTIN_SUBGROUP_SIZE:         return "subgroup_size";
    case SSIR_BUILTIN_SUBGROUP_INVOCATION_ID: return "subgroup_invocation_id";
    case SSIR_BUILTIN_SUBGROUP_ID:            return "subgroup_id";
    case SSIR_BUILTIN_NUM_SUBGROUPS:          return "num_subgroups";
    default: return "unknown_builtin";
    }
}

static const char *builtin_func_to_wgsl(SsirBuiltinId id) {
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
    case SSIR_BUILTIN_INVERSESQRT:   return "inverseSqrt";
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
    case SSIR_BUILTIN_FACEFORWARD:   return "faceForward";
    case SSIR_BUILTIN_REFLECT:       return "reflect";
    case SSIR_BUILTIN_REFRACT:       return "refract";
    case SSIR_BUILTIN_ALL:           return "all";
    case SSIR_BUILTIN_ANY:           return "any";
    case SSIR_BUILTIN_SELECT:        return "select";
    case SSIR_BUILTIN_COUNTBITS:     return "countOneBits";
    case SSIR_BUILTIN_REVERSEBITS:   return "reverseBits";
    case SSIR_BUILTIN_FIRSTLEADINGBIT: return "firstLeadingBit";
    case SSIR_BUILTIN_FIRSTTRAILINGBIT: return "firstTrailingBit";
    case SSIR_BUILTIN_EXTRACTBITS:   return "extractBits";
    case SSIR_BUILTIN_INSERTBITS:    return "insertBits";
    case SSIR_BUILTIN_DPDX:          return "dpdx";
    case SSIR_BUILTIN_DPDY:          return "dpdy";
    case SSIR_BUILTIN_FWIDTH:        return "fwidth";
    case SSIR_BUILTIN_DPDX_COARSE:   return "dpdxCoarse";
    case SSIR_BUILTIN_DPDY_COARSE:   return "dpdyCoarse";
    case SSIR_BUILTIN_DPDX_FINE:     return "dpdxFine";
    case SSIR_BUILTIN_DPDY_FINE:     return "dpdyFine";
    case SSIR_BUILTIN_FMA:           return "fma";
    case SSIR_BUILTIN_ISINF:         return NULL; /* special */
    case SSIR_BUILTIN_ISNAN:         return NULL; /* special */
    case SSIR_BUILTIN_DEGREES:       return "degrees";
    case SSIR_BUILTIN_RADIANS:       return "radians";
    case SSIR_BUILTIN_MODF:          return "modf";
    case SSIR_BUILTIN_FREXP:         return "frexp";
    case SSIR_BUILTIN_LDEXP:         return "ldexp";
    case SSIR_BUILTIN_DETERMINANT:   return "determinant";
    case SSIR_BUILTIN_TRANSPOSE:     return "transpose";
    case SSIR_BUILTIN_PACK4X8SNORM:  return "pack4x8snorm";
    case SSIR_BUILTIN_PACK4X8UNORM:  return "pack4x8unorm";
    case SSIR_BUILTIN_PACK2X16SNORM: return "pack2x16snorm";
    case SSIR_BUILTIN_PACK2X16UNORM: return "pack2x16unorm";
    case SSIR_BUILTIN_PACK2X16FLOAT: return "pack2x16float";
    case SSIR_BUILTIN_UNPACK4X8SNORM:  return "unpack4x8snorm";
    case SSIR_BUILTIN_UNPACK4X8UNORM:  return "unpack4x8unorm";
    case SSIR_BUILTIN_UNPACK2X16SNORM: return "unpack2x16snorm";
    case SSIR_BUILTIN_UNPACK2X16UNORM: return "unpack2x16unorm";
    case SSIR_BUILTIN_UNPACK2X16FLOAT: return "unpack2x16float";
    case SSIR_BUILTIN_SUBGROUP_BALLOT: return "subgroupBallot";
    case SSIR_BUILTIN_SUBGROUP_BROADCAST: return "subgroupBroadcast";
    case SSIR_BUILTIN_SUBGROUP_ADD: return "subgroupAdd";
    case SSIR_BUILTIN_SUBGROUP_MIN: return "subgroupMin";
    case SSIR_BUILTIN_SUBGROUP_MAX: return "subgroupMax";
    case SSIR_BUILTIN_SUBGROUP_ALL: return "subgroupAll";
    case SSIR_BUILTIN_SUBGROUP_ANY: return "subgroupAny";
    case SSIR_BUILTIN_SUBGROUP_SHUFFLE: return "subgroupShuffle";
    case SSIR_BUILTIN_SUBGROUP_PREFIX_ADD: return "subgroupPrefixExclusiveAdd";
    default: return "unknown_builtin";
    }
}

static const char *address_space_to_wgsl(SsirAddressSpace space) {
    switch (space) {
    case SSIR_ADDR_FUNCTION:        return NULL;  /* implicit */
    case SSIR_ADDR_PRIVATE:         return "private";
    case SSIR_ADDR_WORKGROUP:       return "workgroup";
    case SSIR_ADDR_UNIFORM:         return "uniform";
    case SSIR_ADDR_UNIFORM_CONSTANT: return NULL;  /* implicit for samplers/textures */
    case SSIR_ADDR_STORAGE:         return "storage";
    case SSIR_ADDR_INPUT:           return NULL;  /* not a WGSL address space */
    case SSIR_ADDR_OUTPUT:          return NULL;  /* not a WGSL address space */
    case SSIR_ADDR_PUSH_CONSTANT:   return "push_constant";
    default: return "unknown";
    }
}

static const char *texture_dim_to_wgsl(SsirTextureDim dim) {
    switch (dim) {
    case SSIR_TEX_1D:              return "1d";
    case SSIR_TEX_2D:              return "2d";
    case SSIR_TEX_3D:              return "3d";
    case SSIR_TEX_CUBE:            return "cube";
    case SSIR_TEX_2D_ARRAY:        return "2d_array";
    case SSIR_TEX_CUBE_ARRAY:      return "cube_array";
    case SSIR_TEX_MULTISAMPLED_2D: return "multisampled_2d";
    case SSIR_TEX_1D_ARRAY:        return "1d_array";
    case SSIR_TEX_BUFFER:          return "1d"; /* WGSL has no buffer textures; approximate as 1d */
    case SSIR_TEX_MULTISAMPLED_2D_ARRAY: return "multisampled_2d"; /* WGSL has no MS array; approximate */
    default: return "2d";
    }
}

/* ============================================================================
 * Type Emission
 * ============================================================================ */

static void emit_type(SsirToWgslContext *ctx, uint32_t type_id, StwStringBuffer *out);

static void emit_scalar_type(SsirToWgslContext *ctx, const SsirType *t, StwStringBuffer *out) {
    switch (t->kind) {
    case SSIR_TYPE_VOID: stw_sb_append(out, "void"); break;
    case SSIR_TYPE_BOOL: stw_sb_append(out, "bool"); break;
    case SSIR_TYPE_I32:  stw_sb_append(out, "i32"); break;
    case SSIR_TYPE_U32:  stw_sb_append(out, "u32"); break;
    case SSIR_TYPE_F32:  stw_sb_append(out, "f32"); break;
    case SSIR_TYPE_F16:  stw_sb_append(out, "f16"); break;
    case SSIR_TYPE_F64:  stw_sb_append(out, "f64"); break;
    case SSIR_TYPE_I8:   stw_sb_append(out, "i32"); break;
    case SSIR_TYPE_U8:   stw_sb_append(out, "u32"); break;
    case SSIR_TYPE_I16:  stw_sb_append(out, "i32"); break;
    case SSIR_TYPE_U16:  stw_sb_append(out, "u32"); break;
    case SSIR_TYPE_I64:  stw_sb_append(out, "i32"); break;
    case SSIR_TYPE_U64:  stw_sb_append(out, "u32"); break;
    default: stw_sb_append(out, "unknown"); break;
    }
}

static void emit_type(SsirToWgslContext *ctx, uint32_t type_id, StwStringBuffer *out) {
    SsirType *t = ssir_get_type((SsirModule *)ctx->mod, type_id);
    if (!t) {
        stw_sb_appendf(out, "_type%u", type_id);
        return;
    }

    switch (t->kind) {
    case SSIR_TYPE_VOID:
    case SSIR_TYPE_BOOL:
    case SSIR_TYPE_I32:
    case SSIR_TYPE_U32:
    case SSIR_TYPE_F32:
    case SSIR_TYPE_F16:
    case SSIR_TYPE_F64:
    case SSIR_TYPE_I8:
    case SSIR_TYPE_U8:
    case SSIR_TYPE_I16:
    case SSIR_TYPE_U16:
    case SSIR_TYPE_I64:
    case SSIR_TYPE_U64:
        emit_scalar_type(ctx, t, out);
        break;

    case SSIR_TYPE_VEC: {
        stw_sb_appendf(out, "vec%u<", t->vec.size);
        emit_type(ctx, t->vec.elem, out);
        stw_sb_append(out, ">");
        break;
    }

    case SSIR_TYPE_MAT: {
        stw_sb_appendf(out, "mat%ux%u<", t->mat.cols, t->mat.rows);
        /* Get the scalar type from the column vector */
        SsirType *col_t = ssir_get_type((SsirModule *)ctx->mod, t->mat.elem);
        if (col_t && col_t->kind == SSIR_TYPE_VEC) {
            emit_type(ctx, col_t->vec.elem, out);
        } else {
            stw_sb_append(out, "f32");
        }
        stw_sb_append(out, ">");
        break;
    }

    case SSIR_TYPE_ARRAY:
        stw_sb_append(out, "array<");
        emit_type(ctx, t->array.elem, out);
        stw_sb_appendf(out, ", %u>", t->array.length);
        break;

    case SSIR_TYPE_RUNTIME_ARRAY:
        stw_sb_append(out, "array<");
        emit_type(ctx, t->runtime_array.elem, out);
        stw_sb_append(out, ">");
        break;

    case SSIR_TYPE_STRUCT:
        stw_sb_append(out, t->struc.name ? t->struc.name : "_Struct");
        break;

    case SSIR_TYPE_PTR:
        /* In WGSL, pointers are typically unwrapped */
        emit_type(ctx, t->ptr.pointee, out);
        break;

    case SSIR_TYPE_SAMPLER:
        stw_sb_append(out, "sampler");
        break;

    case SSIR_TYPE_SAMPLER_COMPARISON:
        stw_sb_append(out, "sampler_comparison");
        break;

    case SSIR_TYPE_TEXTURE: {
        stw_sb_appendf(out, "texture_%s<", texture_dim_to_wgsl(t->texture.dim));
        emit_type(ctx, t->texture.sampled_type, out);
        stw_sb_append(out, ">");
        break;
    }

    case SSIR_TYPE_TEXTURE_STORAGE: {
        stw_sb_appendf(out, "texture_storage_%s<", texture_dim_to_wgsl(t->texture_storage.dim));
        /* Would need format enum to string conversion */
        stw_sb_append(out, "rgba8unorm");
        stw_sb_append(out, ", ");
        switch (t->texture_storage.access) {
        case SSIR_ACCESS_READ: stw_sb_append(out, "read"); break;
        case SSIR_ACCESS_WRITE: stw_sb_append(out, "write"); break;
        case SSIR_ACCESS_READ_WRITE: stw_sb_append(out, "read_write"); break;
        }
        stw_sb_append(out, ">");
        break;
    }

    case SSIR_TYPE_TEXTURE_DEPTH:
        stw_sb_appendf(out, "texture_depth_%s", texture_dim_to_wgsl(t->texture_depth.dim));
        break;

    default:
        stw_sb_appendf(out, "_unknown_type_%u", type_id);
        break;
    }
}

/* ============================================================================
 * Constant Emission
 * ============================================================================ */

static void stw_emit_constant(SsirToWgslContext *ctx, SsirConstant *c, StwStringBuffer *out);

static void stw_emit_constant(SsirToWgslContext *ctx, SsirConstant *c, StwStringBuffer *out) {
    switch (c->kind) {
    case SSIR_CONST_BOOL:
        stw_sb_append(out, c->bool_val ? "true" : "false");
        break;

    case SSIR_CONST_I32:
        stw_sb_appendf(out, "%di", c->i32_val);
        break;

    case SSIR_CONST_U32:
        stw_sb_appendf(out, "%uu", c->u32_val);
        break;

    case SSIR_CONST_F32: {
        float f = c->f32_val;
        if (floorf(f) == f && fabsf(f) < 1e6f) {
            stw_sb_appendf(out, "%.1f", (double)f);
        } else {
            stw_sb_appendf(out, "%g", (double)f);
        }
        break;
    }

    case SSIR_CONST_F16:
        /* Half-precision - emit as hex for now */
        stw_sb_appendf(out, "0x%xh", c->f16_val);
        break;

    case SSIR_CONST_F64: {
        double d = c->f64_val;
        if (floor(d) == d && fabs(d) < 1e15)
            stw_sb_appendf(out, "%.1f", d);
        else
            stw_sb_appendf(out, "%g", d);
        break;
    }
    case SSIR_CONST_I8:
        stw_sb_appendf(out, "%di", (int)c->i8_val);
        break;
    case SSIR_CONST_U8:
        stw_sb_appendf(out, "%uu", (unsigned)c->u8_val);
        break;
    case SSIR_CONST_I16:
        stw_sb_appendf(out, "%di", (int)c->i16_val);
        break;
    case SSIR_CONST_U16:
        stw_sb_appendf(out, "%uu", (unsigned)c->u16_val);
        break;
    case SSIR_CONST_I64:
        stw_sb_appendf(out, "%lldi", (long long)c->i64_val);
        break;
    case SSIR_CONST_U64:
        stw_sb_appendf(out, "%lluu", (unsigned long long)c->u64_val);
        break;

    case SSIR_CONST_COMPOSITE: {
        /* Emit type constructor */
        emit_type(ctx, c->type, out);
        stw_sb_append(out, "(");
        for (uint32_t i = 0; i < c->composite.count; i++) {
            if (i > 0) stw_sb_append(out, ", ");
            SsirConstant *elem = ssir_get_constant((SsirModule *)ctx->mod, c->composite.components[i]);
            if (elem) {
                stw_emit_constant(ctx, elem, out);
            } else {
                stw_sb_appendf(out, "_const%u", c->composite.components[i]);
            }
        }
        stw_sb_append(out, ")");
        break;
    }

    case SSIR_CONST_NULL:
        /* Zero value for type */
        emit_type(ctx, c->type, out);
        stw_sb_append(out, "()");
        break;

    default:
        stw_sb_appendf(out, "_const%u", c->id);
        break;
    }
}

/* ============================================================================
 * Expression Emission
 * ============================================================================ */

/* Forward declaration */
static void stw_emit_expression(SsirToWgslContext *ctx, uint32_t id, StwStringBuffer *out);

/* Find instruction by result ID in current function */
static SsirInst *find_instruction(SsirToWgslContext *ctx, uint32_t id) {
    if (ctx->inst_map && id < ctx->inst_map_cap) return ctx->inst_map[id];
    return NULL;
}

/* Check if ID is a function parameter */
static SsirFunctionParam *find_param(SsirToWgslContext *ctx, uint32_t id) {
    if (!ctx->current_func) return NULL;
    for (uint32_t i = 0; i < ctx->current_func->param_count; i++) {
        if (ctx->current_func->params[i].id == id) {
            return &ctx->current_func->params[i];
        }
    }
    return NULL;
}

/* Check if ID is a local variable */
static SsirLocalVar *find_local(SsirToWgslContext *ctx, uint32_t id) {
    if (!ctx->current_func) return NULL;
    for (uint32_t i = 0; i < ctx->current_func->local_count; i++) {
        if (ctx->current_func->locals[i].id == id) {
            return &ctx->current_func->locals[i];
        }
    }
    return NULL;
}

static void stw_emit_binary_op(SsirToWgslContext *ctx, SsirInst *inst, const char *op, StwStringBuffer *out) {
    stw_sb_append(out, "(");
    stw_emit_expression(ctx, inst->operands[0], out);
    stw_sb_append(out, op);
    stw_emit_expression(ctx, inst->operands[1], out);
    stw_sb_append(out, ")");
}

static void stw_emit_unary_op(SsirToWgslContext *ctx, SsirInst *inst, const char *op, StwStringBuffer *out) {
    stw_sb_append(out, "(");
    stw_sb_append(out, op);
    stw_emit_expression(ctx, inst->operands[0], out);
    stw_sb_append(out, ")");
}

static void stw_emit_expression(SsirToWgslContext *ctx, uint32_t id, StwStringBuffer *out) {
    /* Check if it's a constant */
    SsirConstant *c = ssir_get_constant((SsirModule *)ctx->mod, id);
    if (c) {
        stw_emit_constant(ctx, c, out);
        return;
    }

    /* Check if it's a global variable */
    SsirGlobalVar *g = ssir_get_global((SsirModule *)ctx->mod, id);
    if (g) {
        stw_sb_append(out, stw_get_id_name(ctx, id));
        return;
    }

    /* Check if it's a function parameter */
    SsirFunctionParam *param = find_param(ctx, id);
    if (param) {
        stw_sb_append(out, stw_get_id_name(ctx, id));
        return;
    }

    /* Check if it's a local variable */
    SsirLocalVar *local = find_local(ctx, id);
    if (local) {
        stw_sb_append(out, stw_get_id_name(ctx, id));
        return;
    }

    /* Look for instruction in current function */
    SsirInst *inst = find_instruction(ctx, id);
    if (!inst) {
        stw_sb_append(out, stw_get_id_name(ctx, id));
        return;
    }

    /* If materialized as a temporary (multi-use), emit the name */
    if (id < ctx->id_names_cap && ctx->id_names[id]) {
        stw_sb_append(out, ctx->id_names[id]);
        return;
    }

    switch (inst->op) {
    /* Arithmetic */
    case SSIR_OP_ADD: stw_emit_binary_op(ctx, inst, " + ", out); break;
    case SSIR_OP_SUB: stw_emit_binary_op(ctx, inst, " - ", out); break;
    case SSIR_OP_MUL: stw_emit_binary_op(ctx, inst, " * ", out); break;
    case SSIR_OP_DIV: stw_emit_binary_op(ctx, inst, " / ", out); break;
    case SSIR_OP_MOD: stw_emit_binary_op(ctx, inst, " % ", out); break;
    case SSIR_OP_REM: {
        SsirModule *rmod = (SsirModule *)ctx->mod;
        SsirType *rem_t = ssir_get_type(rmod, inst->type);
        if (rem_t && ssir_type_is_float(rem_t)) {
            stw_sb_append(out, "(");
            stw_emit_expression(ctx, inst->operands[0], out);
            stw_sb_append(out, " - ");
            stw_emit_expression(ctx, inst->operands[1], out);
            stw_sb_append(out, " * trunc(");
            stw_emit_expression(ctx, inst->operands[0], out);
            stw_sb_append(out, " / ");
            stw_emit_expression(ctx, inst->operands[1], out);
            stw_sb_append(out, "))");
        } else {
            stw_emit_expression(ctx, inst->operands[0], out);
            stw_sb_append(out, " % ");
            stw_emit_expression(ctx, inst->operands[1], out);
        }
        break;
    }
    case SSIR_OP_NEG: stw_emit_unary_op(ctx, inst, "-", out); break;

    /* Matrix */
    case SSIR_OP_MAT_MUL: stw_emit_binary_op(ctx, inst, " * ", out); break;
    case SSIR_OP_MAT_TRANSPOSE:
        stw_sb_append(out, "transpose(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    /* Bitwise */
    case SSIR_OP_BIT_AND: stw_emit_binary_op(ctx, inst, " & ", out); break;
    case SSIR_OP_BIT_OR:  stw_emit_binary_op(ctx, inst, " | ", out); break;
    case SSIR_OP_BIT_XOR: stw_emit_binary_op(ctx, inst, " ^ ", out); break;
    case SSIR_OP_BIT_NOT: stw_emit_unary_op(ctx, inst, "~", out); break;
    case SSIR_OP_SHL:     stw_emit_binary_op(ctx, inst, " << ", out); break;
    case SSIR_OP_SHR:    stw_emit_binary_op(ctx, inst, " >> ", out); break;
    case SSIR_OP_SHR_LOGICAL: {
        SsirModule *wmod = (SsirModule *)ctx->mod;
        SsirType *shr_type = ssir_get_type(wmod, inst->type);
        bool shr_signed = shr_type && (ssir_type_is_signed(shr_type) ||
            (shr_type->kind == SSIR_TYPE_VEC &&
             ssir_type_is_signed(ssir_get_type(wmod, shr_type->vec.elem))));
        if (shr_signed) {
            stw_sb_append(out, "bitcast<");
            emit_type(ctx, inst->type, out);
            stw_sb_append(out, ">(bitcast<");
            if (shr_type->kind == SSIR_TYPE_VEC) {
                uint32_t uvec = ssir_type_vec(wmod,
                    ssir_type_u32(wmod), shr_type->vec.size);
                emit_type(ctx, uvec, out);
            } else {
                stw_sb_append(out, "u32");
            }
            stw_sb_append(out, ">(");
            stw_emit_expression(ctx, inst->operands[0], out);
            stw_sb_append(out, ") >> ");
            stw_emit_expression(ctx, inst->operands[1], out);
            stw_sb_append(out, ")");
        } else {
            stw_emit_binary_op(ctx, inst, " >> ", out);
        }
        break;
    }

    /* Comparison */
    case SSIR_OP_EQ: stw_emit_binary_op(ctx, inst, " == ", out); break;
    case SSIR_OP_NE: stw_emit_binary_op(ctx, inst, " != ", out); break;
    case SSIR_OP_LT: stw_emit_binary_op(ctx, inst, " < ", out); break;
    case SSIR_OP_LE: stw_emit_binary_op(ctx, inst, " <= ", out); break;
    case SSIR_OP_GT: stw_emit_binary_op(ctx, inst, " > ", out); break;
    case SSIR_OP_GE: stw_emit_binary_op(ctx, inst, " >= ", out); break;

    /* Logical */
    case SSIR_OP_AND: stw_emit_binary_op(ctx, inst, " && ", out); break;
    case SSIR_OP_OR:  stw_emit_binary_op(ctx, inst, " || ", out); break;
    case SSIR_OP_NOT: stw_emit_unary_op(ctx, inst, "!", out); break;

    /* Composite */
    case SSIR_OP_CONSTRUCT: {
        emit_type(ctx, inst->type, out);
        stw_sb_append(out, "(");
        /* Components are either in operands or extra */
        uint32_t count = inst->operand_count;
        const uint32_t *comps = inst->operands;
        if (inst->extra_count > 0) {
            count = inst->extra_count;
            comps = inst->extra;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (i > 0) stw_sb_append(out, ", ");
            stw_emit_expression(ctx, comps[i], out);
        }
        stw_sb_append(out, ")");
        break;
    }

    case SSIR_OP_EXTRACT: {
        stw_emit_expression(ctx, inst->operands[0], out);
        /* Check if extracting from vector - use swizzle */
        SsirType *base_t = ssir_get_type((SsirModule *)ctx->mod, inst->type);
        uint32_t idx = inst->operands[1];
        if (idx < 4) {
            const char swizzle[] = "xyzw";
            stw_sb_appendf(out, ".%c", swizzle[idx]);
        } else {
            stw_sb_appendf(out, "[%u]", idx);
        }
        break;
    }

    case SSIR_OP_INSERT:
    case SSIR_OP_INSERT_DYN:
        stw_sb_append(out, stw_get_id_name(ctx, id));
        break;

    case SSIR_OP_SHUFFLE: {
        emit_type(ctx, inst->type, out);
        stw_sb_append(out, "(");
        /* Indices in extra array */
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) stw_sb_append(out, ", ");
            uint32_t src_idx = inst->extra[i];
            /* First operand is v1, second is v2 */
            uint32_t v1_size = 4; /* Assume vec4 for simplicity */
            if (src_idx < v1_size) {
                stw_emit_expression(ctx, inst->operands[0], out);
            } else {
                stw_emit_expression(ctx, inst->operands[1], out);
                src_idx -= v1_size;
            }
            if (src_idx < 4) {
                const char swizzle[] = "xyzw";
                stw_sb_appendf(out, ".%c", swizzle[src_idx]);
            }
        }
        stw_sb_append(out, ")");
        break;
    }

    case SSIR_OP_SPLAT: {
        emit_type(ctx, inst->type, out);
        stw_sb_append(out, "(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;
    }

    case SSIR_OP_EXTRACT_DYN:
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, "[");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, "]");
        break;

    /* Memory */
    case SSIR_OP_LOAD:
        /* Dereference pointer - in WGSL just use the variable */
        stw_emit_expression(ctx, inst->operands[0], out);
        break;

    case SSIR_OP_ACCESS: {
        SsirModule *amod = (SsirModule *)ctx->mod;
        stw_emit_expression(ctx, inst->operands[0], out);
        /* Trace struct type through the access chain for member names */
        uint32_t cur_type_id = 0;
        {
            SsirGlobalVar *ag = ssir_get_global(amod, inst->operands[0]);
            if (ag) {
                SsirType *pt = ssir_get_type(amod, ag->type);
                if (pt && pt->kind == SSIR_TYPE_PTR)
                    cur_type_id = pt->ptr.pointee;
            }
            if (!cur_type_id) {
                SsirInst *ai = find_instruction(ctx, inst->operands[0]);
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
            SsirConstant *idx_c = ssir_get_constant(amod, idx);
            SsirType *cur_st = cur_type_id ? ssir_get_type(amod, cur_type_id) : NULL;
            if (idx_c && (idx_c->kind == SSIR_CONST_U32 || idx_c->kind == SSIR_CONST_I32)) {
                uint32_t midx = (idx_c->kind == SSIR_CONST_U32) ? idx_c->u32_val : (uint32_t)idx_c->i32_val;
                const char *mname = NULL;
                if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT &&
                    cur_st->struc.member_names && midx < cur_st->struc.member_count)
                    mname = cur_st->struc.member_names[midx];
                if (mname)
                    stw_sb_appendf(out, ".%s", mname);
                else
                    stw_sb_appendf(out, ".member%u", midx);
                /* Advance type through struct member */
                if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT && midx < cur_st->struc.member_count)
                    cur_type_id = cur_st->struc.members[midx];
                else
                    cur_type_id = 0;
            } else {
                stw_sb_append(out, "[");
                stw_emit_expression(ctx, idx, out);
                stw_sb_append(out, "]");
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
        stw_sb_append(out, "arrayLength(&");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    /* Call */
    case SSIR_OP_CALL: {
        uint32_t callee_id = inst->operands[0];
        SsirFunction *callee = ssir_get_function((SsirModule *)ctx->mod, callee_id);
        if (callee && callee->name) {
            stw_sb_append(out, callee->name);
        } else {
            stw_sb_append(out, stw_get_id_name(ctx, callee_id));
        }
        stw_sb_append(out, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) stw_sb_append(out, ", ");
            stw_emit_expression(ctx, inst->extra[i], out);
        }
        stw_sb_append(out, ")");
        break;
    }

    case SSIR_OP_BUILTIN: {
        SsirBuiltinId builtin_id = (SsirBuiltinId)inst->operands[0];
        /* WGSL doesn't have isinf/isnan — emit inline equivalents */
        if (builtin_id == SSIR_BUILTIN_ISNAN && inst->extra_count > 0) {
            stw_sb_append(out, "(");
            stw_emit_expression(ctx, inst->extra[0], out);
            stw_sb_append(out, " != ");
            stw_emit_expression(ctx, inst->extra[0], out);
            stw_sb_append(out, ")");
            break;
        }
        if (builtin_id == SSIR_BUILTIN_ISINF && inst->extra_count > 0) {
            stw_sb_append(out, "(abs(");
            stw_emit_expression(ctx, inst->extra[0], out);
            stw_sb_append(out, ") == ");
            /* Use a very large float literal representing infinity */
            stw_sb_append(out, "0x1p+128f");
            stw_sb_append(out, ")");
            break;
        }
        const char *wname = builtin_func_to_wgsl(builtin_id);
        stw_sb_append(out, wname ? wname : "unknown_builtin");
        stw_sb_append(out, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) stw_sb_append(out, ", ");
            stw_emit_expression(ctx, inst->extra[i], out);
        }
        stw_sb_append(out, ")");
        break;
    }

    /* Conversion */
    case SSIR_OP_CONVERT:
        emit_type(ctx, inst->type, out);
        stw_sb_append(out, "(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_BITCAST:
        stw_sb_append(out, "bitcast<");
        emit_type(ctx, inst->type, out);
        stw_sb_append(out, ">(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    /* Texture */
    case SSIR_OP_TEX_SAMPLE:
        stw_sb_append(out, "textureSample(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_BIAS:
        stw_sb_append(out, "textureSampleBias(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_LEVEL:
        stw_sb_append(out, "textureSampleLevel(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_GRAD:
        stw_sb_append(out, "textureSampleGrad(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP:
        stw_sb_append(out, "textureSampleCompare(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
        stw_sb_append(out, "textureSampleCompareLevel(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_OFFSET:
        stw_sb_append(out, "textureSample(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
        stw_sb_append(out, "textureSampleBias(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
        stw_sb_append(out, "textureSampleLevel(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
        stw_sb_append(out, "textureSampleGrad(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[5], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
        stw_sb_append(out, "textureSampleCompare(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_GATHER:
        stw_sb_append(out, "textureGather(");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_GATHER_CMP:
        stw_sb_append(out, "textureGatherCompare(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_GATHER_OFFSET:
        stw_sb_append(out, "textureGather(");
        stw_emit_expression(ctx, inst->operands[3], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[4], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_QUERY_LEVELS:
        stw_sb_append(out, "textureNumLevels(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_QUERY_SAMPLES:
        stw_sb_append(out, "textureNumSamples(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_LOAD:
        stw_sb_append(out, "textureLoad(");
        stw_emit_expression(ctx, inst->operands[0], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[1], out);
        stw_sb_append(out, ", ");
        stw_emit_expression(ctx, inst->operands[2], out);
        stw_sb_append(out, ")");
        break;

    case SSIR_OP_TEX_SIZE:
        stw_sb_append(out, "textureDimensions(");
        stw_emit_expression(ctx, inst->operands[0], out);
        if (inst->operand_count > 1 && inst->operands[1] != 0) {
            stw_sb_append(out, ", ");
            stw_emit_expression(ctx, inst->operands[1], out);
        }
        stw_sb_append(out, ")");
        break;

    /* Phi - not directly representable, use the name */
    case SSIR_OP_PHI:
        stw_sb_append(out, stw_get_id_name(ctx, id));
        break;

    default:
        /* Fallback to ID name */
        stw_sb_append(out, stw_get_id_name(ctx, id));
        break;
    }
}

/* ============================================================================
 * Struct Emission
 * ============================================================================ */

static void emit_struct(SsirToWgslContext *ctx, SsirType *t) {
    if (t->kind != SSIR_TYPE_STRUCT) return;

    stw_sb_appendf(&ctx->sb, "struct %s {\n", t->struc.name ? t->struc.name : "_Struct");
    ctx->sb.indent++;

    for (uint32_t i = 0; i < t->struc.member_count; i++) {
        stw_sb_indent(&ctx->sb);
        if (t->struc.member_names && t->struc.member_names[i])
            stw_sb_appendf(&ctx->sb, "%s: ", t->struc.member_names[i]);
        else
            stw_sb_appendf(&ctx->sb, "member%u: ", i);
        emit_type(ctx, t->struc.members[i], &ctx->sb);
        stw_sb_append(&ctx->sb, ",\n");
    }

    ctx->sb.indent--;
    stw_sb_append(&ctx->sb, "}\n\n");
}

/* ============================================================================
 * Global Variable Emission
 * ============================================================================ */

static void emit_global(SsirToWgslContext *ctx, SsirGlobalVar *g) {
    /* Skip input/output globals - they become function parameters/returns */
    SsirType *ptr_t = ssir_get_type((SsirModule *)ctx->mod, g->type);
    if (ptr_t && ptr_t->kind == SSIR_TYPE_PTR) {
        if (ptr_t->ptr.space == SSIR_ADDR_INPUT || ptr_t->ptr.space == SSIR_ADDR_OUTPUT) {
            return;
        }
    }

    /* Emit attributes */
    if (g->has_group) stw_sb_appendf(&ctx->sb, "@group(%u) ", g->group);
    if (g->has_binding) stw_sb_appendf(&ctx->sb, "@binding(%u) ", g->binding);

    /* Get pointee type */
    uint32_t pointee = ptr_t ? ptr_t->ptr.pointee : g->type;
    const char *addr_space = ptr_t ? address_space_to_wgsl(ptr_t->ptr.space) : NULL;

    if (addr_space) {
        stw_sb_appendf(&ctx->sb, "var<%s> %s: ", addr_space, stw_get_id_name(ctx, g->id));
    } else {
        stw_sb_appendf(&ctx->sb, "var %s: ", stw_get_id_name(ctx, g->id));
    }
    emit_type(ctx, pointee, &ctx->sb);
    stw_sb_append(&ctx->sb, ";\n");
}

/* ============================================================================
 * Function Emission
 * ============================================================================ */

static SsirEntryPoint *find_entry_point(SsirToWgslContext *ctx, uint32_t func_id) {
    for (uint32_t i = 0; i < ctx->mod->entry_point_count; i++) {
        if (ctx->mod->entry_points[i].function == func_id) {
            return &ctx->mod->entry_points[i];
        }
    }
    return NULL;
}

/* Track which blocks have been emitted */
typedef struct {
    bool *emitted;
    uint32_t count;
} BlockEmitState;

static void stw_emit_block(SsirToWgslContext *ctx, SsirBlock *blk, SsirFunction *fn, BlockEmitState *state);

static void emit_statement(SsirToWgslContext *ctx, SsirInst *inst, SsirBlock *blk, SsirFunction *fn, BlockEmitState *state) {
    switch (inst->op) {
    case SSIR_OP_STORE:
        stw_sb_indent(&ctx->sb);
        stw_emit_expression(ctx, inst->operands[0], &ctx->sb);
        stw_sb_append(&ctx->sb, " = ");
        stw_emit_expression(ctx, inst->operands[1], &ctx->sb);
        stw_sb_append(&ctx->sb, ";\n");
        break;

    case SSIR_OP_TEX_STORE:
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "textureStore(");
        stw_emit_expression(ctx, inst->operands[0], &ctx->sb);
        stw_sb_append(&ctx->sb, ", ");
        stw_emit_expression(ctx, inst->operands[1], &ctx->sb);
        stw_sb_append(&ctx->sb, ", ");
        stw_emit_expression(ctx, inst->operands[2], &ctx->sb);
        stw_sb_append(&ctx->sb, ");\n");
        break;

    case SSIR_OP_RETURN_VOID:
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "return;\n");
        break;

    case SSIR_OP_RETURN:
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "return ");
        stw_emit_expression(ctx, inst->operands[0], &ctx->sb);
        stw_sb_append(&ctx->sb, ";\n");
        break;

    case SSIR_OP_BRANCH:
        /* Simple branch - continue to target block */
        {
            uint32_t target = inst->operands[0];
            for (uint32_t i = 0; i < fn->block_count; i++) {
                if (fn->blocks[i].id == target && !state->emitted[i]) {
                    stw_emit_block(ctx, &fn->blocks[i], fn, state);
                    break;
                }
            }
        }
        break;

    case SSIR_OP_BRANCH_COND: {
        uint32_t cond = inst->operands[0];
        uint32_t true_block = inst->operands[1];
        uint32_t false_block = inst->operands[2];

        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "if (");
        stw_emit_expression(ctx, cond, &ctx->sb);
        stw_sb_append(&ctx->sb, ") {\n");
        ctx->sb.indent++;

        /* Emit true block */
        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == true_block && !state->emitted[i]) {
                stw_emit_block(ctx, &fn->blocks[i], fn, state);
                break;
            }
        }

        ctx->sb.indent--;
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "} else {\n");
        ctx->sb.indent++;

        /* Emit false block */
        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == false_block && !state->emitted[i]) {
                stw_emit_block(ctx, &fn->blocks[i], fn, state);
                break;
            }
        }

        ctx->sb.indent--;
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "}\n");
        break;
    }

    case SSIR_OP_BARRIER:
        stw_sb_indent(&ctx->sb);
        switch ((SsirBarrierScope)inst->operands[0]) {
        case SSIR_BARRIER_WORKGROUP: stw_sb_append(&ctx->sb, "workgroupBarrier();\n"); break;
        case SSIR_BARRIER_STORAGE:   stw_sb_append(&ctx->sb, "storageBarrier();\n"); break;
        case SSIR_BARRIER_SUBGROUP:  stw_sb_append(&ctx->sb, "/* subgroup barrier */;\n"); break;
        case SSIR_BARRIER_IMAGE:     stw_sb_append(&ctx->sb, "textureBarrier();\n"); break;
        }
        break;

    case SSIR_OP_DISCARD:
        stw_sb_indent(&ctx->sb);
        stw_sb_append(&ctx->sb, "discard;\n");
        break;

    case SSIR_OP_INSERT_DYN:
        stw_sb_indent(&ctx->sb);
        stw_sb_appendf(&ctx->sb, "var %s = ", stw_get_id_name(ctx, inst->result));
        stw_emit_expression(ctx, inst->operands[0], &ctx->sb);
        stw_sb_append(&ctx->sb, ";\n");
        stw_sb_indent(&ctx->sb);
        stw_sb_appendf(&ctx->sb, "%s[", stw_get_id_name(ctx, inst->result));
        stw_emit_expression(ctx, inst->operands[2], &ctx->sb);
        stw_sb_append(&ctx->sb, "] = ");
        stw_emit_expression(ctx, inst->operands[1], &ctx->sb);
        stw_sb_append(&ctx->sb, ";\n");
        break;

    case SSIR_OP_LOOP_MERGE:
    case SSIR_OP_SELECTION_MERGE:
    case SSIR_OP_UNREACHABLE:
        /* Skip these - they're control flow hints */
        break;

    default:
        /* Value-producing instruction - emit as let if needed */
        if (inst->result != 0) {
            bool has_side_effects = (inst->op == SSIR_OP_CALL ||
                                     inst->op == SSIR_OP_ATOMIC);
            bool must_materialize = (inst->op == SSIR_OP_PHI || has_side_effects);
            uint32_t uses = (ctx->use_counts && inst->result < ctx->mod->next_id)
                            ? ctx->use_counts[inst->result] : 2;
            if (must_materialize || uses > 1) {
                stw_sb_indent(&ctx->sb);
                if (has_side_effects && uses == 0) {
                    stw_emit_expression(ctx, inst->result, &ctx->sb);
                } else {
                    stw_sb_appendf(&ctx->sb, "let %s = ", stw_get_id_name(ctx, inst->result));
                    stw_emit_expression(ctx, inst->result, &ctx->sb);
                    set_id_name(ctx, inst->result, stw_get_id_name(ctx, inst->result));
                }
                stw_sb_append(&ctx->sb, ";\n");
            }
        }
        break;
    }
}

static void stw_emit_block(SsirToWgslContext *ctx, SsirBlock *blk, SsirFunction *fn, BlockEmitState *state) {
    /* Find block index and mark as emitted */
    for (uint32_t i = 0; i < fn->block_count; i++) {
        if (fn->blocks[i].id == blk->id) {
            if (state->emitted[i]) return;
            state->emitted[i] = true;
            break;
        }
    }

    /* Emit all instructions in the block */
    for (uint32_t i = 0; i < blk->inst_count; i++) {
        emit_statement(ctx, &blk->insts[i], blk, fn, state);
    }
}

static void stw_emit_function(SsirToWgslContext *ctx, SsirFunction *fn) {
    SsirEntryPoint *ep = find_entry_point(ctx, fn->id);

    /* Entry point attributes */
    if (ep) {
        switch (ep->stage) {
        case SSIR_STAGE_VERTEX:
            stw_sb_append(&ctx->sb, "@vertex\n");
            break;
        case SSIR_STAGE_FRAGMENT:
            stw_sb_append(&ctx->sb, "@fragment\n");
            break;
        case SSIR_STAGE_COMPUTE:
            stw_sb_appendf(&ctx->sb, "@compute @workgroup_size(%u, %u, %u)\n",
                       ep->workgroup_size[0], ep->workgroup_size[1], ep->workgroup_size[2]);
            break;
        case SSIR_STAGE_GEOMETRY:
        case SSIR_STAGE_TESS_CONTROL:
        case SSIR_STAGE_TESS_EVAL:
            break;
        }
    }

    /* Function signature */
    stw_sb_appendf(&ctx->sb, "fn %s(", fn->name ? fn->name : stw_get_id_name(ctx, fn->id));

    /* Parameters - for entry points, include interface variables */
    int first_param = 1;
    if (ep) {
        /* Add input interface variables as parameters */
        for (uint32_t i = 0; i < ep->interface_count; i++) {
            SsirGlobalVar *g = ssir_get_global((SsirModule *)ctx->mod, ep->interface[i]);
            if (!g) continue;
            SsirType *ptr_t = ssir_get_type((SsirModule *)ctx->mod, g->type);
            if (!ptr_t || ptr_t->kind != SSIR_TYPE_PTR) continue;
            if (ptr_t->ptr.space != SSIR_ADDR_INPUT) continue;

            if (!first_param) stw_sb_append(&ctx->sb, ", ");
            first_param = 0;

            /* Emit parameter attributes */
            if (g->has_location) {
                stw_sb_appendf(&ctx->sb, "@location(%u) ", g->location);
            }
            if (g->builtin != SSIR_BUILTIN_NONE) {
                stw_sb_appendf(&ctx->sb, "@builtin(%s) ", builtin_var_to_wgsl(g->builtin));
            }
            if (g->invariant) {
                stw_sb_append(&ctx->sb, "@invariant ");
            }
            if (g->interp != SSIR_INTERP_NONE || g->interp_sampling != SSIR_INTERP_SAMPLING_NONE) {
                const char *itype = "perspective";
                if (g->interp == SSIR_INTERP_FLAT) itype = "flat";
                else if (g->interp == SSIR_INTERP_LINEAR) itype = "linear";
                if (g->interp_sampling == SSIR_INTERP_SAMPLING_CENTROID)
                    stw_sb_appendf(&ctx->sb, "@interpolate(%s, centroid) ", itype);
                else if (g->interp_sampling == SSIR_INTERP_SAMPLING_SAMPLE)
                    stw_sb_appendf(&ctx->sb, "@interpolate(%s, sample) ", itype);
                else
                    stw_sb_appendf(&ctx->sb, "@interpolate(%s) ", itype);
            }

            stw_sb_appendf(&ctx->sb, "%s: ", stw_get_id_name(ctx, g->id));
            emit_type(ctx, ptr_t->ptr.pointee, &ctx->sb);
        }
    }

    /* Regular function parameters */
    for (uint32_t i = 0; i < fn->param_count; i++) {
        if (!first_param) stw_sb_append(&ctx->sb, ", ");
        first_param = 0;

        stw_sb_appendf(&ctx->sb, "%s: ", stw_get_id_name(ctx, fn->params[i].id));
        emit_type(ctx, fn->params[i].type, &ctx->sb);
    }

    stw_sb_append(&ctx->sb, ")");

    /* Return type - for entry points, check output interface variables first */
    int has_return_type = 0;
    if (ep) {
        for (uint32_t i = 0; i < ep->interface_count; i++) {
            SsirGlobalVar *g = ssir_get_global((SsirModule *)ctx->mod, ep->interface[i]);
            if (!g) continue;
            SsirType *ptr_t = ssir_get_type((SsirModule *)ctx->mod, g->type);
            if (!ptr_t || ptr_t->kind != SSIR_TYPE_PTR) continue;
            if (ptr_t->ptr.space != SSIR_ADDR_OUTPUT) continue;

            stw_sb_append(&ctx->sb, " -> ");

            /* Emit return attributes */
            if (g->has_location) {
                stw_sb_appendf(&ctx->sb, "@location(%u) ", g->location);
            }
            if (g->builtin != SSIR_BUILTIN_NONE) {
                stw_sb_appendf(&ctx->sb, "@builtin(%s) ", builtin_var_to_wgsl(g->builtin));
            }

            emit_type(ctx, ptr_t->ptr.pointee, &ctx->sb);
            has_return_type = 1;
            break;
        }
    }

    /* Fall back to function return type if no output interface variable */
    if (!has_return_type) {
        SsirType *ret_t = ssir_get_type((SsirModule *)ctx->mod, fn->return_type);
        if (ret_t && ret_t->kind != SSIR_TYPE_VOID) {
            stw_sb_append(&ctx->sb, " -> ");
            emit_type(ctx, fn->return_type, &ctx->sb);
        }
    }

    stw_sb_append(&ctx->sb, " {\n");
    ctx->sb.indent++;

    /* Set current function for expression lookup */
    ctx->current_func = fn;

    /* Compute use counts for inlining decisions */
    STW_FREE(ctx->use_counts);
    ctx->use_counts = (uint32_t *)STW_MALLOC(ctx->mod->next_id * sizeof(uint32_t));
    if (ctx->use_counts) {
        memset(ctx->use_counts, 0, ctx->mod->next_id * sizeof(uint32_t));
        ssir_count_uses((SsirFunction *)fn, ctx->use_counts, ctx->mod->next_id);
    }

    /* Build instruction lookup map */
    STW_FREE(ctx->inst_map);
    ctx->inst_map_cap = ctx->mod->next_id;
    ctx->inst_map = (SsirInst **)STW_MALLOC(ctx->inst_map_cap * sizeof(SsirInst *));
    if (ctx->inst_map) {
        memset(ctx->inst_map, 0, ctx->inst_map_cap * sizeof(SsirInst *));
        for (uint32_t bi = 0; bi < fn->block_count; bi++) {
            SsirBlock *blk = &fn->blocks[bi];
            for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                uint32_t rid = blk->insts[ii].result;
                if (rid && rid < ctx->inst_map_cap)
                    ctx->inst_map[rid] = &blk->insts[ii];
            }
        }
    }

    /* Local variable declarations */
    for (uint32_t i = 0; i < fn->local_count; i++) {
        SsirLocalVar *local = &fn->locals[i];
        SsirType *ptr_t = ssir_get_type((SsirModule *)ctx->mod, local->type);
        uint32_t pointee = (ptr_t && ptr_t->kind == SSIR_TYPE_PTR) ? ptr_t->ptr.pointee : local->type;

        stw_sb_indent(&ctx->sb);
        stw_sb_appendf(&ctx->sb, "var %s: ", stw_get_id_name(ctx, local->id));
        emit_type(ctx, pointee, &ctx->sb);

        if (local->has_initializer) {
            stw_sb_append(&ctx->sb, " = ");
            SsirConstant *init = ssir_get_constant((SsirModule *)ctx->mod, local->initializer);
            if (init) {
                stw_emit_constant(ctx, init, &ctx->sb);
            }
        }

        stw_sb_append(&ctx->sb, ";\n");
    }

    /* Emit function body (basic blocks) */
    if (fn->block_count > 0) {
        BlockEmitState state;
        state.count = fn->block_count;
        state.emitted = (bool *)STW_MALLOC(fn->block_count * sizeof(bool));
        if (state.emitted) {
            memset(state.emitted, 0, fn->block_count * sizeof(bool));
            stw_emit_block(ctx, &fn->blocks[0], fn, &state);
            STW_FREE(state.emitted);
        }
    }

    ctx->current_func = NULL;

    ctx->sb.indent--;
    stw_sb_append(&ctx->sb, "}\n\n");
}

/* ============================================================================
 * Name Assignment
 * ============================================================================ */

static void assign_names(SsirToWgslContext *ctx) {
    /* Assign names to globals */
    for (uint32_t i = 0; i < ctx->mod->global_count; i++) {
        SsirGlobalVar *g = &ctx->mod->globals[i];
        if (g->name && ctx->opts.preserve_names) {
            set_id_name(ctx, g->id, g->name);
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "_g%u", g->id);
            set_id_name(ctx, g->id, buf);
        }
    }

    /* Assign names to functions and their contents */
    for (uint32_t i = 0; i < ctx->mod->function_count; i++) {
        SsirFunction *fn = &ctx->mod->functions[i];
        if (fn->name && ctx->opts.preserve_names) {
            set_id_name(ctx, fn->id, fn->name);
        }

        /* Parameters */
        for (uint32_t j = 0; j < fn->param_count; j++) {
            SsirFunctionParam *p = &fn->params[j];
            if (p->name && ctx->opts.preserve_names) {
                set_id_name(ctx, p->id, p->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_p%u", p->id);
                set_id_name(ctx, p->id, buf);
            }
        }

        /* Locals */
        for (uint32_t j = 0; j < fn->local_count; j++) {
            SsirLocalVar *l = &fn->locals[j];
            if (l->name && ctx->opts.preserve_names) {
                set_id_name(ctx, l->id, l->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_l%u", l->id);
                set_id_name(ctx, l->id, buf);
            }
        }
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

SsirToWgslResult ssir_to_wgsl(const SsirModule *mod,
                              const SsirToWgslOptions *opts,
                              char **out_wgsl,
                              char **out_error) {
    if (!mod || !out_wgsl) {
        if (out_error) *out_error = strdup("Invalid input: null module or output pointer");
        return SSIR_TO_WGSL_ERR_INVALID_INPUT;
    }

    SsirToWgslContext ctx;
    ctx_init(&ctx, mod, opts);

    if (!ctx.id_names) {
        if (out_error) *out_error = strdup("Out of memory");
        return SSIR_TO_WGSL_ERR_OOM;
    }

    /* Phase 1: Assign names */
    assign_names(&ctx);

    /* Phase 2: Emit structs */
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_STRUCT) {
            emit_struct(&ctx, &mod->types[i]);
        }
    }

    /* Phase 2b: Emit specialization constants (WGSL override) */
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        SsirConstant *k = &mod->constants[i];
        if (!k->is_specialization) continue;
        stw_sb_appendf(&ctx.sb, "@id(%u) override ", k->spec_id);
        const char *sname = k->name ? k->name : "spec_const";
        stw_sb_append(&ctx.sb, sname);
        stw_sb_append(&ctx.sb, ": ");
        emit_type(&ctx, k->type, &ctx.sb);
        stw_sb_append(&ctx.sb, " = ");
        stw_emit_constant(&ctx, k, &ctx.sb);
        stw_sb_append(&ctx.sb, ";\n");
    }

    /* Phase 3: Emit global variables */
    for (uint32_t i = 0; i < mod->global_count; i++) {
        emit_global(&ctx, &mod->globals[i]);
    }

    if (mod->global_count > 0) {
        stw_sb_newline(&ctx.sb);
    }

    /* Phase 4: Emit functions */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        stw_emit_function(&ctx, &mod->functions[i]);
    }

    /* Return result */
    *out_wgsl = ctx.sb.data ? strdup(ctx.sb.data) : strdup("");

    ctx_free(&ctx);
    return SSIR_TO_WGSL_OK;
}

void ssir_to_wgsl_free(void *p) {
    STW_FREE(p);
}

const char *ssir_to_wgsl_result_string(SsirToWgslResult result) {
    switch (result) {
    case SSIR_TO_WGSL_OK:               return "Success";
    case SSIR_TO_WGSL_ERR_INVALID_INPUT: return "Invalid input";
    case SSIR_TO_WGSL_ERR_UNSUPPORTED:   return "Unsupported feature";
    case SSIR_TO_WGSL_ERR_INTERNAL:      return "Internal error";
    case SSIR_TO_WGSL_ERR_OOM:           return "Out of memory";
    default:                             return "Unknown error";
    }
}
