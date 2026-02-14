/*
 * SSIR to SPIR-V Converter - Implementation
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef STS_MALLOC
#define STS_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STS_REALLOC
#define STS_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STS_FREE
#define STS_FREE(p) free((p))
#endif

/* ============================================================================
 * Word Buffer
 * ============================================================================ */

typedef struct {
    uint32_t *data;
    size_t len;
    size_t cap;
} StsWordBuf;

//wb nonnull
static void sts_wb_init(StsWordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_init: wb is NULL");
    wb->data = NULL;
    wb->len = 0;
    wb->cap = 0;
}

//wb nonnull
static void sts_wb_free(StsWordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_free: wb is NULL");
    STS_FREE(wb->data);
    wb->data = NULL;
    wb->len = wb->cap = 0;
}

//wb nonnull
static int sts_wb_reserve(StsWordBuf *wb, size_t need) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_reserve: wb is NULL");
    if (wb->len + need <= wb->cap) return 1;
    size_t ncap = wb->cap ? wb->cap : 64;
    while (ncap < wb->len + need) ncap *= 2;
    void *nd = STS_REALLOC(wb->data, ncap * sizeof(uint32_t));
    if (!nd) return 0;
    wb->data = (uint32_t *)nd;
    wb->cap = ncap;
    return 1;
}

//wb nonnull
static int wb_push(StsWordBuf *wb, uint32_t w) {
    wgsl_compiler_assert(wb != NULL, "wb_push: wb is NULL");
    if (!sts_wb_reserve(wb, 1)) return 0;
    wb->data[wb->len++] = w;
    return 1;
}

//wb nonnull, src nonnull
static int sts_wb_push_many(StsWordBuf *wb, const uint32_t *src, size_t n) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_push_many: wb is NULL");
    wgsl_compiler_assert(src != NULL, "sts_wb_push_many: src is NULL");
    if (!sts_wb_reserve(wb, n)) return 0;
    memcpy(wb->data + wb->len, src, n * sizeof(uint32_t));
    wb->len += n;
    return 1;
}

/* ============================================================================
 * SPIR-V Sections
 * ============================================================================ */

typedef struct {
    StsWordBuf capabilities;
    StsWordBuf extensions;
    StsWordBuf ext_inst_imports;
    StsWordBuf memory_model;
    StsWordBuf entry_points;
    StsWordBuf execution_modes;
    StsWordBuf debug_names;
    StsWordBuf annotations;
    StsWordBuf types_constants;
    StsWordBuf globals;
    StsWordBuf functions;
} StsSpvSections;

//s nonnull
static void sections_init(StsSpvSections *s) {
    wgsl_compiler_assert(s != NULL, "sections_init: s is NULL");
    sts_wb_init(&s->capabilities);
    sts_wb_init(&s->extensions);
    sts_wb_init(&s->ext_inst_imports);
    sts_wb_init(&s->memory_model);
    sts_wb_init(&s->entry_points);
    sts_wb_init(&s->execution_modes);
    sts_wb_init(&s->debug_names);
    sts_wb_init(&s->annotations);
    sts_wb_init(&s->types_constants);
    sts_wb_init(&s->globals);
    sts_wb_init(&s->functions);
}

//s nonnull
static void sections_free(StsSpvSections *s) {
    wgsl_compiler_assert(s != NULL, "sections_free: s is NULL");
    sts_wb_free(&s->capabilities);
    sts_wb_free(&s->extensions);
    sts_wb_free(&s->ext_inst_imports);
    sts_wb_free(&s->memory_model);
    sts_wb_free(&s->entry_points);
    sts_wb_free(&s->execution_modes);
    sts_wb_free(&s->debug_names);
    sts_wb_free(&s->annotations);
    sts_wb_free(&s->types_constants);
    sts_wb_free(&s->globals);
    sts_wb_free(&s->functions);
}

/* ============================================================================
 * Converter Context
 * ============================================================================ */

typedef struct {
    const SsirModule *mod;
    SsirToSpirvOptions opts;
    StsSpvSections sections;

    /* ID mapping: SSIR ID -> SPIR-V ID */
    uint32_t *id_map;
    uint32_t id_map_size;
    uint32_t next_spv_id;

    /* Type mapping: SSIR ID -> SSIR type ID */
    uint32_t *type_map;
    uint32_t type_map_size;

    /* Type IDs for common types (cached) */
    uint32_t spv_void;
    uint32_t spv_bool;
    uint32_t spv_i32;
    uint32_t spv_u32;
    uint32_t spv_f32;
    uint32_t spv_f16;

    /* GLSL.std.450 import ID */
    uint32_t glsl_ext_id;

    /* Capability tracking */
    int has_shader_cap;
    int has_float16_cap;
    int glsl_ext_emitted;

    /* Pre-computed function type IDs (indexed by function index) */
    uint32_t *func_type_cache;
    uint32_t func_type_cache_count;

    /* Deduplication cache for OpTypeFunction signatures */
    struct {
        uint32_t return_type;
        uint32_t param_count;
        uint32_t param_hash;   /* simple hash of param types for quick reject */
        uint32_t spv_id;
    } func_type_dedup[64];
    uint32_t func_type_dedup_count;
} Ctx;

//c nonnull
static uint32_t sts_fresh_id(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_fresh_id: c is NULL");
    return c->next_spv_id++;
}

//c nonnull
static uint32_t get_spv_id(Ctx *c, uint32_t ssir_id) {
    wgsl_compiler_assert(c != NULL, "get_spv_id: c is NULL");
    if (ssir_id < c->id_map_size && c->id_map[ssir_id] != 0) {
        return c->id_map[ssir_id];
    }
    /* Allocate new SPIR-V ID */
    uint32_t spv_id = sts_fresh_id(c);
    if (ssir_id < c->id_map_size) {
        c->id_map[ssir_id] = spv_id;
    }
    return spv_id;
}

//c nonnull
static void set_spv_id(Ctx *c, uint32_t ssir_id, uint32_t spv_id) {
    wgsl_compiler_assert(c != NULL, "set_spv_id: c is NULL");
    if (ssir_id < c->id_map_size) {
        c->id_map[ssir_id] = spv_id;
    }
}

//c nonnull
static void set_ssir_type(Ctx *c, uint32_t ssir_id, uint32_t ssir_type) {
    wgsl_compiler_assert(c != NULL, "set_ssir_type: c is NULL");
    if (ssir_id < c->type_map_size) {
        c->type_map[ssir_id] = ssir_type;
    }
}

//c nonnull
static uint32_t get_ssir_type(Ctx *c, uint32_t ssir_id) {
    wgsl_compiler_assert(c != NULL, "get_ssir_type: c is NULL");
    if (ssir_id < c->type_map_size) {
        return c->type_map[ssir_id];
    }
    return 0;
}

/* ============================================================================
 * String Literal Encoding
 * ============================================================================ */

//out_words nonnull, out_count nonnull
static uint32_t encode_string(const char *s, uint32_t **out_words, size_t *out_count) {
    wgsl_compiler_assert(out_words != NULL, "encode_string: out_words is NULL");
    wgsl_compiler_assert(out_count != NULL, "encode_string: out_count is NULL");
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    size_t words = (n + 3) / 4;
    uint32_t *buf = (uint32_t *)STS_MALLOC(words * sizeof(uint32_t));
    if (!buf) {
        *out_words = NULL;
        *out_count = 0;
        return 0;
    }
    memset(buf, 0, words * sizeof(uint32_t));
    memcpy(buf, s, n);
    *out_words = buf;
    *out_count = words;
    return (uint32_t)words;
}

/* ============================================================================
 * Instruction Emission Helpers
 * ============================================================================ */

//wb nonnull
static int sts_emit_op(StsWordBuf *wb, SpvOp op, size_t word_count) {
    wgsl_compiler_assert(wb != NULL, "sts_emit_op: wb is NULL");
    return wb_push(wb, ((uint32_t)word_count << 16) | (uint32_t)op);
}

//c nonnull
static int sts_emit_capability(Ctx *c, SpvCapability cap) {
    wgsl_compiler_assert(c != NULL, "sts_emit_capability: c is NULL");
    StsWordBuf *wb = &c->sections.capabilities;
    if (!sts_emit_op(wb, SpvOpCapability, 2)) return 0;
    return wb_push(wb, cap);
}

//c nonnull, name nonnull
static int sts_emit_extension(Ctx *c, const char *name) {
    wgsl_compiler_assert(c != NULL, "sts_emit_extension: c is NULL");
    wgsl_compiler_assert(name != NULL, "sts_emit_extension: name is NULL");
    StsWordBuf *wb = &c->sections.extensions;
    uint32_t *str; size_t wn;
    encode_string(name, &str, &wn);
    if (!sts_emit_op(wb, SpvOpExtension, 1 + wn)) { STS_FREE(str); return 0; }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    return ok;
}

//c nonnull, name nonnull, out_id nonnull
static int sts_emit_ext_inst_import(Ctx *c, const char *name, uint32_t *out_id) {
    wgsl_compiler_assert(c != NULL, "sts_emit_ext_inst_import: c is NULL");
    wgsl_compiler_assert(name != NULL, "sts_emit_ext_inst_import: name is NULL");
    wgsl_compiler_assert(out_id != NULL, "sts_emit_ext_inst_import: out_id is NULL");
    StsWordBuf *wb = &c->sections.ext_inst_imports;
    uint32_t *str; size_t wn;
    encode_string(name, &str, &wn);
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpExtInstImport, 2 + wn)) { STS_FREE(str); return 0; }
    if (!wb_push(wb, id)) { STS_FREE(str); return 0; }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    *out_id = id;
    return ok;
}

//c nonnull
static void sts_ensure_glsl_import(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_ensure_glsl_import: c is NULL");
    if (c->glsl_ext_emitted) return;
    c->glsl_ext_emitted = 1;
    if (!c->glsl_ext_id) c->glsl_ext_id = sts_fresh_id(c);
    StsWordBuf *wb = &c->sections.ext_inst_imports;
    uint32_t *str; size_t wn;
    encode_string("GLSL.std.450", &str, &wn);
    sts_emit_op(wb, SpvOpExtInstImport, 2 + wn);
    wb_push(wb, c->glsl_ext_id);
    sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
}

//c nonnull
static int sts_emit_memory_model(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_memory_model: c is NULL");
    StsWordBuf *wb = &c->sections.memory_model;
    if (!sts_emit_op(wb, SpvOpMemoryModel, 3)) return 0;
    if (!wb_push(wb, SpvAddressingModelLogical)) return 0;
    return wb_push(wb, SpvMemoryModelGLSL450);
}

//c nonnull
static int sts_emit_name(Ctx *c, uint32_t target, const char *name) {
    wgsl_compiler_assert(c != NULL, "sts_emit_name: c is NULL");
    if (!name || !*name) return 1;
    StsWordBuf *wb = &c->sections.debug_names;
    uint32_t *str; size_t wn;
    encode_string(name, &str, &wn);
    if (!sts_emit_op(wb, SpvOpName, 2 + wn)) { STS_FREE(str); return 0; }
    if (!wb_push(wb, target)) { STS_FREE(str); return 0; }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    return ok;
}

//c nonnull
static int sts_emit_member_name(Ctx *c, uint32_t struct_id, uint32_t member, const char *name) {
    wgsl_compiler_assert(c != NULL, "sts_emit_member_name: c is NULL");
    if (!name || !*name) return 1;
    StsWordBuf *wb = &c->sections.debug_names;
    uint32_t *str; size_t wn;
    encode_string(name, &str, &wn);
    if (!sts_emit_op(wb, SpvOpMemberName, 3 + wn)) { STS_FREE(str); return 0; }
    if (!wb_push(wb, struct_id)) { STS_FREE(str); return 0; }
    if (!wb_push(wb, member)) { STS_FREE(str); return 0; }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    return ok;
}

//c nonnull
static int sts_emit_decorate(Ctx *c, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_decorate: c is NULL");
    StsWordBuf *wb = &c->sections.annotations;
    if (!sts_emit_op(wb, SpvOpDecorate, 3 + lit_count)) return 0;
    if (!wb_push(wb, target)) return 0;
    if (!wb_push(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push(wb, literals[i])) return 0;
    }
    return 1;
}

//c nonnull
static int sts_emit_member_decorate(Ctx *c, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_member_decorate: c is NULL");
    StsWordBuf *wb = &c->sections.annotations;
    if (!sts_emit_op(wb, SpvOpMemberDecorate, 4 + lit_count)) return 0;
    if (!wb_push(wb, struct_id)) return 0;
    if (!wb_push(wb, member)) return 0;
    if (!wb_push(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push(wb, literals[i])) return 0;
    }
    return 1;
}

/* ============================================================================
 * Type Emission
 * ============================================================================ */

static uint32_t sts_emit_type(Ctx *c, uint32_t ssir_type_id);

/* Compute the size and alignment of a type (for std430 layout).
 * Returns the size in bytes, and stores alignment in *out_align. */
//c nonnull
static uint32_t compute_type_size(Ctx *c, uint32_t ssir_type_id, uint32_t *out_align) {
    wgsl_compiler_assert(c != NULL, "compute_type_size: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type_id);
    if (!t) {
        if (out_align) *out_align = 4;
        return 4;
    }
    switch (t->kind) {
        case SSIR_TYPE_BOOL:
        case SSIR_TYPE_I32:
        case SSIR_TYPE_U32:
        case SSIR_TYPE_F32:
            if (out_align) *out_align = 4;
            return 4;
        case SSIR_TYPE_F16:
            if (out_align) *out_align = 2;
            return 2;
        case SSIR_TYPE_VEC: {
            uint32_t elem_size = compute_type_size(c, t->vec.elem, NULL);
            uint32_t size = t->vec.size;
            /* vec2 aligns to 2N, vec3/vec4 align to 4N */
            uint32_t align_factor = (size == 2) ? 2 : 4;
            if (out_align) *out_align = elem_size * align_factor;
            return elem_size * size;
        }
        case SSIR_TYPE_MAT: {
            /* Matrix is array of columns (vectors) */
            uint32_t col_align;
            uint32_t col_size = compute_type_size(c, t->mat.elem, &col_align);
            /* Round up column size to alignment */
            uint32_t stride = (col_size + col_align - 1) & ~(col_align - 1);
            if (out_align) *out_align = col_align;
            return stride * t->mat.cols;
        }
        case SSIR_TYPE_ARRAY: {
            uint32_t elem_align;
            uint32_t elem_size = compute_type_size(c, t->array.elem, &elem_align);
            /* Stride is at least element size rounded up to element alignment */
            uint32_t stride = (elem_size + elem_align - 1) & ~(elem_align - 1);
            if (out_align) *out_align = elem_align;
            return stride * t->array.length;
        }
        case SSIR_TYPE_RUNTIME_ARRAY: {
            uint32_t elem_align;
            compute_type_size(c, t->runtime_array.elem, &elem_align);
            if (out_align) *out_align = elem_align;
            return 0; /* Runtime arrays have unknown size */
        }
        case SSIR_TYPE_STRUCT: {
            /* Compute struct size based on members */
            uint32_t max_align = 1;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                uint32_t mem_align;
                uint32_t mem_size = compute_type_size(c, t->struc.members[i], &mem_align);
                if (mem_align > max_align) max_align = mem_align;
                offset = (offset + mem_align - 1) & ~(mem_align - 1);
                offset += mem_size;
            }
            /* Final size is offset rounded up to struct alignment */
            offset = (offset + max_align - 1) & ~(max_align - 1);
            if (out_align) *out_align = max_align;
            return offset;
        }
        default:
            if (out_align) *out_align = 4;
            return 4;
    }
}

/* Compute the array stride for a given element type */
//c nonnull
static uint32_t compute_array_stride(Ctx *c, uint32_t elem_type_id) {
    wgsl_compiler_assert(c != NULL, "compute_array_stride: c is NULL");
    uint32_t elem_align;
    uint32_t elem_size = compute_type_size(c, elem_type_id, &elem_align);
    /* Stride is element size rounded up to element alignment */
    return (elem_size + elem_align - 1) & ~(elem_align - 1);
}

//c nonnull
static uint32_t sts_emit_type_void(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_void: c is NULL");
    if (c->spv_void) return c->spv_void;
    StsWordBuf *wb = &c->sections.types_constants;
    c->spv_void = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeVoid, 2)) return 0;
    if (!wb_push(wb, c->spv_void)) return 0;
    return c->spv_void;
}

//c nonnull
static uint32_t sts_emit_type_bool(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_bool: c is NULL");
    if (c->spv_bool) return c->spv_bool;
    StsWordBuf *wb = &c->sections.types_constants;
    c->spv_bool = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeBool, 2)) return 0;
    if (!wb_push(wb, c->spv_bool)) return 0;
    return c->spv_bool;
}

//c nonnull
static uint32_t sts_emit_type_int(Ctx *c, uint32_t width, uint32_t signedness) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_int: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeInt, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, width)) return 0;
    if (!wb_push(wb, signedness)) return 0;
    if (width == 32 && signedness == 1) c->spv_i32 = id;
    if (width == 32 && signedness == 0) c->spv_u32 = id;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_float(Ctx *c, uint32_t width) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_float: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeFloat, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, width)) return 0;
    if (width == 32) c->spv_f32 = id;
    if (width == 16) c->spv_f16 = id;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_vector(Ctx *c, uint32_t elem_type, uint32_t count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_vector: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeVector, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    if (!wb_push(wb, count)) return 0;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_matrix(Ctx *c, uint32_t col_type, uint32_t col_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_matrix: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeMatrix, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, col_type)) return 0;
    if (!wb_push(wb, col_count)) return 0;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_array(Ctx *c, uint32_t elem_type, uint32_t length_id) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_array: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeArray, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    if (!wb_push(wb, length_id)) return 0;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_runtime_array(Ctx *c, uint32_t elem_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_runtime_array: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeRuntimeArray, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    return id;
}

//c nonnull, member_types nonnull
static uint32_t sts_emit_type_struct(Ctx *c, uint32_t *member_types, uint32_t member_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_struct: c is NULL");
    wgsl_compiler_assert(member_types != NULL, "sts_emit_type_struct: member_types is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeStruct, 2 + member_count)) return 0;
    if (!wb_push(wb, id)) return 0;
    for (uint32_t i = 0; i < member_count; ++i) {
        if (!wb_push(wb, member_types[i])) return 0;
    }
    return id;
}

static SpvStorageClass addr_space_to_storage_class(SsirAddressSpace space) {
    switch (space) {
        case SSIR_ADDR_FUNCTION:        return SpvStorageClassFunction;
        case SSIR_ADDR_PRIVATE:         return SpvStorageClassPrivate;
        case SSIR_ADDR_WORKGROUP:       return SpvStorageClassWorkgroup;
        case SSIR_ADDR_UNIFORM:         return SpvStorageClassUniform;
        case SSIR_ADDR_UNIFORM_CONSTANT: return SpvStorageClassUniformConstant;
        case SSIR_ADDR_STORAGE:         return SpvStorageClassStorageBuffer;
        case SSIR_ADDR_INPUT:           return SpvStorageClassInput;
        case SSIR_ADDR_OUTPUT:          return SpvStorageClassOutput;
        case SSIR_ADDR_PUSH_CONSTANT:   return SpvStorageClassPushConstant;
        case SSIR_ADDR_PHYSICAL_STORAGE_BUFFER: return SpvStorageClassPhysicalStorageBuffer;
        default:                        return SpvStorageClassFunction;
    }
}

//c nonnull
static uint32_t sts_emit_type_pointer(Ctx *c, SpvStorageClass sc, uint32_t pointee) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_pointer: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypePointer, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, sc)) return 0;
    if (!wb_push(wb, pointee)) return 0;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_function(Ctx *c, uint32_t return_type, uint32_t *param_types, uint32_t param_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_function: c is NULL");

    /* Compute a simple hash of param types for quick comparison */
    uint32_t param_hash = 0;
    for (uint32_t i = 0; i < param_count; ++i)
        param_hash = param_hash * 31 + param_types[i];

    /* Check dedup cache for matching signature */
    for (uint32_t d = 0; d < c->func_type_dedup_count; ++d) {
        if (c->func_type_dedup[d].return_type == return_type &&
            c->func_type_dedup[d].param_count == param_count &&
            c->func_type_dedup[d].param_hash == param_hash) {
            return c->func_type_dedup[d].spv_id;
        }
    }

    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeFunction, 3 + param_count)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, return_type)) return 0;
    for (uint32_t i = 0; i < param_count; ++i) {
        if (!wb_push(wb, param_types[i])) return 0;
    }

    /* Store in dedup cache */
    if (c->func_type_dedup_count < 64) {
        c->func_type_dedup[c->func_type_dedup_count].return_type = return_type;
        c->func_type_dedup[c->func_type_dedup_count].param_count = param_count;
        c->func_type_dedup[c->func_type_dedup_count].param_hash = param_hash;
        c->func_type_dedup[c->func_type_dedup_count].spv_id = id;
        c->func_type_dedup_count++;
    }

    return id;
}

//c nonnull
static uint32_t sts_emit_type_sampler(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_sampler: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeSampler, 2)) return 0;
    if (!wb_push(wb, id)) return 0;
    return id;
}

static SpvDim texture_dim_to_spv(SsirTextureDim dim) {
    switch (dim) {
        case SSIR_TEX_1D:             return SpvDim1D;
        case SSIR_TEX_2D:             return SpvDim2D;
        case SSIR_TEX_3D:             return SpvDim3D;
        case SSIR_TEX_CUBE:           return SpvDimCube;
        case SSIR_TEX_2D_ARRAY:       return SpvDim2D;
        case SSIR_TEX_CUBE_ARRAY:     return SpvDimCube;
        case SSIR_TEX_MULTISAMPLED_2D: return SpvDim2D;
        case SSIR_TEX_1D_ARRAY:       return SpvDim1D;
        case SSIR_TEX_BUFFER:         return SpvDimBuffer;
        case SSIR_TEX_MULTISAMPLED_2D_ARRAY: return SpvDim2D;
        default:                       return SpvDim2D;
    }
}

//c nonnull
static uint32_t sts_emit_type_image(Ctx *c, SsirTextureDim dim, uint32_t sampled_type,
                                uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, SpvImageFormat format) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_image: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeImage, 9)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, sampled_type)) return 0;
    if (!wb_push(wb, texture_dim_to_spv(dim))) return 0;
    if (!wb_push(wb, depth)) return 0;
    if (!wb_push(wb, arrayed)) return 0;
    if (!wb_push(wb, ms)) return 0;
    if (!wb_push(wb, sampled)) return 0;
    if (!wb_push(wb, format)) return 0;
    return id;
}

//c nonnull
static uint32_t sts_emit_type_sampled_image(Ctx *c, uint32_t image_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_sampled_image: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeSampledImage, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, image_type)) return 0;
    return id;
}

/* Emit SSIR type, returns SPIR-V type ID */
//c nonnull
static uint32_t sts_emit_type(Ctx *c, uint32_t ssir_type_id) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type_id);
    if (!t) return 0;

    /* Check if already emitted */
    uint32_t cached = 0;
    if (ssir_type_id < c->id_map_size) {
        cached = c->id_map[ssir_type_id];
        if (cached != 0) return cached;
    }

    uint32_t spv_id = 0;
    switch (t->kind) {
        case SSIR_TYPE_VOID:
            spv_id = sts_emit_type_void(c);
            break;
        case SSIR_TYPE_BOOL:
            spv_id = sts_emit_type_bool(c);
            break;
        case SSIR_TYPE_I32:
            spv_id = sts_emit_type_int(c, 32, 1);
            break;
        case SSIR_TYPE_U32:
            spv_id = sts_emit_type_int(c, 32, 0);
            break;
        case SSIR_TYPE_F32:
            spv_id = sts_emit_type_float(c, 32);
            break;
        case SSIR_TYPE_F16:
            spv_id = sts_emit_type_float(c, 16);
            if (!c->has_float16_cap) {
                sts_emit_capability(c, SpvCapabilityFloat16);
                c->has_float16_cap = 1;
            }
            break;
        case SSIR_TYPE_F64:
            spv_id = sts_emit_type_float(c, 64);
            sts_emit_capability(c, SpvCapabilityFloat64);
            break;
        case SSIR_TYPE_I8:
            spv_id = sts_emit_type_int(c, 8, 1);
            sts_emit_capability(c, SpvCapabilityInt8);
            break;
        case SSIR_TYPE_U8:
            spv_id = sts_emit_type_int(c, 8, 0);
            sts_emit_capability(c, SpvCapabilityInt8);
            break;
        case SSIR_TYPE_I16:
            spv_id = sts_emit_type_int(c, 16, 1);
            sts_emit_capability(c, SpvCapabilityInt16);
            break;
        case SSIR_TYPE_U16:
            spv_id = sts_emit_type_int(c, 16, 0);
            sts_emit_capability(c, SpvCapabilityInt16);
            break;
        case SSIR_TYPE_I64:
            spv_id = sts_emit_type_int(c, 64, 1);
            sts_emit_capability(c, SpvCapabilityInt64);
            break;
        case SSIR_TYPE_U64:
            spv_id = sts_emit_type_int(c, 64, 0);
            sts_emit_capability(c, SpvCapabilityInt64);
            break;
        case SSIR_TYPE_VEC: {
            uint32_t elem_spv = sts_emit_type(c, t->vec.elem);
            spv_id = sts_emit_type_vector(c, elem_spv, t->vec.size);
            break;
        }
        case SSIR_TYPE_MAT: {
            /* Matrix element is column type (vector) */
            uint32_t col_spv = sts_emit_type(c, t->mat.elem);
            spv_id = sts_emit_type_matrix(c, col_spv, t->mat.cols);
            break;
        }
        case SSIR_TYPE_ARRAY: {
            uint32_t elem_spv = sts_emit_type(c, t->array.elem);
            /* Emit length as constant */
            uint32_t len_id = sts_fresh_id(c);
            StsWordBuf *wb = &c->sections.types_constants;
            uint32_t u32_type = c->spv_u32 ? c->spv_u32 : sts_emit_type_int(c, 32, 0);
            sts_emit_op(wb, SpvOpConstant, 4);
            wb_push(wb, u32_type);
            wb_push(wb, len_id);
            wb_push(wb, t->array.length);
            spv_id = sts_emit_type_array(c, elem_spv, len_id);
            /* Decorate array stride */
            uint32_t stride = t->array.stride ? t->array.stride : compute_array_stride(c, t->array.elem);
            sts_emit_decorate(c, spv_id, SpvDecorationArrayStride, &stride, 1);
            break;
        }
        case SSIR_TYPE_RUNTIME_ARRAY: {
            uint32_t elem_spv = sts_emit_type(c, t->runtime_array.elem);
            spv_id = sts_emit_type_runtime_array(c, elem_spv);
            /* Decorate runtime array stride */
            uint32_t stride = compute_array_stride(c, t->runtime_array.elem);
            sts_emit_decorate(c, spv_id, SpvDecorationArrayStride, &stride, 1);
            break;
        }
        case SSIR_TYPE_STRUCT: {
            uint32_t *member_spv = (uint32_t *)STS_MALLOC(t->struc.member_count * sizeof(uint32_t));
            if (!member_spv) return 0;
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                member_spv[i] = sts_emit_type(c, t->struc.members[i]);
            }
            spv_id = sts_emit_type_struct(c, member_spv, t->struc.member_count);
            STS_FREE(member_spv);
            /* Emit debug names and member decorations */
            if (t->struc.name) {
                sts_emit_name(c, spv_id, t->struc.name);
            }
            if (t->struc.member_names) {
                for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                    if (t->struc.member_names[i])
                        sts_emit_member_name(c, spv_id, i, t->struc.member_names[i]);
                }
            }
            if (t->struc.offsets) {
                for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                    sts_emit_member_decorate(c, spv_id, i, SpvDecorationOffset, &t->struc.offsets[i], 1);
                }
            }
            /* Emit ColMajor/RowMajor + MatrixStride for matrix members */
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                SsirType *mt = ssir_get_type((SsirModule *)c->mod, t->struc.members[i]);
                if (mt && mt->kind == SSIR_TYPE_MAT) {
                    uint8_t major = (t->struc.matrix_major && t->struc.matrix_major[i]) ? t->struc.matrix_major[i] : 1;
                    SpvDecoration maj_dec = (major == 2) ? SpvDecorationRowMajor : SpvDecorationColMajor;
                    sts_emit_member_decorate(c, spv_id, i, maj_dec, NULL, 0);
                    uint32_t stride;
                    if (t->struc.matrix_strides && t->struc.matrix_strides[i]) {
                        stride = t->struc.matrix_strides[i];
                    } else {
                        uint32_t comp_size = 4;
                        SsirType *elem_t = ssir_get_type((SsirModule *)c->mod, mt->mat.elem);
                        if (elem_t && elem_t->kind == SSIR_TYPE_F16) comp_size = 2;
                        stride = mt->mat.rows * comp_size;
                        if (stride < 16) stride = 16;
                    }
                    sts_emit_member_decorate(c, spv_id, i, SpvDecorationMatrixStride, &stride, 1);
                }
            }
            break;
        }
        case SSIR_TYPE_PTR: {
            uint32_t pointee_spv = sts_emit_type(c, t->ptr.pointee);
            SpvStorageClass sc = addr_space_to_storage_class(t->ptr.space);
            spv_id = sts_emit_type_pointer(c, sc, pointee_spv);
            break;
        }
        case SSIR_TYPE_SAMPLER:
            spv_id = sts_emit_type_sampler(c);
            break;
        case SSIR_TYPE_SAMPLER_COMPARISON:
            spv_id = sts_emit_type_sampler(c);
            break;
        case SSIR_TYPE_TEXTURE: {
            uint32_t sampled_spv = sts_emit_type(c, t->texture.sampled_type);
            SsirTextureDim dim = t->texture.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_CUBE_ARRAY ||
                                dim == SSIR_TEX_1D_ARRAY || dim == SSIR_TEX_MULTISAMPLED_2D_ARRAY) ? 1 : 0;
            uint32_t ms = (dim == SSIR_TEX_MULTISAMPLED_2D || dim == SSIR_TEX_MULTISAMPLED_2D_ARRAY) ? 1 : 0;
            spv_id = sts_emit_type_image(c, dim, sampled_spv, 0, arrayed, ms, 1, SpvImageFormatUnknown);
            break;
        }
        case SSIR_TYPE_TEXTURE_STORAGE: {
            uint32_t f32_type = c->spv_f32 ? c->spv_f32 : sts_emit_type_float(c, 32);
            SsirTextureDim dim = t->texture_storage.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_1D_ARRAY) ? 1 : 0;
            uint32_t sampled = 2; /* storage image */
            spv_id = sts_emit_type_image(c, dim, f32_type, 0, arrayed, 0, sampled, (SpvImageFormat)t->texture_storage.format);
            break;
        }
        case SSIR_TYPE_TEXTURE_DEPTH: {
            uint32_t f32_type = c->spv_f32 ? c->spv_f32 : sts_emit_type_float(c, 32);
            SsirTextureDim dim = t->texture_depth.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_CUBE_ARRAY || dim == SSIR_TEX_1D_ARRAY) ? 1 : 0;
            spv_id = sts_emit_type_image(c, dim, f32_type, 1, arrayed, 0, 1, SpvImageFormatUnknown);
            break;
        }
    }

    /* Cache */
    if (spv_id && ssir_type_id < c->id_map_size) {
        c->id_map[ssir_type_id] = spv_id;
    }

    return spv_id;
}

/* ============================================================================
 * Constant Emission
 * ============================================================================ */

//c nonnull
static uint32_t sts_emit_const_u32(Ctx *c, uint32_t value) {
    wgsl_compiler_assert(c != NULL, "sts_emit_const_u32: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t u32_type = c->spv_u32 ? c->spv_u32 : sts_emit_type_int(c, 32, 0);
    uint32_t id = sts_fresh_id(c);
    sts_emit_op(wb, SpvOpConstant, 4);
    wb_push(wb, u32_type);
    wb_push(wb, id);
    wb_push(wb, value);
    return id;
}

//c nonnull, cnst nonnull
static uint32_t sts_emit_constant(Ctx *c, const SsirConstant *cnst) {
    wgsl_compiler_assert(c != NULL, "sts_emit_constant: c is NULL");
    wgsl_compiler_assert(cnst != NULL, "sts_emit_constant: cnst is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t type_spv = sts_emit_type(c, cnst->type);
    uint32_t id = get_spv_id(c, cnst->id);

    bool spec = cnst->is_specialization;
    switch (cnst->kind) {
        case SSIR_CONST_BOOL:
            if (cnst->bool_val) {
                sts_emit_op(wb, spec ? SpvOpSpecConstantTrue : SpvOpConstantTrue, 3);
            } else {
                sts_emit_op(wb, spec ? SpvOpSpecConstantFalse : SpvOpConstantFalse, 3);
            }
            wb_push(wb, type_spv);
            wb_push(wb, id);
            break;
        case SSIR_CONST_I32:
        case SSIR_CONST_U32:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, cnst->u32_val);
            break;
        case SSIR_CONST_F32: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t bits;
            memcpy(&bits, &cnst->f32_val, sizeof(float));
            wb_push(wb, bits);
            break;
        }
        case SSIR_CONST_F16:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, cnst->f16_val);
            break;
        case SSIR_CONST_F64: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 5);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t dw[2];
            memcpy(dw, &cnst->f64_val, sizeof(double));
            wb_push(wb, dw[0]);
            wb_push(wb, dw[1]);
            break;
        }
        case SSIR_CONST_I8:
        case SSIR_CONST_U8:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, (uint32_t)cnst->u8_val);
            break;
        case SSIR_CONST_I16:
        case SSIR_CONST_U16:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, (uint32_t)cnst->u16_val);
            break;
        case SSIR_CONST_I64:
        case SSIR_CONST_U64: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 5);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t qw[2];
            memcpy(qw, &cnst->u64_val, sizeof(uint64_t));
            wb_push(wb, qw[0]);
            wb_push(wb, qw[1]);
            break;
        }
        case SSIR_CONST_COMPOSITE: {
            sts_emit_op(wb, spec ? SpvOpSpecConstantComposite : SpvOpConstantComposite, 3 + cnst->composite.count);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            for (uint32_t i = 0; i < cnst->composite.count; ++i) {
                wb_push(wb, get_spv_id(c, cnst->composite.components[i]));
            }
            break;
        }
        case SSIR_CONST_NULL:
            sts_emit_op(wb, SpvOpConstantNull, 3);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            break;
    }

    /* Emit SpecId decoration for specialization constants */
    if (spec) {
        sts_emit_decorate(c, id, SpvDecorationSpecId, &cnst->spec_id, 1);
    }

    /* Emit debug name for named constants */
    sts_emit_name(c, id, cnst->name);

    /* Record type for this constant */
    set_ssir_type(c, cnst->id, cnst->type);

    return id;
}

/* ============================================================================
 * Global Variable Emission
 * ============================================================================ */

static SpvBuiltIn builtin_var_to_spv(SsirBuiltinVar b) {
    switch (b) {
        case SSIR_BUILTIN_VERTEX_INDEX:          return SpvBuiltInVertexIndex;
        case SSIR_BUILTIN_INSTANCE_INDEX:        return SpvBuiltInInstanceIndex;
        case SSIR_BUILTIN_POSITION:              return SpvBuiltInPosition;
        case SSIR_BUILTIN_FRONT_FACING:          return SpvBuiltInFrontFacing;
        case SSIR_BUILTIN_FRAG_DEPTH:            return SpvBuiltInFragDepth;
        case SSIR_BUILTIN_SAMPLE_INDEX:          return SpvBuiltInSampleId;
        case SSIR_BUILTIN_SAMPLE_MASK:           return SpvBuiltInSampleMask;
        case SSIR_BUILTIN_LOCAL_INVOCATION_ID:   return SpvBuiltInLocalInvocationId;
        case SSIR_BUILTIN_LOCAL_INVOCATION_INDEX: return SpvBuiltInLocalInvocationIndex;
        case SSIR_BUILTIN_GLOBAL_INVOCATION_ID:  return SpvBuiltInGlobalInvocationId;
        case SSIR_BUILTIN_WORKGROUP_ID:          return SpvBuiltInWorkgroupId;
        case SSIR_BUILTIN_NUM_WORKGROUPS:        return SpvBuiltInNumWorkgroups;
        case SSIR_BUILTIN_POINT_SIZE:            return SpvBuiltInPointSize;
        case SSIR_BUILTIN_CLIP_DISTANCE:         return SpvBuiltInClipDistance;
        case SSIR_BUILTIN_CULL_DISTANCE:         return SpvBuiltInCullDistance;
        case SSIR_BUILTIN_LAYER:                 return SpvBuiltInLayer;
        case SSIR_BUILTIN_VIEWPORT_INDEX:        return SpvBuiltInViewportIndex;
        case SSIR_BUILTIN_FRAG_COORD:            return SpvBuiltInFragCoord;
        case SSIR_BUILTIN_HELPER_INVOCATION:     return SpvBuiltInHelperInvocation;
        case SSIR_BUILTIN_PRIMITIVE_ID:          return SpvBuiltInPrimitiveId;
        case SSIR_BUILTIN_BASE_VERTEX:           return SpvBuiltInBaseVertex;
        case SSIR_BUILTIN_BASE_INSTANCE:         return SpvBuiltInBaseInstance;
        case SSIR_BUILTIN_SUBGROUP_SIZE:         return SpvBuiltInSubgroupSize;
        case SSIR_BUILTIN_SUBGROUP_INVOCATION_ID: return SpvBuiltInSubgroupLocalInvocationId;
        case SSIR_BUILTIN_SUBGROUP_ID:           return SpvBuiltInSubgroupId;
        case SSIR_BUILTIN_NUM_SUBGROUPS:         return SpvBuiltInNumSubgroups;
        default:                                  return SpvBuiltInMax;
    }
}

//c nonnull, g nonnull
static uint32_t sts_emit_global_var(Ctx *c, const SsirGlobalVar *g) {
    wgsl_compiler_assert(c != NULL, "sts_emit_global_var: c is NULL");
    wgsl_compiler_assert(g != NULL, "sts_emit_global_var: g is NULL");
    uint32_t type_spv = sts_emit_type(c, g->type);
    uint32_t id = get_spv_id(c, g->id);

    /* Get storage class from pointer type */
    SsirType *ptr_type = ssir_get_type((SsirModule *)c->mod, g->type);
    SpvStorageClass sc = SpvStorageClassPrivate;
    if (ptr_type && ptr_type->kind == SSIR_TYPE_PTR) {
        sc = addr_space_to_storage_class(ptr_type->ptr.space);
    }

    /* Emit OpVariable */
    StsWordBuf *wb = &c->sections.globals;
    sts_emit_op(wb, SpvOpVariable, g->has_initializer ? 5 : 4);
    wb_push(wb, type_spv);
    wb_push(wb, id);
    wb_push(wb, sc);
    if (g->has_initializer) {
        wb_push(wb, get_spv_id(c, g->initializer));
    }

    /* Emit debug name */
    sts_emit_name(c, id, g->name);

    /* Emit decorations */
    if (g->has_group) {
        sts_emit_decorate(c, id, SpvDecorationDescriptorSet, &g->group, 1);
    }
    if (g->has_binding) {
        sts_emit_decorate(c, id, SpvDecorationBinding, &g->binding, 1);
    }
    if (g->has_location) {
        sts_emit_decorate(c, id, SpvDecorationLocation, &g->location, 1);
    }
    if (g->builtin != SSIR_BUILTIN_NONE) {
        uint32_t b = builtin_var_to_spv(g->builtin);
        sts_emit_decorate(c, id, SpvDecorationBuiltIn, &b, 1);
    }
    if (g->interp == SSIR_INTERP_FLAT) {
        sts_emit_decorate(c, id, SpvDecorationFlat, NULL, 0);
    } else if (g->interp == SSIR_INTERP_LINEAR) {
        sts_emit_decorate(c, id, SpvDecorationNoPerspective, NULL, 0);
    }
    if (g->interp_sampling == SSIR_INTERP_SAMPLING_CENTROID) {
        sts_emit_decorate(c, id, SpvDecorationCentroid, NULL, 0);
    } else if (g->interp_sampling == SSIR_INTERP_SAMPLING_SAMPLE) {
        sts_emit_decorate(c, id, SpvDecorationSample, NULL, 0);
    }
    if (g->non_writable) {
        sts_emit_decorate(c, id, SpvDecorationNonWritable, NULL, 0);
    }
    if (g->invariant) {
        sts_emit_decorate(c, id, SpvDecorationInvariant, NULL, 0);
    }

    /* Block decoration for uniform/storage buffers */
    if (sc == SpvStorageClassUniform || sc == SpvStorageClassStorageBuffer) {
        if (ptr_type && ptr_type->kind == SSIR_TYPE_PTR) {
            SsirType *pointee = ssir_get_type((SsirModule *)c->mod, ptr_type->ptr.pointee);
            if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
                uint32_t struct_spv = sts_emit_type(c, ptr_type->ptr.pointee);
                sts_emit_decorate(c, struct_spv, SpvDecorationBlock, NULL, 0);
            }
        }
    }

    /* Record type for this global */
    set_ssir_type(c, g->id, g->type);

    return id;
}

/* ============================================================================
 * Instruction Emission
 * ============================================================================ */

/* Type helper to check if SSIR type is float-based */
//c nonnull
static int is_float_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_float_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_float_ssir_type(c, t->vec.elem);
    }
    if (t->kind == SSIR_TYPE_MAT) {
        SsirType *col = ssir_get_type((SsirModule *)c->mod, t->mat.elem);
        if (col && col->kind == SSIR_TYPE_VEC) {
            return is_float_ssir_type(c, col->vec.elem);
        }
    }
    return 0;
}

//c nonnull
static int is_signed_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_signed_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_I32) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_signed_ssir_type(c, t->vec.elem);
    }
    return 0;
}

//c nonnull
static int is_unsigned_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_unsigned_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_U32) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_unsigned_ssir_type(c, t->vec.elem);
    }
    return 0;
}

//c nonnull
static int is_bool_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_bool_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_BOOL) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_bool_ssir_type(c, t->vec.elem);
    }
    return 0;
}

//c nonnull
static int is_matrix_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_matrix_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    return t && t->kind == SSIR_TYPE_MAT;
}

//c nonnull
static int is_vector_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_vector_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    return t && t->kind == SSIR_TYPE_VEC;
}

//c nonnull
static int is_scalar_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_scalar_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    return t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16 ||
           t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_U32 ||
           t->kind == SSIR_TYPE_BOOL;
}

/* Map SSIR builtin to GLSL.std.450 instruction */
static int builtin_to_glsl_op(SsirBuiltinId id) {
    switch (id) {
        case SSIR_BUILTIN_SIN: return GLSLstd450Sin;
        case SSIR_BUILTIN_COS: return GLSLstd450Cos;
        case SSIR_BUILTIN_TAN: return GLSLstd450Tan;
        case SSIR_BUILTIN_ASIN: return GLSLstd450Asin;
        case SSIR_BUILTIN_ACOS: return GLSLstd450Acos;
        case SSIR_BUILTIN_ATAN: return GLSLstd450Atan;
        case SSIR_BUILTIN_ATAN2: return GLSLstd450Atan2;
        case SSIR_BUILTIN_SINH: return GLSLstd450Sinh;
        case SSIR_BUILTIN_COSH: return GLSLstd450Cosh;
        case SSIR_BUILTIN_TANH: return GLSLstd450Tanh;
        case SSIR_BUILTIN_ASINH: return GLSLstd450Asinh;
        case SSIR_BUILTIN_ACOSH: return GLSLstd450Acosh;
        case SSIR_BUILTIN_ATANH: return GLSLstd450Atanh;
        case SSIR_BUILTIN_EXP: return GLSLstd450Exp;
        case SSIR_BUILTIN_EXP2: return GLSLstd450Exp2;
        case SSIR_BUILTIN_LOG: return GLSLstd450Log;
        case SSIR_BUILTIN_LOG2: return GLSLstd450Log2;
        case SSIR_BUILTIN_POW: return GLSLstd450Pow;
        case SSIR_BUILTIN_SQRT: return GLSLstd450Sqrt;
        case SSIR_BUILTIN_INVERSESQRT: return GLSLstd450InverseSqrt;
        case SSIR_BUILTIN_FLOOR: return GLSLstd450Floor;
        case SSIR_BUILTIN_CEIL: return GLSLstd450Ceil;
        case SSIR_BUILTIN_ROUND: return GLSLstd450Round;
        case SSIR_BUILTIN_TRUNC: return GLSLstd450Trunc;
        case SSIR_BUILTIN_FRACT: return GLSLstd450Fract;
        case SSIR_BUILTIN_SIGN: return GLSLstd450FSign;
        case SSIR_BUILTIN_LENGTH: return GLSLstd450Length;
        case SSIR_BUILTIN_DISTANCE: return GLSLstd450Distance;
        case SSIR_BUILTIN_NORMALIZE: return GLSLstd450Normalize;
        case SSIR_BUILTIN_FACEFORWARD: return GLSLstd450FaceForward;
        case SSIR_BUILTIN_REFLECT: return GLSLstd450Reflect;
        case SSIR_BUILTIN_REFRACT: return GLSLstd450Refract;
        case SSIR_BUILTIN_CROSS: return GLSLstd450Cross;
        case SSIR_BUILTIN_MIX: return GLSLstd450FMix;
        case SSIR_BUILTIN_STEP: return GLSLstd450Step;
        case SSIR_BUILTIN_SMOOTHSTEP: return GLSLstd450SmoothStep;
        case SSIR_BUILTIN_FMA: return GLSLstd450Fma;
        case SSIR_BUILTIN_DEGREES: return GLSLstd450Degrees;
        case SSIR_BUILTIN_RADIANS: return GLSLstd450Radians;
        case SSIR_BUILTIN_MODF: return GLSLstd450Modf;
        case SSIR_BUILTIN_FREXP: return GLSLstd450Frexp;
        case SSIR_BUILTIN_LDEXP: return GLSLstd450Ldexp;
        case SSIR_BUILTIN_DETERMINANT: return GLSLstd450Determinant;
        case SSIR_BUILTIN_PACK4X8SNORM: return GLSLstd450PackSnorm4x8;
        case SSIR_BUILTIN_PACK4X8UNORM: return GLSLstd450PackUnorm4x8;
        case SSIR_BUILTIN_PACK2X16SNORM: return GLSLstd450PackSnorm2x16;
        case SSIR_BUILTIN_PACK2X16UNORM: return GLSLstd450PackUnorm2x16;
        case SSIR_BUILTIN_PACK2X16FLOAT: return GLSLstd450PackHalf2x16;
        case SSIR_BUILTIN_UNPACK4X8SNORM: return GLSLstd450UnpackSnorm4x8;
        case SSIR_BUILTIN_UNPACK4X8UNORM: return GLSLstd450UnpackUnorm4x8;
        case SSIR_BUILTIN_UNPACK2X16SNORM: return GLSLstd450UnpackSnorm2x16;
        case SSIR_BUILTIN_UNPACK2X16UNORM: return GLSLstd450UnpackUnorm2x16;
        case SSIR_BUILTIN_UNPACK2X16FLOAT: return GLSLstd450UnpackHalf2x16;
        default: return -1;
    }
}

//c nonnull
static uint32_t get_unsigned_ssir_type(Ctx *c, uint32_t ssir_type) {
    wgsl_compiler_assert(c != NULL, "get_unsigned_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_I32) return ssir_type_u32((SsirModule *)c->mod);
    if (t->kind == SSIR_TYPE_VEC) {
        uint32_t unsigned_elem = get_unsigned_ssir_type(c, t->vec.elem);
        if (unsigned_elem) return ssir_type_vec((SsirModule *)c->mod, unsigned_elem, t->vec.size);
    }
    return 0;
}

//c nonnull
static void emit_signed_int_binop(Ctx *c, SpvOp spv_op,
                                   uint32_t ssir_result, uint32_t type_spv,
                                   uint32_t ssir_type, uint32_t op0, uint32_t op1) {
    wgsl_compiler_assert(c != NULL, "emit_signed_int_binop: c is NULL");
    StsWordBuf *wb = &c->sections.functions;
    uint32_t unsigned_ssir = get_unsigned_ssir_type(c, ssir_type);
    uint32_t unsigned_spv = sts_emit_type(c, unsigned_ssir);
    uint32_t cast_a = sts_fresh_id(c);
    uint32_t cast_b = sts_fresh_id(c);
    uint32_t op_result = sts_fresh_id(c);
    uint32_t result_spv = sts_fresh_id(c);
    set_spv_id(c, ssir_result, result_spv);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, unsigned_spv); wb_push(wb, cast_a); wb_push(wb, op0);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, unsigned_spv); wb_push(wb, cast_b); wb_push(wb, op1);
    sts_emit_op(wb, spv_op, 5);
    wb_push(wb, unsigned_spv); wb_push(wb, op_result); wb_push(wb, cast_a); wb_push(wb, cast_b);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, type_spv); wb_push(wb, result_spv); wb_push(wb, op_result);
}

//c nonnull, inst nonnull
static int emit_instruction(Ctx *c, const SsirInst *inst, uint32_t func_type_hint) {
    wgsl_compiler_assert(c != NULL, "emit_instruction: c is NULL");
    wgsl_compiler_assert(inst != NULL, "emit_instruction: inst is NULL");
    StsWordBuf *wb = &c->sections.functions;
    uint32_t type_spv = inst->type ? sts_emit_type(c, inst->type) : 0;

    /* Get operand types for type-driven opcode selection.
     * For most ops, the result type matches operand types.
     * For comparison ops (result is bool), we need the actual operand type. */
    uint32_t op0_type = 0;
    uint32_t op1_type = 0;
    if (inst->operand_count > 0) {
        op0_type = get_ssir_type(c, inst->operands[0]);
        if (op0_type == 0) {
            op0_type = inst->type;
        }
    }
    if (inst->operand_count > 1) {
        op1_type = get_ssir_type(c, inst->operands[1]);
        if (op1_type == 0) {
            op1_type = inst->type;
        }
    }

    /* Defer result ID allocation for signed int binops (they allocate lazily) */
    int defer_result = 0;
    if ((inst->op == SSIR_OP_ADD || inst->op == SSIR_OP_SUB || inst->op == SSIR_OP_MUL) &&
        !is_float_ssir_type(c, op0_type) && is_signed_ssir_type(c, op0_type)) {
        defer_result = 1;
    }
    uint32_t result_spv = (!defer_result && inst->result) ? get_spv_id(c, inst->result) : 0;

    /* Record result type for this instruction (for future lookups) */
    if (inst->result && inst->type) {
        set_ssir_type(c, inst->result, inst->type);
    }

    switch (inst->op) {
        /* Arithmetic */
        case SSIR_OP_ADD:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFAdd, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpIAdd, inst->result, type_spv, inst->type,
                                       get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpIAdd, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_SUB:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFSub, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpISub, inst->result, type_spv, inst->type,
                                       get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpISub, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_MUL:
            if (is_float_ssir_type(c, op0_type)) {
                /* Check for vector * scalar or scalar * vector */
                int op0_is_vec = is_vector_ssir_type(c, op0_type);
                int op1_is_vec = is_vector_ssir_type(c, op1_type);
                int op0_is_scalar = is_scalar_ssir_type(c, op0_type);
                int op1_is_scalar = is_scalar_ssir_type(c, op1_type);

                if (op0_is_vec && op1_is_scalar) {
                    /* vec * scalar */
                    sts_emit_op(wb, SpvOpVectorTimesScalar, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[0]));
                    wb_push(wb, get_spv_id(c, inst->operands[1]));
                } else if (op0_is_scalar && op1_is_vec) {
                    /* scalar * vec -> swap operands for VectorTimesScalar */
                    sts_emit_op(wb, SpvOpVectorTimesScalar, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[1])); /* vec first */
                    wb_push(wb, get_spv_id(c, inst->operands[0])); /* scalar second */
                } else {
                    sts_emit_op(wb, SpvOpFMul, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[0]));
                    wb_push(wb, get_spv_id(c, inst->operands[1]));
                }
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpIMul, inst->result, type_spv, inst->type,
                                       get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpIMul, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_DIV:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFDiv, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSDiv, 5);
            } else {
                sts_emit_op(wb, SpvOpUDiv, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_MOD:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFMod, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSMod, 5);
            } else {
                sts_emit_op(wb, SpvOpUMod, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_REM:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFRem, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSRem, 5);
            } else {
                sts_emit_op(wb, SpvOpUMod, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NEG:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFNegate, 4);
            } else {
                sts_emit_op(wb, SpvOpSNegate, 4);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Matrix */
        case SSIR_OP_MAT_MUL: {
            /* Determine operation based on operand types */
            int op0_mat = is_matrix_ssir_type(c, op0_type);
            int op1_mat = is_matrix_ssir_type(c, op1_type);
            int op0_vec = is_vector_ssir_type(c, op0_type);
            int op1_vec = is_vector_ssir_type(c, op1_type);
            int op0_scalar = is_scalar_ssir_type(c, op0_type);
            int op1_scalar = is_scalar_ssir_type(c, op1_type);

            if (op0_mat && op1_mat) {
                sts_emit_op(wb, SpvOpMatrixTimesMatrix, 5);
            } else if (op0_mat && op1_vec) {
                sts_emit_op(wb, SpvOpMatrixTimesVector, 5);
            } else if (op0_vec && op1_mat) {
                sts_emit_op(wb, SpvOpVectorTimesMatrix, 5);
            } else if (op0_mat && op1_scalar) {
                sts_emit_op(wb, SpvOpMatrixTimesScalar, 5);
            } else if (op0_scalar && op1_mat) {
                /* scalar * mat -> emit as MatrixTimesScalar with swapped operands */
                sts_emit_op(wb, SpvOpMatrixTimesScalar, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[1]));
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                break;
            } else {
                sts_emit_op(wb, SpvOpFMul, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;
        }

        case SSIR_OP_MAT_TRANSPOSE:
            sts_emit_op(wb, SpvOpTranspose, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Bitwise */
        case SSIR_OP_BIT_AND:
            sts_emit_op(wb, SpvOpBitwiseAnd, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_OR:
            sts_emit_op(wb, SpvOpBitwiseOr, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_XOR:
            sts_emit_op(wb, SpvOpBitwiseXor, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_NOT:
            sts_emit_op(wb, SpvOpNot, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_SHL:
            sts_emit_op(wb, SpvOpShiftLeftLogical, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_SHR:
            if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpShiftRightArithmetic, 5);
            } else {
                sts_emit_op(wb, SpvOpShiftRightLogical, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_SHR_LOGICAL:
            sts_emit_op(wb, SpvOpShiftRightLogical, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        /* Comparison */
        case SSIR_OP_EQ:
            if (is_bool_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpLogicalEqual, 5);
            } else if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpIEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NE:
            if (is_bool_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpLogicalNotEqual, 5);
            } else if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdNotEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpINotEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_LT:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdLessThan, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSLessThan, 5);
            } else {
                sts_emit_op(wb, SpvOpULessThan, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_LE:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdLessThanEqual, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSLessThanEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpULessThanEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_GT:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdGreaterThan, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSGreaterThan, 5);
            } else {
                sts_emit_op(wb, SpvOpUGreaterThan, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_GE:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdGreaterThanEqual, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSGreaterThanEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpUGreaterThanEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        /* Logical */
        case SSIR_OP_AND:
            sts_emit_op(wb, SpvOpLogicalAnd, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_OR:
            sts_emit_op(wb, SpvOpLogicalOr, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NOT:
            sts_emit_op(wb, SpvOpLogicalNot, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Composite */
        case SSIR_OP_CONSTRUCT: {
            uint32_t count = inst->operand_count;
            if (inst->extra_count > 0) count = inst->extra_count;
            sts_emit_op(wb, SpvOpCompositeConstruct, 3 + count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            if (inst->extra_count > 0) {
                for (uint16_t i = 0; i < inst->extra_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->extra[i]));
                }
            } else {
                for (uint8_t i = 0; i < inst->operand_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->operands[i]));
                }
            }
            break;
        }

        case SSIR_OP_EXTRACT:
            sts_emit_op(wb, SpvOpCompositeExtract, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, inst->operands[1]); /* literal index */
            break;

        case SSIR_OP_INSERT:
            sts_emit_op(wb, SpvOpCompositeInsert, 6);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* object to insert */
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* composite */
            wb_push(wb, inst->operands[2]); /* literal index */
            break;

        case SSIR_OP_SHUFFLE: {
            sts_emit_op(wb, SpvOpVectorShuffle, 5 + inst->extra_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            for (uint16_t i = 0; i < inst->extra_count; ++i) {
                wb_push(wb, inst->extra[i]); /* literal indices */
            }
            break;
        }

        case SSIR_OP_SPLAT: {
            /* OpCompositeConstruct with repeated scalar */
            SsirType *t = ssir_get_type((SsirModule *)c->mod, inst->type);
            uint32_t count = t && t->kind == SSIR_TYPE_VEC ? t->vec.size : 4;
            sts_emit_op(wb, SpvOpCompositeConstruct, 3 + count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            uint32_t scalar = get_spv_id(c, inst->operands[0]);
            for (uint32_t i = 0; i < count; ++i) {
                wb_push(wb, scalar);
            }
            break;
        }

        case SSIR_OP_EXTRACT_DYN:
            sts_emit_op(wb, SpvOpVectorExtractDynamic, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_INSERT_DYN:
            sts_emit_op(wb, SpvOpVectorInsertDynamic, 6);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* vector */
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* component */
            wb_push(wb, get_spv_id(c, inst->operands[2])); /* index */
            break;

        /* Memory */
        case SSIR_OP_LOAD:
            /* PRE: operand_count >= 1 */
            wgsl_compiler_assert(inst->operand_count >= 1, "SSIR_OP_LOAD: operand_count < 1");
            sts_emit_op(wb, SpvOpLoad, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_STORE:
            /* PRE: operand_count >= 2 */
            wgsl_compiler_assert(inst->operand_count >= 2, "SSIR_OP_STORE: operand_count < 2");
            sts_emit_op(wb, SpvOpStore, 3);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_ACCESS: {
            uint32_t idx_count = inst->extra_count;
            sts_emit_op(wb, SpvOpAccessChain, 4 + idx_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            for (uint16_t i = 0; i < idx_count; ++i) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));
            }
            break;
        }

        case SSIR_OP_ARRAY_LEN:
            sts_emit_op(wb, SpvOpArrayLength, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, 0); /* member index - typically 0 for runtime arrays */
            break;

        /* Control Flow */
        case SSIR_OP_BRANCH:
            sts_emit_op(wb, SpvOpBranch, 2);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_BRANCH_COND:
            /* Emit OpSelectionMerge if merge block is specified (operands[3]) */
            if (inst->operand_count >= 4 && inst->operands[3] != 0) {
                sts_emit_op(wb, SpvOpSelectionMerge, 3);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, 0); /* SelectionControlMaskNone */
            }
            sts_emit_op(wb, SpvOpBranchConditional, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;

        case SSIR_OP_SWITCH: {
            uint32_t case_count = inst->extra_count / 2;
            sts_emit_op(wb, SpvOpSwitch, 3 + inst->extra_count);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* default */
            for (uint16_t i = 0; i < inst->extra_count; i += 2) {
                wb_push(wb, inst->extra[i]);     /* literal value */
                wb_push(wb, get_spv_id(c, inst->extra[i + 1])); /* label */
            }
            break;
        }

        case SSIR_OP_PHI: {
            uint32_t pair_count = inst->extra_count / 2;
            sts_emit_op(wb, SpvOpPhi, 3 + inst->extra_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            for (uint16_t i = 0; i < inst->extra_count; i += 2) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));     /* value */
                wb_push(wb, get_spv_id(c, inst->extra[i + 1])); /* parent block */
            }
            break;
        }

        case SSIR_OP_RETURN:
            sts_emit_op(wb, SpvOpReturnValue, 2);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_RETURN_VOID:
            sts_emit_op(wb, SpvOpReturn, 1);
            break;

        case SSIR_OP_UNREACHABLE:
            sts_emit_op(wb, SpvOpUnreachable, 1);
            break;

        case SSIR_OP_LOOP_MERGE:
            sts_emit_op(wb, SpvOpLoopMerge, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* merge block */
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* continue block */
            wb_push(wb, 0); /* LoopControlMaskNone */
            break;

        case SSIR_OP_SELECTION_MERGE:
            sts_emit_op(wb, SpvOpSelectionMerge, 3);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* merge block */
            wb_push(wb, 0); /* SelectionControlMaskNone */
            break;

        case SSIR_OP_DISCARD:
            sts_emit_op(wb, SpvOpKill, 1);
            break;

        /* Call */
        case SSIR_OP_CALL: {
            uint32_t arg_count = inst->extra_count;
            sts_emit_op(wb, SpvOpFunctionCall, 4 + arg_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* callee */
            for (uint16_t i = 0; i < arg_count; ++i) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));
            }
            break;
        }

        case SSIR_OP_BUILTIN: {
            SsirBuiltinId builtin_id = (SsirBuiltinId)inst->operands[0];
            int glsl_op = builtin_to_glsl_op(builtin_id);
            sts_ensure_glsl_import(c);

            if (builtin_id == SSIR_BUILTIN_DOT) {
                /* OpDot is native */
                sts_emit_op(wb, SpvOpDot, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
                wb_push(wb, get_spv_id(c, inst->extra[1]));
            } else if (builtin_id == SSIR_BUILTIN_ABS) {
                /* Type-dependent */
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FAbs);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SAbs);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                }
            } else if (builtin_id == SSIR_BUILTIN_MIN) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                }
            } else if (builtin_id == SSIR_BUILTIN_MAX) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                }
            } else if (builtin_id == SSIR_BUILTIN_CLAMP) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                }
            } else if (builtin_id == SSIR_BUILTIN_SELECT) {
                sts_emit_op(wb, SpvOpSelect, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[2])); /* condition */
                wb_push(wb, get_spv_id(c, inst->extra[1])); /* true value */
                wb_push(wb, get_spv_id(c, inst->extra[0])); /* false value */
            } else if (builtin_id == SSIR_BUILTIN_ALL) {
                sts_emit_op(wb, SpvOpAll, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ANY) {
                sts_emit_op(wb, SpvOpAny, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_DPDX || builtin_id == SSIR_BUILTIN_DPDX_COARSE || builtin_id == SSIR_BUILTIN_DPDX_FINE) {
                sts_emit_op(wb, SpvOpDPdx, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_DPDY || builtin_id == SSIR_BUILTIN_DPDY_COARSE || builtin_id == SSIR_BUILTIN_DPDY_FINE) {
                sts_emit_op(wb, SpvOpDPdy, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_FWIDTH) {
                sts_emit_op(wb, SpvOpFwidth, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ISINF) {
                sts_emit_op(wb, SpvOpIsInf, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ISNAN) {
                sts_emit_op(wb, SpvOpIsNan, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_TRANSPOSE) {
                sts_emit_op(wb, SpvOpTranspose, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (glsl_op >= 0) {
                /* Generic GLSL.std.450 function */
                sts_emit_op(wb, SpvOpExtInst, 5 + inst->extra_count);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, c->glsl_ext_id);
                wb_push(wb, glsl_op);
                for (uint16_t i = 0; i < inst->extra_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->extra[i]));
                }
            }
            break;
        }

        /* Conversion */
        case SSIR_OP_CONVERT: {
            /* Determine conversion type */
            int from_float = 0, to_float = 0, from_signed = 0, to_signed = 0;
            /* TODO: need operand type info for proper selection */
            /* For now, use result type to guide */
            if (is_float_ssir_type(c, inst->type)) {
                sts_emit_op(wb, SpvOpConvertSToF, 4); /* assume from signed */
            } else if (is_signed_ssir_type(c, inst->type)) {
                sts_emit_op(wb, SpvOpConvertFToS, 4); /* assume from float */
            } else {
                sts_emit_op(wb, SpvOpConvertFToU, 4);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;
        }

        case SSIR_OP_BITCAST:
            sts_emit_op(wb, SpvOpBitcast, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Texture operations */
        case SSIR_OP_TEX_SAMPLE:
        case SSIR_OP_TEX_SAMPLE_BIAS:
        case SSIR_OP_TEX_SAMPLE_LEVEL:
        case SSIR_OP_TEX_SAMPLE_GRAD:
        case SSIR_OP_TEX_SAMPLE_CMP:
        case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
        case SSIR_OP_TEX_SAMPLE_OFFSET:
        case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
        case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
        case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
        case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
        case SSIR_OP_TEX_GATHER:
        case SSIR_OP_TEX_GATHER_CMP:
        case SSIR_OP_TEX_GATHER_OFFSET: {
            /* Create OpSampledImage from texture + sampler */
            uint32_t tex_ssir_type = get_ssir_type(c, inst->operands[0]);
            uint32_t img_spv = sts_emit_type(c, tex_ssir_type);
            uint32_t si_type = sts_emit_type_sampled_image(c, img_spv);
            uint32_t si_id = sts_fresh_id(c);
            sts_emit_op(wb, SpvOpSampledImage, 5);
            wb_push(wb, si_type);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));

            uint32_t coord_spv = get_spv_id(c, inst->operands[2]);

            switch (inst->op) {
            case SSIR_OP_TEX_SAMPLE:
                sts_emit_op(wb, SpvOpImageSampleImplicitLod, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                break;
            case SSIR_OP_TEX_SAMPLE_BIAS:
                sts_emit_op(wb, SpvOpImageSampleImplicitLod, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x1); /* Bias */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_SAMPLE_LEVEL:
                sts_emit_op(wb, SpvOpImageSampleExplicitLod, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x2); /* Lod */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_SAMPLE_GRAD:
                sts_emit_op(wb, SpvOpImageSampleExplicitLod, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x4); /* Grad */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            case SSIR_OP_TEX_SAMPLE_CMP:
                sts_emit_op(wb, SpvOpImageSampleDrefImplicitLod, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
                sts_emit_op(wb, SpvOpImageSampleDrefExplicitLod, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, 0x2); /* Lod */
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            case SSIR_OP_TEX_SAMPLE_OFFSET:
                sts_emit_op(wb, SpvOpImageSampleImplicitLod, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x10); /* ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
                sts_emit_op(wb, SpvOpImageSampleImplicitLod, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x1 | 0x10); /* Bias | ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
                sts_emit_op(wb, SpvOpImageSampleExplicitLod, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x2 | 0x10); /* Lod | ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
                sts_emit_op(wb, SpvOpImageSampleExplicitLod, 9);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, 0x4 | 0x10); /* Grad | ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                wb_push(wb, get_spv_id(c, inst->operands[5]));
                break;
            case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
                sts_emit_op(wb, SpvOpImageSampleDrefImplicitLod, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, 0x10); /* ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            case SSIR_OP_TEX_GATHER:
                sts_emit_op(wb, SpvOpImageGather, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_GATHER_CMP:
                sts_emit_op(wb, SpvOpImageDrefGather, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                break;
            case SSIR_OP_TEX_GATHER_OFFSET:
                sts_emit_op(wb, SpvOpImageGather, 8);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, si_id);
                wb_push(wb, coord_spv);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, 0x10); /* ConstOffset */
                wb_push(wb, get_spv_id(c, inst->operands[4]));
                break;
            default: break;
            }
            break;
        }

        case SSIR_OP_TEX_LOAD:
            sts_emit_op(wb, SpvOpImageRead, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_TEX_STORE:
            sts_emit_op(wb, SpvOpImageWrite, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;

        case SSIR_OP_TEX_SIZE:
            sts_emit_op(wb, SpvOpImageQuerySizeLod, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_TEX_QUERY_LOD: {
            uint32_t tex_ssir_type = get_ssir_type(c, inst->operands[0]);
            uint32_t img_spv = sts_emit_type(c, tex_ssir_type);
            uint32_t si_type = sts_emit_type_sampled_image(c, img_spv);
            uint32_t si_id = sts_fresh_id(c);
            sts_emit_op(wb, SpvOpSampledImage, 5);
            wb_push(wb, si_type);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            sts_emit_op(wb, SpvOpImageQueryLod, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;
        }

        case SSIR_OP_TEX_QUERY_LEVELS:
            sts_emit_op(wb, SpvOpImageQueryLevels, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_TEX_QUERY_SAMPLES:
            sts_emit_op(wb, SpvOpImageQuerySamples, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Sync */
        case SSIR_OP_BARRIER: {
            SsirBarrierScope bscope = (SsirBarrierScope)inst->operands[0];
            switch (bscope) {
            case SSIR_BARRIER_WORKGROUP: {
                uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                uint32_t sem = sts_emit_const_u32(c, 0x108); /* AcquireRelease | WorkgroupMemory */
                sts_emit_op(wb, SpvOpControlBarrier, 4);
                wb_push(wb, wg); wb_push(wb, wg); wb_push(wb, sem);
                break;
            }
            case SSIR_BARRIER_STORAGE: {
                uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                uint32_t sem = sts_emit_const_u32(c, 0x48); /* AcquireRelease | UniformMemory */
                sts_emit_op(wb, SpvOpMemoryBarrier, 3);
                wb_push(wb, wg); wb_push(wb, sem);
                break;
            }
            case SSIR_BARRIER_SUBGROUP: {
                uint32_t sg = sts_emit_const_u32(c, SpvScopeSubgroup);
                uint32_t sem = sts_emit_const_u32(c, 0x108); /* AcquireRelease | WorkgroupMemory */
                sts_emit_op(wb, SpvOpControlBarrier, 4);
                wb_push(wb, sg); wb_push(wb, sg); wb_push(wb, sem);
                break;
            }
            case SSIR_BARRIER_IMAGE: {
                uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                uint32_t sem = sts_emit_const_u32(c, 0x808); /* AcquireRelease | ImageMemory */
                sts_emit_op(wb, SpvOpMemoryBarrier, 3);
                wb_push(wb, wg); wb_push(wb, sem);
                break;
            }
            }
            break;
        }

        case SSIR_OP_ATOMIC: {
            SsirAtomicOp aop = (SsirAtomicOp)inst->operands[0];
            uint32_t a_ptr = get_spv_id(c, inst->operands[1]);
            uint32_t a_val = inst->operands[2] ? get_spv_id(c, inst->operands[2]) : 0;
            uint32_t a_cmp = inst->operands[3] ? get_spv_id(c, inst->operands[3]) : 0;
            /* Scope and semantics: use from extended form or defaults */
            uint32_t scope_val = (inst->operand_count >= 6) ? inst->operands[4] : 0; /* SSIR_SCOPE_DEVICE */
            uint32_t sem_val = (inst->operand_count >= 6) ? inst->operands[5] : 0; /* SSIR_SEMANTICS_RELAXED */
            /* Convert SSIR scope to SPIR-V scope */
            SpvScope spv_scope;
            switch ((SsirMemoryScope)scope_val) {
            case SSIR_SCOPE_WORKGROUP: spv_scope = SpvScopeWorkgroup; break;
            case SSIR_SCOPE_SUBGROUP: spv_scope = SpvScopeSubgroup; break;
            case SSIR_SCOPE_INVOCATION: spv_scope = SpvScopeInvocation; break;
            default: spv_scope = SpvScopeDevice; break;
            }
            /* Convert SSIR semantics to SPIR-V memory semantics */
            SpvMemorySemanticsMask spv_sem;
            switch ((SsirMemorySemantics)sem_val) {
            case SSIR_SEMANTICS_ACQUIRE: spv_sem = SpvMemorySemanticsMaskNone | 0x2; break; /* Acquire */
            case SSIR_SEMANTICS_RELEASE: spv_sem = SpvMemorySemanticsMaskNone | 0x4; break; /* Release */
            case SSIR_SEMANTICS_ACQUIRE_RELEASE: spv_sem = SpvMemorySemanticsMaskNone | 0x8; break; /* AcquireRelease */
            case SSIR_SEMANTICS_SEQ_CST: spv_sem = SpvMemorySemanticsMaskNone | 0x10; break; /* SequentiallyConsistent */
            default: spv_sem = SpvMemorySemanticsMaskNone; break; /* Relaxed */
            }
            /* Emit scope and semantics as constants */
            uint32_t scope_id = sts_emit_const_u32(c, (uint32_t)spv_scope);
            uint32_t sem_id = sts_emit_const_u32(c, (uint32_t)spv_sem);
            uint32_t sem_equal_id = sem_id; /* For compare exchange, equal semantics */
            uint32_t sem_unequal_id = sts_emit_const_u32(c, (uint32_t)SpvMemorySemanticsMaskNone); /* Unequal gets relaxed */
            switch (aop) {
            case SSIR_ATOMIC_LOAD:
                sts_emit_op(wb, SpvOpAtomicLoad, 5);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id);
                break;
            case SSIR_ATOMIC_STORE:
                sts_emit_op(wb, SpvOpAtomicStore, 5);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_ADD:
                sts_emit_op(wb, SpvOpAtomicIAdd, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_SUB:
                sts_emit_op(wb, SpvOpAtomicISub, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_MAX:
                sts_emit_op(wb, SpvOpAtomicUMax, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_MIN:
                sts_emit_op(wb, SpvOpAtomicUMin, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_AND:
                sts_emit_op(wb, SpvOpAtomicAnd, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_OR:
                sts_emit_op(wb, SpvOpAtomicOr, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_XOR:
                sts_emit_op(wb, SpvOpAtomicXor, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_EXCHANGE:
                sts_emit_op(wb, SpvOpAtomicExchange, 7);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_id); wb_push(wb, a_val);
                break;
            case SSIR_ATOMIC_COMPARE_EXCHANGE:
                sts_emit_op(wb, SpvOpAtomicCompareExchange, 9);
                wb_push(wb, type_spv); wb_push(wb, result_spv);
                wb_push(wb, a_ptr); wb_push(wb, scope_id); wb_push(wb, sem_equal_id);
                wb_push(wb, sem_unequal_id); wb_push(wb, a_val); wb_push(wb, a_cmp);
                break;
            }
            break;
        }

        default:
            break;
    }

    return 1;
}

/* ============================================================================
 * Function Emission
 * ============================================================================ */

//c nonnull, func nonnull
static int sts_emit_function(Ctx *c, const SsirFunction *func, uint32_t func_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_function: c is NULL");
    wgsl_compiler_assert(func != NULL, "sts_emit_function: func is NULL");
    StsWordBuf *wb = &c->sections.functions;

    uint32_t return_spv = sts_emit_type(c, func->return_type);

    /* Emit OpFunction */
    uint32_t func_spv = get_spv_id(c, func->id);
    sts_emit_op(wb, SpvOpFunction, 5);
    wb_push(wb, return_spv);
    wb_push(wb, func_spv);
    wb_push(wb, SpvFunctionControlMaskNone);
    wb_push(wb, func_type);

    /* Debug name */
    sts_emit_name(c, func_spv, func->name);

    /* Emit parameters */
    for (uint32_t i = 0; i < func->param_count; ++i) {
        uint32_t param_type_spv = sts_emit_type(c, func->params[i].type);
        uint32_t param_spv = get_spv_id(c, func->params[i].id);
        sts_emit_op(wb, SpvOpFunctionParameter, 3);
        wb_push(wb, param_type_spv);
        wb_push(wb, param_spv);
        sts_emit_name(c, param_spv, func->params[i].name);
    }

    /* Emit blocks */
    for (uint32_t bi = 0; bi < func->block_count; ++bi) {
        const SsirBlock *block = &func->blocks[bi];
        uint32_t block_spv = get_spv_id(c, block->id);

        /* Emit OpLabel */
        sts_emit_op(wb, SpvOpLabel, 2);
        wb_push(wb, block_spv);

        /* Emit local variables in first block */
        if (bi == 0) {
            for (uint32_t li = 0; li < func->local_count; ++li) {
                const SsirLocalVar *local = &func->locals[li];
                uint32_t local_type_spv = sts_emit_type(c, local->type);
                uint32_t local_spv = get_spv_id(c, local->id);
                sts_emit_op(wb, SpvOpVariable, 4);
                wb_push(wb, local_type_spv);
                wb_push(wb, local_spv);
                wb_push(wb, SpvStorageClassFunction);
                sts_emit_name(c, local_spv, local->name);
                set_ssir_type(c, local->id, local->type);
            }
        }

        /* Emit instructions */
        for (uint32_t ii = 0; ii < block->inst_count; ++ii) {
            emit_instruction(c, &block->insts[ii], func->return_type);
        }
    }

    /* Emit OpFunctionEnd */
    sts_emit_op(wb, SpvOpFunctionEnd, 1);

    return 1;
}

/* ============================================================================
 * Entry Point Emission
 * ============================================================================ */

static SpvExecutionModel stage_to_execution_model(SsirStage stage) {
    switch (stage) {
        case SSIR_STAGE_VERTEX:   return SpvExecutionModelVertex;
        case SSIR_STAGE_FRAGMENT: return SpvExecutionModelFragment;
        case SSIR_STAGE_COMPUTE:  return SpvExecutionModelGLCompute;
        case SSIR_STAGE_GEOMETRY: return SpvExecutionModelGeometry;
        case SSIR_STAGE_TESS_CONTROL: return SpvExecutionModelTessellationControl;
        case SSIR_STAGE_TESS_EVAL:    return SpvExecutionModelTessellationEvaluation;
        default:                  return SpvExecutionModelVertex;
    }
}

//c nonnull, ep nonnull
static int sts_emit_entry_point(Ctx *c, const SsirEntryPoint *ep) {
    wgsl_compiler_assert(c != NULL, "sts_emit_entry_point: c is NULL");
    wgsl_compiler_assert(ep != NULL, "sts_emit_entry_point: ep is NULL");
    /* Emit OpEntryPoint */
    StsWordBuf *wb = &c->sections.entry_points;
    uint32_t *name_words; size_t name_count;
    encode_string(ep->name, &name_words, &name_count);

    uint32_t func_spv = get_spv_id(c, ep->function);

    /* For SPIR-V 1.3, only Input/Output globals go in the interface */
    uint32_t iface_count = 0;
    uint32_t *iface_ids = NULL;
    if (ep->interface_count > 0) {
        iface_ids = (uint32_t *)STS_MALLOC(ep->interface_count * sizeof(uint32_t));
        /* PRE: iface_ids != NULL */
        wgsl_compiler_assert(iface_ids != NULL, "sts_emit_entry_point: iface_ids alloc failed");
        for (uint32_t i = 0; i < ep->interface_count; ++i) {
            SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, ep->interface[i]);
            if (g) {
                SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
                if (pt && pt->kind == SSIR_TYPE_PTR &&
                    (pt->ptr.space == SSIR_ADDR_INPUT || pt->ptr.space == SSIR_ADDR_OUTPUT)) {
                    iface_ids[iface_count++] = get_spv_id(c, ep->interface[i]);
                }
            }
        }
    }

    sts_emit_op(wb, SpvOpEntryPoint, 3 + name_count + iface_count);
    wb_push(wb, stage_to_execution_model(ep->stage));
    wb_push(wb, func_spv);
    sts_wb_push_many(wb, name_words, name_count);
    STS_FREE(name_words);

    for (uint32_t i = 0; i < iface_count; ++i) {
        wb_push(wb, iface_ids[i]);
    }
    STS_FREE(iface_ids);

    /* Emit execution modes */
    wb = &c->sections.execution_modes;

    if (ep->stage == SSIR_STAGE_FRAGMENT) {
        /* OriginUpperLeft for fragment shaders (default unless explicitly disabled) */
        if (!ep->origin_upper_left) {
            /* Still emit OriginUpperLeft as default, since most APIs expect it */
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeOriginUpperLeft);
        } else {
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeOriginUpperLeft);
        }
        if (ep->depth_replacing) {
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeDepthReplacing);
        }
        if (ep->early_fragment_tests) {
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeEarlyFragmentTests);
        }
    }

    if (ep->stage == SSIR_STAGE_COMPUTE) {
        /* LocalSize for compute shaders */
        sts_emit_op(wb, SpvOpExecutionMode, 6);
        wb_push(wb, func_spv);
        wb_push(wb, SpvExecutionModeLocalSize);
        wb_push(wb, ep->workgroup_size[0]);
        wb_push(wb, ep->workgroup_size[1]);
        wb_push(wb, ep->workgroup_size[2]);
    }

    return 1;
}

/* ============================================================================
 * Main Conversion
 * ============================================================================ */

SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
                                 const SsirToSpirvOptions *opts,
                                 uint32_t **out_words,
                                 size_t *out_count) {
    if (!mod || !out_words || !out_count) {
        return SSIR_TO_SPIRV_ERR_INVALID_INPUT;
    }

    *out_words = NULL;
    *out_count = 0;

    /* Initialize context */
    Ctx c = {0};
    c.mod = mod;
    if (opts) {
        c.opts = *opts;
    }
    sections_init(&c.sections);
    c.next_spv_id = 1;

    /* Allocate ID map */
    c.id_map_size = mod->next_id + 1;
    c.id_map = (uint32_t *)STS_MALLOC(c.id_map_size * sizeof(uint32_t));
    if (!c.id_map) {
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }
    memset(c.id_map, 0, c.id_map_size * sizeof(uint32_t));

    /* Allocate type map (SSIR ID -> SSIR type) */
    c.type_map_size = mod->next_id + 1;
    c.type_map = (uint32_t *)STS_MALLOC(c.type_map_size * sizeof(uint32_t));
    if (!c.type_map) {
        STS_FREE(c.id_map);
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }
    memset(c.type_map, 0, c.type_map_size * sizeof(uint32_t));

    /* Emit capabilities */
    sts_emit_capability(&c, SpvCapabilityShader);
    c.has_shader_cap = 1;

    /* GLSL.std.450 import ID allocated lazily on first use */
    c.glsl_ext_id = 0;

    /* Emit memory model */
    sts_emit_memory_model(&c);

    /* Pre-emit function signature types and pre-allocate function/block IDs
     * Order: return type, function ID, param types, function type, block IDs */
    if (mod->function_count > 0) {
        c.func_type_cache = (uint32_t *)STS_MALLOC(mod->function_count * sizeof(uint32_t));
        /* PRE: func_type_cache != NULL */
        wgsl_compiler_assert(c.func_type_cache != NULL, "ssir_to_spirv: func_type_cache alloc failed");
        c.func_type_cache_count = mod->function_count;
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            const SsirFunction *func = &mod->functions[i];
            uint32_t return_spv = sts_emit_type(&c, func->return_type);
            get_spv_id(&c, func->id);
            uint32_t *param_types = NULL;
            if (func->param_count > 0) {
                param_types = (uint32_t *)STS_MALLOC(func->param_count * sizeof(uint32_t));
                /* PRE: param_types != NULL */
                wgsl_compiler_assert(param_types != NULL, "ssir_to_spirv: param_types alloc failed");
                for (uint32_t j = 0; j < func->param_count; ++j) {
                    /* PRE: func->params[j] valid (guarded by param_count loop) */
                    wgsl_compiler_assert(j < func->param_count, "ssir_to_spirv: params[%u] OOB", j);
                    param_types[j] = sts_emit_type(&c, func->params[j].type);
                }
            }
            c.func_type_cache[i] = sts_emit_type_function(&c, return_spv, param_types, func->param_count);
            STS_FREE(param_types);
            for (uint32_t bi = 0; bi < func->block_count; ++bi)
                get_spv_id(&c, func->blocks[bi].id);
        }
    }

    /* Emit constants with correct type ordering:
     * For composite constants, emit their type just before the first component constant */
    {
        /* Build map: for each composite, find the earliest component constant index */
        uint32_t *comp_type_at = NULL;  /* comp_type_at[i] = ssir type to emit before constant i, or 0 */
        if (mod->constant_count > 0) {
            comp_type_at = (uint32_t *)STS_MALLOC(mod->constant_count * sizeof(uint32_t));
            if (comp_type_at) memset(comp_type_at, 0, mod->constant_count * sizeof(uint32_t));
        }
        if (comp_type_at) {
            for (uint32_t ci = 0; ci < mod->constant_count; ++ci) {
                if (mod->constants[ci].kind != SSIR_CONST_COMPOSITE) continue;
                /* Find earliest component index */
                uint32_t earliest = ci;
                for (uint32_t k = 0; k < mod->constants[ci].composite.count; ++k) {
                    uint32_t comp_id = mod->constants[ci].composite.components[k];
                    for (uint32_t j = 0; j < mod->constant_count; ++j) {
                        if (mod->constants[j].id == comp_id && j < earliest) {
                            earliest = j;
                            break;
                        }
                    }
                }
                if (earliest < ci && comp_type_at[earliest] == 0)
                    comp_type_at[earliest] = mod->constants[ci].type;
            }
        }
        for (uint32_t i = 0; i < mod->constant_count; ++i) {
            if (comp_type_at && comp_type_at[i])
                sts_emit_type(&c, comp_type_at[i]);
            sts_emit_constant(&c, &mod->constants[i]);
        }
        STS_FREE(comp_type_at);
    }

    /* Emit global variables */
    for (uint32_t i = 0; i < mod->global_count; ++i) {
        sts_emit_global_var(&c, &mod->globals[i]);
    }

    /* Emit functions */
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        sts_emit_function(&c, &mod->functions[i], c.func_type_cache[i]);
    }

    /* Emit names from name table (for let/const SSA results) */
    for (uint32_t i = 0; i < mod->name_count; ++i) {
        uint32_t spv_id = get_spv_id(&c, mod->names[i].id);
        if (spv_id) sts_emit_name(&c, spv_id, mod->names[i].name);
    }

    /* Emit entry points */
    for (uint32_t i = 0; i < mod->entry_point_count; ++i) {
        sts_emit_entry_point(&c, &mod->entry_points[i]);
    }

    /* Finalize SPIR-V binary */
    size_t total = 5; /* header */
    total += c.sections.capabilities.len;
    total += c.sections.extensions.len;
    total += c.sections.ext_inst_imports.len;
    total += c.sections.memory_model.len;
    total += c.sections.entry_points.len;
    total += c.sections.execution_modes.len;
    total += c.sections.debug_names.len;
    total += c.sections.annotations.len;
    total += c.sections.types_constants.len;
    total += c.sections.globals.len;
    total += c.sections.functions.len;

    uint32_t *words = (uint32_t *)STS_MALLOC(total * sizeof(uint32_t));
    if (!words) {
        STS_FREE(c.id_map);
        STS_FREE(c.type_map);
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }

    size_t pos = 0;

    /* Header */
    words[pos++] = SpvMagicNumber;
    words[pos++] = c.opts.spirv_version ? c.opts.spirv_version : 0x00010300;
    words[pos++] = 0; /* generator */
    words[pos++] = c.next_spv_id; /* bound */
    words[pos++] = 0; /* reserved */

    /* Copy sections */
    #define COPY_SEC(sec) do { \
        if (c.sections.sec.len > 0) { \
            memcpy(words + pos, c.sections.sec.data, c.sections.sec.len * sizeof(uint32_t)); \
            pos += c.sections.sec.len; \
        } \
    } while(0)

    COPY_SEC(capabilities);
    COPY_SEC(extensions);
    COPY_SEC(ext_inst_imports);
    COPY_SEC(memory_model);
    COPY_SEC(entry_points);
    COPY_SEC(execution_modes);
    COPY_SEC(debug_names);
    COPY_SEC(annotations);
    COPY_SEC(types_constants);
    COPY_SEC(globals);
    COPY_SEC(functions);

    #undef COPY_SEC

    /* Cleanup */
    STS_FREE(c.func_type_cache);
    STS_FREE(c.id_map);
    STS_FREE(c.type_map);
    sections_free(&c.sections);

    *out_words = words;
    *out_count = total;

    return SSIR_TO_SPIRV_OK;
}

void ssir_to_spirv_free(void *p) {
    STS_FREE(p);
}

const char *ssir_to_spirv_result_string(SsirToSpirvResult result) {
    switch (result) {
        case SSIR_TO_SPIRV_OK:               return "OK";
        case SSIR_TO_SPIRV_ERR_INVALID_INPUT: return "Invalid input";
        case SSIR_TO_SPIRV_ERR_UNSUPPORTED:   return "Unsupported feature";
        case SSIR_TO_SPIRV_ERR_INTERNAL:      return "Internal error";
        case SSIR_TO_SPIRV_ERR_OOM:           return "Out of memory";
        default:                              return "Unknown error";
    }
}
