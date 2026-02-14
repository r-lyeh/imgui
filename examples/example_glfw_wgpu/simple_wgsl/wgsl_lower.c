// BEGIN FILE wgsl_lower.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "simple_wgsl.h"

// Use official SPIR-V headers
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

// ---------- Internal helpers ----------

#ifndef WGSL_MALLOC
#define WGSL_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef WGSL_REALLOC
#define WGSL_REALLOC(P, SZ) realloc((P), (SZ))
#endif
#ifndef WGSL_FREE
#define WGSL_FREE(P) free((P))
#endif

// ---------- Word Buffer ----------

typedef struct {
    uint32_t *data;
    size_t    len;
    size_t    cap;
} WordBuf;

//wb nonnull
static void wb_init(WordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "wb_init: wb is NULL");
    wb->data = NULL;
    wb->len = 0;
    wb->cap = 0;
}
//wb nonnull
static void wb_free(WordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "wb_free: wb is NULL");
    WGSL_FREE(wb->data);
    wb->data = NULL;
    wb->len = wb->cap = 0;
}
//wb nonnull
static int wb_reserve(WordBuf *wb, size_t need) {
    wgsl_compiler_assert(wb != NULL, "wb_reserve: wb is NULL");
    if (wb->len + need <= wb->cap) return 1;
    size_t ncap = wb->cap ? wb->cap : 64;
    while (ncap < wb->len + need) ncap *= 2;
    void *nd = WGSL_REALLOC(wb->data, ncap * sizeof(uint32_t));
    if (!nd) return 0;
    wb->data = (uint32_t*)nd;
    wb->cap = ncap;
    return 1;
}
//wb nonnull
static int wb_push_u32(WordBuf *wb, uint32_t w) {
    wgsl_compiler_assert(wb != NULL, "wb_push_u32: wb is NULL");
    if (!wb_reserve(wb, 1)) return 0;
    wb->data[wb->len++] = w;
    return 1;
}
//wb nonnull
//src nonnull
static int wb_push_many(WordBuf *wb, const uint32_t *src, size_t n) {
    wgsl_compiler_assert(wb != NULL, "wb_push_many: wb is NULL");
    wgsl_compiler_assert(src != NULL, "wb_push_many: src is NULL");
    if (!wb_reserve(wb, n)) return 0;
    memcpy(wb->data + wb->len, src, n * sizeof(uint32_t));
    wb->len += n;
    return 1;
}

//s nonnull
//out_words nonnull
//out_count nonnull
static uint32_t make_string_lit(const char *s, uint32_t **out_words, size_t *out_count) {
    wgsl_compiler_assert(s != NULL, "make_string_lit: s is NULL");
    wgsl_compiler_assert(out_words != NULL, "make_string_lit: out_words is NULL");
    wgsl_compiler_assert(out_count != NULL, "make_string_lit: out_count is NULL");
    size_t n = strlen(s) + 1; // include null
    size_t words = (n + 3) / 4;
    uint32_t *buf = (uint32_t*)WGSL_MALLOC(words * sizeof(uint32_t));
    if (!buf) { *out_words = NULL; *out_count = 0; return 0; }
    memset(buf, 0, words * sizeof(uint32_t));
    memcpy(buf, s, n);
    *out_words = buf;
    *out_count = words;
    return (uint32_t)words;
}

// ---------- SPIR-V Section Buffers ----------

typedef struct {
    WordBuf capabilities;
    WordBuf extensions;
    WordBuf ext_inst_imports;
    WordBuf memory_model;
    WordBuf entry_points;
    WordBuf execution_modes;
    WordBuf debug_names;
    WordBuf debug_strings;
    WordBuf annotations;
    WordBuf types_constants;
    WordBuf globals;
    WordBuf functions;
} SpvSections;

//s nonnull
static void spv_sections_init(SpvSections *s) {
    wgsl_compiler_assert(s != NULL, "spv_sections_init: s is NULL");
    wb_init(&s->capabilities);
    wb_init(&s->extensions);
    wb_init(&s->ext_inst_imports);
    wb_init(&s->memory_model);
    wb_init(&s->entry_points);
    wb_init(&s->execution_modes);
    wb_init(&s->debug_names);
    wb_init(&s->debug_strings);
    wb_init(&s->annotations);
    wb_init(&s->types_constants);
    wb_init(&s->globals);
    wb_init(&s->functions);
}

//s nonnull
static void spv_sections_free(SpvSections *s) {
    wgsl_compiler_assert(s != NULL, "spv_sections_free: s is NULL");
    wb_free(&s->capabilities);
    wb_free(&s->extensions);
    wb_free(&s->ext_inst_imports);
    wb_free(&s->memory_model);
    wb_free(&s->entry_points);
    wb_free(&s->execution_modes);
    wb_free(&s->debug_names);
    wb_free(&s->debug_strings);
    wb_free(&s->annotations);
    wb_free(&s->types_constants);
    wb_free(&s->globals);
    wb_free(&s->functions);
}

// ---------- Type Cache ----------

typedef enum {
    TC_VOID = 0,
    TC_BOOL,
    TC_INT,
    TC_FLOAT,
    TC_VECTOR,
    TC_MATRIX,
    TC_ARRAY,
    TC_RUNTIME_ARRAY,
    TC_STRUCT,
    TC_POINTER,
    TC_FUNCTION,
    TC_SAMPLER,
    TC_IMAGE,
    TC_SAMPLED_IMAGE,
} TypeCacheKind;

typedef struct TypeCacheEntry {
    TypeCacheKind kind;
    uint32_t      spv_id;

    union {
        struct { uint32_t width; uint32_t signedness; } int_type;
        struct { uint32_t width; } float_type;
        struct { uint32_t component_type_id; uint32_t count; } vector_type;
        struct { uint32_t column_type_id; uint32_t column_count; } matrix_type;
        struct { uint32_t element_type_id; uint32_t length_id; } array_type;
        struct { uint32_t element_type_id; } runtime_array_type;
        struct { const char *name; uint32_t *member_type_ids; uint32_t *member_offsets; int member_count; } struct_type;
        struct { SpvStorageClass storage_class; uint32_t pointee_type_id; } pointer_type;
        struct { uint32_t return_type_id; uint32_t *param_type_ids; int param_count; } function_type;
        struct {
            SpvDim dim;
            uint32_t sampled_type_id;
            uint32_t depth;
            uint32_t arrayed;
            uint32_t ms;
            uint32_t sampled;
            SpvImageFormat format;
        } image_type;
        struct { int comparison; } sampler_type;
    };

    struct TypeCacheEntry *next;
} TypeCacheEntry;

#define TYPE_CACHE_SIZE 256

typedef struct {
    TypeCacheEntry *buckets[TYPE_CACHE_SIZE];
} TypeCache;

//tc nonnull
static void type_cache_init(TypeCache *tc) {
    wgsl_compiler_assert(tc != NULL, "type_cache_init: tc is NULL");
    memset(tc->buckets, 0, sizeof(tc->buckets));
}

//tc nonnull
static void type_cache_free(TypeCache *tc) {
    wgsl_compiler_assert(tc != NULL, "type_cache_free: tc is NULL");
    for (int i = 0; i < TYPE_CACHE_SIZE; ++i) {
        TypeCacheEntry *e = tc->buckets[i];
        while (e) {
            TypeCacheEntry *next = e->next;
            if (e->kind == TC_STRUCT) {
                if (e->struct_type.member_type_ids) {
                    WGSL_FREE(e->struct_type.member_type_ids);
                }
                if (e->struct_type.member_offsets) {
                    WGSL_FREE(e->struct_type.member_offsets);
                }
            }
            if (e->kind == TC_FUNCTION && e->function_type.param_type_ids) {
                WGSL_FREE(e->function_type.param_type_ids);
            }
            WGSL_FREE(e);
            e = next;
        }
    }
}

//key nonnull
static uint32_t type_hash(TypeCacheKind kind, const void *key, size_t key_size) {
    wgsl_compiler_assert(key != NULL, "type_hash: key is NULL");
    uint32_t h = (uint32_t)kind * 31;
    const uint8_t *p = (const uint8_t*)key;
    for (size_t i = 0; i < key_size; ++i) {
        h = h * 31 + p[i];
    }
    return h % TYPE_CACHE_SIZE;
}

// ---------- SPIR-V Module Context ----------

struct WgslLower {
    const WgslAstNode     *program;
    const WgslResolver    *resolver;
    WgslLowerOptions       opts;

    SpvSections            sections;
    uint32_t               next_id;
    uint32_t               next_spec_id;

    // SSIR module (new IR-based approach)
    SsirModule            *ssir;

    // reflection
    WgslLowerModuleFeatures features;
    SpvCapability          cap_buf[8];
    int                    cap_count;
    const char            *ext_buf[4];
    int                    ext_count;

    WgslLowerEntrypointInfo *eps;
    int                      ep_count;

    char last_error[256];

    // Type cache
    TypeCache type_cache;

    // SSIR type ID mappings (SPV type ID -> SSIR type ID)
    uint32_t ssir_void;
    uint32_t ssir_bool;
    uint32_t ssir_i32;
    uint32_t ssir_u32;
    uint32_t ssir_f32;
    uint32_t ssir_f16;
    uint32_t ssir_vec2f;
    uint32_t ssir_vec3f;
    uint32_t ssir_vec4f;
    uint32_t ssir_vec2i;
    uint32_t ssir_vec3i;
    uint32_t ssir_vec4i;
    uint32_t ssir_vec2u;
    uint32_t ssir_vec3u;
    uint32_t ssir_vec4u;

    // Common type IDs
    uint32_t id_void;
    uint32_t id_bool;
    uint32_t id_i32;
    uint32_t id_u32;
    uint32_t id_f32;
    uint32_t id_f16;
    uint32_t id_vec2f;
    uint32_t id_vec3f;
    uint32_t id_vec4f;
    uint32_t id_vec2i;
    uint32_t id_vec3i;
    uint32_t id_vec4i;
    uint32_t id_vec2u;
    uint32_t id_vec3u;
    uint32_t id_vec4u;

    // Function types
    uint32_t id_fn_void_void;

    // GLSL.std.450 import
    uint32_t id_extinst_glsl;

    // Constants cache
    struct {
        uint32_t id;
        uint32_t type_id;
        union {
            int32_t i32_val;
            uint32_t u32_val;
            float f32_val;
            int bool_val;
        };
    } *const_cache;
    int const_cache_count;
    int const_cache_cap;

    // Global variable mapping
    struct {
        int symbol_id;
        uint32_t spv_id;
        uint32_t ssir_id;       // SSIR global variable ID
        uint32_t type_id;       // The element type (struct type for storage buffers)
        SpvStorageClass sc;     // Storage class
        const char *name;       // Variable name for lookup
    } *global_map;
    int global_map_count;
    int global_map_cap;

    // Override (specialization constant) cache - processed once, reused per function
    struct {
        const char *name;
        uint32_t spv_id;
        uint32_t type_id;
    } override_cache[32];
    int override_cache_count;

    // global_map count after lower_io_globals (shared globals boundary)
    int shared_globals_end;

    // Struct type cache - maps struct names to SPIR-V type IDs
    struct {
        const char *name;
        uint32_t spv_id;
        uint32_t inner_type_id; // The type ID without Block decoration (for storage buffers)
    } *struct_cache;
    int struct_cache_count;
    int struct_cache_cap;

    // SSIR ID mapping (SPV ID -> SSIR ID)
    uint32_t *ssir_id_map;
    uint32_t ssir_id_map_cap;

    // Per-function context
    struct {
        uint32_t func_id;
        uint32_t return_type_id;
        uint32_t label_id;
        int      has_returned;

        // Output variable for return value (for vertex/fragment)
        uint32_t output_var_id;
        uint32_t output_type_id;
        int      uses_struct_output;  // Flag for struct return type with multiple outputs

        // SSIR function context
        uint32_t ssir_func_id;
        uint32_t ssir_block_id;

        // Local variable context
        struct {
            struct {
                const char *name;
                uint32_t ptr_id;
                uint32_t type_id;
                uint32_t ssir_id;  // SSIR local variable ID
                int is_value;      // 1 if this is a let/const (value, not pointer)
            } *vars;
            int count;
            int cap;
        } locals;

        // Pending variable initializers (since all OpVariable must come first)
        struct {
            uint32_t var_id;
            const WgslAstNode *init_expr;
        } *pending_inits;
        int pending_init_count;
        int pending_init_cap;

        // Loop context for break/continue
        struct {
            uint32_t merge_block;
            uint32_t continue_block;
            uint32_t ssir_merge_block;
            uint32_t ssir_continue_block;
        } loop_stack[16];
        int loop_depth;

        // Flag to indicate we're in the variable declaration pass
        int collecting_vars;
    } fn_ctx;
};

static void set_error(WgslLower *l, const char *msg) {
    if (!l) return;
    size_t n = strlen(msg);
    if (n >= sizeof(l->last_error)) n = sizeof(l->last_error) - 1;
    memcpy(l->last_error, msg, n);
    l->last_error[n] = 0;
}

//l nonnull
static uint32_t fresh_id(WgslLower *l) { wgsl_compiler_assert(l != NULL, "fresh_id: l is NULL"); return l->next_id++; }

// ---------- SSIR ID Mapping ----------
// Maps SPIR-V IDs to SSIR IDs for values (constants, variables, expression results)

//l nonnull
static void ssir_id_map_set(WgslLower *l, uint32_t spv_id, uint32_t ssir_id) {
    wgsl_compiler_assert(l != NULL, "ssir_id_map_set: l is NULL");
    if (spv_id == 0) return;
    // Grow map if needed
    if (spv_id >= l->ssir_id_map_cap) {
        uint32_t new_cap = l->ssir_id_map_cap ? l->ssir_id_map_cap : 256;
        while (new_cap <= spv_id) new_cap *= 2;
        uint32_t *new_map = (uint32_t*)WGSL_REALLOC(l->ssir_id_map, new_cap * sizeof(uint32_t));
        if (!new_map) return;
        // Zero-initialize new entries
        memset(new_map + l->ssir_id_map_cap, 0, (new_cap - l->ssir_id_map_cap) * sizeof(uint32_t));
        l->ssir_id_map = new_map;
        l->ssir_id_map_cap = new_cap;
    }
    l->ssir_id_map[spv_id] = ssir_id;
}

//l nonnull
static uint32_t ssir_id_map_get(WgslLower *l, uint32_t spv_id) {
    wgsl_compiler_assert(l != NULL, "ssir_id_map_get: l is NULL");
    if (spv_id == 0 || spv_id >= l->ssir_id_map_cap) return 0;
    return l->ssir_id_map[spv_id];
}

// ---------- SPV Type to SSIR Type Conversion ----------

static SsirAddressSpace spv_sc_to_ssir_addr(SpvStorageClass sc) {
    switch (sc) {
        case SpvStorageClassFunction:        return SSIR_ADDR_FUNCTION;
        case SpvStorageClassPrivate:         return SSIR_ADDR_PRIVATE;
        case SpvStorageClassWorkgroup:       return SSIR_ADDR_WORKGROUP;
        case SpvStorageClassUniform:         return SSIR_ADDR_UNIFORM;
        case SpvStorageClassUniformConstant: return SSIR_ADDR_UNIFORM_CONSTANT;
        case SpvStorageClassStorageBuffer:   return SSIR_ADDR_STORAGE;
        case SpvStorageClassInput:           return SSIR_ADDR_INPUT;
        case SpvStorageClassOutput:          return SSIR_ADDR_OUTPUT;
        case SpvStorageClassPushConstant:    return SSIR_ADDR_PUSH_CONSTANT;
        default:                             return SSIR_ADDR_FUNCTION;
    }
}

//l nonnull
static uint32_t spv_type_to_ssir(WgslLower *l, uint32_t spv_type) {
    wgsl_compiler_assert(l != NULL, "spv_type_to_ssir: l is NULL");
    // Check well-known types first
    if (spv_type == l->id_void) return l->ssir_void;
    if (spv_type == l->id_bool) return l->ssir_bool;
    if (spv_type == l->id_i32) return l->ssir_i32;
    if (spv_type == l->id_u32) return l->ssir_u32;
    if (spv_type == l->id_f32) return l->ssir_f32;
    if (spv_type == l->id_f16) return l->ssir_f16;
    if (spv_type == l->id_vec2f) return l->ssir_vec2f;
    if (spv_type == l->id_vec3f) return l->ssir_vec3f;
    if (spv_type == l->id_vec4f) return l->ssir_vec4f;
    if (spv_type == l->id_vec2i) return l->ssir_vec2i;
    if (spv_type == l->id_vec3i) return l->ssir_vec3i;
    if (spv_type == l->id_vec4i) return l->ssir_vec4i;
    if (spv_type == l->id_vec2u) return l->ssir_vec2u;
    if (spv_type == l->id_vec3u) return l->ssir_vec3u;
    if (spv_type == l->id_vec4u) return l->ssir_vec4u;

    // Look up in type cache for composite types
    for (int bucket = 0; bucket < TYPE_CACHE_SIZE; ++bucket) {
        for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
            if (e->spv_id != spv_type) continue;

            switch (e->kind) {
                case TC_BOOL:
                    return l->ssir_bool;
                case TC_INT:
                    if (e->int_type.signedness)
                        return l->ssir_i32;
                    else
                        return l->ssir_u32;
                case TC_FLOAT:
                    if (e->float_type.width == 16)
                        return l->ssir_f16;
                    return l->ssir_f32;
                case TC_VECTOR: {
                    uint32_t elem = spv_type_to_ssir(l, e->vector_type.component_type_id);
                    return ssir_type_vec(l->ssir, elem, e->vector_type.count);
                }
                case TC_MATRIX: {
                    uint32_t col = spv_type_to_ssir(l, e->matrix_type.column_type_id);
                    // Find the column vector's size to get row count
                    uint8_t rows = 4; // default
                    for (int b2 = 0; b2 < TYPE_CACHE_SIZE; ++b2) {
                        for (TypeCacheEntry *e2 = l->type_cache.buckets[b2]; e2; e2 = e2->next) {
                            if (e2->spv_id == e->matrix_type.column_type_id && e2->kind == TC_VECTOR) {
                                rows = (uint8_t)e2->vector_type.count;
                                break;
                            }
                        }
                    }
                    return ssir_type_mat(l->ssir, col, e->matrix_type.column_count, rows);
                }
                case TC_ARRAY: {
                    uint32_t elem = spv_type_to_ssir(l, e->array_type.element_type_id);
                    // Note: length_id is a constant ID, not a literal - we'd need to look it up
                    // For now, use a placeholder length (this may need improvement)
                    return ssir_type_array(l->ssir, elem, 0);
                }
                case TC_RUNTIME_ARRAY: {
                    uint32_t elem = spv_type_to_ssir(l, e->runtime_array_type.element_type_id);
                    return ssir_type_runtime_array(l->ssir, elem);
                }
                case TC_STRUCT: {
                    // Convert struct members to SSIR types
                    int mcount = e->struct_type.member_count;
                    uint32_t *ssir_members = (uint32_t *)WGSL_MALLOC(mcount * sizeof(uint32_t));
                    if (!ssir_members) return l->ssir_f32;
                    for (int m = 0; m < mcount; ++m) {
                        ssir_members[m] = spv_type_to_ssir(l, e->struct_type.member_type_ids[m]);
                    }
                    // Pass member_offsets to SSIR for Block decoration support
                    uint32_t result = ssir_type_struct(l->ssir, e->struct_type.name, ssir_members, mcount, e->struct_type.member_offsets);
                    WGSL_FREE(ssir_members);
                    return result;
                }
                case TC_POINTER: {
                    uint32_t pointee = spv_type_to_ssir(l, e->pointer_type.pointee_type_id);
                    SsirAddressSpace space = spv_sc_to_ssir_addr(e->pointer_type.storage_class);
                    return ssir_type_ptr(l->ssir, pointee, space);
                }
                case TC_IMAGE: {
                    /* Map SpvDim to SsirTextureDim */
                    SsirTextureDim dim;
                    switch (e->image_type.dim) {
                        case SpvDim1D: dim = SSIR_TEX_1D; break;
                        case SpvDim2D:
                            if (e->image_type.arrayed)
                                dim = SSIR_TEX_2D_ARRAY;
                            else if (e->image_type.ms)
                                dim = SSIR_TEX_MULTISAMPLED_2D;
                            else
                                dim = SSIR_TEX_2D;
                            break;
                        case SpvDim3D: dim = SSIR_TEX_3D; break;
                        case SpvDimCube:
                            dim = e->image_type.arrayed ? SSIR_TEX_CUBE_ARRAY : SSIR_TEX_CUBE;
                            break;
                        default: dim = SSIR_TEX_2D; break;
                    }
                    if (e->image_type.depth) {
                        return ssir_type_texture_depth(l->ssir, dim);
                    } else if (e->image_type.sampled == 2) {
                        /* Storage image */
                        return ssir_type_texture_storage(l->ssir, dim,
                            (uint32_t)e->image_type.format, SSIR_ACCESS_READ_WRITE);
                    } else {
                        uint32_t sampled = spv_type_to_ssir(l, e->image_type.sampled_type_id);
                        return ssir_type_texture(l->ssir, dim, sampled);
                    }
                }
                case TC_SAMPLER:
                    if (e->sampler_type.comparison)
                        return ssir_type_sampler_comparison(l->ssir);
                    else
                        return ssir_type_sampler(l->ssir);
                default:
                    break;
            }
        }
    }

    // Default to f32 for truly unknown types
    return l->ssir_f32;
}

// ---------- SPIR-V Instruction Emission Helpers ----------

//wb nonnull
static int emit_op(WordBuf *wb, SpvOp op, size_t word_count) {
    wgsl_compiler_assert(wb != NULL, "emit_op: wb is NULL");
    return wb_push_u32(wb, ((uint32_t)word_count << 16) | (uint32_t)op);
}

//l nonnull
static int emit_capability(WgslLower *l, SpvCapability cap) {
    wgsl_compiler_assert(l != NULL, "emit_capability: l is NULL");
    WordBuf *wb = &l->sections.capabilities;
    if (!emit_op(wb, SpvOpCapability, 2)) return 0;
    if (!wb_push_u32(wb, cap)) return 0;
    // Track for reflection
    if (l->cap_count < 8) {
        l->cap_buf[l->cap_count++] = cap;
    }
    return 1;
}

//l nonnull
//name nonnull
static int emit_extension(WgslLower *l, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_extension: l is NULL");
    wgsl_compiler_assert(name != NULL, "emit_extension: name is NULL");
    WordBuf *wb = &l->sections.extensions;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name, &strw, &wn);
    if (!emit_op(wb, SpvOpExtension, 1 + wn)) { WGSL_FREE(strw); return 0; }
    int ok = wb_push_many(wb, strw, wn);
    WGSL_FREE(strw);
    return ok;
}

//l nonnull
//name nonnull
//out_id nonnull
static int emit_ext_inst_import(WgslLower *l, const char *name, uint32_t *out_id) {
    wgsl_compiler_assert(l != NULL, "emit_ext_inst_import: l is NULL");
    wgsl_compiler_assert(name != NULL, "emit_ext_inst_import: name is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_ext_inst_import: out_id is NULL");
    WordBuf *wb = &l->sections.ext_inst_imports;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name, &strw, &wn);
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpExtInstImport, 2 + wn)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, id)) { WGSL_FREE(strw); return 0; }
    int ok = wb_push_many(wb, strw, wn);
    WGSL_FREE(strw);
    *out_id = id;
    return ok;
}

//l nonnull
static int emit_memory_model(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_memory_model: l is NULL");
    WordBuf *wb = &l->sections.memory_model;
    if (!emit_op(wb, SpvOpMemoryModel, 3)) return 0;
    if (!wb_push_u32(wb, SpvAddressingModelLogical)) return 0;
    if (!wb_push_u32(wb, SpvMemoryModelGLSL450)) return 0;
    return 1;
}

//l nonnull
static int emit_name(WgslLower *l, uint32_t target, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_name: l is NULL");
    if (!l->opts.enable_debug_names || !name || !*name) return 1;
    WordBuf *wb = &l->sections.debug_names;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name, &strw, &wn);
    if (!emit_op(wb, SpvOpName, 2 + wn)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, target)) { WGSL_FREE(strw); return 0; }
    int ok = wb_push_many(wb, strw, wn);
    WGSL_FREE(strw);
    return ok;
}

//l nonnull
static int emit_member_name(WgslLower *l, uint32_t struct_id, uint32_t member, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_member_name: l is NULL");
    if (!l->opts.enable_debug_names || !name || !*name) return 1;
    WordBuf *wb = &l->sections.debug_names;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name, &strw, &wn);
    if (!emit_op(wb, SpvOpMemberName, 3 + wn)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, struct_id)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, member)) { WGSL_FREE(strw); return 0; }
    int ok = wb_push_many(wb, strw, wn);
    WGSL_FREE(strw);
    return ok;
}

//l nonnull
static int emit_decorate(WgslLower *l, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(l != NULL, "emit_decorate: l is NULL");
    WordBuf *wb = &l->sections.annotations;
    if (!emit_op(wb, SpvOpDecorate, 3 + lit_count)) return 0;
    if (!wb_push_u32(wb, target)) return 0;
    if (!wb_push_u32(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push_u32(wb, literals[i])) return 0;
    }
    return 1;
}

//l nonnull
static int emit_member_decorate(WgslLower *l, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(l != NULL, "emit_member_decorate: l is NULL");
    WordBuf *wb = &l->sections.annotations;
    if (!emit_op(wb, SpvOpMemberDecorate, 4 + lit_count)) return 0;
    if (!wb_push_u32(wb, struct_id)) return 0;
    if (!wb_push_u32(wb, member)) return 0;
    if (!wb_push_u32(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push_u32(wb, literals[i])) return 0;
    }
    return 1;
}

// ---------- Type Emission ----------

//l nonnull
static uint32_t emit_type_void(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_type_void: l is NULL");
    if (l->id_void) return l->id_void;
    WordBuf *wb = &l->sections.types_constants;
    l->id_void = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeVoid, 2)) return 0;
    if (!wb_push_u32(wb, l->id_void)) return 0;
    return l->id_void;
}

//l nonnull
static uint32_t emit_type_bool(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_type_bool: l is NULL");
    if (l->id_bool) return l->id_bool;
    WordBuf *wb = &l->sections.types_constants;
    l->id_bool = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeBool, 2)) return 0;
    if (!wb_push_u32(wb, l->id_bool)) return 0;
    return l->id_bool;
}

//l nonnull
static uint32_t emit_type_int(WgslLower *l, uint32_t width, uint32_t signedness) {
    wgsl_compiler_assert(l != NULL, "emit_type_int: l is NULL");
    // Check cache
    uint32_t key[2] = { width, signedness };
    uint32_t bucket = type_hash(TC_INT, key, sizeof(key));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_INT && e->int_type.width == width && e->int_type.signedness == signedness) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeInt, 4)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, width)) return 0;
    if (!wb_push_u32(wb, signedness)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_INT;
        e->spv_id = id;
        e->int_type.width = width;
        e->int_type.signedness = signedness;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    if (width == 32 && signedness == 1) l->id_i32 = id;
    if (width == 32 && signedness == 0) l->id_u32 = id;

    return id;
}

//l nonnull
static uint32_t emit_type_float(WgslLower *l, uint32_t width) {
    wgsl_compiler_assert(l != NULL, "emit_type_float: l is NULL");
    // Check cache
    uint32_t bucket = type_hash(TC_FLOAT, &width, sizeof(width));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_FLOAT && e->float_type.width == width) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeFloat, 3)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, width)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_FLOAT;
        e->spv_id = id;
        e->float_type.width = width;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    if (width == 32) l->id_f32 = id;
    if (width == 16) l->id_f16 = id;

    return id;
}

//l nonnull
static uint32_t emit_type_vector(WgslLower *l, uint32_t component_type, uint32_t count) {
    wgsl_compiler_assert(l != NULL, "emit_type_vector: l is NULL");
    // Check cache
    uint32_t key[2] = { component_type, count };
    uint32_t bucket = type_hash(TC_VECTOR, key, sizeof(key));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_VECTOR && e->vector_type.component_type_id == component_type && e->vector_type.count == count) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeVector, 4)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, component_type)) return 0;
    if (!wb_push_u32(wb, count)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_VECTOR;
        e->spv_id = id;
        e->vector_type.component_type_id = component_type;
        e->vector_type.count = count;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    // Cache common vector types
    if (component_type == l->id_f32) {
        if (count == 2) l->id_vec2f = id;
        if (count == 3) l->id_vec3f = id;
        if (count == 4) l->id_vec4f = id;
    }
    if (component_type == l->id_i32) {
        if (count == 2) l->id_vec2i = id;
        if (count == 3) l->id_vec3i = id;
        if (count == 4) l->id_vec4i = id;
    }
    if (component_type == l->id_u32) {
        if (count == 2) l->id_vec2u = id;
        if (count == 3) l->id_vec3u = id;
        if (count == 4) l->id_vec4u = id;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_matrix(WgslLower *l, uint32_t column_type, uint32_t column_count) {
    wgsl_compiler_assert(l != NULL, "emit_type_matrix: l is NULL");
    // Check cache
    uint32_t key[2] = { column_type, column_count };
    uint32_t bucket = type_hash(TC_MATRIX, key, sizeof(key));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_MATRIX && e->matrix_type.column_type_id == column_type && e->matrix_type.column_count == column_count) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeMatrix, 4)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, column_type)) return 0;
    if (!wb_push_u32(wb, column_count)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_MATRIX;
        e->spv_id = id;
        e->matrix_type.column_type_id = column_type;
        e->matrix_type.column_count = column_count;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_array(WgslLower *l, uint32_t element_type, uint32_t length_id) {
    wgsl_compiler_assert(l != NULL, "emit_type_array: l is NULL");
    // Check cache
    uint32_t key[2] = { element_type, length_id };
    uint32_t bucket = type_hash(TC_ARRAY, key, sizeof(key));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_ARRAY && e->array_type.element_type_id == element_type && e->array_type.length_id == length_id) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeArray, 4)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, element_type)) return 0;
    if (!wb_push_u32(wb, length_id)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_ARRAY;
        e->spv_id = id;
        e->array_type.element_type_id = element_type;
        e->array_type.length_id = length_id;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_runtime_array(WgslLower *l, uint32_t element_type) {
    wgsl_compiler_assert(l != NULL, "emit_type_runtime_array: l is NULL");
    // Check cache
    uint32_t bucket = type_hash(TC_RUNTIME_ARRAY, &element_type, sizeof(element_type));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_RUNTIME_ARRAY && e->runtime_array_type.element_type_id == element_type) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeRuntimeArray, 3)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, element_type)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_RUNTIME_ARRAY;
        e->spv_id = id;
        e->runtime_array_type.element_type_id = element_type;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    return id;
}

//l nonnull
//member_types nonnull
static uint32_t emit_type_struct_with_offsets(WgslLower *l, const char *name, uint32_t *member_types, int member_count, uint32_t *offsets) {
    wgsl_compiler_assert(l != NULL, "emit_type_struct_with_offsets: l is NULL");
    wgsl_compiler_assert(member_types != NULL, "emit_type_struct_with_offsets: member_types is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeStruct, 2 + member_count)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    for (int i = 0; i < member_count; ++i) {
        if (!wb_push_u32(wb, member_types[i])) return 0;
    }
    emit_name(l, id, name);

    // Add to type cache so spv_type_to_ssir can find it
    uint32_t bucket = type_hash(TC_STRUCT, &id, sizeof(id)) % TYPE_CACHE_SIZE;
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_STRUCT;
        e->spv_id = id;
        e->struct_type.name = name;
        e->struct_type.member_type_ids = (uint32_t*)WGSL_MALLOC(member_count * sizeof(uint32_t));
        if (e->struct_type.member_type_ids) {
            memcpy(e->struct_type.member_type_ids, member_types, member_count * sizeof(uint32_t));
        }
        e->struct_type.member_offsets = NULL;
        if (offsets) {
            e->struct_type.member_offsets = (uint32_t*)WGSL_MALLOC(member_count * sizeof(uint32_t));
            if (e->struct_type.member_offsets) {
                memcpy(e->struct_type.member_offsets, offsets, member_count * sizeof(uint32_t));
            }
        }
        e->struct_type.member_count = member_count;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    return id;
}

//l nonnull
//member_types nonnull
static uint32_t emit_type_struct(WgslLower *l, const char *name, uint32_t *member_types, int member_count) {
    wgsl_compiler_assert(l != NULL, "emit_type_struct: l is NULL");
    wgsl_compiler_assert(member_types != NULL, "emit_type_struct: member_types is NULL");
    return emit_type_struct_with_offsets(l, name, member_types, member_count, NULL);
}

//l nonnull
static uint32_t emit_type_pointer(WgslLower *l, SpvStorageClass storage_class, uint32_t pointee_type) {
    wgsl_compiler_assert(l != NULL, "emit_type_pointer: l is NULL");
    // Check cache
    uint32_t key[2] = { storage_class, pointee_type };
    uint32_t bucket = type_hash(TC_POINTER, key, sizeof(key));
    for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
        if (e->kind == TC_POINTER && e->pointer_type.storage_class == storage_class && e->pointer_type.pointee_type_id == pointee_type) {
            return e->spv_id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypePointer, 4)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, storage_class)) return 0;
    if (!wb_push_u32(wb, pointee_type)) return 0;

    // Cache
    TypeCacheEntry *e = (TypeCacheEntry*)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (e) {
        e->kind = TC_POINTER;
        e->spv_id = id;
        e->pointer_type.storage_class = storage_class;
        e->pointer_type.pointee_type_id = pointee_type;
        e->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = e;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_function(WgslLower *l, uint32_t return_type, uint32_t *param_types, int param_count) {
    wgsl_compiler_assert(l != NULL, "emit_type_function: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeFunction, 3 + param_count)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, return_type)) return 0;
    for (int i = 0; i < param_count; ++i) {
        if (!wb_push_u32(wb, param_types[i])) return 0;
    }
    return id;
}

//l nonnull
static uint32_t emit_type_sampler_with_comparison(WgslLower *l, int comparison) {
    wgsl_compiler_assert(l != NULL, "emit_type_sampler_with_comparison: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeSampler, 2)) return 0;
    if (!wb_push_u32(wb, id)) return 0;

    // Add to type cache
    TypeCacheEntry *entry = (TypeCacheEntry *)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (entry) {
        entry->kind = TC_SAMPLER;
        entry->spv_id = id;
        entry->sampler_type.comparison = comparison;
        uint32_t bucket = type_hash(TC_SAMPLER, &comparison, sizeof(comparison));
        entry->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = entry;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_sampler(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_type_sampler: l is NULL");
    return emit_type_sampler_with_comparison(l, 0);
}

//l nonnull
static uint32_t emit_type_image(WgslLower *l, uint32_t sampled_type, SpvDim dim, uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, SpvImageFormat format) {
    wgsl_compiler_assert(l != NULL, "emit_type_image: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeImage, 9)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, sampled_type)) return 0;
    if (!wb_push_u32(wb, dim)) return 0;
    if (!wb_push_u32(wb, depth)) return 0;
    if (!wb_push_u32(wb, arrayed)) return 0;
    if (!wb_push_u32(wb, ms)) return 0;
    if (!wb_push_u32(wb, sampled)) return 0;
    if (!wb_push_u32(wb, format)) return 0;

    // Add to type cache so spv_type_to_ssir can find it
    TypeCacheEntry *entry = (TypeCacheEntry *)WGSL_MALLOC(sizeof(TypeCacheEntry));
    if (entry) {
        entry->kind = TC_IMAGE;
        entry->spv_id = id;
        entry->image_type.dim = dim;
        entry->image_type.sampled_type_id = sampled_type;
        entry->image_type.depth = depth;
        entry->image_type.arrayed = arrayed;
        entry->image_type.ms = ms;
        entry->image_type.sampled = sampled;
        entry->image_type.format = format;
        uint32_t key[7] = { sampled_type, (uint32_t)dim, depth, arrayed, ms, sampled, (uint32_t)format };
        uint32_t bucket = type_hash(TC_IMAGE, key, sizeof(key));
        entry->next = l->type_cache.buckets[bucket];
        l->type_cache.buckets[bucket] = entry;
    }

    return id;
}

//l nonnull
static uint32_t emit_type_sampled_image(WgslLower *l, uint32_t image_type) {
    wgsl_compiler_assert(l != NULL, "emit_type_sampled_image: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpTypeSampledImage, 3)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, image_type)) return 0;
    return id;
}

// ---------- Constant Emission ----------

//l nonnull
static uint32_t emit_const_bool(WgslLower *l, int value) {
    wgsl_compiler_assert(l != NULL, "emit_const_bool: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t bool_type = emit_type_bool(l);
    if (!emit_op(wb, value ? SpvOpConstantTrue : SpvOpConstantFalse, 3)) return 0;
    if (!wb_push_u32(wb, bool_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    // Create SSIR constant and map ID
    uint32_t ssir_const = ssir_const_bool(l->ssir, value ? true : false);
    ssir_id_map_set(l, id, ssir_const);
    return id;
}

//l nonnull
static uint32_t emit_const_i32(WgslLower *l, int32_t value) {
    wgsl_compiler_assert(l != NULL, "emit_const_i32: l is NULL");
    // Check cache
    for (int i = 0; i < l->const_cache_count; ++i) {
        if (l->const_cache[i].type_id == l->id_i32 && l->const_cache[i].i32_val == value) {
            return l->const_cache[i].id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_int(l, 32, 1);
    if (!emit_op(wb, SpvOpConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, (uint32_t)value)) return 0;

    // Create SSIR constant and map ID
    uint32_t ssir_const = ssir_const_i32(l->ssir, value);
    ssir_id_map_set(l, id, ssir_const);

    // Cache
    if (l->const_cache_count >= l->const_cache_cap) {
        int new_cap = l->const_cache_cap ? l->const_cache_cap * 2 : 32;
        void *p = WGSL_REALLOC(l->const_cache, new_cap * sizeof(l->const_cache[0]));
        if (p) {
            l->const_cache = (__typeof__(l->const_cache))p;
            l->const_cache_cap = new_cap;
        }
    }
    if (l->const_cache_count < l->const_cache_cap) {
        l->const_cache[l->const_cache_count].id = id;
        l->const_cache[l->const_cache_count].type_id = type;
        l->const_cache[l->const_cache_count].i32_val = value;
        l->const_cache_count++;
    }

    return id;
}

//l nonnull
static uint32_t emit_const_u32(WgslLower *l, uint32_t value) {
    wgsl_compiler_assert(l != NULL, "emit_const_u32: l is NULL");
    // Check cache
    for (int i = 0; i < l->const_cache_count; ++i) {
        if (l->const_cache[i].type_id == l->id_u32 && l->const_cache[i].u32_val == value) {
            return l->const_cache[i].id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_int(l, 32, 0);
    if (!emit_op(wb, SpvOpConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, value)) return 0;

    // Create SSIR constant and map ID
    uint32_t ssir_const = ssir_const_u32(l->ssir, value);
    ssir_id_map_set(l, id, ssir_const);

    // Cache
    if (l->const_cache_count >= l->const_cache_cap) {
        int new_cap = l->const_cache_cap ? l->const_cache_cap * 2 : 32;
        void *p = WGSL_REALLOC(l->const_cache, new_cap * sizeof(l->const_cache[0]));
        if (p) {
            l->const_cache = (__typeof__(l->const_cache))p;
            l->const_cache_cap = new_cap;
        }
    }
    if (l->const_cache_count < l->const_cache_cap) {
        l->const_cache[l->const_cache_count].id = id;
        l->const_cache[l->const_cache_count].type_id = type;
        l->const_cache[l->const_cache_count].u32_val = value;
        l->const_cache_count++;
    }

    return id;
}

//l nonnull
static uint32_t emit_const_f32(WgslLower *l, float value) {
    wgsl_compiler_assert(l != NULL, "emit_const_f32: l is NULL");
    // Check cache
    uint32_t bits_check;
    memcpy(&bits_check, &value, sizeof(bits_check));
    for (int i = 0; i < l->const_cache_count; ++i) {
        if (l->const_cache[i].type_id == l->id_f32) {
            uint32_t cached_bits;
            memcpy(&cached_bits, &l->const_cache[i].f32_val, sizeof(cached_bits));
            if (cached_bits == bits_check) return l->const_cache[i].id;
        }
    }

    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_float(l, 32);
    if (!emit_op(wb, SpvOpConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, bits_check)) return 0;

    // Create SSIR constant and map ID
    uint32_t ssir_const = ssir_const_f32(l->ssir, value);
    ssir_id_map_set(l, id, ssir_const);

    // Cache
    if (l->const_cache_count >= l->const_cache_cap) {
        int new_cap = l->const_cache_cap ? l->const_cache_cap * 2 : 32;
        void *p = WGSL_REALLOC(l->const_cache, new_cap * sizeof(l->const_cache[0]));
        if (p) {
            l->const_cache = (__typeof__(l->const_cache))p;
            l->const_cache_cap = new_cap;
        }
    }
    if (l->const_cache_count < l->const_cache_cap) {
        l->const_cache[l->const_cache_count].id = id;
        l->const_cache[l->const_cache_count].type_id = type;
        l->const_cache[l->const_cache_count].f32_val = value;
        l->const_cache_count++;
    }

    return id;
}

// ---------- Specialization Constant Emission ----------

//l nonnull
static uint32_t emit_spec_const_f32(WgslLower *l, float value, uint32_t spec_id, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_spec_const_f32: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_float(l, 32);
    uint32_t bits;
    memcpy(&bits, &value, sizeof(float));
    if (!emit_op(wb, SpvOpSpecConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, bits)) return 0;
    emit_decorate(l, id, SpvDecorationSpecId, &spec_id, 1);
    // Always emit name for spec constants (needed for reflection)
    if (name && *name) {
        WordBuf *dbg = &l->sections.debug_names;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit(name, &strw, &wn);
        if (emit_op(dbg, SpvOpName, 2 + wn)) {
            wb_push_u32(dbg, id);
            wb_push_many(dbg, strw, wn);
        }
        WGSL_FREE(strw);
    }
    uint32_t ssir_const = ssir_const_spec_f32(l->ssir, value, spec_id);
    ssir_id_map_set(l, id, ssir_const);
    return id;
}

//l nonnull
static uint32_t emit_spec_const_i32(WgslLower *l, int32_t value, uint32_t spec_id, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_spec_const_i32: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_int(l, 32, 1);
    if (!emit_op(wb, SpvOpSpecConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, (uint32_t)value)) return 0;
    emit_decorate(l, id, SpvDecorationSpecId, &spec_id, 1);
    if (name && *name) {
        WordBuf *dbg = &l->sections.debug_names;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit(name, &strw, &wn);
        if (emit_op(dbg, SpvOpName, 2 + wn)) {
            wb_push_u32(dbg, id);
            wb_push_many(dbg, strw, wn);
        }
        WGSL_FREE(strw);
    }
    uint32_t ssir_const = ssir_const_spec_i32(l->ssir, value, spec_id);
    ssir_id_map_set(l, id, ssir_const);
    return id;
}

//l nonnull
static uint32_t emit_spec_const_u32(WgslLower *l, uint32_t value, uint32_t spec_id, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_spec_const_u32: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t type = emit_type_int(l, 32, 0);
    if (!emit_op(wb, SpvOpSpecConstant, 4)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, value)) return 0;
    emit_decorate(l, id, SpvDecorationSpecId, &spec_id, 1);
    if (name && *name) {
        WordBuf *dbg = &l->sections.debug_names;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit(name, &strw, &wn);
        if (emit_op(dbg, SpvOpName, 2 + wn)) {
            wb_push_u32(dbg, id);
            wb_push_many(dbg, strw, wn);
        }
        WGSL_FREE(strw);
    }
    uint32_t ssir_const = ssir_const_spec_u32(l->ssir, value, spec_id);
    ssir_id_map_set(l, id, ssir_const);
    return id;
}

//l nonnull
static uint32_t emit_spec_const_bool(WgslLower *l, int value, uint32_t spec_id, const char *name) {
    wgsl_compiler_assert(l != NULL, "emit_spec_const_bool: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    uint32_t bool_type = emit_type_bool(l);
    if (!emit_op(wb, value ? SpvOpSpecConstantTrue : SpvOpSpecConstantFalse, 3)) return 0;
    if (!wb_push_u32(wb, bool_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    emit_decorate(l, id, SpvDecorationSpecId, &spec_id, 1);
    if (name && *name) {
        WordBuf *dbg = &l->sections.debug_names;
        uint32_t *strw = NULL; size_t wn = 0;
        make_string_lit(name, &strw, &wn);
        if (emit_op(dbg, SpvOpName, 2 + wn)) {
            wb_push_u32(dbg, id);
            wb_push_many(dbg, strw, wn);
        }
        WGSL_FREE(strw);
    }
    uint32_t ssir_const = ssir_const_spec_bool(l->ssir, value ? true : false, spec_id);
    ssir_id_map_set(l, id, ssir_const);
    return id;
}

//l nonnull
//constituents nonnull
static uint32_t emit_const_composite(WgslLower *l, uint32_t type, uint32_t *constituents, int count) {
    wgsl_compiler_assert(l != NULL, "emit_const_composite: l is NULL");
    wgsl_compiler_assert(constituents != NULL, "emit_const_composite: constituents is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpConstantComposite, 3 + count)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    for (int i = 0; i < count; ++i) {
        if (!wb_push_u32(wb, constituents[i])) return 0;
    }
    return id;
}

//l nonnull
static uint32_t emit_const_null(WgslLower *l, uint32_t type) {
    wgsl_compiler_assert(l != NULL, "emit_const_null: l is NULL");
    WordBuf *wb = &l->sections.types_constants;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpConstantNull, 3)) return 0;
    if (!wb_push_u32(wb, type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    return id;
}

// ---------- Type Lowering from AST ----------

static SpvStorageClass wgsl_address_space_to_storage_class(const char *space) {
    if (!space || !*space) return SpvStorageClassFunction;
    if (strcmp(space, "uniform") == 0) return SpvStorageClassUniform;
    if (strcmp(space, "storage") == 0) return SpvStorageClassStorageBuffer;
    if (strcmp(space, "private") == 0) return SpvStorageClassPrivate;
    if (strcmp(space, "workgroup") == 0) return SpvStorageClassWorkgroup;
    if (strcmp(space, "function") == 0) return SpvStorageClassFunction;
    if (strcmp(space, "in") == 0) return SpvStorageClassInput;
    if (strcmp(space, "out") == 0) return SpvStorageClassOutput;
    return SpvStorageClassUniformConstant; // for samplers/textures
}

static uint32_t lower_type(WgslLower *l, const WgslAstNode *type_node);

//l nonnull
//name nonnull
static uint32_t lower_scalar_type(WgslLower *l, const char *name) {
    wgsl_compiler_assert(l != NULL, "lower_scalar_type: l is NULL");
    wgsl_compiler_assert(name != NULL, "lower_scalar_type: name is NULL");
    if (strcmp(name, "void") == 0) return emit_type_void(l);
    if (strcmp(name, "bool") == 0) return emit_type_bool(l);
    if (strcmp(name, "i32") == 0) return emit_type_int(l, 32, 1);
    if (strcmp(name, "u32") == 0) return emit_type_int(l, 32, 0);
    if (strcmp(name, "f32") == 0) return emit_type_float(l, 32);
    if (strcmp(name, "f16") == 0) return emit_type_float(l, 16);
    return 0;
}

//l nonnull
static uint32_t lower_type(WgslLower *l, const WgslAstNode *type_node) {
    wgsl_compiler_assert(l != NULL, "lower_type: l is NULL");
    if (!type_node) return emit_type_void(l);
    if (type_node->type != WGSL_NODE_TYPE) return 0;

    const TypeNode *tn = &type_node->type_node;
    const char *name = tn->name;

    // Scalar types
    uint32_t scalar = lower_scalar_type(l, name);
    if (scalar) return scalar;

    // Vector types
    if (strcmp(name, "vec2") == 0 || strcmp(name, "vec2f") == 0) {
        uint32_t elem = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (strcmp(name, "vec2") == 0 && tn->type_arg_count > 0) {
            elem = lower_type(l, tn->type_args[0]);
        }
        return emit_type_vector(l, elem, 2);
    }
    if (strcmp(name, "vec3") == 0 || strcmp(name, "vec3f") == 0) {
        uint32_t elem = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (strcmp(name, "vec3") == 0 && tn->type_arg_count > 0) {
            elem = lower_type(l, tn->type_args[0]);
        }
        return emit_type_vector(l, elem, 3);
    }
    if (strcmp(name, "vec4") == 0 || strcmp(name, "vec4f") == 0) {
        uint32_t elem = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (strcmp(name, "vec4") == 0 && tn->type_arg_count > 0) {
            elem = lower_type(l, tn->type_args[0]);
        }
        return emit_type_vector(l, elem, 4);
    }
    if (strcmp(name, "vec2i") == 0) {
        uint32_t elem = l->id_i32 ? l->id_i32 : emit_type_int(l, 32, 1);
        return emit_type_vector(l, elem, 2);
    }
    if (strcmp(name, "vec3i") == 0) {
        uint32_t elem = l->id_i32 ? l->id_i32 : emit_type_int(l, 32, 1);
        return emit_type_vector(l, elem, 3);
    }
    if (strcmp(name, "vec4i") == 0) {
        uint32_t elem = l->id_i32 ? l->id_i32 : emit_type_int(l, 32, 1);
        return emit_type_vector(l, elem, 4);
    }
    if (strcmp(name, "vec2u") == 0) {
        uint32_t elem = l->id_u32 ? l->id_u32 : emit_type_int(l, 32, 0);
        return emit_type_vector(l, elem, 2);
    }
    if (strcmp(name, "vec3u") == 0) {
        uint32_t elem = l->id_u32 ? l->id_u32 : emit_type_int(l, 32, 0);
        return emit_type_vector(l, elem, 3);
    }
    if (strcmp(name, "vec4u") == 0) {
        uint32_t elem = l->id_u32 ? l->id_u32 : emit_type_int(l, 32, 0);
        return emit_type_vector(l, elem, 4);
    }

    // Matrix types
    if (strncmp(name, "mat", 3) == 0) {
        int cols = 0, rows = 0;
        // Parse matCxR or matCxRf patterns
        if (sscanf(name, "mat%dx%d", &cols, &rows) == 2 ||
            sscanf(name, "mat%dx%df", &cols, &rows) == 2) {
            uint32_t f32_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
            uint32_t col_type = emit_type_vector(l, f32_type, rows);
            return emit_type_matrix(l, col_type, cols);
        }
    }

    // Array types
    if (strcmp(name, "array") == 0) {
        if (tn->type_arg_count > 0) {
            uint32_t elem = lower_type(l, tn->type_args[0]);
            if (tn->expr_arg_count > 0 && tn->expr_args[0]->type == WGSL_NODE_LITERAL) {
                // Fixed-size array
                const char *lex = tn->expr_args[0]->literal.lexeme;
                int len = lex ? atoi(lex) : 0;
                uint32_t len_id = emit_const_u32(l, len);
                return emit_type_array(l, elem, len_id);
            } else {
                // Runtime array
                return emit_type_runtime_array(l, elem);
            }
        }
    }

    // Sampler types
    if (strcmp(name, "sampler") == 0) {
        return emit_type_sampler(l);
    }
    if (strcmp(name, "sampler_comparison") == 0) {
        return emit_type_sampler_with_comparison(l, 1);
    }

    // Texture types
    if (strcmp(name, "texture_1d") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (tn->type_arg_count > 0) sampled_type = lower_type(l, tn->type_args[0]);
        return emit_type_image(l, sampled_type, SpvDim1D, 0, 0, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_2d") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (tn->type_arg_count > 0) sampled_type = lower_type(l, tn->type_args[0]);
        return emit_type_image(l, sampled_type, SpvDim2D, 0, 0, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_3d") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (tn->type_arg_count > 0) sampled_type = lower_type(l, tn->type_args[0]);
        return emit_type_image(l, sampled_type, SpvDim3D, 0, 0, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_cube") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (tn->type_arg_count > 0) sampled_type = lower_type(l, tn->type_args[0]);
        return emit_type_image(l, sampled_type, SpvDimCube, 0, 0, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_2d_array") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        if (tn->type_arg_count > 0) sampled_type = lower_type(l, tn->type_args[0]);
        return emit_type_image(l, sampled_type, SpvDim2D, 0, 1, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_depth_2d") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        return emit_type_image(l, sampled_type, SpvDim2D, 1, 0, 0, 1, SpvImageFormatUnknown);
    }
    if (strcmp(name, "texture_storage_2d") == 0) {
        uint32_t sampled_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
        // Storage textures use format from first type arg
        return emit_type_image(l, sampled_type, SpvDim2D, 0, 0, 0, 2, SpvImageFormatRgba8);
    }

    // Check if this is a struct type
    for (int i = 0; i < l->struct_cache_count; ++i) {
        if (strcmp(l->struct_cache[i].name, name) == 0) {
            return l->struct_cache[i].spv_id;
        }
    }

    // Unknown type - return void
    return emit_type_void(l);
}

// ---------- Struct Type Lowering ----------

// Compute alignment and size for a type name (std430 layout rules)
//type_name nonnull
//out_size nonnull
//out_align nonnull
static void get_type_layout(const char *type_name, int *out_size, int *out_align) {
    wgsl_compiler_assert(type_name != NULL, "get_type_layout: type_name is NULL");
    wgsl_compiler_assert(out_size != NULL, "get_type_layout: out_size is NULL");
    wgsl_compiler_assert(out_align != NULL, "get_type_layout: out_align is NULL");
    // Default to 4-byte alignment
    int size = 4, align_val = 4;

    // Scalar types
    if (strcmp(type_name, "bool") == 0 || strcmp(type_name, "i32") == 0 ||
        strcmp(type_name, "u32") == 0 || strcmp(type_name, "f32") == 0) {
        size = 4; align_val = 4;
    } else if (strcmp(type_name, "f16") == 0) {
        size = 2; align_val = 2;
    }
    // Vector types
    else if (strcmp(type_name, "vec2") == 0 || strcmp(type_name, "vec2f") == 0 ||
             strcmp(type_name, "vec2i") == 0 || strcmp(type_name, "vec2u") == 0) {
        size = 8; align_val = 8;
    } else if (strcmp(type_name, "vec3") == 0 || strcmp(type_name, "vec3f") == 0 ||
               strcmp(type_name, "vec3i") == 0 || strcmp(type_name, "vec3u") == 0) {
        size = 12; align_val = 16;  // vec3 aligns to 16 in std430
    } else if (strcmp(type_name, "vec4") == 0 || strcmp(type_name, "vec4f") == 0 ||
               strcmp(type_name, "vec4i") == 0 || strcmp(type_name, "vec4u") == 0) {
        size = 16; align_val = 16;
    }
    // Matrix types (column-major, each column is a vec)
    else if (strncmp(type_name, "mat", 3) == 0) {
        int cols = 0, rows = 0;
        if (sscanf(type_name, "mat%dx%d", &cols, &rows) == 2 ||
            sscanf(type_name, "mat%dx%df", &cols, &rows) == 2) {
            // Column alignment depends on rows (vec2=8, vec3/vec4=16)
            int col_align = (rows == 2) ? 8 : 16;
            int col_size = (rows == 2) ? 8 : (rows == 3 ? 16 : 16);  // vec3 padded to 16
            size = col_size * cols;
            align_val = col_align;
        }
    }
    // Array type - runtime array has indeterminate size
    else if (strcmp(type_name, "array") == 0) {
        // Runtime arrays inherit element alignment, size is unknown
        size = 0; align_val = 4;
    }

    *out_size = size;
    *out_align = align_val;
}

// Helper to add struct to cache
//l nonnull
static int add_struct_to_cache(WgslLower *l, const char *name, uint32_t spv_id, uint32_t inner_type_id) {
    wgsl_compiler_assert(l != NULL, "add_struct_to_cache: l is NULL");
    if (l->struct_cache_count >= l->struct_cache_cap) {
        int new_cap = l->struct_cache_cap ? l->struct_cache_cap * 2 : 16;
        void *p = WGSL_REALLOC(l->struct_cache, new_cap * sizeof(l->struct_cache[0]));
        if (!p) return 0;
        l->struct_cache = (__typeof__(l->struct_cache))p;
        l->struct_cache_cap = new_cap;
    }
    l->struct_cache[l->struct_cache_count].name = name;
    l->struct_cache[l->struct_cache_count].spv_id = spv_id;
    l->struct_cache[l->struct_cache_count].inner_type_id = inner_type_id;
    l->struct_cache_count++;
    return 1;
}

// Forward declarations for type helpers used by lower_structs
static int get_vector_component_count(uint32_t type_id, WgslLower *l);
static int is_matrix_type(uint32_t type_id, WgslLower *l);
static int get_matrix_info(uint32_t type_id, WgslLower *l, uint32_t *out_col_type, int *out_col_count);

// Lower all struct declarations in the program
//l nonnull
static int lower_structs(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_structs: l is NULL");
    if (!l->program || l->program->type != WGSL_NODE_PROGRAM) return 1;

    const Program *prog = &l->program->program;

    for (int d = 0; d < prog->decl_count; ++d) {
        const WgslAstNode *decl = prog->decls[d];
        if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

        const StructDecl *sd = &decl->struct_decl;
        if (!sd->name || sd->field_count == 0) continue;

        // Collect member types and compute offsets
        uint32_t *member_types = (uint32_t*)WGSL_MALLOC(sd->field_count * sizeof(uint32_t));
        if (!member_types) return 0;

        int current_offset = 0;
        int *offsets = (int*)WGSL_MALLOC(sd->field_count * sizeof(int));
        if (!offsets) { WGSL_FREE(member_types); return 0; }

        for (int f = 0; f < sd->field_count; ++f) {
            const WgslAstNode *field_node = sd->fields[f];
            if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) {
                member_types[f] = emit_type_void(l);
                offsets[f] = current_offset;
                continue;
            }

            const StructField *field = &field_node->struct_field;

            // Lower the field type
            uint32_t field_type = lower_type(l, field->type);
            member_types[f] = field_type;

            // Get layout info for offset calculation
            int field_size = 4, field_align = 4;
            if (field->type && field->type->type == WGSL_NODE_TYPE) {
                const char *type_name = field->type->type_node.name;
                if (!type_name) type_name = "";

                // Check for runtime array
                if (strcmp(type_name, "array") == 0) {
                    // Runtime array - must be last member
                    // Inherit alignment from element type
                    const TypeNode *tn = &field->type->type_node;
                    if (tn->type_arg_count > 0 && tn->type_args[0]->type == WGSL_NODE_TYPE) {
                        get_type_layout(tn->type_args[0]->type_node.name, &field_size, &field_align);
                    }
                    field_size = 0;  // Runtime array size is indeterminate
                } else {
                    get_type_layout(type_name, &field_size, &field_align);
                }
            }

            // Align current offset
            if (field_align > 0) {
                current_offset = (current_offset + field_align - 1) & ~(field_align - 1);
            }
            offsets[f] = current_offset;
            current_offset += field_size;
        }

        // Convert offsets to uint32_t array for the function
        uint32_t *uint_offsets = (uint32_t*)WGSL_MALLOC(sd->field_count * sizeof(uint32_t));
        if (!uint_offsets) { WGSL_FREE(offsets); WGSL_FREE(member_types); return 0; }
        for (int f = 0; f < sd->field_count; ++f) {
            uint_offsets[f] = (uint32_t)offsets[f];
        }

        // Emit the struct type with offsets stored in cache
        uint32_t struct_id = emit_type_struct_with_offsets(l, sd->name, member_types, sd->field_count, uint_offsets);
        WGSL_FREE(uint_offsets);

        // Add Block decoration for storage buffer compatibility
        emit_decorate(l, struct_id, SpvDecorationBlock, NULL, 0);

        // Add member offset decorations
        for (int f = 0; f < sd->field_count; ++f) {
            uint32_t offset_val = (uint32_t)offsets[f];
            emit_member_decorate(l, struct_id, f, SpvDecorationOffset, &offset_val, 1);

            // Add member name
            const WgslAstNode *field_node = sd->fields[f];
            if (field_node && field_node->type == WGSL_NODE_STRUCT_FIELD) {
                emit_member_name(l, struct_id, f, field_node->struct_field.name);
            }
        }

        // Add ColMajor + MatrixStride decorations for matrix members
        for (int f = 0; f < sd->field_count; ++f) {
            if (is_matrix_type(member_types[f], l)) {
                emit_member_decorate(l, struct_id, f, SpvDecorationColMajor, NULL, 0);
                uint32_t col_type;
                int col_count;
                if (get_matrix_info(member_types[f], l, &col_type, &col_count)) {
                    int vec_count = get_vector_component_count(col_type, l);
                    // MatrixStride = column vector size rounded up to vec4 alignment (16 bytes)
                    uint32_t stride = (uint32_t)(vec_count * 4);
                    if (stride < 16) stride = 16;  // std140: columns are vec4-aligned
                    emit_member_decorate(l, struct_id, f, SpvDecorationMatrixStride, &stride, 1);
                }
            }
        }

        // Add ArrayStride decoration for runtime array members
        for (int f = 0; f < sd->field_count; ++f) {
            const WgslAstNode *field_node = sd->fields[f];
            if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) continue;

            const StructField *field = &field_node->struct_field;
            if (!field->type || field->type->type != WGSL_NODE_TYPE) continue;

            const char *type_name = field->type->type_node.name;
            if (strcmp(type_name, "array") == 0) {
                // Get element type and compute stride
                const TypeNode *tn = &field->type->type_node;
                if (tn->type_arg_count > 0 && tn->type_args[0]->type == WGSL_NODE_TYPE) {
                    int elem_size = 4, elem_align = 4;
                    get_type_layout(tn->type_args[0]->type_node.name, &elem_size, &elem_align);
                    // Stride must be at least element size and aligned to element alignment
                    int stride = (elem_size + elem_align - 1) & ~(elem_align - 1);
                    if (stride < elem_size) stride = elem_size;
                    if (stride == 0) stride = 4;  // Minimum stride
                    uint32_t stride_val = (uint32_t)stride;
                    emit_decorate(l, member_types[f], SpvDecorationArrayStride, &stride_val, 1);
                }
            }
        }

        // Add to struct cache
        add_struct_to_cache(l, sd->name, struct_id, struct_id);

        WGSL_FREE(member_types);
        WGSL_FREE(offsets);
    }

    return 1;
}

// ---------- Entry Point and Execution Mode Emission ----------

static SpvExecutionModel stage_to_model(WgslStage s) {
    switch (s) {
        case WGSL_STAGE_VERTEX:   return SpvExecutionModelVertex;
        case WGSL_STAGE_FRAGMENT: return SpvExecutionModelFragment;
        case WGSL_STAGE_COMPUTE:  return SpvExecutionModelGLCompute;
        default:                  return SpvExecutionModelMax;
    }
}

//l nonnull
static int emit_entry_point(WgslLower *l, uint32_t func_id, SpvExecutionModel model, const char *name, uint32_t *interface_ids, int interface_count) {
    wgsl_compiler_assert(l != NULL, "emit_entry_point: l is NULL");
    WordBuf *wb = &l->sections.entry_points;
    uint32_t *strw = NULL; size_t wn = 0;
    make_string_lit(name ? name : "main", &strw, &wn);
    if (!emit_op(wb, SpvOpEntryPoint, 3 + wn + interface_count)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, model)) { WGSL_FREE(strw); return 0; }
    if (!wb_push_u32(wb, func_id)) { WGSL_FREE(strw); return 0; }
    int ok = wb_push_many(wb, strw, wn);
    WGSL_FREE(strw);
    if (!ok) return 0;
    for (int i = 0; i < interface_count; ++i) {
        if (!wb_push_u32(wb, interface_ids[i])) return 0;
    }
    return 1;
}

//l nonnull
static int emit_execution_mode(WgslLower *l, uint32_t func_id, SpvExecutionMode mode, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(l != NULL, "emit_execution_mode: l is NULL");
    WordBuf *wb = &l->sections.execution_modes;
    if (!emit_op(wb, SpvOpExecutionMode, 3 + lit_count)) return 0;
    if (!wb_push_u32(wb, func_id)) return 0;
    if (!wb_push_u32(wb, mode)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push_u32(wb, literals[i])) return 0;
    }
    return 1;
}

// ---------- Global Variable Emission ----------

//l nonnull
static uint32_t emit_global_variable(WgslLower *l, uint32_t ptr_type, SpvStorageClass storage_class, const char *name, uint32_t initializer) {
    wgsl_compiler_assert(l != NULL, "emit_global_variable: l is NULL");
    WordBuf *wb = &l->sections.globals;
    uint32_t id = fresh_id(l);
    int has_init = (initializer != 0);
    if (!emit_op(wb, SpvOpVariable, 4 + (has_init ? 1 : 0))) return 0;
    if (!wb_push_u32(wb, ptr_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, storage_class)) return 0;
    if (has_init) {
        if (!wb_push_u32(wb, initializer)) return 0;
    }
    emit_name(l, id, name);
    return id;
}

// ---------- Function Emission ----------

//l nonnull
static int emit_function_begin(WgslLower *l, uint32_t func_id, uint32_t result_type, uint32_t func_type) {
    wgsl_compiler_assert(l != NULL, "emit_function_begin: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpFunction, 5)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, func_id)) return 0;
    if (!wb_push_u32(wb, SpvFunctionControlMaskNone)) return 0;
    if (!wb_push_u32(wb, func_type)) return 0;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_function_parameter(WgslLower *l, uint32_t result_type, uint32_t *out_id) {
    wgsl_compiler_assert(l != NULL, "emit_function_parameter: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_function_parameter: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpFunctionParameter, 3)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_label(WgslLower *l, uint32_t *out_id) {
    wgsl_compiler_assert(l != NULL, "emit_label: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_label: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpLabel, 2)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    *out_id = id;
    return 1;
}

//l nonnull
static int emit_return(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_return: l is NULL");
    WordBuf *wb = &l->sections.functions;
    int ok = emit_op(wb, SpvOpReturn, 1);

    // Build SSIR return void instruction
    if (ok && l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        ssir_build_return_void(l->ssir, l->fn_ctx.ssir_func_id, l->fn_ctx.ssir_block_id);
    }

    return ok;
}

//l nonnull
static int emit_return_value(WgslLower *l, uint32_t value) {
    wgsl_compiler_assert(l != NULL, "emit_return_value: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpReturnValue, 2)) return 0;
    if (!wb_push_u32(wb, value)) return 0;

    // Build SSIR return instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_val = ssir_id_map_get(l, value);
        if (ssir_val) {
            ssir_build_return(l->ssir, l->fn_ctx.ssir_func_id, l->fn_ctx.ssir_block_id, ssir_val);
        }
    }

    return 1;
}

//l nonnull
static int emit_function_end(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_function_end: l is NULL");
    WordBuf *wb = &l->sections.functions;
    return emit_op(wb, SpvOpFunctionEnd, 1);
}

// ---------- Forward declarations for expression/statement lowering ----------

static uint32_t lower_expression(WgslLower *l, const WgslAstNode *expr);
static int lower_statement(WgslLower *l, const WgslAstNode *stmt);
static int lower_block(WgslLower *l, const WgslAstNode *block);

// ---------- Local variable context for function lowering ----------

typedef struct {
    const char *name;
    uint32_t ptr_id;      // SPIR-V pointer to the variable (or value ID if is_value)
    uint32_t type_id;     // SPIR-V type of the value (not pointer)
    int is_value;         // if true, ptr_id is a value (let/const), not a pointer
} LocalVar;

typedef struct {
    LocalVar *vars;
    int count;
    int cap;
} LocalVarContext;

//ctx nonnull
static void local_ctx_init(LocalVarContext *ctx) {
    wgsl_compiler_assert(ctx != NULL, "local_ctx_init: ctx is NULL");
    ctx->vars = NULL;
    ctx->count = 0;
    ctx->cap = 0;
}

//ctx nonnull
static void local_ctx_free(LocalVarContext *ctx) {
    wgsl_compiler_assert(ctx != NULL, "local_ctx_free: ctx is NULL");
    WGSL_FREE(ctx->vars);
    ctx->vars = NULL;
    ctx->count = 0;
    ctx->cap = 0;
}

//ctx nonnull
//name nonnull
static void local_ctx_add(LocalVarContext *ctx, const char *name, uint32_t ptr_id, uint32_t type_id) {
    wgsl_compiler_assert(ctx != NULL, "local_ctx_add: ctx is NULL");
    wgsl_compiler_assert(name != NULL, "local_ctx_add: name is NULL");
    if (ctx->count >= ctx->cap) {
        int new_cap = ctx->cap ? ctx->cap * 2 : 16;
        void *p = WGSL_REALLOC(ctx->vars, new_cap * sizeof(LocalVar));
        if (!p) return;
        ctx->vars = (LocalVar*)p;
        ctx->cap = new_cap;
    }
    ctx->vars[ctx->count].name = name;
    ctx->vars[ctx->count].ptr_id = ptr_id;
    ctx->vars[ctx->count].type_id = type_id;
    ctx->count++;
}

//ctx nonnull
//name nonnull
static LocalVar* local_ctx_find(LocalVarContext *ctx, const char *name) {
    wgsl_compiler_assert(ctx != NULL, "local_ctx_find: ctx is NULL");
    wgsl_compiler_assert(name != NULL, "local_ctx_find: name is NULL");
    for (int i = ctx->count - 1; i >= 0; --i) {
        if (strcmp(ctx->vars[i].name, name) == 0) {
            return &ctx->vars[i];
        }
    }
    return NULL;
}

// ---------- Instruction Emission ----------

//l nonnull
//out_id nonnull
static int emit_load(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t pointer) {
    wgsl_compiler_assert(l != NULL, "emit_load: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_load: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpLoad, 4)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, pointer)) return 0;

    // Build SSIR load instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_ptr = ssir_id_map_get(l, pointer);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        if (ssir_ptr && ssir_type) {
            uint32_t ssir_result = ssir_build_load(l->ssir, l->fn_ctx.ssir_func_id,
                                                    l->fn_ctx.ssir_block_id, ssir_type, ssir_ptr);
            ssir_id_map_set(l, id, ssir_result);
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
static int emit_store(WgslLower *l, uint32_t pointer, uint32_t object) {
    wgsl_compiler_assert(l != NULL, "emit_store: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpStore, 3)) return 0;
    if (!wb_push_u32(wb, pointer)) return 0;
    if (!wb_push_u32(wb, object)) return 0;

    // Build SSIR store instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_ptr = ssir_id_map_get(l, pointer);
        uint32_t ssir_val = ssir_id_map_get(l, object);
        if (ssir_ptr && ssir_val) {
            ssir_build_store(l->ssir, l->fn_ctx.ssir_func_id,
                            l->fn_ctx.ssir_block_id, ssir_ptr, ssir_val);
        }
    }

    return 1;
}

//l nonnull
//out_id nonnull
//indices nonnull
static int emit_access_chain(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t base, uint32_t *indices, int index_count) {
    wgsl_compiler_assert(l != NULL, "emit_access_chain: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_access_chain: out_id is NULL");
    wgsl_compiler_assert(indices != NULL, "emit_access_chain: indices is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpAccessChain, 4 + index_count)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, base)) return 0;
    for (int i = 0; i < index_count; ++i) {
        if (!wb_push_u32(wb, indices[i])) return 0;
    }

    // Build SSIR access instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id && index_count <= 8) {
        uint32_t ssir_base = ssir_id_map_get(l, base);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        uint32_t ssir_indices[8];
        int have_all = ssir_base != 0;
        for (int i = 0; i < index_count && i < 8; ++i) {
            ssir_indices[i] = ssir_id_map_get(l, indices[i]);
            if (!ssir_indices[i]) have_all = 0;
        }
        if (have_all && ssir_type) {
            uint32_t ssir_result = ssir_build_access(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type,
                                                      ssir_base, ssir_indices, (uint32_t)index_count);
            ssir_id_map_set(l, id, ssir_result);
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_local_variable(WgslLower *l, uint32_t ptr_type, uint32_t elem_type, uint32_t *out_id, uint32_t initializer) {
    wgsl_compiler_assert(l != NULL, "emit_local_variable: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_local_variable: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    int has_init = (initializer != 0);
    if (!emit_op(wb, SpvOpVariable, 4 + (has_init ? 1 : 0))) return 0;
    if (!wb_push_u32(wb, ptr_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, SpvStorageClassFunction)) return 0;
    if (has_init) {
        if (!wb_push_u32(wb, initializer)) return 0;
    }

    // Build SSIR local variable
    if (l->fn_ctx.ssir_func_id) {
        uint32_t ssir_elem_type = spv_type_to_ssir(l, elem_type);
        uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_FUNCTION);
        uint32_t ssir_var = ssir_function_add_local(l->ssir, l->fn_ctx.ssir_func_id,
                                                     NULL, ssir_ptr_type);
        ssir_id_map_set(l, id, ssir_var);
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_binary_op(WgslLower *l, SpvOp op, uint32_t result_type, uint32_t *out_id, uint32_t operand1, uint32_t operand2) {
    wgsl_compiler_assert(l != NULL, "emit_binary_op: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_binary_op: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, op, 5)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, operand1)) return 0;
    if (!wb_push_u32(wb, operand2)) return 0;

    // Build SSIR binary instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_a = ssir_id_map_get(l, operand1);
        uint32_t ssir_b = ssir_id_map_get(l, operand2);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        uint32_t ssir_result = 0;

        if (ssir_a && ssir_b && ssir_type) {
            switch (op) {
                // Arithmetic
                case SpvOpFAdd: case SpvOpIAdd:
                    ssir_result = ssir_build_add(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFSub: case SpvOpISub:
                    ssir_result = ssir_build_sub(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFMul: case SpvOpIMul: case SpvOpVectorTimesScalar:
                    ssir_result = ssir_build_mul(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpMatrixTimesScalar: case SpvOpVectorTimesMatrix:
                case SpvOpMatrixTimesVector: case SpvOpMatrixTimesMatrix:
                    ssir_result = ssir_build_mat_mul(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFDiv: case SpvOpSDiv: case SpvOpUDiv:
                    ssir_result = ssir_build_div(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFRem: case SpvOpSRem: case SpvOpUMod:
                    ssir_result = ssir_build_mod(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                // Bitwise
                case SpvOpBitwiseAnd:
                    ssir_result = ssir_build_bit_and(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpBitwiseOr:
                    ssir_result = ssir_build_bit_or(l->ssir, l->fn_ctx.ssir_func_id,
                                                     l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpBitwiseXor:
                    ssir_result = ssir_build_bit_xor(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpShiftLeftLogical:
                    ssir_result = ssir_build_shl(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpShiftRightArithmetic:
                    ssir_result = ssir_build_shr(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpShiftRightLogical:
                    ssir_result = ssir_build_shr_logical(l->ssir, l->fn_ctx.ssir_func_id,
                                                          l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                // Comparison
                case SpvOpFOrdEqual: case SpvOpIEqual:
                    ssir_result = ssir_build_eq(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFOrdNotEqual: case SpvOpINotEqual:
                    ssir_result = ssir_build_ne(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFOrdLessThan: case SpvOpSLessThan: case SpvOpULessThan:
                    ssir_result = ssir_build_lt(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFOrdLessThanEqual: case SpvOpSLessThanEqual: case SpvOpULessThanEqual:
                    ssir_result = ssir_build_le(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFOrdGreaterThan: case SpvOpSGreaterThan: case SpvOpUGreaterThan:
                    ssir_result = ssir_build_gt(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpFOrdGreaterThanEqual: case SpvOpSGreaterThanEqual: case SpvOpUGreaterThanEqual:
                    ssir_result = ssir_build_ge(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                // Logical
                case SpvOpLogicalAnd:
                    ssir_result = ssir_build_and(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                case SpvOpLogicalOr:
                    ssir_result = ssir_build_or(l->ssir, l->fn_ctx.ssir_func_id,
                                                 l->fn_ctx.ssir_block_id, ssir_type, ssir_a, ssir_b);
                    break;
                // Dot product - use builtin
                case SpvOpDot: {
                    uint32_t args[2] = {ssir_a, ssir_b};
                    ssir_result = ssir_build_builtin(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type,
                                                      SSIR_BUILTIN_DOT, args, 2);
                    break;
                }
                default:
                    break;
            }
            if (ssir_result) {
                ssir_id_map_set(l, id, ssir_result);
            }
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_unary_op(WgslLower *l, SpvOp op, uint32_t result_type, uint32_t *out_id, uint32_t operand) {
    wgsl_compiler_assert(l != NULL, "emit_unary_op: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_unary_op: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, op, 4)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, operand)) return 0;

    // Build SSIR unary instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_a = ssir_id_map_get(l, operand);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        uint32_t ssir_result = 0;

        if (ssir_a && ssir_type) {
            switch (op) {
                case SpvOpFNegate: case SpvOpSNegate:
                    ssir_result = ssir_build_neg(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a);
                    break;
                case SpvOpLogicalNot:
                    ssir_result = ssir_build_not(l->ssir, l->fn_ctx.ssir_func_id,
                                                  l->fn_ctx.ssir_block_id, ssir_type, ssir_a);
                    break;
                case SpvOpNot:
                    ssir_result = ssir_build_bit_not(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a);
                    break;
                case SpvOpConvertSToF: case SpvOpConvertUToF:
                case SpvOpConvertFToS: case SpvOpConvertFToU:
                    ssir_result = ssir_build_convert(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a);
                    break;
                case SpvOpBitcast:
                    ssir_result = ssir_build_bitcast(l->ssir, l->fn_ctx.ssir_func_id,
                                                      l->fn_ctx.ssir_block_id, ssir_type, ssir_a);
                    break;
                default:
                    break;
            }
            if (ssir_result) {
                ssir_id_map_set(l, id, ssir_result);
            }
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
//constituents nonnull
static int emit_composite_construct(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t *constituents, int count) {
    wgsl_compiler_assert(l != NULL, "emit_composite_construct: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_composite_construct: out_id is NULL");
    wgsl_compiler_assert(constituents != NULL, "emit_composite_construct: constituents is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpCompositeConstruct, 3 + count)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    for (int i = 0; i < count; ++i) {
        if (!wb_push_u32(wb, constituents[i])) return 0;
    }

    // Build SSIR construct instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        uint32_t ssir_components[16];
        int have_all = 1;
        for (int i = 0; i < count && i < 16; ++i) {
            ssir_components[i] = ssir_id_map_get(l, constituents[i]);
            if (!ssir_components[i]) have_all = 0;
        }
        if (have_all && ssir_type && count <= 16) {
            uint32_t ssir_result = ssir_build_construct(l->ssir, l->fn_ctx.ssir_func_id,
                                                         l->fn_ctx.ssir_block_id, ssir_type,
                                                         ssir_components, (uint32_t)count);
            ssir_id_map_set(l, id, ssir_result);
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
//indices nonnull
static int emit_composite_extract(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t composite, uint32_t *indices, int index_count) {
    wgsl_compiler_assert(l != NULL, "emit_composite_extract: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_composite_extract: out_id is NULL");
    wgsl_compiler_assert(indices != NULL, "emit_composite_extract: indices is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpCompositeExtract, 4 + index_count)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, composite)) return 0;
    for (int i = 0; i < index_count; ++i) {
        if (!wb_push_u32(wb, indices[i])) return 0;
    }

    // Build SSIR extract instruction (handles one index at a time)
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id && index_count == 1) {
        uint32_t ssir_composite = ssir_id_map_get(l, composite);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        if (ssir_composite && ssir_type) {
            uint32_t ssir_result = ssir_build_extract(l->ssir, l->fn_ctx.ssir_func_id,
                                                       l->fn_ctx.ssir_block_id, ssir_type,
                                                       ssir_composite, indices[0]);
            ssir_id_map_set(l, id, ssir_result);
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_select(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t cond, uint32_t true_val, uint32_t false_val) {
    wgsl_compiler_assert(l != NULL, "emit_select: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_select: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpSelect, 6)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, cond)) return 0;
    if (!wb_push_u32(wb, true_val)) return 0;
    if (!wb_push_u32(wb, false_val)) return 0;

    // Build SSIR select instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_cond = ssir_id_map_get(l, cond);
        uint32_t ssir_true = ssir_id_map_get(l, true_val);
        uint32_t ssir_false = ssir_id_map_get(l, false_val);
        uint32_t ssir_type = spv_type_to_ssir(l, result_type);
        if (ssir_cond && ssir_true && ssir_false && ssir_type) {
            uint32_t args[3] = {ssir_cond, ssir_true, ssir_false};
            uint32_t ssir_result = ssir_build_builtin(l->ssir, l->fn_ctx.ssir_func_id,
                                                       l->fn_ctx.ssir_block_id, ssir_type,
                                                       SSIR_BUILTIN_SELECT, args, 3);
            ssir_id_map_set(l, id, ssir_result);
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
static int emit_branch(WgslLower *l, uint32_t target) {
    wgsl_compiler_assert(l != NULL, "emit_branch: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpBranch, 2)) return 0;
    if (!wb_push_u32(wb, target)) return 0;

    // Build SSIR branch instruction
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
        uint32_t ssir_target = ssir_id_map_get(l, target);
        if (ssir_target) {
            ssir_build_branch(l->ssir, l->fn_ctx.ssir_func_id,
                             l->fn_ctx.ssir_block_id, ssir_target);
        }
    }

    return 1;
}

//l nonnull
static int emit_branch_conditional(WgslLower *l, uint32_t cond, uint32_t true_label, uint32_t false_label) {
    wgsl_compiler_assert(l != NULL, "emit_branch_conditional: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpBranchConditional, 4)) return 0;
    if (!wb_push_u32(wb, cond)) return 0;
    if (!wb_push_u32(wb, true_label)) return 0;
    if (!wb_push_u32(wb, false_label)) return 0;
    // Note: SSIR conditional branch is NOT emitted here - caller is responsible
    // for calling ssir_build_branch_cond_merge with proper merge block info
    return 1;
}

//l nonnull
static int emit_selection_merge(WgslLower *l, uint32_t merge_block) {
    wgsl_compiler_assert(l != NULL, "emit_selection_merge: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpSelectionMerge, 3)) return 0;
    if (!wb_push_u32(wb, merge_block)) return 0;
    if (!wb_push_u32(wb, SpvSelectionControlMaskNone)) return 0;
    return 1;
}

//l nonnull
static int emit_loop_merge(WgslLower *l, uint32_t merge_block, uint32_t continue_block) {
    wgsl_compiler_assert(l != NULL, "emit_loop_merge: l is NULL");
    WordBuf *wb = &l->sections.functions;
    if (!emit_op(wb, SpvOpLoopMerge, 4)) return 0;
    if (!wb_push_u32(wb, merge_block)) return 0;
    if (!wb_push_u32(wb, continue_block)) return 0;
    if (!wb_push_u32(wb, SpvLoopControlMaskNone)) return 0;
    return 1;
}

// Map GLSLstd450 opcode to SSIR builtin ID
static SsirBuiltinId glsl_to_ssir_builtin(uint32_t glsl_op) {
    switch (glsl_op) {
        case GLSLstd450Sin: return SSIR_BUILTIN_SIN;
        case GLSLstd450Cos: return SSIR_BUILTIN_COS;
        case GLSLstd450Tan: return SSIR_BUILTIN_TAN;
        case GLSLstd450Asin: return SSIR_BUILTIN_ASIN;
        case GLSLstd450Acos: return SSIR_BUILTIN_ACOS;
        case GLSLstd450Atan: return SSIR_BUILTIN_ATAN;
        case GLSLstd450Atan2: return SSIR_BUILTIN_ATAN2;
        case GLSLstd450Sinh: return SSIR_BUILTIN_SINH;
        case GLSLstd450Cosh: return SSIR_BUILTIN_COSH;
        case GLSLstd450Tanh: return SSIR_BUILTIN_TANH;
        case GLSLstd450Asinh: return SSIR_BUILTIN_ASINH;
        case GLSLstd450Acosh: return SSIR_BUILTIN_ACOSH;
        case GLSLstd450Atanh: return SSIR_BUILTIN_ATANH;
        case GLSLstd450Exp: return SSIR_BUILTIN_EXP;
        case GLSLstd450Exp2: return SSIR_BUILTIN_EXP2;
        case GLSLstd450Log: return SSIR_BUILTIN_LOG;
        case GLSLstd450Log2: return SSIR_BUILTIN_LOG2;
        case GLSLstd450Pow: return SSIR_BUILTIN_POW;
        case GLSLstd450Sqrt: return SSIR_BUILTIN_SQRT;
        case GLSLstd450InverseSqrt: return SSIR_BUILTIN_INVERSESQRT;
        case GLSLstd450FAbs: case GLSLstd450SAbs: return SSIR_BUILTIN_ABS;
        case GLSLstd450FSign: case GLSLstd450SSign: return SSIR_BUILTIN_SIGN;
        case GLSLstd450Floor: return SSIR_BUILTIN_FLOOR;
        case GLSLstd450Ceil: return SSIR_BUILTIN_CEIL;
        case GLSLstd450Round: return SSIR_BUILTIN_ROUND;
        case GLSLstd450Trunc: return SSIR_BUILTIN_TRUNC;
        case GLSLstd450Fract: return SSIR_BUILTIN_FRACT;
        case GLSLstd450FMin: case GLSLstd450SMin: case GLSLstd450UMin: return SSIR_BUILTIN_MIN;
        case GLSLstd450FMax: case GLSLstd450SMax: case GLSLstd450UMax: return SSIR_BUILTIN_MAX;
        case GLSLstd450FClamp: case GLSLstd450SClamp: case GLSLstd450UClamp: return SSIR_BUILTIN_CLAMP;
        case GLSLstd450FMix: return SSIR_BUILTIN_MIX;
        case GLSLstd450Step: return SSIR_BUILTIN_STEP;
        case GLSLstd450SmoothStep: return SSIR_BUILTIN_SMOOTHSTEP;
        case GLSLstd450Cross: return SSIR_BUILTIN_CROSS;
        case GLSLstd450Length: return SSIR_BUILTIN_LENGTH;
        case GLSLstd450Distance: return SSIR_BUILTIN_DISTANCE;
        case GLSLstd450Normalize: return SSIR_BUILTIN_NORMALIZE;
        case GLSLstd450FaceForward: return SSIR_BUILTIN_FACEFORWARD;
        case GLSLstd450Reflect: return SSIR_BUILTIN_REFLECT;
        case GLSLstd450Refract: return SSIR_BUILTIN_REFRACT;
        case GLSLstd450Radians: return SSIR_BUILTIN_COUNT; // No direct mapping
        case GLSLstd450Degrees: return SSIR_BUILTIN_COUNT; // No direct mapping
        default: return SSIR_BUILTIN_COUNT;
    }
}

//l nonnull
//out_id nonnull
//operands nonnull
static int emit_ext_inst(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t set, uint32_t instruction, uint32_t *operands, int count) {
    wgsl_compiler_assert(l != NULL, "emit_ext_inst: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_ext_inst: out_id is NULL");
    wgsl_compiler_assert(operands != NULL, "emit_ext_inst: operands is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpExtInst, 5 + count)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, set)) return 0;
    if (!wb_push_u32(wb, instruction)) return 0;
    for (int i = 0; i < count; ++i) {
        if (!wb_push_u32(wb, operands[i])) return 0;
    }

    // Build SSIR builtin instruction for GLSL.std.450
    if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id && count <= 8) {
        SsirBuiltinId ssir_builtin = glsl_to_ssir_builtin(instruction);
        if (ssir_builtin != SSIR_BUILTIN_COUNT) {
            uint32_t ssir_type = spv_type_to_ssir(l, result_type);
            uint32_t ssir_args[8];
            int have_all = 1;
            for (int i = 0; i < count && i < 8; ++i) {
                ssir_args[i] = ssir_id_map_get(l, operands[i]);
                if (!ssir_args[i]) have_all = 0;
            }
            if (have_all && ssir_type) {
                uint32_t ssir_result = ssir_build_builtin(l->ssir, l->fn_ctx.ssir_func_id,
                                                          l->fn_ctx.ssir_block_id, ssir_type,
                                                          ssir_builtin, ssir_args, (uint32_t)count);
                ssir_id_map_set(l, id, ssir_result);
            }
        }
    }

    *out_id = id;
    return 1;
}

//l nonnull
//out_id nonnull
static int emit_function_call(WgslLower *l, uint32_t result_type, uint32_t *out_id, uint32_t function, uint32_t *args, int arg_count) {
    wgsl_compiler_assert(l != NULL, "emit_function_call: l is NULL");
    wgsl_compiler_assert(out_id != NULL, "emit_function_call: out_id is NULL");
    WordBuf *wb = &l->sections.functions;
    uint32_t id = fresh_id(l);
    if (!emit_op(wb, SpvOpFunctionCall, 4 + arg_count)) return 0;
    if (!wb_push_u32(wb, result_type)) return 0;
    if (!wb_push_u32(wb, id)) return 0;
    if (!wb_push_u32(wb, function)) return 0;
    for (int i = 0; i < arg_count; ++i) {
        if (!wb_push_u32(wb, args[i])) return 0;
    }
    *out_id = id;
    return 1;
}

//l nonnull
//out_words nonnull
//out_count nonnull
static int finalize_spirv(WgslLower *l, uint32_t **out_words, size_t *out_count) {
    wgsl_compiler_assert(l != NULL, "finalize_spirv: l is NULL");
    wgsl_compiler_assert(out_words != NULL, "finalize_spirv: out_words is NULL");
    wgsl_compiler_assert(out_count != NULL, "finalize_spirv: out_count is NULL");
    SsirToSpirvOptions opts = {0};
    opts.spirv_version = l->opts.spirv_version ? l->opts.spirv_version : 0x00010300;
    opts.enable_debug_names = l->opts.enable_debug_names;
    return ssir_to_spirv(l->ssir, &opts, out_words, out_count) == SSIR_TO_SPIRV_OK;
}

// ---------- Expression Type Inference ----------

// Determine if a type is float-based (includes matrices which are float-based in WGSL)
//l nonnull
static int is_float_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_float_type: l is NULL");
    if (type_id == l->id_f32 || type_id == l->id_f16 ||
        type_id == l->id_vec2f || type_id == l->id_vec3f || type_id == l->id_vec4f) {
        return 1;
    }
    // Check if it's a matrix type (matrices are always float-based in WGSL)
    for (int bucket = 0; bucket < TYPE_CACHE_SIZE; ++bucket) {
        for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
            if (e->spv_id == type_id && e->kind == TC_MATRIX) {
                return 1;
            }
        }
    }
    return 0;
}

// Determine if a type is signed integer
//l nonnull
static int is_signed_int_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_signed_int_type: l is NULL");
    return type_id == l->id_i32 ||
           type_id == l->id_vec2i || type_id == l->id_vec3i || type_id == l->id_vec4i;
}

// Determine if a type is unsigned integer
//l nonnull
static int is_unsigned_int_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_unsigned_int_type: l is NULL");
    return type_id == l->id_u32 ||
           type_id == l->id_vec2u || type_id == l->id_vec3u || type_id == l->id_vec4u;
}

// Determine if a type is boolean
//l nonnull
static int is_bool_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_bool_type: l is NULL");
    return type_id == l->id_bool;
}

// Get vector component count (returns 0 for non-vector types)
//l nonnull
static int get_vector_component_count(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "get_vector_component_count: l is NULL");
    if (type_id == l->id_vec2f || type_id == l->id_vec2i || type_id == l->id_vec2u) return 2;
    if (type_id == l->id_vec3f || type_id == l->id_vec3i || type_id == l->id_vec3u) return 3;
    if (type_id == l->id_vec4f || type_id == l->id_vec4i || type_id == l->id_vec4u) return 4;
    return 0;
}

// Check if a type is a matrix type
//l nonnull
static int is_matrix_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_matrix_type: l is NULL");
    for (int bucket = 0; bucket < TYPE_CACHE_SIZE; ++bucket) {
        for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
            if (e->spv_id == type_id && e->kind == TC_MATRIX) {
                return 1;
            }
        }
    }
    return 0;
}

// Get matrix info (column type and count) - returns 1 if found
//l nonnull
static int get_matrix_info(uint32_t type_id, WgslLower *l, uint32_t *out_col_type, int *out_col_count) {
    wgsl_compiler_assert(l != NULL, "get_matrix_info: l is NULL");
    for (int bucket = 0; bucket < TYPE_CACHE_SIZE; ++bucket) {
        for (TypeCacheEntry *e = l->type_cache.buckets[bucket]; e; e = e->next) {
            if (e->spv_id == type_id && e->kind == TC_MATRIX) {
                if (out_col_type) *out_col_type = e->matrix_type.column_type_id;
                if (out_col_count) *out_col_count = e->matrix_type.column_count;
                return 1;
            }
        }
    }
    return 0;
}

// Check if a type is a scalar (not a vector)
//l nonnull
static int is_scalar_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "is_scalar_type: l is NULL");
    return type_id == l->id_f32 || type_id == l->id_f16 ||
           type_id == l->id_i32 || type_id == l->id_u32 || type_id == l->id_bool;
}

// Get the scalar component type of a vector (or the type itself if scalar)
//l nonnull
static uint32_t get_scalar_type(uint32_t type_id, WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "get_scalar_type: l is NULL");
    if (type_id == l->id_vec2f || type_id == l->id_vec3f || type_id == l->id_vec4f) return l->id_f32;
    if (type_id == l->id_vec2i || type_id == l->id_vec3i || type_id == l->id_vec4i) return l->id_i32;
    if (type_id == l->id_vec2u || type_id == l->id_vec3u || type_id == l->id_vec4u) return l->id_u32;
    return type_id; // Already a scalar
}

// Splat a scalar ExprResult to match a vector type.
// If scalar_expr is already a vector or target is scalar, returns 0 (no-op).
// On success, updates scalar_expr->id and scalar_expr->type_id in place and returns 1.
static int maybe_splat_scalar(WgslLower *l, uint32_t vec_type_id,
                              uint32_t *inout_id, uint32_t *inout_type_id) {
    if (!is_scalar_type(*inout_type_id, l)) return 0;
    int vec_count = get_vector_component_count(vec_type_id, l);
    if (vec_count <= 0) return 0;
    uint32_t components[4];
    for (int i = 0; i < vec_count && i < 4; ++i)
        components[i] = *inout_id;
    uint32_t broadcasted;
    if (emit_composite_construct(l, vec_type_id, &broadcasted, components, vec_count)) {
        *inout_id = broadcasted;
        *inout_type_id = vec_type_id;
        return 1;
    }
    return 0;
}

// ---------- Expression Result Tracking ----------

typedef struct {
    uint32_t id;          // SPIR-V result ID
    uint32_t type_id;     // SPIR-V type ID
    uint32_t ssir_id;     // SSIR result ID (parallel tracking)
    uint32_t ssir_type;   // SSIR type ID
} ExprResult;

// ---------- Local Variable Helpers ----------

//l nonnull
//name nonnull
static void fn_ctx_add_local_ex(WgslLower *l, const char *name, uint32_t ptr_id, uint32_t type_id, int is_value) {
    wgsl_compiler_assert(l != NULL, "fn_ctx_add_local_ex: l is NULL");
    wgsl_compiler_assert(name != NULL, "fn_ctx_add_local_ex: name is NULL");
    if (l->fn_ctx.locals.count >= l->fn_ctx.locals.cap) {
        int new_cap = l->fn_ctx.locals.cap ? l->fn_ctx.locals.cap * 2 : 16;
        void *p = WGSL_REALLOC(l->fn_ctx.locals.vars, new_cap * sizeof(l->fn_ctx.locals.vars[0]));
        if (!p) return;
        l->fn_ctx.locals.vars = (__typeof__(l->fn_ctx.locals.vars))p;
        l->fn_ctx.locals.cap = new_cap;
    }
    l->fn_ctx.locals.vars[l->fn_ctx.locals.count].name = name;
    l->fn_ctx.locals.vars[l->fn_ctx.locals.count].ptr_id = ptr_id;
    l->fn_ctx.locals.vars[l->fn_ctx.locals.count].type_id = type_id;
    l->fn_ctx.locals.vars[l->fn_ctx.locals.count].is_value = is_value;
    l->fn_ctx.locals.count++;
}

//l nonnull
//name nonnull
static void fn_ctx_add_local(WgslLower *l, const char *name, uint32_t ptr_id, uint32_t type_id) {
    wgsl_compiler_assert(l != NULL, "fn_ctx_add_local: l is NULL");
    wgsl_compiler_assert(name != NULL, "fn_ctx_add_local: name is NULL");
    fn_ctx_add_local_ex(l, name, ptr_id, type_id, 0);
}

//l nonnull
//name nonnull
static int fn_ctx_find_local(WgslLower *l, const char *name, uint32_t *out_ptr_id, uint32_t *out_type_id) {
    wgsl_compiler_assert(l != NULL, "fn_ctx_find_local: l is NULL");
    wgsl_compiler_assert(name != NULL, "fn_ctx_find_local: name is NULL");
    for (int i = l->fn_ctx.locals.count - 1; i >= 0; --i) {
        if (strcmp(l->fn_ctx.locals.vars[i].name, name) == 0) {
            if (out_ptr_id) *out_ptr_id = l->fn_ctx.locals.vars[i].ptr_id;
            if (out_type_id) *out_type_id = l->fn_ctx.locals.vars[i].type_id;
            return 1;
        }
    }
    return 0;
}

//l nonnull
static void fn_ctx_clear_locals(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "fn_ctx_clear_locals: l is NULL");
    WGSL_FREE(l->fn_ctx.locals.vars);
    l->fn_ctx.locals.vars = NULL;
    l->fn_ctx.locals.count = 0;
    l->fn_ctx.locals.cap = 0;

    WGSL_FREE(l->fn_ctx.pending_inits);
    l->fn_ctx.pending_inits = NULL;
    l->fn_ctx.pending_init_count = 0;
    l->fn_ctx.pending_init_cap = 0;
}

// ---------- Expression Lowering ----------

// Lower a literal expression
//l nonnull
//node nonnull
static ExprResult lower_literal(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_literal: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_literal: node is NULL");
    ExprResult r = {0, 0};
    const Literal *lit = &node->literal;

    if (lit->kind == WGSL_LIT_INT) {
        // Parse integer literal
        int64_t val = 0;
        const char *s = lit->lexeme;
        if (!s) s = "0";
        int is_unsigned = 0;

        // Check for suffix
        size_t len = strlen(s);
        if (len > 0 && (s[len-1] == 'u' || s[len-1] == 'U')) {
            is_unsigned = 1;
        }

        // Parse value
        if (len > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            val = strtoll(s + 2, NULL, 16);
        } else {
            val = strtoll(s, NULL, 10);
        }

        if (is_unsigned) {
            r.id = emit_const_u32(l, (uint32_t)val);
            r.type_id = l->id_u32;
        } else {
            r.id = emit_const_i32(l, (int32_t)val);
            r.type_id = l->id_i32;
        }
    } else if (lit->kind == WGSL_LIT_FLOAT) {
        float val = (float)strtod(lit->lexeme ? lit->lexeme : "0", NULL);
        r.id = emit_const_f32(l, val);
        r.type_id = l->id_f32;
    }

    return r;
}

// Helper to get element type from a runtime array type ID
//l nonnull
static uint32_t get_runtime_array_element_type(WgslLower *l, uint32_t type_id) {
    wgsl_compiler_assert(l != NULL, "get_runtime_array_element_type: l is NULL");
    // Search all buckets for runtime array types
    for (int b = 0; b < TYPE_CACHE_SIZE; ++b) {
        for (TypeCacheEntry *e = l->type_cache.buckets[b]; e; e = e->next) {
            if (e->kind == TC_RUNTIME_ARRAY && e->spv_id == type_id) {
                return e->runtime_array_type.element_type_id;
            }
        }
    }
    return 0;  // Not found
}

// Helper to find global variable by name
//l nonnull
//name nonnull
static int find_global_by_name(WgslLower *l, const char *name, uint32_t *out_var_id, uint32_t *out_type_id, SpvStorageClass *out_sc) {
    wgsl_compiler_assert(l != NULL, "find_global_by_name: l is NULL");
    wgsl_compiler_assert(name != NULL, "find_global_by_name: name is NULL");
    // Search backwards so that per-EP entries (added later) shadow earlier ones
    // with the same name (e.g., "in.uv" from vertex vs fragment)
    for (int i = l->global_map_count - 1; i >= 0; --i) {
        if (l->global_map[i].name && strcmp(l->global_map[i].name, name) == 0) {
            if (out_var_id) *out_var_id = l->global_map[i].spv_id;
            if (out_type_id) *out_type_id = l->global_map[i].type_id;
            if (out_sc) *out_sc = l->global_map[i].sc;
            return 1;
        }
    }
    return 0;
}

// Helper to find struct member index by name
//l nonnull
//struct_type_name nonnull
//member_name nonnull
static int find_struct_member_index(WgslLower *l, const char *struct_type_name, const char *member_name) {
    wgsl_compiler_assert(l != NULL, "find_struct_member_index: l is NULL");
    wgsl_compiler_assert(struct_type_name != NULL, "find_struct_member_index: struct_type_name is NULL");
    wgsl_compiler_assert(member_name != NULL, "find_struct_member_index: member_name is NULL");
    if (!l->program || l->program->type != WGSL_NODE_PROGRAM) return -1;

    const Program *prog = &l->program->program;
    for (int d = 0; d < prog->decl_count; ++d) {
        const WgslAstNode *decl = prog->decls[d];
        if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

        const StructDecl *sd = &decl->struct_decl;
        if (!sd->name || strcmp(sd->name, struct_type_name) != 0) continue;

        // Found the struct, now find the member
        for (int f = 0; f < sd->field_count; ++f) {
            const WgslAstNode *field_node = sd->fields[f];
            if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) continue;

            const StructField *field = &field_node->struct_field;
            if (field->name && strcmp(field->name, member_name) == 0) {
                return f;
            }
        }
    }
    return -1;
}

// Helper to get struct member type
//l nonnull
//struct_type_name nonnull
static uint32_t get_struct_member_type(WgslLower *l, const char *struct_type_name, int member_index) {
    wgsl_compiler_assert(l != NULL, "get_struct_member_type: l is NULL");
    wgsl_compiler_assert(struct_type_name != NULL, "get_struct_member_type: struct_type_name is NULL");
    if (!l->program || l->program->type != WGSL_NODE_PROGRAM) return 0;

    const Program *prog = &l->program->program;
    for (int d = 0; d < prog->decl_count; ++d) {
        const WgslAstNode *decl = prog->decls[d];
        if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

        const StructDecl *sd = &decl->struct_decl;
        if (!sd->name || strcmp(sd->name, struct_type_name) != 0) continue;

        if (member_index >= 0 && member_index < sd->field_count) {
            const WgslAstNode *field_node = sd->fields[member_index];
            if (field_node && field_node->type == WGSL_NODE_STRUCT_FIELD) {
                return lower_type(l, field_node->struct_field.type);
            }
        }
    }
    return 0;
}

// Helper to find struct type name from SPIR-V type ID
//l nonnull
static const char *find_struct_name_by_type_id(WgslLower *l, uint32_t type_id) {
    wgsl_compiler_assert(l != NULL, "find_struct_name_by_type_id: l is NULL");
    for (int i = 0; i < l->struct_cache_count; ++i) {
        if (l->struct_cache[i].spv_id == type_id) {
            return l->struct_cache[i].name;
        }
    }
    return NULL;
}

// Forward declarations for mutual recursion
static ExprResult lower_expr_full(WgslLower *l, const WgslAstNode *node);
static ExprResult lower_ptr_expr(WgslLower *l, const WgslAstNode *node, SpvStorageClass *out_sc);

// Lower an identifier expression (returns loaded value)
//l nonnull
//node nonnull
static ExprResult lower_ident(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_ident: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_ident: node is NULL");
    ExprResult r = {0, 0};
    const char *name = node->ident.name;

    // Check local variables first
    for (int i = l->fn_ctx.locals.count - 1; i >= 0; --i) {
        if (strcmp(l->fn_ctx.locals.vars[i].name, name) == 0) {
            if (l->fn_ctx.locals.vars[i].is_value) {
                r.id = l->fn_ctx.locals.vars[i].ptr_id;
                r.type_id = l->fn_ctx.locals.vars[i].type_id;
            } else {
                uint32_t loaded;
                if (emit_load(l, l->fn_ctx.locals.vars[i].type_id, &loaded, l->fn_ctx.locals.vars[i].ptr_id)) {
                    r.id = loaded;
                    r.type_id = l->fn_ctx.locals.vars[i].type_id;
                }
            }
            return r;
        }
    }

    // Check global variables
    uint32_t var_id, elem_type_id;
    SpvStorageClass sc;
    if (find_global_by_name(l, name, &var_id, &elem_type_id, &sc)) {
        // For Input/Output variables, load the value directly
        if (sc == SpvStorageClassInput || sc == SpvStorageClassOutput) {
            uint32_t loaded;
            if (emit_load(l, elem_type_id, &loaded, var_id)) {
                r.id = loaded;
                r.type_id = elem_type_id;
            }
            return r;
        }

        // For storage buffers, return pointer info - actual load happens at element access
        // This is a special case where we return the pointer as the "value"
        // Member access will handle building the access chain
        r.id = var_id;
        r.type_id = elem_type_id;  // The struct type
        return r;
    }

    return r;
}

// Lower an expression to get a pointer (for lvalues and access chains)
// Returns the pointer ID in r.id and the element type in r.type_id
//l nonnull
//node nonnull
static ExprResult lower_ptr_expr(WgslLower *l, const WgslAstNode *node, SpvStorageClass *out_sc) {
    wgsl_compiler_assert(l != NULL, "lower_ptr_expr: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_ptr_expr: node is NULL");
    ExprResult r = {0, 0};
    if (out_sc) *out_sc = SpvStorageClassFunction;

    switch (node->type) {
        case WGSL_NODE_IDENT: {
            const char *name = node->ident.name;

            // Check local variables
            uint32_t ptr_id, type_id;
            if (fn_ctx_find_local(l, name, &ptr_id, &type_id)) {
                r.id = ptr_id;
                r.type_id = type_id;
                if (out_sc) *out_sc = SpvStorageClassFunction;
                return r;
            }

            // Check global variables
            uint32_t var_id, elem_type_id;
            SpvStorageClass sc;
            if (find_global_by_name(l, name, &var_id, &elem_type_id, &sc)) {
                r.id = var_id;
                r.type_id = elem_type_id;
                if (out_sc) *out_sc = sc;
                return r;
            }
            break;
        }

        case WGSL_NODE_MEMBER: {
            const Member *mem = &node->member;

            // Check if this member is a struct output field (for vertex shader struct outputs)
            // The output variables are stored in global_map by field name
            if (l->fn_ctx.uses_struct_output) {
                uint32_t var_id, elem_type_id;
                SpvStorageClass sc;
                if (find_global_by_name(l, mem->member, &var_id, &elem_type_id, &sc) && sc == SpvStorageClassOutput) {
                    r.id = var_id;
                    r.type_id = elem_type_id;
                    if (out_sc) *out_sc = sc;
                    return r;
                }
            }

            // Get the base pointer
            SpvStorageClass base_sc;
            ExprResult base = lower_ptr_expr(l, mem->object, &base_sc);
            if (!base.id) break;

            // Find the struct type name
            const char *struct_name = find_struct_name_by_type_id(l, base.type_id);
            if (!struct_name) break;

            // Find the member index
            int member_idx = find_struct_member_index(l, struct_name, mem->member);
            if (member_idx < 0) break;

            // Get the member type
            uint32_t member_type_id = get_struct_member_type(l, struct_name, member_idx);
            if (!member_type_id) break;

            // Create access chain for member access
            uint32_t idx_const = emit_const_u32(l, (uint32_t)member_idx);
            uint32_t ptr_type = emit_type_pointer(l, base_sc, member_type_id);
            uint32_t result_id;
            if (emit_access_chain(l, ptr_type, &result_id, base.id, &idx_const, 1)) {
                r.id = result_id;
                r.type_id = member_type_id;
                if (out_sc) *out_sc = base_sc;
            }
            break;
        }

        case WGSL_NODE_INDEX: {
            const Index *idx = &node->index;

            // Get the base pointer
            SpvStorageClass base_sc;
            ExprResult base = lower_ptr_expr(l, idx->object, &base_sc);
            if (!base.id) break;

            // Lower the index expression
            ExprResult index = lower_expr_full(l, idx->index);
            if (!index.id) break;

            // Determine element type from runtime array type
            uint32_t elem_type = get_runtime_array_element_type(l, base.type_id);
            if (!elem_type) elem_type = l->id_f32;  // Fallback to f32

            // Create access chain for array index
            uint32_t ptr_type = emit_type_pointer(l, base_sc, elem_type);
            uint32_t result_id;
            if (emit_access_chain(l, ptr_type, &result_id, base.id, &index.id, 1)) {
                r.id = result_id;
                r.type_id = elem_type;
                if (out_sc) *out_sc = base_sc;
            }
            break;
        }

        default:
            break;
    }

    return r;
}

// Get the lvalue (pointer) for an expression
//l nonnull
//node nonnull
static ExprResult lower_lvalue(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_lvalue: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_lvalue: node is NULL");
    SpvStorageClass sc;
    return lower_ptr_expr(l, node, &sc);
}

// Lower a binary expression
//l nonnull
//node nonnull
static ExprResult lower_binary(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_binary: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_binary: node is NULL");
    ExprResult r = {0, 0};
    const Binary *bin = &node->binary;

    ExprResult left = lower_expression(l, bin->left) ? (ExprResult){lower_expression(l, bin->left), 0} : (ExprResult){0, 0};
    // Actually need to get both id and type
    // Let me restructure this...

    return r;
}

// Forward declare for mutual recursion
static ExprResult lower_expr_full(WgslLower *l, const WgslAstNode *node);

// Lower a call expression
//l nonnull
//node nonnull
static ExprResult lower_call(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_call: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_call: node is NULL");
    ExprResult r = {0, 0};
    const Call *call = &node->call;

    // Get callee name
    const char *callee_name = NULL;
    if (call->callee && call->callee->type == WGSL_NODE_IDENT) {
        callee_name = call->callee->ident.name;
    } else if (call->callee && call->callee->type == WGSL_NODE_TYPE) {
        callee_name = call->callee->type_node.name;
    }

    if (!callee_name) return r;

    // Generic vector constructor: vec2/vec3/vec4 with any element type
    {
        int vec_size = 0;
        if (strncmp(callee_name, "vec2", 4) == 0) vec_size = 2;
        else if (strncmp(callee_name, "vec3", 4) == 0) vec_size = 3;
        else if (strncmp(callee_name, "vec4", 4) == 0) vec_size = 4;

        if (vec_size > 0) {
            // Determine element type from suffix or type args
            uint32_t elem_type = 0;
            char suffix = callee_name[4];
            if (suffix == 'f' || suffix == '\0') {
                // Check type args for vec3<u32> style
                if (suffix == '\0' && call->callee && call->callee->type == WGSL_NODE_TYPE &&
                    call->callee->type_node.type_arg_count > 0 && call->callee->type_node.type_args[0]) {
                    const char *ta = call->callee->type_node.type_args[0]->type_node.name;
                    if (strcmp(ta, "f32") == 0) elem_type = l->id_f32;
                    else if (strcmp(ta, "f16") == 0) elem_type = l->id_f16 ? l->id_f16 : emit_type_float(l, 16);
                    else if (strcmp(ta, "i32") == 0) elem_type = l->id_i32;
                    else if (strcmp(ta, "u32") == 0) elem_type = l->id_u32;
                    else if (strcmp(ta, "bool") == 0) elem_type = l->id_bool;
                } else {
                    elem_type = l->id_f32;
                }
            } else if (suffix == 'i') { elem_type = l->id_i32; }
            else if (suffix == 'u') { elem_type = l->id_u32; }
            else if (suffix == 'h') { elem_type = l->id_f16 ? l->id_f16 : emit_type_float(l, 16); }

            if (!elem_type) elem_type = l->id_f32;

            uint32_t constituents[4] = {0, 0, 0, 0};
            int idx = 0;
            for (int i = 0; i < call->arg_count && idx < vec_size; ++i) {
                ExprResult arg = lower_expr_full(l, call->args[i]);
                if (arg.id) {
                    int vec_count = get_vector_component_count(arg.type_id, l);
                    if (vec_count > 0) {
                        uint32_t scalar_type = get_scalar_type(arg.type_id, l);
                        for (int j = 0; j < vec_count && idx < vec_size; ++j) {
                            uint32_t comp_id;
                            uint32_t index = j;
                            if (emit_composite_extract(l, scalar_type, &comp_id, arg.id, &index, 1)) {
                                constituents[idx++] = comp_id;
                            }
                        }
                    } else {
                        constituents[idx++] = arg.id;
                    }
                }
            }
            // Single-arg splat
            if (call->arg_count == 1 && idx == 1) {
                for (int i = 1; i < vec_size; ++i)
                    constituents[i] = constituents[0];
                idx = vec_size;
            }
            uint32_t vec_type = emit_type_vector(l, elem_type, vec_size);

            // Check if all constituents are constants -> use OpConstantComposite
            int all_const = 1;
            for (int i = 0; i < vec_size; ++i) {
                int found = 0;
                for (int j = 0; j < l->const_cache_count; ++j) {
                    if (l->const_cache[j].id == constituents[i]) { found = 1; break; }
                }
                if (!found) { all_const = 0; break; }
            }

            if (all_const && l->ssir) {
                uint32_t ssir_type = spv_type_to_ssir(l, vec_type);
                uint32_t ssir_components[4];
                for (int i = 0; i < vec_size; ++i)
                    ssir_components[i] = ssir_id_map_get(l, constituents[i]);
                uint32_t ssir_const_id = ssir_const_composite(l->ssir, ssir_type, ssir_components, vec_size);
                uint32_t id = fresh_id(l);
                ssir_id_map_set(l, id, ssir_const_id);
                r.id = id;
                r.type_id = vec_type;
            } else {
                uint32_t result;
                if (emit_composite_construct(l, vec_type, &result, constituents, vec_size)) {
                    r.id = result;
                    r.type_id = vec_type;
                }
            }
            return r;
        }
    }

    // Matrix constructors (mat2x2, mat3x3, mat4x4, etc.)
    if (strncmp(callee_name, "mat", 3) == 0) {
        int cols = 0, rows = 0;
        // Parse matCxR or matCxRf patterns
        if (sscanf(callee_name, "mat%dx%d", &cols, &rows) == 2 ||
            sscanf(callee_name, "mat%dx%df", &cols, &rows) == 2) {
            if (cols >= 2 && cols <= 4 && rows >= 2 && rows <= 4) {
                uint32_t elem_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
                char mat_suffix = callee_name[strlen(callee_name) - 1];
                if (mat_suffix == 'h') elem_type = l->id_f16 ? l->id_f16 : emit_type_float(l, 16);
                else if (call->callee && call->callee->type == WGSL_NODE_TYPE &&
                         call->callee->type_node.type_arg_count > 0 && call->callee->type_node.type_args[0]) {
                    const char *ta = call->callee->type_node.type_args[0]->type_node.name;
                    if (strcmp(ta, "f16") == 0) elem_type = l->id_f16 ? l->id_f16 : emit_type_float(l, 16);
                }
                uint32_t col_type = emit_type_vector(l, elem_type, rows);
                uint32_t mat_type = emit_type_matrix(l, col_type, cols);

                uint32_t columns[4] = {0, 0, 0, 0};

                if (call->arg_count == cols) {
                    // Column vectors passed directly
                    for (int i = 0; i < cols; ++i) {
                        ExprResult arg = lower_expr_full(l, call->args[i]);
                        if (arg.id) {
                            columns[i] = arg.id;
                        }
                    }
                } else if (call->arg_count == cols * rows) {
                    // All scalars - build column vectors
                    for (int c = 0; c < cols; ++c) {
                        uint32_t components[4] = {0, 0, 0, 0};
                        for (int row = 0; row < rows; ++row) {
                            ExprResult arg = lower_expr_full(l, call->args[c * rows + row]);
                            if (arg.id) {
                                components[row] = arg.id;
                            }
                        }
                        uint32_t col_id;
                        if (emit_composite_construct(l, col_type, &col_id, components, rows)) {
                            columns[c] = col_id;
                        }
                    }
                }

                // Build the matrix
                uint32_t result;
                if (emit_composite_construct(l, mat_type, &result, columns, cols)) {
                    r.id = result;
                    r.type_id = mat_type;
                }
                return r;
            }
        }
    }

    // Scalar type conversions: f32(), i32(), u32()
    if (strcmp(callee_name, "f32") == 0 && call->arg_count >= 1) {
        ExprResult arg = lower_expr_full(l, call->args[0]);
        if (arg.id) {
            if (is_float_type(arg.type_id, l)) {
                r.id = arg.id;
                r.type_id = l->id_f32;
            } else {
                uint32_t result;
                SpvOp op = is_signed_int_type(arg.type_id, l) ? SpvOpConvertSToF : SpvOpConvertUToF;
                if (emit_unary_op(l, op, l->id_f32, &result, arg.id)) {
                    r.id = result;
                    r.type_id = l->id_f32;
                }
            }
        }
        return r;
    }

    if (strcmp(callee_name, "i32") == 0 && call->arg_count >= 1) {
        ExprResult arg = lower_expr_full(l, call->args[0]);
        if (arg.id) {
            if (is_signed_int_type(arg.type_id, l)) {
                r.id = arg.id;
                r.type_id = l->id_i32;
            } else if (is_float_type(arg.type_id, l)) {
                uint32_t result;
                if (emit_unary_op(l, SpvOpConvertFToS, l->id_i32, &result, arg.id)) {
                    r.id = result;
                    r.type_id = l->id_i32;
                }
            } else {
                uint32_t result;
                if (emit_unary_op(l, SpvOpBitcast, l->id_i32, &result, arg.id)) {
                    r.id = result;
                    r.type_id = l->id_i32;
                }
            }
        }
        return r;
    }

    if (strcmp(callee_name, "u32") == 0 && call->arg_count >= 1) {
        ExprResult arg = lower_expr_full(l, call->args[0]);
        if (arg.id) {
            if (is_unsigned_int_type(arg.type_id, l)) {
                r.id = arg.id;
                r.type_id = l->id_u32;
            } else if (is_float_type(arg.type_id, l)) {
                uint32_t result;
                if (emit_unary_op(l, SpvOpConvertFToU, l->id_u32, &result, arg.id)) {
                    r.id = result;
                    r.type_id = l->id_u32;
                }
            } else {
                uint32_t result;
                if (emit_unary_op(l, SpvOpBitcast, l->id_u32, &result, arg.id)) {
                    r.id = result;
                    r.type_id = l->id_u32;
                }
            }
        }
        return r;
    }

    // GLSL.std.450 built-in functions
    if (strcmp(callee_name, "abs") == 0 && call->arg_count >= 1) {
        ExprResult arg = lower_expr_full(l, call->args[0]);
        if (arg.id) {
            uint32_t result;
            int glsl_op = is_float_type(arg.type_id, l) ? GLSLstd450FAbs : GLSLstd450SAbs;
            if (emit_ext_inst(l, arg.type_id, &result, l->id_extinst_glsl, glsl_op, &arg.id, 1)) {
                r.id = result;
                r.type_id = arg.type_id;
            }
        }
        return r;
    }

    if (strcmp(callee_name, "min") == 0 && call->arg_count >= 2) {
        ExprResult a = lower_expr_full(l, call->args[0]);
        ExprResult b = lower_expr_full(l, call->args[1]);
        if (a.id && b.id) {
            uint32_t result;
            int glsl_op = is_float_type(a.type_id, l) ? GLSLstd450FMin :
                         (is_signed_int_type(a.type_id, l) ? GLSLstd450SMin : GLSLstd450UMin);
            uint32_t operands[2] = {a.id, b.id};
            if (emit_ext_inst(l, a.type_id, &result, l->id_extinst_glsl, glsl_op, operands, 2)) {
                r.id = result;
                r.type_id = a.type_id;
            }
        }
        return r;
    }

    if (strcmp(callee_name, "max") == 0 && call->arg_count >= 2) {
        ExprResult a = lower_expr_full(l, call->args[0]);
        ExprResult b = lower_expr_full(l, call->args[1]);
        if (a.id && b.id) {
            uint32_t result;
            int glsl_op = is_float_type(a.type_id, l) ? GLSLstd450FMax :
                         (is_signed_int_type(a.type_id, l) ? GLSLstd450SMax : GLSLstd450UMax);
            uint32_t operands[2] = {a.id, b.id};
            if (emit_ext_inst(l, a.type_id, &result, l->id_extinst_glsl, glsl_op, operands, 2)) {
                r.id = result;
                r.type_id = a.type_id;
            }
        }
        return r;
    }

    if (strcmp(callee_name, "clamp") == 0 && call->arg_count >= 3) {
        ExprResult x = lower_expr_full(l, call->args[0]);
        ExprResult lo = lower_expr_full(l, call->args[1]);
        ExprResult hi = lower_expr_full(l, call->args[2]);
        if (x.id && lo.id && hi.id) {
            maybe_splat_scalar(l, x.type_id, &lo.id, &lo.type_id);
            maybe_splat_scalar(l, x.type_id, &hi.id, &hi.type_id);
            uint32_t result;
            int glsl_op = is_float_type(x.type_id, l) ? GLSLstd450FClamp :
                         (is_signed_int_type(x.type_id, l) ? GLSLstd450SClamp : GLSLstd450UClamp);
            uint32_t operands[3] = {x.id, lo.id, hi.id};
            if (emit_ext_inst(l, x.type_id, &result, l->id_extinst_glsl, glsl_op, operands, 3)) {
                r.id = result;
                r.type_id = x.type_id;
            }
        }
        return r;
    }

    // Math functions (float only)
    struct { const char *name; int glsl_op; } math_funcs[] = {
        {"floor", GLSLstd450Floor}, {"ceil", GLSLstd450Ceil}, {"round", GLSLstd450Round},
        {"trunc", GLSLstd450Trunc}, {"fract", GLSLstd450Fract}, {"sqrt", GLSLstd450Sqrt},
        {"inverseSqrt", GLSLstd450InverseSqrt}, {"exp", GLSLstd450Exp}, {"exp2", GLSLstd450Exp2},
        {"log", GLSLstd450Log}, {"log2", GLSLstd450Log2}, {"sin", GLSLstd450Sin},
        {"cos", GLSLstd450Cos}, {"tan", GLSLstd450Tan}, {"asin", GLSLstd450Asin},
        {"acos", GLSLstd450Acos}, {"atan", GLSLstd450Atan}, {"sinh", GLSLstd450Sinh},
        {"cosh", GLSLstd450Cosh}, {"tanh", GLSLstd450Tanh}, {"asinh", GLSLstd450Asinh},
        {"acosh", GLSLstd450Acosh}, {"atanh", GLSLstd450Atanh}, {"sign", GLSLstd450FSign},
        {"radians", GLSLstd450Radians}, {"degrees", GLSLstd450Degrees},
        {"length", GLSLstd450Length}, {"normalize", GLSLstd450Normalize},
        {NULL, 0}
    };

    for (int i = 0; math_funcs[i].name; ++i) {
        if (strcmp(callee_name, math_funcs[i].name) == 0 && call->arg_count >= 1) {
            ExprResult arg = lower_expr_full(l, call->args[0]);
            if (arg.id) {
                uint32_t result;
                uint32_t result_type = arg.type_id;
                // length returns scalar
                if (strcmp(callee_name, "length") == 0) {
                    result_type = l->id_f32;
                }
                if (emit_ext_inst(l, result_type, &result, l->id_extinst_glsl, math_funcs[i].glsl_op, &arg.id, 1)) {
                    r.id = result;
                    r.type_id = result_type;
                }
            }
            return r;
        }
    }

    // Two-argument math functions
    struct { const char *name; int glsl_op; } math2_funcs[] = {
        {"pow", GLSLstd450Pow}, {"atan2", GLSLstd450Atan2}, {"distance", GLSLstd450Distance},
        {"reflect", GLSLstd450Reflect}, {"step", GLSLstd450Step},
        {NULL, 0}
    };

    for (int i = 0; math2_funcs[i].name; ++i) {
        if (strcmp(callee_name, math2_funcs[i].name) == 0 && call->arg_count >= 2) {
            ExprResult a = lower_expr_full(l, call->args[0]);
            ExprResult b = lower_expr_full(l, call->args[1]);
            if (a.id && b.id) {
                // step(scalar_edge, vecN_x): splat edge to match x
                if (strcmp(callee_name, "step") == 0)
                    maybe_splat_scalar(l, b.type_id, &a.id, &a.type_id);
                uint32_t result;
                uint32_t result_type = a.type_id;
                if (strcmp(callee_name, "distance") == 0) {
                    result_type = l->id_f32;
                }
                uint32_t operands[2] = {a.id, b.id};
                if (emit_ext_inst(l, result_type, &result, l->id_extinst_glsl, math2_funcs[i].glsl_op, operands, 2)) {
                    r.id = result;
                    r.type_id = result_type;
                }
            }
            return r;
        }
    }

    // mix, smoothstep, fma, refract
    if (strcmp(callee_name, "mix") == 0 && call->arg_count >= 3) {
        ExprResult a = lower_expr_full(l, call->args[0]);
        ExprResult b = lower_expr_full(l, call->args[1]);
        ExprResult t = lower_expr_full(l, call->args[2]);
        if (a.id && b.id && t.id) {
            maybe_splat_scalar(l, a.type_id, &t.id, &t.type_id);
            uint32_t result;
            uint32_t operands[3] = {a.id, b.id, t.id};
            if (emit_ext_inst(l, a.type_id, &result, l->id_extinst_glsl, GLSLstd450FMix, operands, 3)) {
                r.id = result;
                r.type_id = a.type_id;
            }
        }
        return r;
    }

    if (strcmp(callee_name, "smoothstep") == 0 && call->arg_count >= 3) {
        ExprResult e0 = lower_expr_full(l, call->args[0]);
        ExprResult e1 = lower_expr_full(l, call->args[1]);
        ExprResult x = lower_expr_full(l, call->args[2]);
        if (e0.id && e1.id && x.id) {
            maybe_splat_scalar(l, x.type_id, &e0.id, &e0.type_id);
            maybe_splat_scalar(l, x.type_id, &e1.id, &e1.type_id);
            uint32_t result;
            uint32_t operands[3] = {e0.id, e1.id, x.id};
            if (emit_ext_inst(l, x.type_id, &result, l->id_extinst_glsl, GLSLstd450SmoothStep, operands, 3)) {
                r.id = result;
                r.type_id = x.type_id;
            }
        }
        return r;
    }

    // cross product
    if (strcmp(callee_name, "cross") == 0 && call->arg_count >= 2) {
        ExprResult a = lower_expr_full(l, call->args[0]);
        ExprResult b = lower_expr_full(l, call->args[1]);
        if (a.id && b.id) {
            uint32_t result;
            uint32_t operands[2] = {a.id, b.id};
            if (emit_ext_inst(l, a.type_id, &result, l->id_extinst_glsl, GLSLstd450Cross, operands, 2)) {
                r.id = result;
                r.type_id = a.type_id;
            }
        }
        return r;
    }

    // dot product - native SPIR-V instruction
    if (strcmp(callee_name, "dot") == 0 && call->arg_count >= 2) {
        ExprResult a = lower_expr_full(l, call->args[0]);
        ExprResult b = lower_expr_full(l, call->args[1]);
        if (a.id && b.id) {
            uint32_t result;
            if (emit_binary_op(l, SpvOpDot, l->id_f32, &result, a.id, b.id)) {
                r.id = result;
                r.type_id = l->id_f32;
            }
        }
        return r;
    }

    // textureSample(texture, sampler, coord)
    if (strcmp(callee_name, "textureSample") == 0 && call->arg_count >= 3) {
        // Get texture and sampler variable IDs (UniformConstant pointers)
        ExprResult tex = lower_expr_full(l, call->args[0]);
        ExprResult samp = lower_expr_full(l, call->args[1]);
        ExprResult coord = lower_expr_full(l, call->args[2]);
        if (!tex.id || !samp.id || !coord.id) return r;

        // Load texture and sampler from UniformConstant pointers
        uint32_t loaded_tex, loaded_samp;
        if (!emit_load(l, tex.type_id, &loaded_tex, tex.id)) return r;
        if (!emit_load(l, samp.type_id, &loaded_samp, samp.id)) return r;

        // Result is always vec4f for textureSample
        uint32_t result_type = l->id_vec4f;

        // SpvSections path: OpSampledImage + OpImageSampleImplicitLod
        WordBuf *wb = &l->sections.functions;
        uint32_t si_type = emit_type_sampled_image(l, tex.type_id);
        uint32_t si_id = fresh_id(l);
        emit_op(wb, SpvOpSampledImage, 5);
        wb_push_u32(wb, si_type);
        wb_push_u32(wb, si_id);
        wb_push_u32(wb, loaded_tex);
        wb_push_u32(wb, loaded_samp);

        uint32_t result_id = fresh_id(l);
        emit_op(wb, SpvOpImageSampleImplicitLod, 5);
        wb_push_u32(wb, result_type);
        wb_push_u32(wb, result_id);
        wb_push_u32(wb, si_id);
        wb_push_u32(wb, coord.id);

        // SSIR path: ssir_build_tex_sample
        if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
            uint32_t ssir_tex = ssir_id_map_get(l, loaded_tex);
            uint32_t ssir_samp = ssir_id_map_get(l, loaded_samp);
            uint32_t ssir_coord = ssir_id_map_get(l, coord.id);
            uint32_t ssir_result_type = spv_type_to_ssir(l, result_type);
            if (ssir_tex && ssir_samp && ssir_coord && ssir_result_type) {
                uint32_t ssir_result = ssir_build_tex_sample(l->ssir, l->fn_ctx.ssir_func_id,
                                                              l->fn_ctx.ssir_block_id,
                                                              ssir_result_type, ssir_tex, ssir_samp,
                                                              ssir_coord);
                ssir_id_map_set(l, result_id, ssir_result);
            }
        }

        r.id = result_id;
        r.type_id = result_type;
        return r;
    }

    return r;
}

// Full expression lowering
//l nonnull
static ExprResult lower_expr_full(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_expr_full: l is NULL");
    ExprResult r = {0, 0};
    if (!node) return r;

    switch (node->type) {
        case WGSL_NODE_LITERAL:
            return lower_literal(l, node);

        case WGSL_NODE_IDENT:
            return lower_ident(l, node);

        case WGSL_NODE_CALL:
            return lower_call(l, node);

        case WGSL_NODE_BINARY: {
            const Binary *bin = &node->binary;
            ExprResult left = lower_expr_full(l, bin->left);
            ExprResult right = lower_expr_full(l, bin->right);
            if (!left.id || !right.id) return r;

            SpvOp op = SpvOpNop;
            uint32_t result_type = left.type_id;
            int is_float = is_float_type(left.type_id, l);
            int is_signed = is_signed_int_type(left.type_id, l);

            // Arithmetic operators
            if (strcmp(bin->op, "+") == 0) {
                // Handle scalar + vector (broadcast scalar to vector)
                int left_vec_count = get_vector_component_count(left.type_id, l);
                int right_vec_count = get_vector_component_count(right.type_id, l);
                int left_is_scalar = is_scalar_type(left.type_id, l);
                int right_is_scalar = is_scalar_type(right.type_id, l);

                if (left_is_scalar && right_vec_count > 0) {
                    // Broadcast left scalar to vector
                    uint32_t components[4];
                    for (int i = 0; i < right_vec_count && i < 4; ++i) {
                        components[i] = left.id;
                    }
                    uint32_t broadcasted;
                    if (emit_composite_construct(l, right.type_id, &broadcasted, components, right_vec_count)) {
                        left.id = broadcasted;
                        left.type_id = right.type_id;
                    }
                    result_type = right.type_id;
                } else if (left_vec_count > 0 && right_is_scalar) {
                    // Broadcast right scalar to vector
                    uint32_t components[4];
                    for (int i = 0; i < left_vec_count && i < 4; ++i) {
                        components[i] = right.id;
                    }
                    uint32_t broadcasted;
                    if (emit_composite_construct(l, left.type_id, &broadcasted, components, left_vec_count)) {
                        right.id = broadcasted;
                        right.type_id = left.type_id;
                    }
                }

                // Special handling for matrix addition (column-wise)
                if (is_matrix_type(left.type_id, l)) {
                    uint32_t col_type;
                    int col_count;
                    if (get_matrix_info(left.type_id, l, &col_type, &col_count)) {
                        uint32_t columns[4];
                        for (int c = 0; c < col_count && c < 4; ++c) {
                            uint32_t left_col, right_col;
                            uint32_t idx = c;
                            emit_composite_extract(l, col_type, &left_col, left.id, &idx, 1);
                            emit_composite_extract(l, col_type, &right_col, right.id, &idx, 1);
                            emit_binary_op(l, SpvOpFAdd, col_type, &columns[c], left_col, right_col);
                        }
                        uint32_t result;
                        if (emit_composite_construct(l, left.type_id, &result, columns, col_count)) {
                            r.id = result;
                            r.type_id = left.type_id;
                        }
                        return r;
                    }
                }
                op = is_float ? SpvOpFAdd : SpvOpIAdd;
            } else if (strcmp(bin->op, "-") == 0) {
                // Handle scalar - vector or vector - scalar (broadcast scalar to vector)
                int left_vec_count = get_vector_component_count(left.type_id, l);
                int right_vec_count = get_vector_component_count(right.type_id, l);
                int left_is_scalar = is_scalar_type(left.type_id, l);
                int right_is_scalar = is_scalar_type(right.type_id, l);

                if (left_is_scalar && right_vec_count > 0) {
                    // Broadcast left scalar to vector
                    uint32_t components[4];
                    for (int i = 0; i < right_vec_count && i < 4; ++i) {
                        components[i] = left.id;
                    }
                    uint32_t broadcasted;
                    if (emit_composite_construct(l, right.type_id, &broadcasted, components, right_vec_count)) {
                        left.id = broadcasted;
                        left.type_id = right.type_id;
                    }
                    result_type = right.type_id;
                } else if (left_vec_count > 0 && right_is_scalar) {
                    // Broadcast right scalar to vector
                    uint32_t components[4];
                    for (int i = 0; i < left_vec_count && i < 4; ++i) {
                        components[i] = right.id;
                    }
                    uint32_t broadcasted;
                    if (emit_composite_construct(l, left.type_id, &broadcasted, components, left_vec_count)) {
                        right.id = broadcasted;
                        right.type_id = left.type_id;
                    }
                }

                // Special handling for matrix subtraction (column-wise)
                if (is_matrix_type(left.type_id, l)) {
                    uint32_t col_type;
                    int col_count;
                    if (get_matrix_info(left.type_id, l, &col_type, &col_count)) {
                        uint32_t columns[4];
                        for (int c = 0; c < col_count && c < 4; ++c) {
                            uint32_t left_col, right_col;
                            uint32_t idx = c;
                            emit_composite_extract(l, col_type, &left_col, left.id, &idx, 1);
                            emit_composite_extract(l, col_type, &right_col, right.id, &idx, 1);
                            emit_binary_op(l, SpvOpFSub, col_type, &columns[c], left_col, right_col);
                        }
                        uint32_t result;
                        if (emit_composite_construct(l, left.type_id, &result, columns, col_count)) {
                            r.id = result;
                            r.type_id = left.type_id;
                        }
                        return r;
                    }
                }
                op = is_float ? SpvOpFSub : SpvOpISub;
            } else if (strcmp(bin->op, "*") == 0) {
                // Check for various multiplication cases
                int left_vec_count = get_vector_component_count(left.type_id, l);
                int right_vec_count = get_vector_component_count(right.type_id, l);
                int left_is_scalar = is_scalar_type(left.type_id, l);
                int right_is_scalar = is_scalar_type(right.type_id, l);
                int left_is_matrix = is_matrix_type(left.type_id, l);
                int right_is_matrix = is_matrix_type(right.type_id, l);

                if (left_is_matrix && right_is_matrix) {
                    // matrix × matrix: mat<K,R> * mat<C,K> -> mat<C,R>
                    op = SpvOpMatrixTimesMatrix;
                    uint32_t left_col_type;
                    int right_col_count;
                    if (get_matrix_info(left.type_id, l, &left_col_type, NULL) &&
                        get_matrix_info(right.type_id, l, NULL, &right_col_count)) {
                        result_type = emit_type_matrix(l, left_col_type, right_col_count);
                    } else {
                        result_type = left.type_id;
                    }
                } else if (left_is_matrix && right_vec_count > 0) {
                    // matrix × vector: mat<C,R> * vec<C> -> vec<R>
                    op = SpvOpMatrixTimesVector;
                    uint32_t col_type;
                    if (get_matrix_info(left.type_id, l, &col_type, NULL)) {
                        result_type = col_type;
                    } else {
                        result_type = right.type_id;
                    }
                } else if (left_vec_count > 0 && right_is_matrix) {
                    // vector × matrix: vec<R> * mat<C,R> -> vec<C>
                    op = SpvOpVectorTimesMatrix;
                    int col_count;
                    uint32_t col_type;
                    if (get_matrix_info(right.type_id, l, &col_type, &col_count)) {
                        uint32_t scalar = get_scalar_type(left.type_id, l);
                        result_type = emit_type_vector(l, scalar, col_count);
                    } else {
                        result_type = left.type_id;
                    }
                } else if (left_is_matrix && right_is_scalar) {
                    // matrix × scalar
                    op = SpvOpMatrixTimesScalar;
                    result_type = left.type_id;
                } else if (left_is_scalar && right_is_matrix) {
                    // scalar × matrix -> swap operands
                    ExprResult tmp = left;
                    left = right;
                    right = tmp;
                    op = SpvOpMatrixTimesScalar;
                    result_type = left.type_id;
                } else if (left_vec_count > 0 && right_is_scalar) {
                    // vector × scalar -> use VectorTimesScalar
                    op = SpvOpVectorTimesScalar;
                    result_type = left.type_id;
                } else if (left_is_scalar && right_vec_count > 0) {
                    // scalar × vector -> swap and use VectorTimesScalar
                    // VectorTimesScalar requires vector as first operand
                    ExprResult tmp = left;
                    left = right;
                    right = tmp;
                    op = SpvOpVectorTimesScalar;
                    result_type = left.type_id;
                } else {
                    op = is_float ? SpvOpFMul : SpvOpIMul;
                }
            } else if (strcmp(bin->op, "/") == 0) {
                op = is_float ? SpvOpFDiv : (is_signed ? SpvOpSDiv : SpvOpUDiv);
            } else if (strcmp(bin->op, "%") == 0) {
                op = is_float ? SpvOpFRem : (is_signed ? SpvOpSRem : SpvOpUMod);
            }
            // Comparison operators
            else if (strcmp(bin->op, "==") == 0) {
                op = is_float ? SpvOpFOrdEqual : SpvOpIEqual;
                result_type = l->id_bool;
            } else if (strcmp(bin->op, "!=") == 0) {
                op = is_float ? SpvOpFOrdNotEqual : SpvOpINotEqual;
                result_type = l->id_bool;
            } else if (strcmp(bin->op, "<") == 0) {
                op = is_float ? SpvOpFOrdLessThan : (is_signed ? SpvOpSLessThan : SpvOpULessThan);
                result_type = l->id_bool;
            } else if (strcmp(bin->op, "<=") == 0) {
                op = is_float ? SpvOpFOrdLessThanEqual : (is_signed ? SpvOpSLessThanEqual : SpvOpULessThanEqual);
                result_type = l->id_bool;
            } else if (strcmp(bin->op, ">") == 0) {
                op = is_float ? SpvOpFOrdGreaterThan : (is_signed ? SpvOpSGreaterThan : SpvOpUGreaterThan);
                result_type = l->id_bool;
            } else if (strcmp(bin->op, ">=") == 0) {
                op = is_float ? SpvOpFOrdGreaterThanEqual : (is_signed ? SpvOpSGreaterThanEqual : SpvOpUGreaterThanEqual);
                result_type = l->id_bool;
            }
            // Bitwise operators
            else if (strcmp(bin->op, "&") == 0) {
                op = SpvOpBitwiseAnd;
            } else if (strcmp(bin->op, "|") == 0) {
                op = SpvOpBitwiseOr;
            } else if (strcmp(bin->op, "^") == 0) {
                op = SpvOpBitwiseXor;
            } else if (strcmp(bin->op, "<<") == 0) {
                op = SpvOpShiftLeftLogical;
            } else if (strcmp(bin->op, ">>") == 0) {
                op = is_signed ? SpvOpShiftRightArithmetic : SpvOpShiftRightLogical;
            }
            // Logical operators
            else if (strcmp(bin->op, "&&") == 0) {
                op = SpvOpLogicalAnd;
                result_type = l->id_bool;
            } else if (strcmp(bin->op, "||") == 0) {
                op = SpvOpLogicalOr;
                result_type = l->id_bool;
            }

            if (op != SpvOpNop) {
                uint32_t result;
                if (emit_binary_op(l, op, result_type, &result, left.id, right.id)) {
                    r.id = result;
                    r.type_id = result_type;
                }
            }
            return r;
        }

        case WGSL_NODE_UNARY: {
            const Unary *un = &node->unary;
            ExprResult operand = lower_expr_full(l, un->expr);
            if (!operand.id) return r;

            SpvOp op = SpvOpNop;
            uint32_t result_type = operand.type_id;

            if (strcmp(un->op, "-") == 0) {
                op = is_float_type(operand.type_id, l) ? SpvOpFNegate : SpvOpSNegate;
            } else if (strcmp(un->op, "!") == 0) {
                op = SpvOpLogicalNot;
                result_type = l->id_bool;
            } else if (strcmp(un->op, "~") == 0) {
                op = SpvOpNot;
            }

            if (op != SpvOpNop) {
                uint32_t result;
                if (emit_unary_op(l, op, result_type, &result, operand.id)) {
                    r.id = result;
                    r.type_id = result_type;
                }
            }
            return r;
        }

        case WGSL_NODE_MEMBER: {
            const Member *mem = &node->member;

            // Check for vertex input compound name (e.g., "in.position" for vertex inputs)
            if (mem->object && mem->object->type == WGSL_NODE_IDENT) {
                char compound_name[256];
                snprintf(compound_name, sizeof(compound_name), "%s.%s", mem->object->ident.name, mem->member);

                // Look up in global_map
                uint32_t var_id, elem_type_id;
                SpvStorageClass sc;
                if (find_global_by_name(l, compound_name, &var_id, &elem_type_id, &sc)) {
                    // Found the compound name - load from it
                    uint32_t loaded;
                    if (emit_load(l, elem_type_id, &loaded, var_id)) {
                        r.id = loaded;
                        r.type_id = elem_type_id;
                    }
                    return r;
                }
            }

            // First, check if this is struct member access (for storage buffers, uniforms, inputs/outputs, or local vars)
            // Try to use pointer-based access chain
            SpvStorageClass sc;
            ExprResult ptr = lower_ptr_expr(l, node, &sc);
            if (ptr.id && (sc == SpvStorageClassStorageBuffer || sc == SpvStorageClassUniform ||
                           sc == SpvStorageClassInput || sc == SpvStorageClassOutput ||
                           sc == SpvStorageClassFunction)) {
                // This is struct member access - load from the pointer
                uint32_t loaded;
                if (emit_load(l, ptr.type_id, &loaded, ptr.id)) {
                    r.id = loaded;
                    r.type_id = ptr.type_id;
                }
                return r;
            }

            // Otherwise, try regular value-based access (swizzle, etc.)
            ExprResult obj = lower_expr_full(l, mem->object);
            if (!obj.id) return r;

            // Handle swizzle operations on vectors
            const char *swizzle = mem->member;
            if (!swizzle) return r;
            int swizzle_len = (int)strlen(swizzle);
            if (swizzle_len > 4) swizzle_len = 4;

            // Map swizzle characters to indices
            int indices[4] = {-1, -1, -1, -1};
            int valid = 1;
            for (int i = 0; i < swizzle_len; ++i) {
                char c = swizzle[i];
                if (c == 'x' || c == 'r' || c == 's') indices[i] = 0;
                else if (c == 'y' || c == 'g' || c == 't') indices[i] = 1;
                else if (c == 'z' || c == 'b' || c == 'p') indices[i] = 2;
                else if (c == 'w' || c == 'a' || c == 'q') indices[i] = 3;
                else { valid = 0; break; }
            }

            if (!valid) return r;

            // Determine element type
            uint32_t elem_type = l->id_f32;
            if (obj.type_id == l->id_vec2i || obj.type_id == l->id_vec3i || obj.type_id == l->id_vec4i) {
                elem_type = l->id_i32;
            } else if (obj.type_id == l->id_vec2u || obj.type_id == l->id_vec3u || obj.type_id == l->id_vec4u) {
                elem_type = l->id_u32;
            }

            if (swizzle_len == 1) {
                // Single component extraction
                uint32_t result;
                uint32_t idx = (uint32_t)indices[0];
                if (emit_composite_extract(l, elem_type, &result, obj.id, &idx, 1)) {
                    r.id = result;
                    r.type_id = elem_type;
                }
            } else {
                // Multi-component swizzle using VectorShuffle
                uint32_t result_type = emit_type_vector(l, elem_type, swizzle_len);
                WordBuf *wb = &l->sections.functions;
                uint32_t result_id = fresh_id(l);
                if (!emit_op(wb, SpvOpVectorShuffle, 5 + swizzle_len)) return r;
                if (!wb_push_u32(wb, result_type)) return r;
                if (!wb_push_u32(wb, result_id)) return r;
                if (!wb_push_u32(wb, obj.id)) return r;
                if (!wb_push_u32(wb, obj.id)) return r;
                for (int i = 0; i < swizzle_len; ++i) {
                    if (!wb_push_u32(wb, (uint32_t)indices[i])) return r;
                }
                r.id = result_id;
                r.type_id = result_type;

                // Build SSIR shuffle instruction
                if (l->fn_ctx.ssir_func_id && l->fn_ctx.ssir_block_id) {
                    uint32_t ssir_obj = ssir_id_map_get(l, obj.id);
                    uint32_t ssir_type = spv_type_to_ssir(l, result_type);
                    if (ssir_obj && ssir_type) {
                        uint32_t ssir_indices[4];
                        for (int i = 0; i < swizzle_len; ++i) {
                            ssir_indices[i] = (uint32_t)indices[i];
                        }
                        uint32_t ssir_result = ssir_build_shuffle(l->ssir, l->fn_ctx.ssir_func_id,
                                                                   l->fn_ctx.ssir_block_id, ssir_type,
                                                                   ssir_obj, ssir_obj,
                                                                   ssir_indices, (uint32_t)swizzle_len);
                        ssir_id_map_set(l, result_id, ssir_result);
                    }
                }
            }
            return r;
        }

        case WGSL_NODE_INDEX: {
            const Index *idx = &node->index;

            // First, check if this is storage buffer array indexing
            SpvStorageClass sc;
            ExprResult ptr = lower_ptr_expr(l, node, &sc);
            if (ptr.id && (sc == SpvStorageClassStorageBuffer || sc == SpvStorageClassUniform)) {
                // This is storage buffer array access - load from the pointer
                uint32_t loaded;
                if (emit_load(l, ptr.type_id, &loaded, ptr.id)) {
                    r.id = loaded;
                    r.type_id = ptr.type_id;
                }
                return r;
            }

            // Otherwise, try regular value-based access (vector indexing)
            ExprResult arr = lower_expr_full(l, idx->object);
            ExprResult index = lower_expr_full(l, idx->index);
            if (!arr.id || !index.id) return r;

            // Use VectorExtractDynamic for vectors
            uint32_t elem_type = l->id_f32;
            if (arr.type_id == l->id_vec2i || arr.type_id == l->id_vec3i || arr.type_id == l->id_vec4i) {
                elem_type = l->id_i32;
            } else if (arr.type_id == l->id_vec2u || arr.type_id == l->id_vec3u || arr.type_id == l->id_vec4u) {
                elem_type = l->id_u32;
            }

            WordBuf *wb = &l->sections.functions;
            uint32_t result_id = fresh_id(l);
            if (!emit_op(wb, SpvOpVectorExtractDynamic, 5)) return r;
            if (!wb_push_u32(wb, elem_type)) return r;
            if (!wb_push_u32(wb, result_id)) return r;
            if (!wb_push_u32(wb, arr.id)) return r;
            if (!wb_push_u32(wb, index.id)) return r;
            r.id = result_id;
            r.type_id = elem_type;
            return r;
        }

        case WGSL_NODE_TERNARY: {
            const Ternary *tern = &node->ternary;
            ExprResult cond = lower_expr_full(l, tern->cond);
            ExprResult then_val = lower_expr_full(l, tern->then_expr);
            ExprResult else_val = lower_expr_full(l, tern->else_expr);
            if (!cond.id || !then_val.id || !else_val.id) return r;

            uint32_t result;
            if (emit_select(l, then_val.type_id, &result, cond.id, then_val.id, else_val.id)) {
                r.id = result;
                r.type_id = then_val.type_id;
            }
            return r;
        }

        default:
            return r;
    }
}

// Simple wrapper for backward compatibility
//l nonnull
static uint32_t lower_expression(WgslLower *l, const WgslAstNode *expr) {
    wgsl_compiler_assert(l != NULL, "lower_expression: l is NULL");
    ExprResult r = lower_expr_full(l, expr);
    return r.id;
}

// ---------- Statement Lowering ----------

// Forward declarations for variable collection
static void collect_variables_from_block(WgslLower *l, const WgslAstNode *block);

// Infer type from expression (lightweight version for variable declaration)
//l nonnull
static uint32_t infer_expr_type(WgslLower *l, const WgslAstNode *expr) {
    wgsl_compiler_assert(l != NULL, "infer_expr_type: l is NULL");
    if (!expr) return l->id_f32;

    switch (expr->type) {
        case WGSL_NODE_LITERAL: {
            const Literal *lit = &expr->literal;
            if (lit->kind == WGSL_LIT_INT) {
                const char *s = lit->lexeme;
                if (!s) return l->id_i32;
                size_t len = strlen(s);
                if (len > 0 && (s[len-1] == 'u' || s[len-1] == 'U')) {
                    return l->id_u32;
                }
                return l->id_i32;
            }
            return l->id_f32;
        }
        case WGSL_NODE_BINARY: {
            const Binary *bin = &expr->binary;
            if (!bin->op) return l->id_f32;
            // Comparison operators return bool
            if (strcmp(bin->op, "==") == 0 || strcmp(bin->op, "!=") == 0 ||
                strcmp(bin->op, "<") == 0 || strcmp(bin->op, "<=") == 0 ||
                strcmp(bin->op, ">") == 0 || strcmp(bin->op, ">=") == 0 ||
                strcmp(bin->op, "&&") == 0 || strcmp(bin->op, "||") == 0) {
                return l->id_bool;
            }
            // Otherwise inherit from left operand
            return infer_expr_type(l, bin->left);
        }
        case WGSL_NODE_UNARY: {
            const Unary *un = &expr->unary;
            if (un->op && strcmp(un->op, "!") == 0) {
                return l->id_bool;
            }
            return infer_expr_type(l, un->expr);
        }
        case WGSL_NODE_CALL: {
            const Call *call = &expr->call;
            const char *callee_name = NULL;
            if (call->callee && call->callee->type == WGSL_NODE_IDENT) {
                callee_name = call->callee->ident.name;
            } else if (call->callee && call->callee->type == WGSL_NODE_TYPE) {
                callee_name = call->callee->type_node.name;
            }
            if (callee_name) {
                if (strcmp(callee_name, "vec2f") == 0 || strcmp(callee_name, "vec2") == 0) {
                    return l->id_vec2f ? l->id_vec2f : emit_type_vector(l, l->id_f32, 2);
                }
                if (strcmp(callee_name, "vec3f") == 0 || strcmp(callee_name, "vec3") == 0) {
                    return l->id_vec3f ? l->id_vec3f : emit_type_vector(l, l->id_f32, 3);
                }
                if (strcmp(callee_name, "vec4f") == 0 || strcmp(callee_name, "vec4") == 0) {
                    return l->id_vec4f ? l->id_vec4f : emit_type_vector(l, l->id_f32, 4);
                }
                // Matrix constructors
                if (strncmp(callee_name, "mat", 3) == 0) {
                    int cols = 0, rows = 0;
                    if (sscanf(callee_name, "mat%dx%d", &cols, &rows) == 2 ||
                        sscanf(callee_name, "mat%dx%df", &cols, &rows) == 2) {
                        if (cols >= 2 && cols <= 4 && rows >= 2 && rows <= 4) {
                            uint32_t f32_type = l->id_f32 ? l->id_f32 : emit_type_float(l, 32);
                            uint32_t col_type = emit_type_vector(l, f32_type, rows);
                            return emit_type_matrix(l, col_type, cols);
                        }
                    }
                }
                // dot, length, distance return scalar
                if (strcmp(callee_name, "dot") == 0 || strcmp(callee_name, "length") == 0 ||
                    strcmp(callee_name, "distance") == 0) {
                    return l->id_f32;
                }
                // cross returns vec3
                if (strcmp(callee_name, "cross") == 0) {
                    return l->id_vec3f ? l->id_vec3f : emit_type_vector(l, l->id_f32, 3);
                }
                // Most other functions preserve input type
                if (call->arg_count > 0) {
                    return infer_expr_type(l, call->args[0]);
                }
            }
            return l->id_f32;
        }
        case WGSL_NODE_IDENT: {
            // Look up in locals
            const char *name = expr->ident.name;
            if (!name) return l->id_f32;
            uint32_t ptr_id, type_id;
            if (fn_ctx_find_local(l, name, &ptr_id, &type_id)) {
                return type_id;
            }
            return l->id_f32;
        }
        case WGSL_NODE_MEMBER: {
            const Member *mem = &expr->member;

            // Check for vertex input compound name (e.g., "in.position")
            if (mem->object && mem->object->type == WGSL_NODE_IDENT && mem->object->ident.name && mem->member) {
                char compound_name[256];
                snprintf(compound_name, sizeof(compound_name), "%s.%s", mem->object->ident.name, mem->member);

                uint32_t var_id, elem_type_id;
                SpvStorageClass sc;
                if (find_global_by_name(l, compound_name, &var_id, &elem_type_id, &sc)) {
                    return elem_type_id;
                }
            }

            uint32_t obj_type = infer_expr_type(l, mem->object);

            // Check if object is a vector type (swizzle applies to vectors)
            int vec_count = get_vector_component_count(obj_type, l);
            if (vec_count > 0) {
                // Determine element type from object type
                uint32_t elem_type = get_scalar_type(obj_type, l);

                size_t swizzle_len = mem->member ? strlen(mem->member) : 0;
                if (swizzle_len == 1) {
                    return elem_type;
                } else if (swizzle_len <= 4) {
                    return emit_type_vector(l, elem_type, (int)swizzle_len);
                }
            }

            // For struct member access, try to get the member type from the struct
            // Use lower_ptr_expr to get the actual member type
            SpvStorageClass sc;
            ExprResult ptr = lower_ptr_expr(l, (WgslAstNode*)expr, &sc);
            if (ptr.id && ptr.type_id) {
                return ptr.type_id;
            }

            return l->id_f32;
        }
        default:
            return l->id_f32;
    }
}

// Collect variable declarations from a block (for pre-emit)
//l nonnull
static void collect_variables_from_stmt(WgslLower *l, const WgslAstNode *stmt) {
    wgsl_compiler_assert(l != NULL, "collect_variables_from_stmt: l is NULL");
    if (!stmt) return;

    switch (stmt->type) {
        case WGSL_NODE_VAR_DECL: {
            const VarDecl *vd = &stmt->var_decl;

            // let/const/override are handled elsewhere, not as local variables
            if (vd->kind == WGSL_DECL_LET || vd->kind == WGSL_DECL_CONST || vd->kind == WGSL_DECL_OVERRIDE) {
                break;
            }

            // Lower the type
            uint32_t type_id = l->id_f32; // default
            if (vd->type) {
                type_id = lower_type(l, vd->type);
            } else if (vd->init) {
                type_id = infer_expr_type(l, vd->init);
            }

            uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassFunction, type_id);

            uint32_t var_id;
            if (emit_local_variable(l, ptr_type, type_id, &var_id, 0)) {
                fn_ctx_add_local(l, vd->name, var_id, type_id);
                emit_name(l, var_id, vd->name);

                if (vd->init) {
                    if (l->fn_ctx.pending_init_count >= l->fn_ctx.pending_init_cap) {
                        int new_cap = l->fn_ctx.pending_init_cap ? l->fn_ctx.pending_init_cap * 2 : 16;
                        void *p = WGSL_REALLOC(l->fn_ctx.pending_inits, new_cap * sizeof(l->fn_ctx.pending_inits[0]));
                        if (p) {
                            l->fn_ctx.pending_inits = (__typeof__(l->fn_ctx.pending_inits))p;
                            l->fn_ctx.pending_init_cap = new_cap;
                        }
                    }
                    if (l->fn_ctx.pending_init_count < l->fn_ctx.pending_init_cap) {
                        l->fn_ctx.pending_inits[l->fn_ctx.pending_init_count].var_id = var_id;
                        l->fn_ctx.pending_inits[l->fn_ctx.pending_init_count].init_expr = vd->init;
                        l->fn_ctx.pending_init_count++;
                    }
                }
            }
            break;
        }
        case WGSL_NODE_BLOCK:
            collect_variables_from_block(l, stmt);
            break;
        case WGSL_NODE_IF: {
            const IfStmt *if_stmt = &stmt->if_stmt;
            collect_variables_from_block(l, if_stmt->then_branch);
            if (if_stmt->else_branch) {
                collect_variables_from_block(l, if_stmt->else_branch);
            }
            break;
        }
        case WGSL_NODE_WHILE:
            collect_variables_from_block(l, stmt->while_stmt.body);
            break;
        case WGSL_NODE_FOR: {
            const ForStmt *for_stmt = &stmt->for_stmt;
            if (for_stmt->init) {
                collect_variables_from_stmt(l, for_stmt->init);
            }
            collect_variables_from_block(l, for_stmt->body);
            break;
        }
        case WGSL_NODE_DO_WHILE:
            collect_variables_from_block(l, stmt->do_while_stmt.body);
            break;
        case WGSL_NODE_SWITCH:
            for (int i = 0; i < stmt->switch_stmt.case_count; i++) {
                const WgslAstNode *c = stmt->switch_stmt.cases[i];
                if (!c) continue;
                for (int s = 0; s < c->case_clause.stmt_count; s++)
                    collect_variables_from_stmt(l, c->case_clause.stmts[s]);
            }
            break;
        default:
            break;
    }
}

//l nonnull
static void collect_variables_from_block(WgslLower *l, const WgslAstNode *block) {
    wgsl_compiler_assert(l != NULL, "collect_variables_from_block: l is NULL");
    if (!block) return;

    if (block->type == WGSL_NODE_BLOCK) {
        const Block *b = &block->block;
        for (int i = 0; i < b->stmt_count; ++i) {
            collect_variables_from_stmt(l, b->stmts[i]);
        }
    } else {
        collect_variables_from_stmt(l, block);
    }
}

// Emit pending initializers after all OpVariable instructions
//l nonnull
static void emit_pending_initializers(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "emit_pending_initializers: l is NULL");
    for (int i = 0; i < l->fn_ctx.pending_init_count; ++i) {
        ExprResult init = lower_expr_full(l, l->fn_ctx.pending_inits[i].init_expr);
        if (init.id) {
            emit_store(l, l->fn_ctx.pending_inits[i].var_id, init.id);
        }
    }
    // Clear pending inits
    WGSL_FREE(l->fn_ctx.pending_inits);
    l->fn_ctx.pending_inits = NULL;
    l->fn_ctx.pending_init_count = 0;
    l->fn_ctx.pending_init_cap = 0;
}

//l nonnull
//node nonnull
static int lower_var_decl(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_var_decl: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_var_decl: node is NULL");
    const VarDecl *vd = &node->var_decl;

    // var declarations are handled by collect_variables + pending_inits
    if (vd->kind == WGSL_DECL_VAR) return 1;

    // let/const: evaluate init expression and bind as SSA value
    if (!vd->init) return 1;

    ExprResult init = lower_expr_full(l, vd->init);
    if (!init.id) return 0;

    // Register as a value binding (not a pointer)
    fn_ctx_add_local_ex(l, vd->name, init.id, init.type_id, 1);

    // Emit OpName for this value via SSIR
    uint32_t ssir_id = ssir_id_map_get(l, init.id);
    if (ssir_id && l->ssir) {
        ssir_set_name(l->ssir, ssir_id, vd->name);
    }

    return 1;
}

//l nonnull
//node nonnull
static int lower_assignment(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_assignment: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_assignment: node is NULL");
    const Assign *asgn = &node->assign;

    ExprResult lhs = lower_lvalue(l, asgn->lhs);
    ExprResult rhs = lower_expr_full(l, asgn->rhs);

    if (!lhs.id || !rhs.id) return 0;

    /* Compound assignment: load, op, store */
    if (asgn->op && strcmp(asgn->op, "=") != 0) {
        uint32_t lhs_val;
        if (!emit_load(l, lhs.type_id, &lhs_val, lhs.id)) return 0;

        int is_float = is_float_type(lhs.type_id, l);
        SpvOp op = 0;
        if (strcmp(asgn->op, "+=") == 0) op = is_float ? SpvOpFAdd : SpvOpIAdd;
        else if (strcmp(asgn->op, "-=") == 0) op = is_float ? SpvOpFSub : SpvOpISub;
        else if (strcmp(asgn->op, "*=") == 0) op = is_float ? SpvOpFMul : SpvOpIMul;
        else if (strcmp(asgn->op, "/=") == 0) op = is_float ? SpvOpFDiv : SpvOpSDiv;
        else if (strcmp(asgn->op, "%=") == 0) op = is_float ? SpvOpFRem : SpvOpSRem;
        else if (strcmp(asgn->op, "&=") == 0) op = SpvOpBitwiseAnd;
        else if (strcmp(asgn->op, "|=") == 0) op = SpvOpBitwiseOr;
        else if (strcmp(asgn->op, "^=") == 0) op = SpvOpBitwiseXor;
        else if (strcmp(asgn->op, "<<=") == 0) op = SpvOpShiftLeftLogical;
        else if (strcmp(asgn->op, ">>=") == 0) op = SpvOpShiftRightArithmetic;

        if (op) {
            uint32_t result;
            if (!emit_binary_op(l, op, lhs.type_id, &result, lhs_val, rhs.id)) return 0;
            return emit_store(l, lhs.id, result);
        }
        /* Unknown compound op - fall through to plain store */
    }

    return emit_store(l, lhs.id, rhs.id);
}

//l nonnull
//node nonnull
static int lower_return(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_return: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_return: node is NULL");
    const ReturnNode *ret = &node->return_stmt;

    // For struct outputs, the values are already written via member assignments
    // Just emit void return
    if (l->fn_ctx.uses_struct_output) {
        return emit_return(l);
    }

    if (ret->expr) {
        ExprResult val = lower_expr_full(l, ret->expr);
        if (val.id) {
            // If we have an output variable, store to it and return void
            if (l->fn_ctx.output_var_id) {
                emit_store(l, l->fn_ctx.output_var_id, val.id);
                return emit_return(l);
            } else {
                return emit_return_value(l, val.id);
            }
        }
    }

    return emit_return(l);
}

//l nonnull
//node nonnull
static int lower_if_stmt(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_if_stmt: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_if_stmt: node is NULL");
    const IfStmt *if_stmt = &node->if_stmt;

    // Lower condition
    ExprResult cond = lower_expr_full(l, if_stmt->cond);
    if (!cond.id) return 0;

    // Allocate block IDs
    uint32_t then_label = fresh_id(l);
    uint32_t else_label = if_stmt->else_branch ? fresh_id(l) : 0;
    uint32_t merge_label = fresh_id(l);

    // Create SSIR blocks for then, else (if needed), and merge
    uint32_t ssir_then_block = 0, ssir_else_block = 0, ssir_merge_block = 0;
    if (l->fn_ctx.ssir_func_id) {
        ssir_then_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "if.then");
        if (if_stmt->else_branch) {
            ssir_else_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "if.else");
        }
        ssir_merge_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "if.merge");

        // Map SPV labels to SSIR blocks
        ssir_id_map_set(l, then_label, ssir_then_block);
        if (ssir_else_block) ssir_id_map_set(l, else_label, ssir_else_block);
        ssir_id_map_set(l, merge_label, ssir_merge_block);

        // Emit SSIR conditional branch with merge block
        uint32_t ssir_cond = ssir_id_map_get(l, cond.id);
        uint32_t ssir_false_target = ssir_else_block ? ssir_else_block : ssir_merge_block;
        if (ssir_cond) {
            ssir_build_branch_cond_merge(l->ssir, l->fn_ctx.ssir_func_id, l->fn_ctx.ssir_block_id,
                                         ssir_cond, ssir_then_block, ssir_false_target, ssir_merge_block);
        }
    }

    // Emit selection merge
    emit_selection_merge(l, merge_label);

    // Emit conditional branch
    uint32_t false_target = else_label ? else_label : merge_label;
    emit_branch_conditional(l, cond.id, then_label, false_target);

    // Then block
    WordBuf *wb = &l->sections.functions;
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, then_label);

    // Update SSIR current block to then block
    uint32_t saved_ssir_block = l->fn_ctx.ssir_block_id;
    if (ssir_then_block) l->fn_ctx.ssir_block_id = ssir_then_block;

    if (!lower_block(l, if_stmt->then_branch)) return 0;

    // Branch to merge (if not already terminated)
    if (!l->fn_ctx.has_returned) {
        emit_branch(l, merge_label);
        // Note: emit_branch already emits SSIR branch
    }
    l->fn_ctx.has_returned = 0;

    // Else block
    if (if_stmt->else_branch) {
        emit_op(wb, SpvOpLabel, 2);
        wb_push_u32(wb, else_label);

        // Update SSIR current block to else block
        if (ssir_else_block) l->fn_ctx.ssir_block_id = ssir_else_block;

        if (!lower_block(l, if_stmt->else_branch)) return 0;

        if (!l->fn_ctx.has_returned) {
            emit_branch(l, merge_label);
            // Note: emit_branch already emits SSIR branch
        }
        l->fn_ctx.has_returned = 0;
    }

    // Merge block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, merge_label);

    // Update SSIR current block to merge block
    if (ssir_merge_block) l->fn_ctx.ssir_block_id = ssir_merge_block;

    return 1;
}

//l nonnull
//node nonnull
static int lower_while_stmt(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_while_stmt: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_while_stmt: node is NULL");
    const WhileStmt *while_stmt = &node->while_stmt;

    // Allocate block IDs
    uint32_t header_label = fresh_id(l);
    uint32_t cond_label = fresh_id(l);
    uint32_t body_label = fresh_id(l);
    uint32_t continue_label = fresh_id(l);
    uint32_t merge_label = fresh_id(l);

    // Create SSIR blocks for loop
    // NOTE: We defer creation of continue/merge blocks until after processing the body,
    // so that nested control flow (if-else) blocks are created before them, ensuring
    // proper dominator ordering in the emitted SPIR-V.
    uint32_t ssir_header_block = 0, ssir_cond_block = 0, ssir_body_block = 0;
    uint32_t ssir_continue_block = 0, ssir_merge_block = 0;
    if (l->fn_ctx.ssir_func_id) {
        ssir_header_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "while.header");
        ssir_cond_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "while.cond");
        ssir_body_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "while.body");
        // Pre-allocate IDs for continue/merge but defer block creation
        ssir_continue_block = ssir_module_alloc_id(l->ssir);
        ssir_merge_block = ssir_module_alloc_id(l->ssir);

        // Map SPV labels to SSIR blocks
        ssir_id_map_set(l, header_label, ssir_header_block);
        ssir_id_map_set(l, cond_label, ssir_cond_block);
        ssir_id_map_set(l, body_label, ssir_body_block);
        ssir_id_map_set(l, continue_label, ssir_continue_block);
        ssir_id_map_set(l, merge_label, ssir_merge_block);
    }

    // Push loop context
    if (l->fn_ctx.loop_depth < 16) {
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].merge_block = merge_label;
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].continue_block = continue_label;
    }
    l->fn_ctx.loop_depth++;

    // Branch to header
    emit_branch(l, header_label);

    // Header block with loop merge (must be followed immediately by branch)
    WordBuf *wb = &l->sections.functions;
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, header_label);
    emit_loop_merge(l, merge_label, continue_label);

    // SSIR: Set current block to header and emit loop merge
    if (ssir_header_block) {
        l->fn_ctx.ssir_block_id = ssir_header_block;
        ssir_build_loop_merge(l->ssir, l->fn_ctx.ssir_func_id, ssir_header_block,
                              ssir_merge_block, ssir_continue_block);
        // emit_branch will emit the SSIR branch
    }

    emit_branch(l, cond_label);

    // Condition evaluation block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, cond_label);

    // SSIR: Set current block to cond
    if (ssir_cond_block) l->fn_ctx.ssir_block_id = ssir_cond_block;

    ExprResult cond = lower_expr_full(l, while_stmt->cond);
    if (!cond.id) {
        l->fn_ctx.loop_depth--;
        return 0;
    }

    // SSIR: Emit conditional branch (no merge - loop merge is in header block)
    if (ssir_cond_block) {
        uint32_t ssir_cond_val = ssir_id_map_get(l, cond.id);
        if (ssir_cond_val) {
            ssir_build_branch_cond(l->ssir, l->fn_ctx.ssir_func_id, ssir_cond_block,
                                   ssir_cond_val, ssir_body_block, ssir_merge_block);
        }
    }

    emit_branch_conditional(l, cond.id, body_label, merge_label);

    // Body block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, body_label);

    // SSIR: Set current block to body
    if (ssir_body_block) l->fn_ctx.ssir_block_id = ssir_body_block;

    if (!lower_block(l, while_stmt->body)) {
        l->fn_ctx.loop_depth--;
        return 0;
    }

    // Branch to continue
    if (!l->fn_ctx.has_returned) {
        emit_branch(l, continue_label);
    }
    l->fn_ctx.has_returned = 0;

    // Continue block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, continue_label);

    // SSIR: Create continue block now (after body processing) and set as current
    if (ssir_continue_block) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_continue_block, "while.continue");
        l->fn_ctx.ssir_block_id = ssir_continue_block;
        // emit_branch will emit the SSIR branch
    }

    // Branch back to header
    emit_branch(l, header_label);

    // Merge block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, merge_label);

    // SSIR: Create merge block now and set as current
    if (ssir_merge_block) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_merge_block, "while.merge");
        l->fn_ctx.ssir_block_id = ssir_merge_block;
    }

    // Pop loop context
    l->fn_ctx.loop_depth--;

    return 1;
}

//l nonnull
//node nonnull
static int lower_for_stmt(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_for_stmt: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_for_stmt: node is NULL");
    const ForStmt *for_stmt = &node->for_stmt;

    // Lower initializer (store is emitted here since var was already declared)
    // Actually, the initializer in a for loop creates a new variable that should be
    // handled in the variable collection pass. The init statement lowering here
    // should just be a no-op for var decls.
    // But we may have assignment as init, so handle those:
    if (for_stmt->init && for_stmt->init->type == WGSL_NODE_ASSIGN) {
        lower_statement(l, for_stmt->init);
    }

    // Allocate block IDs
    uint32_t header_label = fresh_id(l);
    uint32_t cond_label = fresh_id(l);
    uint32_t body_label = fresh_id(l);
    uint32_t continue_label = fresh_id(l);
    uint32_t merge_label = fresh_id(l);

    // Create SSIR blocks for loop
    // NOTE: We defer creation of continue/merge blocks until after processing the body,
    // so that nested control flow (if-else) blocks are created before them, ensuring
    // proper dominator ordering in the emitted SPIR-V.
    uint32_t ssir_header_block = 0, ssir_cond_block = 0, ssir_body_block = 0;
    uint32_t ssir_continue_block = 0, ssir_merge_block = 0;
    if (l->fn_ctx.ssir_func_id) {
        ssir_header_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "for.header");
        ssir_cond_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "for.cond");
        ssir_body_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "for.body");
        // Pre-allocate IDs for continue/merge but defer block creation
        ssir_continue_block = ssir_module_alloc_id(l->ssir);
        ssir_merge_block = ssir_module_alloc_id(l->ssir);

        // Map SPV labels to SSIR blocks
        ssir_id_map_set(l, header_label, ssir_header_block);
        ssir_id_map_set(l, cond_label, ssir_cond_block);
        ssir_id_map_set(l, body_label, ssir_body_block);
        ssir_id_map_set(l, continue_label, ssir_continue_block);
        ssir_id_map_set(l, merge_label, ssir_merge_block);
    }

    // Push loop context
    if (l->fn_ctx.loop_depth < 16) {
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].merge_block = merge_label;
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].continue_block = continue_label;
    }
    l->fn_ctx.loop_depth++;

    // Branch to header
    emit_branch(l, header_label);

    // Header block with loop merge (must be followed immediately by branch)
    WordBuf *wb = &l->sections.functions;
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, header_label);
    emit_loop_merge(l, merge_label, continue_label);

    // SSIR: Set current block to header and emit loop merge
    if (ssir_header_block) {
        l->fn_ctx.ssir_block_id = ssir_header_block;
        ssir_build_loop_merge(l->ssir, l->fn_ctx.ssir_func_id, ssir_header_block,
                              ssir_merge_block, ssir_continue_block);
        // emit_branch will emit the SSIR branch
    }

    emit_branch(l, cond_label);

    // Condition evaluation block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, cond_label);

    // SSIR: Set current block to cond
    if (ssir_cond_block) l->fn_ctx.ssir_block_id = ssir_cond_block;

    if (for_stmt->cond) {
        ExprResult cond = lower_expr_full(l, for_stmt->cond);
        if (cond.id) {
            // SSIR: Emit conditional branch (no merge - loop merge is in header block)
            if (ssir_cond_block) {
                uint32_t ssir_cond_val = ssir_id_map_get(l, cond.id);
                if (ssir_cond_val) {
                    ssir_build_branch_cond(l->ssir, l->fn_ctx.ssir_func_id, ssir_cond_block,
                                           ssir_cond_val, ssir_body_block, ssir_merge_block);
                }
            }
            emit_branch_conditional(l, cond.id, body_label, merge_label);
        } else {
            // emit_branch will emit SSIR branch
            emit_branch(l, body_label);
        }
    } else {
        // emit_branch will emit SSIR branch
        emit_branch(l, body_label);
    }

    // Body block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, body_label);

    // SSIR: Set current block to body
    if (ssir_body_block) l->fn_ctx.ssir_block_id = ssir_body_block;

    if (!lower_block(l, for_stmt->body)) {
        l->fn_ctx.loop_depth--;
        return 0;
    }

    // Branch to continue
    if (!l->fn_ctx.has_returned) {
        emit_branch(l, continue_label);
    }
    l->fn_ctx.has_returned = 0;

    // Continue block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, continue_label);

    // SSIR: Create continue block now (after body processing) and set as current
    if (ssir_continue_block) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_continue_block, "for.continue");
        l->fn_ctx.ssir_block_id = ssir_continue_block;
    }

    // Lower continuation expression
    if (for_stmt->cont) {
        lower_statement(l, for_stmt->cont);
    }

    // Branch back to header (emit_branch will also emit SSIR branch)
    emit_branch(l, header_label);

    // Merge block
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, merge_label);

    // SSIR: Create merge block now and set as current
    if (ssir_merge_block) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_merge_block, "for.merge");
        l->fn_ctx.ssir_block_id = ssir_merge_block;
    }

    // Pop loop context
    l->fn_ctx.loop_depth--;

    return 1;
}

//l nonnull
//node nonnull
static int lower_do_while_stmt(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_do_while_stmt: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_do_while_stmt: node is NULL");
    const DoWhileStmt *dw = &node->do_while_stmt;
    WordBuf *wb = &l->sections.functions;

    uint32_t header_label = fresh_id(l);
    uint32_t body_label = fresh_id(l);
    uint32_t continue_label = fresh_id(l);
    uint32_t merge_label = fresh_id(l);

    uint32_t ssir_header = 0, ssir_body = 0, ssir_continue = 0, ssir_merge = 0;
    if (l->fn_ctx.ssir_func_id) {
        ssir_header = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "dowhile.header");
        ssir_body = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "dowhile.body");
        ssir_continue = ssir_module_alloc_id(l->ssir);
        ssir_merge = ssir_module_alloc_id(l->ssir);
        ssir_id_map_set(l, header_label, ssir_header);
        ssir_id_map_set(l, body_label, ssir_body);
        ssir_id_map_set(l, continue_label, ssir_continue);
        ssir_id_map_set(l, merge_label, ssir_merge);
    }

    if (l->fn_ctx.loop_depth < 16) {
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].merge_block = merge_label;
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].continue_block = continue_label;
    }
    l->fn_ctx.loop_depth++;

    emit_branch(l, header_label);

    /* Header: loop merge then branch to body */
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, header_label);
    emit_loop_merge(l, merge_label, continue_label);

    if (ssir_header) {
        l->fn_ctx.ssir_block_id = ssir_header;
        ssir_build_loop_merge(l->ssir, l->fn_ctx.ssir_func_id, ssir_header, ssir_merge, ssir_continue);
    }

    emit_branch(l, body_label);

    /* Body */
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, body_label);
    if (ssir_body) l->fn_ctx.ssir_block_id = ssir_body;

    if (!lower_block(l, dw->body)) { l->fn_ctx.loop_depth--; return 0; }

    if (!l->fn_ctx.has_returned) emit_branch(l, continue_label);
    l->fn_ctx.has_returned = 0;

    /* Continue: evaluate condition */
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, continue_label);
    if (ssir_continue) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_continue, "dowhile.continue");
        l->fn_ctx.ssir_block_id = ssir_continue;
    }

    ExprResult cond = lower_expr_full(l, dw->cond);
    if (!cond.id) { l->fn_ctx.loop_depth--; return 0; }

    if (ssir_continue) {
        uint32_t ssir_cond = ssir_id_map_get(l, cond.id);
        if (ssir_cond)
            ssir_build_branch_cond(l->ssir, l->fn_ctx.ssir_func_id, ssir_continue,
                                   ssir_cond, ssir_header, ssir_merge);
    }
    emit_branch_conditional(l, cond.id, header_label, merge_label);

    /* Merge */
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, merge_label);
    if (ssir_merge) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_merge, "dowhile.merge");
        l->fn_ctx.ssir_block_id = ssir_merge;
    }

    l->fn_ctx.loop_depth--;
    return 1;
}

//l nonnull
//node nonnull
static int lower_switch_stmt(WgslLower *l, const WgslAstNode *node) {
    wgsl_compiler_assert(l != NULL, "lower_switch_stmt: l is NULL");
    wgsl_compiler_assert(node != NULL, "lower_switch_stmt: node is NULL");
    const SwitchStmt *sw = &node->switch_stmt;
    WordBuf *wb = &l->sections.functions;

    ExprResult sel = lower_expr_full(l, sw->expr);
    if (!sel.id) return 0;

    uint32_t merge_label = fresh_id(l);
    uint32_t default_label = merge_label;

    /* Allocate labels for each case */
    uint32_t *case_labels = (uint32_t *)WGSL_MALLOC((size_t)sw->case_count * sizeof(uint32_t));
    for (int i = 0; i < sw->case_count; i++) {
        case_labels[i] = fresh_id(l);
        if (!sw->cases[i] || sw->cases[i]->case_clause.expr == NULL)
            default_label = case_labels[i];
    }

    /* SSIR blocks */
    uint32_t ssir_merge = 0;
    if (l->fn_ctx.ssir_func_id)
        ssir_merge = ssir_module_alloc_id(l->ssir);

    /* Push switch context (reuse loop_stack for break target) */
    if (l->fn_ctx.loop_depth < 16) {
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].merge_block = merge_label;
        l->fn_ctx.loop_stack[l->fn_ctx.loop_depth].continue_block = 0;
    }
    l->fn_ctx.loop_depth++;

    /* Selection merge */
    emit_selection_merge(l, merge_label);

    /* Build OpSwitch: word count = 3 + 2*literal_count */
    int literal_count = 0;
    for (int i = 0; i < sw->case_count; i++) {
        const WgslAstNode *c = sw->cases[i];
        if (c && c->case_clause.expr) literal_count++;
    }

    emit_op(wb, SpvOpSwitch, 3 + 2 * literal_count);
    wb_push_u32(wb, sel.id);
    wb_push_u32(wb, default_label);
    for (int i = 0; i < sw->case_count; i++) {
        const WgslAstNode *c = sw->cases[i];
        if (!c || !c->case_clause.expr) continue;
        int val = 0;
        if (c->case_clause.expr->type == WGSL_NODE_LITERAL && c->case_clause.expr->literal.lexeme)
            val = atoi(c->case_clause.expr->literal.lexeme);
        wb_push_u32(wb, (uint32_t)val);
        wb_push_u32(wb, case_labels[i]);
    }

    /* Lower each case body */
    for (int i = 0; i < sw->case_count; i++) {
        const WgslAstNode *c = sw->cases[i];
        if (!c) continue;

        emit_op(wb, SpvOpLabel, 2);
        wb_push_u32(wb, case_labels[i]);

        if (l->fn_ctx.ssir_func_id) {
            uint32_t ssir_blk = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "switch.case");
            ssir_id_map_set(l, case_labels[i], ssir_blk);
            l->fn_ctx.ssir_block_id = ssir_blk;
        }

        for (int s = 0; s < c->case_clause.stmt_count; s++)
            lower_statement(l, c->case_clause.stmts[s]);

        if (!l->fn_ctx.has_returned)
            emit_branch(l, merge_label);
        l->fn_ctx.has_returned = 0;
    }

    /* Merge block */
    emit_op(wb, SpvOpLabel, 2);
    wb_push_u32(wb, merge_label);
    if (ssir_merge) {
        ssir_block_create_with_id(l->ssir, l->fn_ctx.ssir_func_id, ssir_merge, "switch.merge");
        l->fn_ctx.ssir_block_id = ssir_merge;
    }

    l->fn_ctx.loop_depth--;
    WGSL_FREE(case_labels);
    return 1;
}

//l nonnull
static int lower_break_stmt(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_break_stmt: l is NULL");
    if (l->fn_ctx.loop_depth <= 0) return 1;
    uint32_t merge = l->fn_ctx.loop_stack[l->fn_ctx.loop_depth - 1].merge_block;
    if (merge) emit_branch(l, merge);
    l->fn_ctx.has_returned = 1; /* prevent further emission in this block */
    return 1;
}

//l nonnull
static int lower_continue_stmt(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_continue_stmt: l is NULL");
    if (l->fn_ctx.loop_depth <= 0) return 1;
    uint32_t cont = l->fn_ctx.loop_stack[l->fn_ctx.loop_depth - 1].continue_block;
    if (cont) emit_branch(l, cont);
    l->fn_ctx.has_returned = 1;
    return 1;
}

//l nonnull
static int lower_discard_stmt(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_discard_stmt: l is NULL");
    WordBuf *wb = &l->sections.functions;
    emit_op(wb, SpvOpKill, 1);
    l->fn_ctx.has_returned = 1;
    return 1;
}

//l nonnull
static int lower_statement(WgslLower *l, const WgslAstNode *stmt) {
    wgsl_compiler_assert(l != NULL, "lower_statement: l is NULL");
    if (!stmt) return 1;

    switch (stmt->type) {
        case WGSL_NODE_VAR_DECL:
            return lower_var_decl(l, stmt);

        case WGSL_NODE_ASSIGN:
            return lower_assignment(l, stmt);

        case WGSL_NODE_RETURN:
            l->fn_ctx.has_returned = 1;
            return lower_return(l, stmt);

        case WGSL_NODE_EXPR_STMT: {
            // Check if the expression is an assignment
            const WgslAstNode *expr = stmt->expr_stmt.expr;
            if (expr && expr->type == WGSL_NODE_ASSIGN) {
                return lower_assignment(l, expr);
            }
            // Just evaluate the expression, ignore result
            lower_expression(l, expr);
            return 1;
        }

        case WGSL_NODE_IF:
            return lower_if_stmt(l, stmt);

        case WGSL_NODE_WHILE:
            return lower_while_stmt(l, stmt);

        case WGSL_NODE_FOR:
            return lower_for_stmt(l, stmt);

        case WGSL_NODE_DO_WHILE:
            return lower_do_while_stmt(l, stmt);

        case WGSL_NODE_SWITCH:
            return lower_switch_stmt(l, stmt);

        case WGSL_NODE_BREAK:
            return lower_break_stmt(l);

        case WGSL_NODE_CONTINUE:
            return lower_continue_stmt(l);

        case WGSL_NODE_DISCARD:
            return lower_discard_stmt(l);

        case WGSL_NODE_BLOCK:
            return lower_block(l, stmt);

        default:
            return 1; // Ignore unknown statements
    }
}

//l nonnull
static int lower_block(WgslLower *l, const WgslAstNode *block) {
    wgsl_compiler_assert(l != NULL, "lower_block: l is NULL");
    if (!block) return 1;

    if (block->type == WGSL_NODE_BLOCK) {
        const Block *b = &block->block;
        for (int i = 0; i < b->stmt_count; ++i) {
            if (!lower_statement(l, b->stmts[i])) return 0;
            if (l->fn_ctx.has_returned) break; // Stop after return
        }
    } else {
        // Single statement
        return lower_statement(l, block);
    }

    return 1;
}

// ---------- Lowering Entry Points and Functions ----------

// Check if a type is a texture or sampler
static int is_sampler_or_texture_type(const WgslAstNode *type_node) {
    if (!type_node || type_node->type != WGSL_NODE_TYPE) return 0;
    const char *name = type_node->type_node.name;
    if (!name) return 0;

    // Sampler types
    if (strcmp(name, "sampler") == 0) return 1;
    if (strcmp(name, "sampler_comparison") == 0) return 1;

    // Texture types
    if (strncmp(name, "texture_", 8) == 0) return 1;

    return 0;
}

//l nonnull
static int lower_globals(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_globals: l is NULL");
    int bind_count = 0;
    const WgslSymbolInfo *bindings = wgsl_resolver_binding_vars(l->resolver, &bind_count);
    if (!bindings || bind_count == 0) return 1;

    for (int i = 0; i < bind_count; ++i) {
        const WgslSymbolInfo *sym = &bindings[i];
        if (!sym->decl_node) continue;

        const WgslAstNode *decl = sym->decl_node;
        if (decl->type != WGSL_NODE_GLOBAL_VAR) continue;

        const GlobalVar *gv = &decl->global_var;

        // Determine type and storage class
        uint32_t elem_type = lower_type(l, gv->type);

        // For samplers and textures, always use UniformConstant
        SpvStorageClass sc;
        if (is_sampler_or_texture_type(gv->type)) {
            sc = SpvStorageClassUniformConstant;
        } else {
            sc = wgsl_address_space_to_storage_class(gv->address_space);
        }
        uint32_t ptr_type = emit_type_pointer(l, sc, elem_type);

        // Emit variable
        uint32_t var_id = emit_global_variable(l, ptr_type, sc, gv->name, 0);

        // Emit decorations
        if (sym->has_group) {
            uint32_t set = (uint32_t)sym->group_index;
            emit_decorate(l, var_id, SpvDecorationDescriptorSet, &set, 1);
        }
        if (sym->has_binding) {
            uint32_t binding = (uint32_t)sym->binding_index;
            emit_decorate(l, var_id, SpvDecorationBinding, &binding, 1);
        }

        // Create SSIR global variable
        uint32_t ssir_var = 0;
        {
            uint32_t ssir_elem_type = spv_type_to_ssir(l, elem_type);
            SsirAddressSpace ssir_addr = spv_sc_to_ssir_addr(sc);
            uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, ssir_addr);
            ssir_var = ssir_global_var(l->ssir, gv->name, ssir_ptr_type);
            if (sym->has_group) {
                ssir_global_set_group(l->ssir, ssir_var, (uint32_t)sym->group_index);
            }
            if (sym->has_binding) {
                ssir_global_set_binding(l->ssir, ssir_var, (uint32_t)sym->binding_index);
            }
            ssir_id_map_set(l, var_id, ssir_var);
        }

        // Track mapping
        if (l->global_map_count >= l->global_map_cap) {
            int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 16;
            void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
            if (p) {
                l->global_map = (__typeof__(l->global_map))p;
                l->global_map_cap = new_cap;
            }
        }
        if (l->global_map_count < l->global_map_cap) {
            l->global_map[l->global_map_count].symbol_id = sym->id;
            l->global_map[l->global_map_count].spv_id = var_id;
            l->global_map[l->global_map_count].ssir_id = ssir_var;
            l->global_map[l->global_map_count].type_id = elem_type;
            l->global_map[l->global_map_count].sc = sc;
            l->global_map[l->global_map_count].name = gv->name;
            l->global_map_count++;
        }
    }

    wgsl_resolve_free((void*)bindings);
    return 1;
}

//l nonnull
static void add_global_map_entry(WgslLower *l, int symbol_id, uint32_t spv_id,
                                 uint32_t ssir_id, uint32_t type_id,
                                 SpvStorageClass sc, const char *name) {
    wgsl_compiler_assert(l != NULL, "add_global_map_entry: l is NULL");
    if (l->global_map_count >= l->global_map_cap) {
        int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 16;
        void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
        if (!p) return;
        l->global_map = (__typeof__(l->global_map))p;
        l->global_map_cap = new_cap;
    }
    l->global_map[l->global_map_count].symbol_id = symbol_id;
    l->global_map[l->global_map_count].spv_id = spv_id;
    l->global_map[l->global_map_count].ssir_id = ssir_id;
    l->global_map[l->global_map_count].type_id = type_id;
    l->global_map[l->global_map_count].sc = sc;
    l->global_map[l->global_map_count].name = name;
    l->global_map_count++;
}

//name nonnull
//out_spv nonnull
//out_ssir nonnull
static int resolve_builtin_name(const char *name, SpvBuiltIn *out_spv, SsirBuiltinVar *out_ssir) {
    wgsl_compiler_assert(name != NULL, "resolve_builtin_name: name is NULL");
    wgsl_compiler_assert(out_spv != NULL, "resolve_builtin_name: out_spv is NULL");
    wgsl_compiler_assert(out_ssir != NULL, "resolve_builtin_name: out_ssir is NULL");
    struct { const char *name; SpvBuiltIn spv; SsirBuiltinVar ssir; } table[] = {
        {"position",              SpvBuiltInPosition,             SSIR_BUILTIN_POSITION},
        {"vertex_index",          SpvBuiltInVertexIndex,          SSIR_BUILTIN_VERTEX_INDEX},
        {"instance_index",        SpvBuiltInInstanceIndex,        SSIR_BUILTIN_INSTANCE_INDEX},
        {"front_facing",          SpvBuiltInFrontFacing,          SSIR_BUILTIN_FRONT_FACING},
        {"frag_depth",            SpvBuiltInFragDepth,            SSIR_BUILTIN_FRAG_DEPTH},
        {"sample_index",          SpvBuiltInSampleId,             SSIR_BUILTIN_SAMPLE_INDEX},
        {"sample_mask",           SpvBuiltInSampleMask,           SSIR_BUILTIN_SAMPLE_MASK},
        {"global_invocation_id",  SpvBuiltInGlobalInvocationId,   SSIR_BUILTIN_GLOBAL_INVOCATION_ID},
        {"local_invocation_id",   SpvBuiltInLocalInvocationId,    SSIR_BUILTIN_LOCAL_INVOCATION_ID},
        {"local_invocation_index",SpvBuiltInLocalInvocationIndex, SSIR_BUILTIN_LOCAL_INVOCATION_INDEX},
        {"workgroup_id",          SpvBuiltInWorkgroupId,          SSIR_BUILTIN_WORKGROUP_ID},
        {"num_workgroups",        SpvBuiltInNumWorkgroups,        SSIR_BUILTIN_NUM_WORKGROUPS},
    };
    for (int i = 0; i < (int)(sizeof(table)/sizeof(table[0])); i++) {
        if (strcmp(name, table[i].name) == 0) {
            *out_spv = table[i].spv;
            *out_ssir = table[i].ssir;
            return 1;
        }
    }
    return 0;
}

//l nonnull
static int lower_io_globals(WgslLower *l) {
    wgsl_compiler_assert(l != NULL, "lower_io_globals: l is NULL");
    if (!l->program || l->program->type != WGSL_NODE_PROGRAM) return 1;
    const Program *prog = &l->program->program;

    for (int i = 0; i < prog->decl_count; i++) {
        const WgslAstNode *decl = prog->decls[i];
        if (!decl || decl->type != WGSL_NODE_GLOBAL_VAR) continue;
        const GlobalVar *gv = &decl->global_var;
        if (!gv->address_space) continue;
        if (strcmp(gv->address_space, "in") != 0 && strcmp(gv->address_space, "out") != 0)
            continue;
        if (!gv->type) continue;

        SpvStorageClass sc = wgsl_address_space_to_storage_class(gv->address_space);
        uint32_t elem_type = lower_type(l, gv->type);
        uint32_t ptr_type = emit_type_pointer(l, sc, elem_type);
        uint32_t var_id = emit_global_variable(l, ptr_type, sc, gv->name, 0);

        uint32_t ssir_var = 0;
        if (l->ssir) {
            uint32_t ssir_elem = spv_type_to_ssir(l, elem_type);
            SsirAddressSpace ssir_addr = spv_sc_to_ssir_addr(sc);
            uint32_t ssir_ptr = ssir_type_ptr(l->ssir, ssir_elem, ssir_addr);
            ssir_var = ssir_global_var(l->ssir, gv->name, ssir_ptr);
            ssir_id_map_set(l, var_id, ssir_var);
        }

        /* Apply @location or @builtin decorations */
        for (int a = 0; a < gv->attr_count; a++) {
            const WgslAstNode *attr = gv->attrs[a];
            if (!attr || attr->type != WGSL_NODE_ATTRIBUTE) continue;

            if (strcmp(attr->attribute.name, "location") == 0) {
                int loc = 0;
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL)
                    loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                uint32_t loc_val = (uint32_t)loc;
                emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);
                if (l->ssir) ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
            } else if (strcmp(attr->attribute.name, "builtin") == 0) {
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                    SpvBuiltIn spv_bi;
                    SsirBuiltinVar ssir_bi;
                    if (resolve_builtin_name(attr->attribute.args[0]->ident.name, &spv_bi, &ssir_bi)) {
                        uint32_t bi_val = (uint32_t)spv_bi;
                        emit_decorate(l, var_id, SpvDecorationBuiltIn, &bi_val, 1);
                        if (l->ssir) ssir_global_set_builtin(l->ssir, ssir_var, ssir_bi);
                    }
                }
            }
        }

        /* Find the resolver symbol id for this global */
        int sym_id = -1;
        int sym_count = 0;
        const WgslSymbolInfo *all_syms = wgsl_resolver_all_symbols(l->resolver, &sym_count);
        if (all_syms) {
            for (int s = 0; s < sym_count; s++) {
                if (all_syms[s].decl_node == decl) { sym_id = all_syms[s].id; break; }
            }
            wgsl_resolve_free((void*)all_syms);
        }

        add_global_map_entry(l, sym_id, var_id, ssir_var, elem_type, sc, gv->name);
    }
    return 1;
}

//l nonnull
//ep nonnull
static int lower_function(WgslLower *l, const WgslResolverEntrypoint *ep, uint32_t *out_func_id, uint32_t **out_interface, int *out_interface_count) {
    wgsl_compiler_assert(l != NULL, "lower_function: l is NULL");
    wgsl_compiler_assert(ep != NULL, "lower_function: ep is NULL");
    const WgslAstNode *fn_node = ep->function_node;
    if (!fn_node || fn_node->type != WGSL_NODE_FUNCTION) return 0;

    const Function *fn = &fn_node->function;
    SpvExecutionModel model = stage_to_model(ep->stage);
    (void)model; // May be unused

    // Clear function context
    memset(&l->fn_ctx, 0, sizeof(l->fn_ctx));

    // Record global_map baseline (entries before this are shared or from other EPs)
    int global_map_baseline = l->global_map_count;

    // Lower return type
    uint32_t return_type = emit_type_void(l);
    if (fn->return_type) {
        return_type = lower_type(l, fn->return_type);
    }
    l->fn_ctx.return_type_id = return_type;

    // Create interface variables for vertex/fragment I/O
    uint32_t *interface_ids = NULL;
    int interface_count = 0;
    int interface_cap = 0;

    // Helper to add interface variable
    #define ADD_INTERFACE(id) do { \
        if (interface_count >= interface_cap) { \
            interface_cap = interface_cap ? interface_cap * 2 : 8; \
            interface_ids = (uint32_t*)WGSL_REALLOC(interface_ids, interface_cap * sizeof(uint32_t)); \
        } \
        if (interface_ids) interface_ids[interface_count++] = id; \
    } while(0)

    // For fragment shaders, handle outputs
    if (ep->stage == WGSL_STAGE_FRAGMENT && fn->return_type) {
        // Check for @location attribute on return
        /* PRE: ret_attrs valid if ret_attr_count > 0 */
        wgsl_compiler_assert(fn->ret_attr_count == 0 || fn->ret_attrs != NULL, "lower_function: ret_attrs NULL with count %d", fn->ret_attr_count);
        for (int i = 0; i < fn->ret_attr_count; ++i) {
            const WgslAstNode *attr = fn->ret_attrs[i];
            /* PRE: attr non-NULL */
            wgsl_compiler_assert(attr != NULL, "lower_function: ret_attrs[%d] is NULL", i);
            if (attr->type == WGSL_NODE_ATTRIBUTE && strcmp(attr->attribute.name, "location") == 0) {
                int loc = 0;
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                    loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                }

                uint32_t out_type = lower_type(l, fn->return_type);
                uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassOutput, out_type);
                uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassOutput, "__frag_output", 0);

                uint32_t loc_val = (uint32_t)loc;
                emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);

                // Create SSIR global for fragment output
                uint32_t ssir_var = 0;
                if (l->ssir) {
                    uint32_t ssir_elem_type = spv_type_to_ssir(l, out_type);
                    uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_OUTPUT);
                    ssir_var = ssir_global_var(l->ssir, "__frag_output", ssir_ptr_type);
                    ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                    ssir_id_map_set(l, var_id, ssir_var);
                }

                ADD_INTERFACE(var_id);
                l->fn_ctx.output_var_id = var_id;
                l->fn_ctx.output_type_id = out_type;

                // Store in global_map for SSIR interface
                if (l->global_map_count >= l->global_map_cap) {
                    int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                    void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                    if (p) { l->global_map = (__typeof__(l->global_map))p; l->global_map_cap = new_cap; }
                }
                if (l->global_map_count < l->global_map_cap) {
                    l->global_map[l->global_map_count].symbol_id = -1;
                    l->global_map[l->global_map_count].spv_id = var_id;
                    l->global_map[l->global_map_count].ssir_id = ssir_var;
                    l->global_map[l->global_map_count].type_id = out_type;
                    l->global_map[l->global_map_count].sc = SpvStorageClassOutput;
                    l->global_map[l->global_map_count].name = "__frag_output";
                    l->global_map_count++;
                }
                break;
            }
        }
    }

    // For vertex shaders, handle position output (direct @builtin on return type)
    if (ep->stage == WGSL_STAGE_VERTEX && fn->return_type) {
        /* PRE: ret_attrs valid if ret_attr_count > 0 */
        wgsl_compiler_assert(fn->ret_attr_count == 0 || fn->ret_attrs != NULL, "lower_function vertex: ret_attrs NULL");
        for (int i = 0; i < fn->ret_attr_count; ++i) {
            const WgslAstNode *attr = fn->ret_attrs[i];
            /* PRE: attr non-NULL */
            wgsl_compiler_assert(attr != NULL, "lower_function vertex: ret_attrs[%d] NULL", i);
            if (attr->type == WGSL_NODE_ATTRIBUTE && strcmp(attr->attribute.name, "builtin") == 0) {
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                    const char *builtin_name = attr->attribute.args[0]->ident.name;
                    if (strcmp(builtin_name, "position") == 0) {
                        uint32_t out_type = lower_type(l, fn->return_type);
                        uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassOutput, out_type);
                        uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassOutput, "gl_Position", 0);

                        uint32_t builtin = SpvBuiltInPosition;
                        emit_decorate(l, var_id, SpvDecorationBuiltIn, &builtin, 1);

                        // Create SSIR global for output builtin
                        uint32_t ssir_var = 0;
                        if (l->ssir) {
                            uint32_t ssir_elem_type = spv_type_to_ssir(l, out_type);
                            uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_OUTPUT);
                            ssir_var = ssir_global_var(l->ssir, "gl_Position", ssir_ptr_type);
                            ssir_global_set_builtin(l->ssir, ssir_var, SSIR_BUILTIN_POSITION);
                            ssir_id_map_set(l, var_id, ssir_var);
                        }

                        ADD_INTERFACE(var_id);
                        l->fn_ctx.output_var_id = var_id;
                        l->fn_ctx.output_type_id = out_type;

                        // Store in global_map for SSIR interface
                        if (l->global_map_count >= l->global_map_cap) {
                            int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                            void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                            if (p) { l->global_map = (__typeof__(l->global_map))p; l->global_map_cap = new_cap; }
                        }
                        if (l->global_map_count < l->global_map_cap) {
                            l->global_map[l->global_map_count].symbol_id = -1;
                            l->global_map[l->global_map_count].spv_id = var_id;
                            l->global_map[l->global_map_count].ssir_id = ssir_var;
                            l->global_map[l->global_map_count].type_id = out_type;
                            l->global_map[l->global_map_count].sc = SpvStorageClassOutput;
                            l->global_map[l->global_map_count].name = "gl_Position";
                            l->global_map_count++;
                        }
                        break;
                    }
                }
            }
        }
    }

    // For vertex shaders, handle struct return type with @builtin/@location members
    if (ep->stage == WGSL_STAGE_VERTEX && fn->return_type && fn->return_type->type == WGSL_NODE_TYPE) {
        const char *return_type_name = fn->return_type->type_node.name;

        // Find the struct definition
        if (l->program && l->program->type == WGSL_NODE_PROGRAM) {
            const Program *prog = &l->program->program;

            for (int d = 0; d < prog->decl_count; ++d) {
                const WgslAstNode *decl = prog->decls[d];
                if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

                const StructDecl *sd = &decl->struct_decl;
                if (!sd->name || strcmp(sd->name, return_type_name) != 0) continue;

                // Found the struct - process each field for @builtin or @location
                for (int f = 0; f < sd->field_count; ++f) {
                    const WgslAstNode *field_node = sd->fields[f];
                    if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) continue;

                    const StructField *field = &field_node->struct_field;
                    uint32_t field_type = lower_type(l, field->type);

                    for (int a = 0; a < field->attr_count; ++a) {
                        const WgslAstNode *attr = field->attrs[a];
                        if (attr->type != WGSL_NODE_ATTRIBUTE) continue;

                        if (strcmp(attr->attribute.name, "builtin") == 0) {
                            if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                                const char *builtin_name = attr->attribute.args[0]->ident.name;
                                if (strcmp(builtin_name, "position") == 0) {
                                    uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassOutput, field_type);
                                    uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassOutput, "gl_Position", 0);

                                    uint32_t builtin = SpvBuiltInPosition;
                                    emit_decorate(l, var_id, SpvDecorationBuiltIn, &builtin, 1);
                                    ADD_INTERFACE(var_id);

                                    // Create SSIR global for output builtin
                                    uint32_t ssir_var = 0;
                                    if (l->ssir) {
                                        uint32_t ssir_elem_type = spv_type_to_ssir(l, field_type);
                                        uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_OUTPUT);
                                        ssir_var = ssir_global_var(l->ssir, "gl_Position", ssir_ptr_type);
                                        ssir_global_set_builtin(l->ssir, ssir_var, SSIR_BUILTIN_POSITION);
                                        ssir_id_map_set(l, var_id, ssir_var);
                                    }

                                    // Store in global_map for "out.position" style access
                                    if (l->global_map_count < l->global_map_cap || 1) {
                                        if (l->global_map_count >= l->global_map_cap) {
                                            int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                                            void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                                            if (p) { l->global_map = (__typeof__(l->global_map))p; l->global_map_cap = new_cap; }
                                        }
                                        l->global_map[l->global_map_count].symbol_id = -1;
                                        l->global_map[l->global_map_count].spv_id = var_id;
                                        l->global_map[l->global_map_count].ssir_id = ssir_var;
                                        l->global_map[l->global_map_count].type_id = field_type;
                                        l->global_map[l->global_map_count].sc = SpvStorageClassOutput;
                                        l->global_map[l->global_map_count].name = field->name;
                                        l->global_map_count++;
                                    }
                                }
                            }
                        } else if (strcmp(attr->attribute.name, "location") == 0) {
                            int loc = 0;
                            if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                                loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                            }

                            uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassOutput, field_type);
                            uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassOutput, field->name, 0);

                            uint32_t loc_val = (uint32_t)loc;
                            emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);
                            ADD_INTERFACE(var_id);

                            // Create SSIR global for output with location
                            uint32_t ssir_var = 0;
                            if (l->ssir) {
                                uint32_t ssir_elem_type = spv_type_to_ssir(l, field_type);
                                uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_OUTPUT);
                                ssir_var = ssir_global_var(l->ssir, field->name, ssir_ptr_type);
                                ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                                ssir_id_map_set(l, var_id, ssir_var);
                            }

                            // Store in global_map
                            if (l->global_map_count >= l->global_map_cap) {
                                int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                                void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                                if (p) { l->global_map = (__typeof__(l->global_map))p; l->global_map_cap = new_cap; }
                            }
                            if (l->global_map_count < l->global_map_cap) {
                                l->global_map[l->global_map_count].symbol_id = -1;
                                l->global_map[l->global_map_count].spv_id = var_id;
                                l->global_map[l->global_map_count].ssir_id = ssir_var;
                                l->global_map[l->global_map_count].type_id = field_type;
                                l->global_map[l->global_map_count].sc = SpvStorageClassOutput;
                                l->global_map[l->global_map_count].name = field->name;
                                l->global_map_count++;
                            }
                        }
                    }
                }
                l->fn_ctx.uses_struct_output = 1;  // Flag to indicate struct output (return becomes void)
                break;
            }
        }
    }

    // For vertex shaders, handle direct parameters with @location attributes
    if (ep->stage == WGSL_STAGE_VERTEX && fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;

            const Param *param = &param_node->param;

            // Check for @location attribute directly on the parameter
            for (int a = 0; a < param->attr_count; ++a) {
                const WgslAstNode *attr = param->attrs[a];
                if (attr->type != WGSL_NODE_ATTRIBUTE || strcmp(attr->attribute.name, "location") != 0)
                    continue;

                int loc = 0;
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                    loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                }

                uint32_t param_type = lower_type(l, param->type);
                uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, param_type);
                uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, param->name, 0);

                // Create SSIR global for input
                uint32_t ssir_var = 0;
                if (l->ssir) {
                    uint32_t ssir_elem_type = spv_type_to_ssir(l, param_type);
                    uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                    ssir_var = ssir_global_var(l->ssir, param->name, ssir_ptr_type);
                    ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                    ssir_id_map_set(l, var_id, ssir_var);
                }

                uint32_t loc_val = (uint32_t)loc;
                emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);

                ADD_INTERFACE(var_id);

                // Store in global_map for later lookup by parameter name
                if (l->global_map_count >= l->global_map_cap) {
                    int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                    void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                    if (p) {
                        l->global_map = (__typeof__(l->global_map))p;
                        l->global_map_cap = new_cap;
                    }
                }
                if (l->global_map_count < l->global_map_cap) {
                    l->global_map[l->global_map_count].symbol_id = -1;
                    l->global_map[l->global_map_count].spv_id = var_id;
                    l->global_map[l->global_map_count].ssir_id = ssir_var;
                    l->global_map[l->global_map_count].type_id = param_type;
                    l->global_map[l->global_map_count].sc = SpvStorageClassInput;
                    l->global_map[l->global_map_count].name = param->name;
                    l->global_map_count++;
                }
            }
        }
    }

    // For vertex shaders, handle struct-typed input parameters with @location attributes
    if (ep->stage == WGSL_STAGE_VERTEX && fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;

            const Param *param = &param_node->param;

            // Check if the parameter type is a struct (TYPE node with a name)
            if (!param->type || param->type->type != WGSL_NODE_TYPE) continue;
            const char *type_name = param->type->type_node.name;

            // Find the struct definition
            if (!l->program || l->program->type != WGSL_NODE_PROGRAM) continue;
            const Program *prog = &l->program->program;

            for (int d = 0; d < prog->decl_count; ++d) {
                const WgslAstNode *decl = prog->decls[d];
                if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

                const StructDecl *sd = &decl->struct_decl;
                if (!sd->name || strcmp(sd->name, type_name) != 0) continue;

                // Found the struct, process each field
                for (int f = 0; f < sd->field_count; ++f) {
                    const WgslAstNode *field_node = sd->fields[f];
                    if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) continue;

                    const StructField *field = &field_node->struct_field;

                    // Look for @location attribute on the field
                    /* PRE: attrs valid if attr_count > 0 */
                    wgsl_compiler_assert(field->attr_count == 0 || field->attrs != NULL, "lower_function: field->attrs NULL");
                    for (int a = 0; a < field->attr_count; ++a) {
                        const WgslAstNode *attr = field->attrs[a];
                        /* PRE: attr non-NULL */
                        wgsl_compiler_assert(attr != NULL, "lower_function: field->attrs[%d] NULL", a);
                        if (attr->type != WGSL_NODE_ATTRIBUTE || strcmp(attr->attribute.name, "location") != 0)
                            continue;

                        // Get location value
                        int loc = 0;
                        if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                            loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                        }

                        // Lower the field type
                        uint32_t field_type = lower_type(l, field->type);

                        // Create Input variable for this field
                        uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, field_type);
                        uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, field->name, 0);

                        // Create SSIR global for input
                        uint32_t ssir_var = 0;
                        if (l->ssir) {
                            uint32_t ssir_elem_type = spv_type_to_ssir(l, field_type);
                            uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                            ssir_var = ssir_global_var(l->ssir, field->name, ssir_ptr_type);
                            ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                            ssir_id_map_set(l, var_id, ssir_var);
                        }

                        // Decorate with location
                        uint32_t loc_val = (uint32_t)loc;
                        emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);

                        ADD_INTERFACE(var_id);

                        // Build compound name: "param.field" for lookup
                        char compound_name[256];
                        snprintf(compound_name, sizeof(compound_name), "%s.%s", param->name, field->name);

                        // Store in global_map for later access
                        if (l->global_map_count >= l->global_map_cap) {
                            int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                            void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                            if (p) {
                                l->global_map = (__typeof__(l->global_map))p;
                                l->global_map_cap = new_cap;
                            }
                        }
                        if (l->global_map_count < l->global_map_cap) {
                            l->global_map[l->global_map_count].symbol_id = -1;
                            l->global_map[l->global_map_count].spv_id = var_id;
                            l->global_map[l->global_map_count].ssir_id = ssir_var;
                            l->global_map[l->global_map_count].type_id = field_type;
                            l->global_map[l->global_map_count].sc = SpvStorageClassInput;
                            // Allocate and copy the compound name
                            char *name_copy = (char*)WGSL_MALLOC(strlen(compound_name) + 1);
                            if (name_copy) {
                                strcpy(name_copy, compound_name);
                                l->global_map[l->global_map_count].name = name_copy;
                            }
                            l->global_map_count++;
                        }
                    }
                }
                break; // Found the struct
            }
        }
    }

    // For fragment shaders, handle struct-typed input parameters with @location/@builtin attributes
    if (ep->stage == WGSL_STAGE_FRAGMENT && fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;

            const Param *param = &param_node->param;

            // Check if the parameter type is a struct (TYPE node with a name)
            if (!param->type || param->type->type != WGSL_NODE_TYPE) continue;
            const char *type_name = param->type->type_node.name;

            // Find the struct definition
            if (!l->program || l->program->type != WGSL_NODE_PROGRAM) continue;
            const Program *prog = &l->program->program;

            for (int d = 0; d < prog->decl_count; ++d) {
                const WgslAstNode *decl = prog->decls[d];
                if (!decl || decl->type != WGSL_NODE_STRUCT) continue;

                const StructDecl *sd = &decl->struct_decl;
                if (!sd->name || strcmp(sd->name, type_name) != 0) continue;

                // Found the struct, process each field
                for (int f = 0; f < sd->field_count; ++f) {
                    const WgslAstNode *field_node = sd->fields[f];
                    if (!field_node || field_node->type != WGSL_NODE_STRUCT_FIELD) continue;

                    const StructField *field = &field_node->struct_field;

                    /* PRE: attrs valid */
                    wgsl_compiler_assert(field->attr_count == 0 || field->attrs != NULL, "lower_function frag: field->attrs NULL");
                    for (int a = 0; a < field->attr_count; ++a) {
                        const WgslAstNode *attr = field->attrs[a];
                        /* PRE: attr non-NULL */
                        wgsl_compiler_assert(attr != NULL, "lower_function frag: attrs[%d] NULL", a);
                        if (attr->type != WGSL_NODE_ATTRIBUTE) continue;

                        if (strcmp(attr->attribute.name, "location") == 0) {
                            // Handle @location field
                            int loc = 0;
                            if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                                loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                            }

                            uint32_t field_type = lower_type(l, field->type);
                            uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, field_type);
                            uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, field->name, 0);

                            // Create SSIR global for input
                            uint32_t ssir_var = 0;
                            if (l->ssir) {
                                uint32_t ssir_elem_type = spv_type_to_ssir(l, field_type);
                                uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                                ssir_var = ssir_global_var(l->ssir, field->name, ssir_ptr_type);
                                ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                                ssir_id_map_set(l, var_id, ssir_var);
                            }

                            // Decorate with location
                            uint32_t loc_val = (uint32_t)loc;
                            emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);

                            // Check for @interpolate(flat)
                            for (int ia = 0; ia < field->attr_count; ++ia) {
                                const WgslAstNode *iattr = field->attrs[ia];
                                if (iattr->type == WGSL_NODE_ATTRIBUTE && strcmp(iattr->attribute.name, "interpolate") == 0) {
                                    if (iattr->attribute.arg_count > 0 && iattr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                                        if (strcmp(iattr->attribute.args[0]->ident.name, "flat") == 0) {
                                            uint32_t flat_val = SpvDecorationFlat;
                                            emit_decorate(l, var_id, SpvDecorationFlat, NULL, 0);
                                            if (l->ssir) ssir_global_set_interpolation(l->ssir, ssir_var, SSIR_INTERP_FLAT);
                                        }
                                    }
                                }
                            }

                            ADD_INTERFACE(var_id);

                            // Build compound name: "param.field" for lookup
                            char compound_name[256];
                            snprintf(compound_name, sizeof(compound_name), "%s.%s", param->name, field->name);

                            // Store in global_map for later access
                            char *name_copy = (char*)WGSL_MALLOC(strlen(compound_name) + 1);
                            if (name_copy) {
                                strcpy(name_copy, compound_name);
                                add_global_map_entry(l, -1, var_id, ssir_var, field_type, SpvStorageClassInput, name_copy);
                            }
                        } else if (strcmp(attr->attribute.name, "builtin") == 0) {
                            // Handle @builtin field (e.g., @builtin(position) -> FragCoord)
                            if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                                const char *builtin_name = attr->attribute.args[0]->ident.name;
                                SpvBuiltIn spv_bi;
                                SsirBuiltinVar ssir_bi;
                                if (resolve_builtin_name(builtin_name, &spv_bi, &ssir_bi)) {
                                    uint32_t field_type = lower_type(l, field->type);
                                    uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, field_type);
                                    uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, field->name, 0);

                                    uint32_t bi_val = (uint32_t)spv_bi;
                                    emit_decorate(l, var_id, SpvDecorationBuiltIn, &bi_val, 1);

                                    uint32_t ssir_var = 0;
                                    if (l->ssir) {
                                        uint32_t ssir_elem_type = spv_type_to_ssir(l, field_type);
                                        uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                                        ssir_var = ssir_global_var(l->ssir, field->name, ssir_ptr_type);
                                        ssir_global_set_builtin(l->ssir, ssir_var, ssir_bi);
                                        ssir_id_map_set(l, var_id, ssir_var);
                                    }

                                    ADD_INTERFACE(var_id);

                                    char compound_name[256];
                                    snprintf(compound_name, sizeof(compound_name), "%s.%s", param->name, field->name);

                                    char *name_copy = (char*)WGSL_MALLOC(strlen(compound_name) + 1);
                                    if (name_copy) {
                                        strcpy(name_copy, compound_name);
                                        add_global_map_entry(l, -1, var_id, ssir_var, field_type, SpvStorageClassInput, name_copy);
                                    }
                                }
                            }
                        }
                    }
                }
                break; // Found the struct
            }
        }
    }

    // For fragment shaders, handle direct parameters with @location attributes (varying inputs)
    if (ep->stage == WGSL_STAGE_FRAGMENT && fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;

            const Param *param = &param_node->param;

            // Check for @location attribute
            /* PRE: attrs valid */
            wgsl_compiler_assert(param->attr_count == 0 || param->attrs != NULL, "lower_function frag param: attrs NULL");
            for (int a = 0; a < param->attr_count; ++a) {
                const WgslAstNode *attr = param->attrs[a];
                /* PRE: attr non-NULL */
                wgsl_compiler_assert(attr != NULL, "lower_function frag param: attrs[%d] NULL", a);
                if (attr->type != WGSL_NODE_ATTRIBUTE || strcmp(attr->attribute.name, "location") != 0)
                    continue;

                int loc = 0;
                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_LITERAL) {
                    loc = attr->attribute.args[0]->literal.lexeme ? atoi(attr->attribute.args[0]->literal.lexeme) : 0;
                }

                uint32_t param_type = lower_type(l, param->type);
                uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, param_type);
                uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, param->name, 0);

                // Create SSIR global for input
                uint32_t ssir_var = 0;
                if (l->ssir) {
                    uint32_t ssir_elem_type = spv_type_to_ssir(l, param_type);
                    uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                    ssir_var = ssir_global_var(l->ssir, param->name, ssir_ptr_type);
                    ssir_global_set_location(l->ssir, ssir_var, (uint32_t)loc);
                    ssir_id_map_set(l, var_id, ssir_var);
                }

                uint32_t loc_val = (uint32_t)loc;
                emit_decorate(l, var_id, SpvDecorationLocation, &loc_val, 1);

                ADD_INTERFACE(var_id);

                // Store in global_map for later lookup by parameter name
                if (l->global_map_count >= l->global_map_cap) {
                    int new_cap = l->global_map_cap ? l->global_map_cap * 2 : 32;
                    void *p = WGSL_REALLOC(l->global_map, new_cap * sizeof(l->global_map[0]));
                    if (p) {
                        l->global_map = (__typeof__(l->global_map))p;
                        l->global_map_cap = new_cap;
                    }
                }
                if (l->global_map_count < l->global_map_cap) {
                    l->global_map[l->global_map_count].symbol_id = -1;
                    l->global_map[l->global_map_count].spv_id = var_id;
                    l->global_map[l->global_map_count].ssir_id = ssir_var;
                    l->global_map[l->global_map_count].type_id = param_type;
                    l->global_map[l->global_map_count].sc = SpvStorageClassInput;
                    l->global_map[l->global_map_count].name = param->name;
                    l->global_map_count++;
                }
            }
        }
    }

    // Handle direct @builtin parameters for all shader stages
    if (fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;

            const Param *param = &param_node->param;

            // Check for @builtin attribute
            /* PRE: attrs valid */
            wgsl_compiler_assert(param->attr_count == 0 || param->attrs != NULL, "lower_function builtin: attrs NULL");
            for (int a = 0; a < param->attr_count; ++a) {
                const WgslAstNode *attr = param->attrs[a];
                /* PRE: attr non-NULL */
                wgsl_compiler_assert(attr != NULL, "lower_function builtin: attrs[%d] NULL", a);
                if (attr->type != WGSL_NODE_ATTRIBUTE || strcmp(attr->attribute.name, "builtin") != 0)
                    continue;

                if (attr->attribute.arg_count > 0 && attr->attribute.args[0]->type == WGSL_NODE_IDENT) {
                    const char *builtin_name = attr->attribute.args[0]->ident.name;

                    SpvBuiltIn spv_bi;
                    SsirBuiltinVar ssir_bi;
                    if (!resolve_builtin_name(builtin_name, &spv_bi, &ssir_bi))
                        continue;

                    uint32_t param_type = lower_type(l, param->type);
                    uint32_t ptr_type = emit_type_pointer(l, SpvStorageClassInput, param_type);
                    uint32_t var_id = emit_global_variable(l, ptr_type, SpvStorageClassInput, param->name, 0);

                    uint32_t bi_val = (uint32_t)spv_bi;
                    emit_decorate(l, var_id, SpvDecorationBuiltIn, &bi_val, 1);

                    ADD_INTERFACE(var_id);

                    // Create SSIR global variable for builtin
                    uint32_t ssir_var = 0;
                    if (l->ssir) {
                        uint32_t ssir_elem_type = spv_type_to_ssir(l, param_type);
                        uint32_t ssir_ptr_type = ssir_type_ptr(l->ssir, ssir_elem_type, SSIR_ADDR_INPUT);
                        ssir_var = ssir_global_var(l->ssir, param->name, ssir_ptr_type);
                        ssir_global_set_builtin(l->ssir, ssir_var, ssir_bi);
                        ssir_id_map_set(l, var_id, ssir_var);
                    }

                    add_global_map_entry(l, -1, var_id, ssir_var, param_type,
                                         SpvStorageClassInput, param->name);
                }
            }
        }
    }

    // Add global variables to the interface.
    // Include: shared resources (StorageBuffer, Uniform) from anywhere,
    //          shared I/O from lower_io_globals (indices < shared_globals_end),
    //          per-EP I/O created during THIS entry point (indices >= global_map_baseline).
    // Exclude: I/O from OTHER entry points (between shared_globals_end and global_map_baseline).
    for (int g = 0; g < l->global_map_count; ++g) {
        SpvStorageClass gsc = l->global_map[g].sc;
        if (gsc == SpvStorageClassStorageBuffer || gsc == SpvStorageClassUniform) {
            ADD_INTERFACE(l->global_map[g].spv_id);
        } else if (gsc == SpvStorageClassInput || gsc == SpvStorageClassOutput) {
            if (g < l->shared_globals_end || g >= global_map_baseline) {
                ADD_INTERFACE(l->global_map[g].spv_id);
            }
        }
    }

    #undef ADD_INTERFACE

    // Create function type (void -> void for now since we use interface vars)
    uint32_t void_type = emit_type_void(l);
    uint32_t func_type = emit_type_function(l, void_type, NULL, 0);

    uint32_t func_id = fresh_id(l);
    l->fn_ctx.func_id = func_id;

    // Create SSIR function
    uint32_t ssir_func = ssir_function_create(l->ssir, fn->name, l->ssir_void);
    l->fn_ctx.ssir_func_id = ssir_func;

    // Create SSIR entry block
    uint32_t ssir_block = ssir_block_create(l->ssir, ssir_func, "entry");
    l->fn_ctx.ssir_block_id = ssir_block;

    // Emit function
    if (!emit_function_begin(l, func_id, void_type, func_type)) {
        fn_ctx_clear_locals(l);
        return 0;
    }

    uint32_t label_id;
    if (!emit_label(l, &label_id)) {
        fn_ctx_clear_locals(l);
        return 0;
    }
    l->fn_ctx.label_id = label_id;

    // First pass: collect all variable declarations and emit OpVariable instructions
    // SPIR-V requires all OpVariable to be at the start of the first block
    if (fn->body) {
        collect_variables_from_block(l, fn->body);
    }

    // Emit pending initializers (stores happen after all OpVariable)
    emit_pending_initializers(l);

    // Pre-load fragment input parameters as local value bindings.
    // This is needed because find_global_by_name returns the first match,
    // and vertex output globals (e.g. "color") may shadow fragment input
    // globals with the same name. By registering as locals, lower_ident
    // finds them before checking the global map.
    if (ep->stage == WGSL_STAGE_FRAGMENT && fn->param_count > 0) {
        for (int p = 0; p < fn->param_count; ++p) {
            const WgslAstNode *param_node = fn->params[p];
            if (!param_node || param_node->type != WGSL_NODE_PARAM) continue;
            const Param *param = &param_node->param;

            // Find the input global for this parameter in global_map
            for (int g = 0; g < l->global_map_count; ++g) {
                if (l->global_map[g].sc == SpvStorageClassInput &&
                    l->global_map[g].name && param->name &&
                    strcmp(l->global_map[g].name, param->name) == 0) {
                    uint32_t loaded;
                    if (emit_load(l, l->global_map[g].type_id, &loaded, l->global_map[g].spv_id)) {
                        fn_ctx_add_local_ex(l, param->name, loaded, l->global_map[g].type_id, 1);
                    }
                    break;
                }
            }
        }
    }

    // Register module-scope const and override declarations as value bindings
    if (l->program && l->program->type == WGSL_NODE_PROGRAM) {
        const Program *mprog = &l->program->program;
        for (int d = 0; d < mprog->decl_count; d++) {
            const WgslAstNode *decl = mprog->decls[d];
            if (!decl || decl->type != WGSL_NODE_VAR_DECL) continue;
            const VarDecl *vd = &decl->var_decl;
            if (vd->kind == WGSL_DECL_CONST && vd->init) {
                ExprResult cval = lower_expr_full(l, vd->init);
                if (cval.id) {
                    fn_ctx_add_local_ex(l, vd->name, cval.id, cval.type_id, 1);
                }
            } else if (vd->kind == WGSL_DECL_OVERRIDE) {
                // Look up from override_cache (processed once globally)
                for (int oc = 0; oc < l->override_cache_count; oc++) {
                    if (strcmp(l->override_cache[oc].name, vd->name) == 0) {
                        fn_ctx_add_local_ex(l, vd->name, l->override_cache[oc].spv_id,
                                            l->override_cache[oc].type_id, 1);
                        break;
                    }
                }
            }
        }
    }

    // Second pass: lower the rest of the function body
    if (fn->body) {
        lower_block(l, fn->body);
    }

    // Emit return if not already returned
    if (!l->fn_ctx.has_returned) {
        emit_return(l);
    }

    if (!emit_function_end(l)) {
        fn_ctx_clear_locals(l);
        return 0;
    }

    emit_name(l, func_id, fn->name);

    // Clean up function context
    fn_ctx_clear_locals(l);

    *out_func_id = func_id;
    *out_interface = interface_ids;
    *out_interface_count = interface_count;
    return 1;
}

// ---------- Public API ----------

//program nonnull
//resolver nonnull
WgslLower *wgsl_lower_create(const WgslAstNode *program,
                             const WgslResolver *resolver,
                             const WgslLowerOptions *opts) {
    wgsl_compiler_assert(program != NULL, "wgsl_lower_create: program is NULL");
    wgsl_compiler_assert(resolver != NULL, "wgsl_lower_create: resolver is NULL");
    WgslLower *l = (WgslLower*)WGSL_MALLOC(sizeof(WgslLower));
    if (!l) return NULL;
    memset(l, 0, sizeof(*l));

    // Declare variables at top to avoid goto scope issues
    int ep_count = 0;
    const WgslResolverEntrypoint *eps = NULL;
    uint32_t void_type, func_type, func_id, label_id;
    int i;
    uint32_t *interface_ids;
    int interface_count;
    SpvExecutionModel model;
    uint32_t wg_size[3];

    l->program  = program;
    l->resolver = resolver;
    if (opts) l->opts = *opts;
    else {
        memset(&l->opts, 0, sizeof(l->opts));
        l->opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
        l->opts.spirv_version = 0x00010300;
    }
    if (l->opts.spirv_version == 0) l->opts.spirv_version = 0x00010300;

    l->next_id = 1;
    spv_sections_init(&l->sections);
    type_cache_init(&l->type_cache);

    // Create SSIR module
    l->ssir = ssir_module_create();
    if (!l->ssir) goto fail;

    // Emit capabilities
    if (!emit_capability(l, SpvCapabilityShader)) goto fail;

    // Import GLSL.std.450
    if (!emit_ext_inst_import(l, "GLSL.std.450", &l->id_extinst_glsl)) goto fail;

    // Memory model
    if (!emit_memory_model(l)) goto fail;

    // Emit basic types
    emit_type_void(l);
    emit_type_bool(l);
    emit_type_int(l, 32, 1);  // i32
    emit_type_int(l, 32, 0);  // u32
    emit_type_float(l, 32);   // f32

    // Common vector types
    emit_type_vector(l, l->id_f32, 2);
    emit_type_vector(l, l->id_f32, 3);
    emit_type_vector(l, l->id_f32, 4);

    // Initialize SSIR basic types
    l->ssir_void = ssir_type_void(l->ssir);
    l->ssir_bool = ssir_type_bool(l->ssir);
    l->ssir_i32 = ssir_type_i32(l->ssir);
    l->ssir_u32 = ssir_type_u32(l->ssir);
    l->ssir_f32 = ssir_type_f32(l->ssir);
    l->ssir_f16 = ssir_type_f16(l->ssir);
    l->ssir_vec2f = ssir_type_vec(l->ssir, l->ssir_f32, 2);
    l->ssir_vec3f = ssir_type_vec(l->ssir, l->ssir_f32, 3);
    l->ssir_vec4f = ssir_type_vec(l->ssir, l->ssir_f32, 4);
    l->ssir_vec2i = ssir_type_vec(l->ssir, l->ssir_i32, 2);
    l->ssir_vec3i = ssir_type_vec(l->ssir, l->ssir_i32, 3);
    l->ssir_vec4i = ssir_type_vec(l->ssir, l->ssir_i32, 4);
    l->ssir_vec2u = ssir_type_vec(l->ssir, l->ssir_u32, 2);
    l->ssir_vec3u = ssir_type_vec(l->ssir, l->ssir_u32, 3);
    l->ssir_vec4u = ssir_type_vec(l->ssir, l->ssir_u32, 4);

    // Lower struct types first (so they're available for globals)
    if (!lower_structs(l)) goto fail;

    // Lower globals
    if (!lower_globals(l)) goto fail;

    // Lower in/out globals (GLSL-style I/O)
    if (!lower_io_globals(l)) goto fail;
    l->shared_globals_end = l->global_map_count;

    // Process override declarations once (specialization constants)
    if (l->program && l->program->type == WGSL_NODE_PROGRAM) {
        const Program *mprog = &l->program->program;
        for (int d = 0; d < mprog->decl_count; d++) {
            const WgslAstNode *decl = mprog->decls[d];
            if (!decl || decl->type != WGSL_NODE_VAR_DECL) continue;
            const VarDecl *vd = &decl->var_decl;
            if (vd->kind != WGSL_DECL_OVERRIDE) continue;
            if (l->override_cache_count >= 32) break;

            // Extract @id(N) from attributes if present
            uint32_t spec_id = l->next_spec_id++;
            for (int a = 0; a < vd->attr_count; a++) {
                if (vd->attrs[a] && vd->attrs[a]->type == WGSL_NODE_ATTRIBUTE &&
                    vd->attrs[a]->attribute.name &&
                    strcmp(vd->attrs[a]->attribute.name, "id") == 0 &&
                    vd->attrs[a]->attribute.arg_count > 0 &&
                    vd->attrs[a]->attribute.args[0]) {
                    const WgslAstNode *arg = vd->attrs[a]->attribute.args[0];
                    if (arg->type == WGSL_NODE_LITERAL && arg->literal.lexeme) {
                        spec_id = (uint32_t)atoi(arg->literal.lexeme);
                        if (spec_id >= l->next_spec_id)
                            l->next_spec_id = spec_id + 1;
                    }
                    break;
                }
            }

            // Emit spec constant (creates both SPIR-V and SSIR entries)
            uint32_t sc_id = 0;
            uint32_t sc_type = 0;
            if (vd->init && vd->init->type == WGSL_NODE_LITERAL) {
                const Literal *lit = &vd->init->literal;
                if (lit->kind == WGSL_LIT_FLOAT) {
                    float val = (float)strtod(lit->lexeme ? lit->lexeme : "0", NULL);
                    sc_id = emit_spec_const_f32(l, val, spec_id, vd->name);
                    sc_type = l->id_f32;
                } else if (lit->kind == WGSL_LIT_INT) {
                    const char *s = lit->lexeme ? lit->lexeme : "0";
                    size_t slen = strlen(s);
                    if (slen > 0 && (s[slen-1] == 'u' || s[slen-1] == 'U')) {
                        sc_id = emit_spec_const_u32(l, (uint32_t)strtoul(s, NULL, 0), spec_id, vd->name);
                        sc_type = l->id_u32;
                    } else {
                        sc_id = emit_spec_const_i32(l, (int32_t)strtol(s, NULL, 0), spec_id, vd->name);
                        sc_type = l->id_i32;
                    }
                }
            } else {
                // No init or non-literal: default to f32 with 0.0
                sc_id = emit_spec_const_f32(l, 0.0f, spec_id, vd->name);
                sc_type = l->id_f32;
            }
            if (sc_id) {
                l->override_cache[l->override_cache_count].name = vd->name;
                l->override_cache[l->override_cache_count].spv_id = sc_id;
                l->override_cache[l->override_cache_count].type_id = sc_type;
                l->override_cache_count++;
            }
        }
    }

    // Get entrypoints
    eps = wgsl_resolver_entrypoints(resolver, &ep_count);

    if (ep_count <= 0 || !eps) {
        // Synthesize a default fragment entry
        l->eps = (WgslLowerEntrypointInfo*)WGSL_MALLOC(sizeof(WgslLowerEntrypointInfo));
        if (!l->eps) goto fail;
        l->ep_count = 1;

        void_type = emit_type_void(l);
        func_type = emit_type_function(l, void_type, NULL, 0);
        func_id = fresh_id(l);

        emit_function_begin(l, func_id, void_type, func_type);
        emit_label(l, &label_id);
        emit_return(l);
        emit_function_end(l);

        emit_entry_point(l, func_id, SpvExecutionModelFragment, "main", NULL, 0);
        emit_execution_mode(l, func_id, SpvExecutionModeOriginUpperLeft, NULL, 0);

        // Create SSIR function and entry point for default case
        {
            uint32_t ssir_func = ssir_function_create(l->ssir, "main", l->ssir_void);
            uint32_t ssir_block = ssir_block_create(l->ssir, ssir_func, "entry");
            ssir_build_return_void(l->ssir, ssir_func, ssir_block);
            ssir_entry_point_create(l->ssir, SSIR_STAGE_FRAGMENT, ssir_func, "main");
        }

        l->eps[0].name = "main";
        l->eps[0].stage = WGSL_STAGE_FRAGMENT;
        l->eps[0].function_id = func_id;
        l->eps[0].interface_count = 0;
        l->eps[0].interface_ids = NULL;
    } else {
        l->eps = (WgslLowerEntrypointInfo*)WGSL_MALLOC(sizeof(WgslLowerEntrypointInfo) * (size_t)ep_count);
        if (!l->eps) goto fail;
        l->ep_count = ep_count;

        for (i = 0; i < ep_count; ++i) {
            func_id = 0;
            interface_ids = NULL;
            interface_count = 0;

            if (!lower_function(l, &eps[i], &func_id, &interface_ids, &interface_count)) {
                // Fallback to empty function
                void_type = emit_type_void(l);
                func_type = emit_type_function(l, void_type, NULL, 0);
                func_id = fresh_id(l);

                emit_function_begin(l, func_id, void_type, func_type);
                emit_label(l, &label_id);
                emit_return(l);
                emit_function_end(l);

                // Create fallback SSIR function
                l->fn_ctx.ssir_func_id = ssir_function_create(l->ssir, eps[i].name, l->ssir_void);
                uint32_t ssir_block = ssir_block_create(l->ssir, l->fn_ctx.ssir_func_id, "entry");
                ssir_build_return_void(l->ssir, l->fn_ctx.ssir_func_id, ssir_block);
            }

            model = stage_to_model(eps[i].stage);
            if (model == SpvExecutionModelMax) model = SpvExecutionModelFragment;

            emit_entry_point(l, func_id, model, eps[i].name, interface_ids, interface_count);

            // Create SSIR entry point
            {
                SsirStage ssir_stage = SSIR_STAGE_FRAGMENT;
                if (eps[i].stage == WGSL_STAGE_VERTEX) ssir_stage = SSIR_STAGE_VERTEX;
                else if (eps[i].stage == WGSL_STAGE_COMPUTE) ssir_stage = SSIR_STAGE_COMPUTE;

                uint32_t ssir_ep = ssir_entry_point_create(l->ssir, ssir_stage,
                                                           l->fn_ctx.ssir_func_id, eps[i].name);

                // Add interface variables matching the per-entry-point SpvSections list
                // (deduplicated, since interface_ids may have duplicates)
                for (int g = 0; g < interface_count; ++g) {
                    uint32_t ssir_id = ssir_id_map_get(l, interface_ids[g]);
                    if (!ssir_id) continue;
                    int dup = 0;
                    for (int k = 0; k < g; ++k) {
                        if (ssir_id_map_get(l, interface_ids[k]) == ssir_id) { dup = 1; break; }
                    }
                    if (!dup) {
                        ssir_entry_point_add_interface(l->ssir, ssir_ep, ssir_id);
                    }
                }

                // Set workgroup size for compute shaders
                if (ssir_stage == SSIR_STAGE_COMPUTE) {
                    ssir_entry_point_set_workgroup_size(l->ssir, ssir_ep, 1, 1, 1);
                }
            }

            // Execution modes
            if (model == SpvExecutionModelFragment) {
                emit_execution_mode(l, func_id, SpvExecutionModeOriginUpperLeft, NULL, 0);
            }
            if (model == SpvExecutionModelGLCompute) {
                // Default workgroup size
                wg_size[0] = 1; wg_size[1] = 1; wg_size[2] = 1;
                emit_execution_mode(l, func_id, SpvExecutionModeLocalSize, wg_size, 3);
            }

            l->eps[i].name = eps[i].name;
            l->eps[i].stage = eps[i].stage;
            l->eps[i].function_id = func_id;
            l->eps[i].interface_count = interface_count;
            l->eps[i].interface_ids = interface_ids;
        }
    }

    // Setup reflection
    l->features.capabilities = l->cap_buf;
    l->features.capability_count = l->cap_count;
    l->ext_buf[0] = "GLSL.std.450";
    l->ext_count = 1;
    l->features.extensions = (const char *const *)l->ext_buf;
    l->features.extension_count = l->ext_count;

    return l;

fail:
    spv_sections_free(&l->sections);
    type_cache_free(&l->type_cache);
    if (l->ssir) ssir_module_destroy(l->ssir);
    WGSL_FREE(l->struct_cache);
    WGSL_FREE(l);
    return NULL;
}

void wgsl_lower_destroy(WgslLower *lower) {
    if (!lower) return;
    spv_sections_free(&lower->sections);
    type_cache_free(&lower->type_cache);
    if (lower->ssir) ssir_module_destroy(lower->ssir);
    WGSL_FREE(lower->ssir_id_map);
    WGSL_FREE(lower->const_cache);
    WGSL_FREE(lower->global_map);
    WGSL_FREE(lower->struct_cache);
    for (int i = 0; i < lower->ep_count; ++i) {
        WGSL_FREE((void*)lower->eps[i].interface_ids);
    }
    WGSL_FREE(lower->eps);
    WGSL_FREE(lower);
}

//program nonnull
//resolver nonnull
//out_word_count nonnull
WgslLowerResult wgsl_lower_emit_spirv(const WgslAstNode *program,
                                      const WgslResolver *resolver,
                                      const WgslLowerOptions *opts,
                                      uint32_t **out_words,
                                      size_t *out_word_count) {
    wgsl_compiler_assert(program != NULL, "wgsl_lower_emit_spirv: program is NULL");
    wgsl_compiler_assert(resolver != NULL, "wgsl_lower_emit_spirv: resolver is NULL");
    wgsl_compiler_assert(out_word_count != NULL, "wgsl_lower_emit_spirv: out_word_count is NULL");
    if (!out_word_count) return WGSL_LOWER_ERR_INVALID_INPUT;
    *out_word_count = 0;
    if (out_words) *out_words = NULL;

    WgslLower *l = wgsl_lower_create(program, resolver, opts);
    if (!l) return WGSL_LOWER_ERR_INTERNAL;

    uint32_t *words = NULL;
    size_t count = 0;
    if (!finalize_spirv(l, &words, &count)) {
        wgsl_lower_destroy(l);
        return WGSL_LOWER_ERR_OOM;
    }

    if (out_words) {
        *out_words = words;
    } else {
        WGSL_FREE(words);
    }
    *out_word_count = count;

    wgsl_lower_destroy(l);
    return WGSL_LOWER_OK;
}

//lower nonnull
//out_word_count nonnull
WgslLowerResult wgsl_lower_serialize(const WgslLower *lower,
                                     uint32_t **out_words,
                                     size_t *out_word_count) {
    wgsl_compiler_assert(lower != NULL, "wgsl_lower_serialize: lower is NULL");
    wgsl_compiler_assert(out_word_count != NULL, "wgsl_lower_serialize: out_word_count is NULL");
    if (!lower || !out_word_count) return WGSL_LOWER_ERR_INVALID_INPUT;

    uint32_t *words = NULL;
    size_t count = 0;
    if (!finalize_spirv((WgslLower*)lower, &words, &count)) {
        return WGSL_LOWER_ERR_OOM;
    }

    if (out_words) {
        *out_words = words;
    } else {
        WGSL_FREE(words);
    }
    *out_word_count = count;
    return WGSL_LOWER_OK;
}

//lower nonnull
//out_written nonnull
WgslLowerResult wgsl_lower_serialize_into(const WgslLower *lower,
                                          uint32_t *out_words,
                                          size_t max_words,
                                          size_t *out_written) {
    wgsl_compiler_assert(lower != NULL, "wgsl_lower_serialize_into: lower is NULL");
    wgsl_compiler_assert(out_written != NULL, "wgsl_lower_serialize_into: out_written is NULL");
    if (!lower || !out_written) return WGSL_LOWER_ERR_INVALID_INPUT;

    uint32_t *words = NULL;
    size_t count = 0;
    if (!finalize_spirv((WgslLower*)lower, &words, &count)) {
        return WGSL_LOWER_ERR_OOM;
    }

    *out_written = count;
    if (!out_words || max_words < count) {
        WGSL_FREE(words);
        return WGSL_LOWER_ERR_INVALID_INPUT;
    }

    memcpy(out_words, words, count * sizeof(uint32_t));
    WGSL_FREE(words);
    return WGSL_LOWER_OK;
}

const char *wgsl_lower_last_error(const WgslLower *lower) {
    if (!lower) return "invalid";
    return lower->last_error[0] ? lower->last_error : "";
}

const WgslLowerModuleFeatures *wgsl_lower_module_features(const WgslLower *lower) {
    if (!lower) return NULL;
    return &lower->features;
}

const WgslLowerEntrypointInfo *wgsl_lower_entrypoints(const WgslLower *lower, int *out_count) {
    if (!lower) return NULL;
    if (out_count) *out_count = lower->ep_count;
    return lower->eps;
}

uint32_t wgsl_lower_node_result_id(const WgslLower *lower, const WgslAstNode *node) {
    (void)lower; (void)node;
    return 0;
}

uint32_t wgsl_lower_symbol_result_id(const WgslLower *lower, int symbol_id) {
    if (!lower) return 0;
    for (int i = 0; i < lower->global_map_count; ++i) {
        if (lower->global_map[i].symbol_id == symbol_id) {
            return lower->global_map[i].spv_id;
        }
    }
    return 0;
}

void wgsl_lower_free(void *p) {
    WGSL_FREE(p);
}

const SsirModule *wgsl_lower_get_ssir(const WgslLower *lower) {
    if (!lower) return NULL;
    return lower->ssir;
}
// END FILE wgsl_lower.c
