/*
 * SSIR (Simple Shader IR) - Implementation
 */

#include "simple_wgsl.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

//data nonnull
//capacity nonnull
static int ssir_grow_array(void **data, uint32_t *capacity, uint32_t elem_size, uint32_t needed) {
    wgsl_compiler_assert(data != NULL, "ssir_grow_array: data is NULL");
    wgsl_compiler_assert(capacity != NULL, "ssir_grow_array: capacity is NULL");
    if (*capacity >= needed) return 1;
    uint32_t new_cap = *capacity ? *capacity : 8;
    while (new_cap < needed) new_cap *= 2;
    void *new_data = SSIR_REALLOC(*data, (size_t)new_cap * elem_size);
    if (!new_data) return 0;
    *data = new_data;
    *capacity = new_cap;
    return 1;
}

//s allowed to be NULL
static char *ssir_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *dup = (char *)SSIR_MALLOC(len);
    if (dup) memcpy(dup, s, len);
    return dup;
}

/* ============================================================================
 * ID Lookup Cache (forward declarations for use in getters)
 * ============================================================================ */

#define STAG_TYPE    1
#define STAG_CONST   2
#define STAG_GLOBAL  3
#define STAG_FUNC    4
#define STAG_MASK    7u

typedef struct {
    uintptr_t *entries; /* Tagged pointers: (ptr | tag) */
    uint32_t cap;
} SsirLookupCache;

//ptr allowed to be NULL
static uintptr_t stag_encode(void *ptr, int tag) {
    return (uintptr_t)ptr | (uintptr_t)tag;
}

//entry is not a pointer parameter
static void *stag_decode(uintptr_t entry, int expected_tag) {
    if ((entry & STAG_MASK) != (uintptr_t)expected_tag) return NULL;
    return (void *)(entry & ~(uintptr_t)STAG_MASK);
}

//mod nonnull
static void ssir_module_free_lookup(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_module_free_lookup: mod is NULL");
    if (mod->_lookup_cache) {
        SsirLookupCache *lc = (SsirLookupCache *)mod->_lookup_cache;
        free(lc->entries);
        free(lc);
        mod->_lookup_cache = NULL;
    }
}

/* ============================================================================
 * Module API
 * ============================================================================ */

SsirModule *ssir_module_create(void) {
    SsirModule *mod = (SsirModule *)SSIR_MALLOC(sizeof(SsirModule));
    if (!mod) return NULL;
    memset(mod, 0, sizeof(SsirModule));
    mod->next_id = 1; /* start IDs at 1 (0 is invalid/null) */
    return mod;
}

//t allowed to be NULL
static void ssir_free_type(SsirType *t) {
    if (!t) return;
    if (t->kind == SSIR_TYPE_STRUCT) {
        SSIR_FREE((void *)t->struc.name);
        SSIR_FREE(t->struc.members);
        SSIR_FREE(t->struc.offsets);
        if (t->struc.member_names) {
            for (uint32_t i = 0; i < t->struc.member_count; i++)
                SSIR_FREE((void *)t->struc.member_names[i]);
            SSIR_FREE(t->struc.member_names);
        }
        SSIR_FREE(t->struc.matrix_major);
        SSIR_FREE(t->struc.matrix_strides);
    }
}

//c allowed to be NULL
static void ssir_free_constant(SsirConstant *c) {
    if (!c) return;
    if (c->kind == SSIR_CONST_COMPOSITE) {
        SSIR_FREE(c->composite.components);
    }
}

//g allowed to be NULL
static void ssir_free_global(SsirGlobalVar *g) {
    if (!g) return;
    SSIR_FREE((void *)g->name);
}

//b allowed to be NULL
static void ssir_free_block(SsirBlock *b) {
    if (!b) return;
    SSIR_FREE((void *)b->name);
    for (uint32_t i = 0; i < b->inst_count; i++) {
        SSIR_FREE(b->insts[i].extra);
    }
    SSIR_FREE(b->insts);
}

//f allowed to be NULL
static void ssir_free_function(SsirFunction *f) {
    if (!f) return;
    SSIR_FREE((void *)f->name);
    for (uint32_t i = 0; i < f->param_count; i++) {
        SSIR_FREE((void *)f->params[i].name);
    }
    SSIR_FREE(f->params);
    for (uint32_t i = 0; i < f->local_count; i++) {
        SSIR_FREE((void *)f->locals[i].name);
    }
    SSIR_FREE(f->locals);
    for (uint32_t i = 0; i < f->block_count; i++) {
        ssir_free_block(&f->blocks[i]);
    }
    SSIR_FREE(f->blocks);
}

//ep allowed to be NULL
static void ssir_free_entry_point(SsirEntryPoint *ep) {
    if (!ep) return;
    SSIR_FREE((void *)ep->name);
    SSIR_FREE(ep->interface);
}

//mod allowed to be NULL
void ssir_module_destroy(SsirModule *mod) {
    if (!mod) return;

    for (uint32_t i = 0; i < mod->type_count; i++) {
        ssir_free_type(&mod->types[i]);
    }
    SSIR_FREE(mod->types);

    for (uint32_t i = 0; i < mod->constant_count; i++) {
        ssir_free_constant(&mod->constants[i]);
    }
    SSIR_FREE(mod->constants);

    for (uint32_t i = 0; i < mod->global_count; i++) {
        ssir_free_global(&mod->globals[i]);
    }
    SSIR_FREE(mod->globals);

    for (uint32_t i = 0; i < mod->function_count; i++) {
        ssir_free_function(&mod->functions[i]);
    }
    SSIR_FREE(mod->functions);

    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        ssir_free_entry_point(&mod->entry_points[i]);
    }
    SSIR_FREE(mod->entry_points);

    for (uint32_t i = 0; i < mod->name_count; i++) {
        SSIR_FREE((void *)mod->names[i].name);
    }
    SSIR_FREE(mod->names);

    ssir_module_free_lookup(mod);

    SSIR_FREE(mod);
}

//mod nonnull
uint32_t ssir_module_alloc_id(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_module_alloc_id: mod is NULL");
    return mod->next_id++;
}

//mod allowed to be NULL
//name allowed to be NULL
void ssir_set_name(SsirModule *mod, uint32_t id, const char *name) {
    if (!mod || !name || !*name) return;
    if (mod->name_count >= mod->name_capacity) {
        uint32_t new_cap = mod->name_capacity ? mod->name_capacity * 2 : 16;
        SsirNameEntry *new_names = (SsirNameEntry *)SSIR_REALLOC(mod->names, new_cap * sizeof(SsirNameEntry));
        if (!new_names) return;
        mod->names = new_names;
        mod->name_capacity = new_cap;
    }
    mod->names[mod->name_count].id = id;
    mod->names[mod->name_count].name = ssir_strdup(name);
    mod->name_count++;
}

//mod allowed to be NULL
void ssir_module_set_clip_space(SsirModule *mod, SsirClipSpaceConvention convention) {
    if (mod) mod->clip_space = convention;
}

/* ============================================================================
 * Type API - Internal
 * ============================================================================ */

//mod nonnull
//t nonnull
static uint32_t ssir_add_type(SsirModule *mod, SsirType *t) {
    wgsl_compiler_assert(mod != NULL, "ssir_add_type: mod is NULL");
    wgsl_compiler_assert(t != NULL, "ssir_add_type: t is NULL");
    if (!ssir_grow_array((void **)&mod->types, &mod->type_capacity,
                         sizeof(SsirType), mod->type_count + 1)) {
        return 0;
    }
    uint32_t id = ssir_module_alloc_id(mod);
    t->id = id;
    mod->types[mod->type_count++] = *t;
    return id;
}

/* Find existing type that matches - returns type ID (not array index) */
//mod nonnull
static uint32_t ssir_find_type(SsirModule *mod, SsirTypeKind kind) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == kind) return mod->types[i].id;
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_vec_type(SsirModule *mod, uint32_t elem, uint8_t size) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_vec_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_VEC &&
            mod->types[i].vec.elem == elem &&
            mod->types[i].vec.size == size) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_mat_type(SsirModule *mod, uint32_t elem, uint8_t cols, uint8_t rows) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_mat_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_MAT &&
            mod->types[i].mat.elem == elem &&
            mod->types[i].mat.cols == cols &&
            mod->types[i].mat.rows == rows) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_array_type(SsirModule *mod, uint32_t elem, uint32_t length) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_array_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_ARRAY &&
            mod->types[i].array.elem == elem &&
            mod->types[i].array.length == length) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_runtime_array_type(SsirModule *mod, uint32_t elem) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_runtime_array_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_RUNTIME_ARRAY &&
            mod->types[i].runtime_array.elem == elem) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_ptr_type(SsirModule *mod, uint32_t pointee, SsirAddressSpace space) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_ptr_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_PTR &&
            mod->types[i].ptr.pointee == pointee &&
            mod->types[i].ptr.space == space) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_texture_type(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_texture_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE &&
            mod->types[i].texture.dim == dim &&
            mod->types[i].texture.sampled_type == sampled_type) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_texture_storage_type(SsirModule *mod, SsirTextureDim dim,
                                               uint32_t format, SsirAccessMode access) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_texture_storage_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE_STORAGE &&
            mod->types[i].texture_storage.dim == dim &&
            mod->types[i].texture_storage.format == format &&
            mod->types[i].texture_storage.access == access) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

//mod nonnull
static uint32_t ssir_find_texture_depth_type(SsirModule *mod, SsirTextureDim dim) {
    wgsl_compiler_assert(mod != NULL, "ssir_find_texture_depth_type: mod is NULL");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE_DEPTH &&
            mod->types[i].texture_depth.dim == dim) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

/* ============================================================================
 * Type API - Public
 * ============================================================================ */

//mod nonnull
uint32_t ssir_type_void(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_void: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_VOID);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_VOID };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_bool(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_bool: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_BOOL);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_BOOL };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_i32(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_i32: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_I32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_I32 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_u32(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_u32: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_U32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_U32 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_f32(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_f32: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_F32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_F32 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_f16(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_f16: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_F16);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_F16 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_f64(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_f64: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_F64);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_F64 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_i8(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_i8: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_I8);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_I8 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_u8(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_u8: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_U8);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_U8 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_i16(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_i16: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_I16);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_I16 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_u16(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_u16: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_U16);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_U16 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_i64(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_i64: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_I64);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_I64 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_u64(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_u64: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_U64);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_U64 };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_vec(SsirModule *mod, uint32_t elem_type, uint8_t size) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_vec: mod is NULL");
    uint32_t id = ssir_find_vec_type(mod, elem_type, size);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_VEC };
    t.vec.elem = elem_type;
    t.vec.size = size;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_mat(SsirModule *mod, uint32_t col_type, uint8_t cols, uint8_t rows) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_mat: mod is NULL");
    uint32_t id = ssir_find_mat_type(mod, col_type, cols, rows);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_MAT };
    t.mat.elem = col_type;
    t.mat.cols = cols;
    t.mat.rows = rows;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_array(SsirModule *mod, uint32_t elem_type, uint32_t length) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_array: mod is NULL");
    uint32_t id = ssir_find_array_type(mod, elem_type, length);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_ARRAY };
    t.array.elem = elem_type;
    t.array.length = length;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_array_stride(SsirModule *mod, uint32_t elem_type, uint32_t length, uint32_t stride) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_array_stride: mod is NULL");
    SsirType t = { .kind = SSIR_TYPE_ARRAY };
    t.array.elem = elem_type;
    t.array.length = length;
    t.array.stride = stride;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_runtime_array(SsirModule *mod, uint32_t elem_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_runtime_array: mod is NULL");
    uint32_t id = ssir_find_runtime_array_type(mod, elem_type);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_RUNTIME_ARRAY };
    t.runtime_array.elem = elem_type;
    return ssir_add_type(mod, &t);
}

//mod nonnull
//name allowed to be NULL
//members allowed to be NULL (if member_count is 0)
//offsets allowed to be NULL
//member_names allowed to be NULL
uint32_t ssir_type_struct_named(SsirModule *mod, const char *name,
                                const uint32_t *members, uint32_t member_count,
                                const uint32_t *offsets,
                                const char *const *member_names) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_struct_named: mod is NULL");
    /* Structs are not deduplicated (they can have the same layout but different names) */
    SsirType t = { .kind = SSIR_TYPE_STRUCT };
    t.struc.name = ssir_strdup(name);
    t.struc.member_names = NULL;
    if (member_count > 0) {
        t.struc.members = (uint32_t *)SSIR_MALLOC(member_count * sizeof(uint32_t));
        if (!t.struc.members) return UINT32_MAX;
        memcpy(t.struc.members, members, member_count * sizeof(uint32_t));

        if (offsets) {
            t.struc.offsets = (uint32_t *)SSIR_MALLOC(member_count * sizeof(uint32_t));
            if (!t.struc.offsets) {
                SSIR_FREE(t.struc.members);
                return UINT32_MAX;
            }
            memcpy(t.struc.offsets, offsets, member_count * sizeof(uint32_t));
        }

        if (member_names) {
            t.struc.member_names = (const char **)SSIR_MALLOC(member_count * sizeof(const char *));
            if (t.struc.member_names) {
                for (uint32_t i = 0; i < member_count; i++)
                    t.struc.member_names[i] = member_names[i] ? ssir_strdup(member_names[i]) : NULL;
            }
        }
    }
    t.struc.member_count = member_count;
    return ssir_add_type(mod, &t);
}

//mod nonnull
//name allowed to be NULL
//members allowed to be NULL (if member_count is 0)
//offsets allowed to be NULL
uint32_t ssir_type_struct(SsirModule *mod, const char *name,
                          const uint32_t *members, uint32_t member_count,
                          const uint32_t *offsets) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_struct: mod is NULL");
    return ssir_type_struct_named(mod, name, members, member_count, offsets, NULL);
}

//mod nonnull
uint32_t ssir_type_ptr(SsirModule *mod, uint32_t pointee_type, SsirAddressSpace space) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_ptr: mod is NULL");
    uint32_t id = ssir_find_ptr_type(mod, pointee_type, space);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_PTR };
    t.ptr.pointee = pointee_type;
    t.ptr.space = space;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_sampler(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_sampler: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_SAMPLER);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_SAMPLER };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_sampler_comparison(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_sampler_comparison: mod is NULL");
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_SAMPLER_COMPARISON);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_SAMPLER_COMPARISON };
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_texture(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_texture: mod is NULL");
    uint32_t id = ssir_find_texture_type(mod, dim, sampled_type);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE };
    t.texture.dim = dim;
    t.texture.sampled_type = sampled_type;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_texture_storage(SsirModule *mod, SsirTextureDim dim,
                                   uint32_t format, SsirAccessMode access) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_texture_storage: mod is NULL");
    uint32_t id = ssir_find_texture_storage_type(mod, dim, format, access);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE_STORAGE };
    t.texture_storage.dim = dim;
    t.texture_storage.format = format;
    t.texture_storage.access = access;
    return ssir_add_type(mod, &t);
}

//mod nonnull
uint32_t ssir_type_texture_depth(SsirModule *mod, SsirTextureDim dim) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_texture_depth: mod is NULL");
    uint32_t id = ssir_find_texture_depth_type(mod, dim);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE_DEPTH };
    t.texture_depth.dim = dim;
    return ssir_add_type(mod, &t);
}

//mod nonnull
SsirType *ssir_get_type(SsirModule *mod, uint32_t type_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_type: mod is NULL");
    if (mod->_lookup_cache) {
        SsirLookupCache *lc = (SsirLookupCache *)mod->_lookup_cache;
        if (type_id < lc->cap)
            return (SsirType *)stag_decode(lc->entries[type_id], STAG_TYPE);
        return NULL;
    }
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].id == type_id) {
            return &mod->types[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Type Classification Helpers
 * ============================================================================ */

//t allowed to be NULL
bool ssir_type_is_scalar(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_BOOL ||
           t->kind == SSIR_TYPE_I32 ||
           t->kind == SSIR_TYPE_U32 ||
           t->kind == SSIR_TYPE_F32 ||
           t->kind == SSIR_TYPE_F16;
}

//t allowed to be NULL
bool ssir_type_is_integer(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_U32 ||
           t->kind == SSIR_TYPE_I8 || t->kind == SSIR_TYPE_U8 ||
           t->kind == SSIR_TYPE_I16 || t->kind == SSIR_TYPE_U16 ||
           t->kind == SSIR_TYPE_I64 || t->kind == SSIR_TYPE_U64;
}

//t allowed to be NULL
bool ssir_type_is_signed(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_I8 ||
           t->kind == SSIR_TYPE_I16 || t->kind == SSIR_TYPE_I64;
}

//t allowed to be NULL
bool ssir_type_is_unsigned(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_U32 || t->kind == SSIR_TYPE_U8 ||
           t->kind == SSIR_TYPE_U16 || t->kind == SSIR_TYPE_U64;
}

//t allowed to be NULL
bool ssir_type_is_float(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16 || t->kind == SSIR_TYPE_F64;
}

//t allowed to be NULL
bool ssir_type_is_bool(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_BOOL;
}

//t allowed to be NULL
bool ssir_type_is_vector(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_VEC;
}

//t allowed to be NULL
bool ssir_type_is_matrix(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_MAT;
}

//t allowed to be NULL
bool ssir_type_is_numeric(const SsirType *t) {
    if (!t) return false;
    return ssir_type_is_integer(t) || ssir_type_is_float(t);
}

//mod nonnull
uint32_t ssir_type_scalar_of(SsirModule *mod, uint32_t type_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_scalar_of: mod is NULL");
    SsirType *t = ssir_get_type(mod, type_id);
    if (!t) return UINT32_MAX;

    switch (t->kind) {
        case SSIR_TYPE_VEC:
            return t->vec.elem;
        case SSIR_TYPE_MAT:
            /* Matrix element type is a vector, get its element */
            return ssir_type_scalar_of(mod, t->mat.elem);
        default:
            /* Already scalar or not applicable */
            return type_id;
    }
}

/* ============================================================================
 * Constant API
 * ============================================================================ */

//mod nonnull
//c nonnull
static uint32_t ssir_add_constant(SsirModule *mod, SsirConstant *c) {
    wgsl_compiler_assert(mod != NULL, "ssir_add_constant: mod is NULL");
    wgsl_compiler_assert(c != NULL, "ssir_add_constant: c is NULL");
    if (!ssir_grow_array((void **)&mod->constants, &mod->constant_capacity,
                         sizeof(SsirConstant), mod->constant_count + 1)) {
        return 0;
    }
    c->id = ssir_module_alloc_id(mod);
    mod->constants[mod->constant_count++] = *c;
    return c->id;
}

//mod nonnull
uint32_t ssir_const_bool(SsirModule *mod, bool val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_bool: mod is NULL");
    /* Check for existing constant */
    uint32_t bool_type = ssir_type_bool(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_BOOL &&
            mod->constants[i].bool_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = bool_type,
        .kind = SSIR_CONST_BOOL,
        .bool_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_i32(SsirModule *mod, int32_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_i32: mod is NULL");
    uint32_t i32_type = ssir_type_i32(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_I32 &&
            mod->constants[i].i32_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = i32_type,
        .kind = SSIR_CONST_I32,
        .i32_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_u32(SsirModule *mod, uint32_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_u32: mod is NULL");
    uint32_t u32_type = ssir_type_u32(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_U32 &&
            mod->constants[i].u32_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = u32_type,
        .kind = SSIR_CONST_U32,
        .u32_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_f32(SsirModule *mod, float val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_f32: mod is NULL");
    uint32_t f32_type = ssir_type_f32(mod);
    /* Note: comparing floats with memcmp to handle NaN/negative zero correctly */
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_F32 &&
            memcmp(&mod->constants[i].f32_val, &val, sizeof(float)) == 0) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = f32_type,
        .kind = SSIR_CONST_F32,
        .f32_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_f16(SsirModule *mod, uint16_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_f16: mod is NULL");
    uint32_t f16_type = ssir_type_f16(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_F16 &&
            mod->constants[i].f16_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = f16_type,
        .kind = SSIR_CONST_F16,
        .f16_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_f64(SsirModule *mod, double val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_f64: mod is NULL");
    uint32_t f64_type = ssir_type_f64(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_F64 &&
            memcmp(&mod->constants[i].f64_val, &val, sizeof(double)) == 0) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = f64_type,
        .kind = SSIR_CONST_F64,
        .f64_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_i8(SsirModule *mod, int8_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_i8: mod is NULL");
    uint32_t i8_type = ssir_type_i8(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_I8 &&
            mod->constants[i].i8_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = i8_type,
        .kind = SSIR_CONST_I8,
        .i8_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_u8(SsirModule *mod, uint8_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_u8: mod is NULL");
    uint32_t u8_type = ssir_type_u8(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_U8 &&
            mod->constants[i].u8_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = u8_type,
        .kind = SSIR_CONST_U8,
        .u8_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_i16(SsirModule *mod, int16_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_i16: mod is NULL");
    uint32_t i16_type = ssir_type_i16(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_I16 &&
            mod->constants[i].i16_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = i16_type,
        .kind = SSIR_CONST_I16,
        .i16_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_u16(SsirModule *mod, uint16_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_u16: mod is NULL");
    uint32_t u16_type = ssir_type_u16(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_U16 &&
            mod->constants[i].u16_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = u16_type,
        .kind = SSIR_CONST_U16,
        .u16_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_i64(SsirModule *mod, int64_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_i64: mod is NULL");
    uint32_t i64_type = ssir_type_i64(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_I64 &&
            mod->constants[i].i64_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = i64_type,
        .kind = SSIR_CONST_I64,
        .i64_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_u64(SsirModule *mod, uint64_t val) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_u64: mod is NULL");
    uint32_t u64_type = ssir_type_u64(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_U64 &&
            mod->constants[i].u64_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = u64_type,
        .kind = SSIR_CONST_U64,
        .u64_val = val
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
//components allowed to be NULL (if count is 0)
uint32_t ssir_const_composite(SsirModule *mod, uint32_t type_id,
                              const uint32_t *components, uint32_t count) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_composite: mod is NULL");
    SsirConstant c = {
        .type = type_id,
        .kind = SSIR_CONST_COMPOSITE,
    };
    if (count > 0) {
        c.composite.components = (uint32_t *)SSIR_MALLOC(count * sizeof(uint32_t));
        if (!c.composite.components) return 0;
        memcpy(c.composite.components, components, count * sizeof(uint32_t));
    }
    c.composite.count = count;
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_null(SsirModule *mod, uint32_t type_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_null: mod is NULL");
    SsirConstant c = {
        .type = type_id,
        .kind = SSIR_CONST_NULL,
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_spec_bool(SsirModule *mod, bool default_val, uint32_t spec_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_spec_bool: mod is NULL");
    SsirConstant c = {
        .type = ssir_type_bool(mod),
        .kind = SSIR_CONST_BOOL,
        .is_specialization = true,
        .spec_id = spec_id,
        .bool_val = default_val,
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_spec_i32(SsirModule *mod, int32_t default_val, uint32_t spec_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_spec_i32: mod is NULL");
    SsirConstant c = {
        .type = ssir_type_i32(mod),
        .kind = SSIR_CONST_I32,
        .is_specialization = true,
        .spec_id = spec_id,
        .i32_val = default_val,
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_spec_u32(SsirModule *mod, uint32_t default_val, uint32_t spec_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_spec_u32: mod is NULL");
    SsirConstant c = {
        .type = ssir_type_u32(mod),
        .kind = SSIR_CONST_U32,
        .is_specialization = true,
        .spec_id = spec_id,
        .u32_val = default_val,
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
uint32_t ssir_const_spec_f32(SsirModule *mod, float default_val, uint32_t spec_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_const_spec_f32: mod is NULL");
    SsirConstant c = {
        .type = ssir_type_f32(mod),
        .kind = SSIR_CONST_F32,
        .is_specialization = true,
        .spec_id = spec_id,
        .f32_val = default_val,
    };
    return ssir_add_constant(mod, &c);
}

//mod nonnull
SsirConstant *ssir_get_constant(SsirModule *mod, uint32_t const_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_constant: mod is NULL");
    if (mod->_lookup_cache) {
        SsirLookupCache *lc = (SsirLookupCache *)mod->_lookup_cache;
        if (const_id < lc->cap)
            return (SsirConstant *)stag_decode(lc->entries[const_id], STAG_CONST);
        return NULL;
    }
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].id == const_id) {
            return &mod->constants[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Global Variable API
 * ============================================================================ */

//mod nonnull
//name allowed to be NULL
uint32_t ssir_global_var(SsirModule *mod, const char *name, uint32_t ptr_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_var: mod is NULL");
    if (!ssir_grow_array((void **)&mod->globals, &mod->global_capacity,
                         sizeof(SsirGlobalVar), mod->global_count + 1)) {
        return 0;
    }
    SsirGlobalVar *g = &mod->globals[mod->global_count++];
    memset(g, 0, sizeof(SsirGlobalVar));
    g->id = ssir_module_alloc_id(mod);
    g->name = ssir_strdup(name);
    g->type = ptr_type;
    return g->id;
}

//mod nonnull
SsirGlobalVar *ssir_get_global(SsirModule *mod, uint32_t global_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_global: mod is NULL");
    if (mod->_lookup_cache) {
        SsirLookupCache *lc = (SsirLookupCache *)mod->_lookup_cache;
        if (global_id < lc->cap)
            return (SsirGlobalVar *)stag_decode(lc->entries[global_id], STAG_GLOBAL);
        return NULL;
    }
    for (uint32_t i = 0; i < mod->global_count; i++) {
        if (mod->globals[i].id == global_id) {
            return &mod->globals[i];
        }
    }
    return NULL;
}

//mod nonnull
void ssir_global_set_group(SsirModule *mod, uint32_t global_id, uint32_t group) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_group: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_group = true;
        g->group = group;
    }
}

//mod nonnull
void ssir_global_set_binding(SsirModule *mod, uint32_t global_id, uint32_t binding) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_binding: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_binding = true;
        g->binding = binding;
    }
}

//mod nonnull
void ssir_global_set_location(SsirModule *mod, uint32_t global_id, uint32_t location) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_location: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_location = true;
        g->location = location;
    }
}

//mod nonnull
void ssir_global_set_builtin(SsirModule *mod, uint32_t global_id, SsirBuiltinVar builtin) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_builtin: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->builtin = builtin;
    }
}

//mod nonnull
void ssir_global_set_interpolation(SsirModule *mod, uint32_t global_id, SsirInterpolation interp) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_interpolation: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->interp = interp;
    }
}

//mod nonnull
void ssir_global_set_interp_sampling(SsirModule *mod, uint32_t global_id, SsirInterpolationSampling sampling) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_interp_sampling: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->interp_sampling = sampling;
    }
}

//mod nonnull
void ssir_global_set_non_writable(SsirModule *mod, uint32_t global_id, bool non_writable) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_non_writable: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->non_writable = non_writable;
    }
}

//mod nonnull
void ssir_global_set_invariant(SsirModule *mod, uint32_t global_id, bool invariant) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_invariant: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->invariant = invariant;
    }
}

//mod nonnull
void ssir_global_set_initializer(SsirModule *mod, uint32_t global_id, uint32_t const_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_global_set_initializer: mod is NULL");
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_initializer = true;
        g->initializer = const_id;
    }
}

/* ============================================================================
 * Function API
 * ============================================================================ */

//mod nonnull
//name allowed to be NULL
uint32_t ssir_function_create(SsirModule *mod, const char *name, uint32_t return_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_function_create: mod is NULL");
    if (!ssir_grow_array((void **)&mod->functions, &mod->function_capacity,
                         sizeof(SsirFunction), mod->function_count + 1)) {
        return 0;
    }
    SsirFunction *f = &mod->functions[mod->function_count++];
    memset(f, 0, sizeof(SsirFunction));
    f->id = ssir_module_alloc_id(mod);
    f->name = ssir_strdup(name);
    f->return_type = return_type;
    return f->id;
}

//mod nonnull
SsirFunction *ssir_get_function(SsirModule *mod, uint32_t func_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_function: mod is NULL");
    if (mod->_lookup_cache) {
        SsirLookupCache *lc = (SsirLookupCache *)mod->_lookup_cache;
        if (func_id < lc->cap)
            return (SsirFunction *)stag_decode(lc->entries[func_id], STAG_FUNC);
        return NULL;
    }
    for (uint32_t i = 0; i < mod->function_count; i++) {
        if (mod->functions[i].id == func_id) {
            return &mod->functions[i];
        }
    }
    return NULL;
}

//mod nonnull
//name allowed to be NULL
uint32_t ssir_function_add_param(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t type) {
    wgsl_compiler_assert(mod != NULL, "ssir_function_add_param: mod is NULL");
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    uint32_t new_count = f->param_count + 1;
    SsirFunctionParam *new_params = (SsirFunctionParam *)SSIR_REALLOC(
        f->params, new_count * sizeof(SsirFunctionParam));
    if (!new_params) return 0;
    f->params = new_params;

    SsirFunctionParam *p = &f->params[f->param_count++];
    p->id = ssir_module_alloc_id(mod);
    p->name = ssir_strdup(name);
    p->type = type;
    return p->id;
}

//mod nonnull
//name allowed to be NULL
uint32_t ssir_function_add_local(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t ptr_type) {
    wgsl_compiler_assert(mod != NULL, "ssir_function_add_local: mod is NULL");
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->locals, &f->local_capacity,
                         sizeof(SsirLocalVar), f->local_count + 1)) {
        return 0;
    }

    SsirLocalVar *l = &f->locals[f->local_count++];
    memset(l, 0, sizeof(SsirLocalVar));
    l->id = ssir_module_alloc_id(mod);
    l->name = ssir_strdup(name);
    l->type = ptr_type;
    return l->id;
}

/* ============================================================================
 * Block API
 * ============================================================================ */

//mod nonnull
//name allowed to be NULL
uint32_t ssir_block_create(SsirModule *mod, uint32_t func_id, const char *name) {
    wgsl_compiler_assert(mod != NULL, "ssir_block_create: mod is NULL");
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->blocks, &f->block_capacity,
                         sizeof(SsirBlock), f->block_count + 1)) {
        return 0;
    }

    SsirBlock *b = &f->blocks[f->block_count++];
    memset(b, 0, sizeof(SsirBlock));
    b->id = ssir_module_alloc_id(mod);
    b->name = ssir_strdup(name);
    return b->id;
}

//mod nonnull
//name allowed to be NULL
uint32_t ssir_block_create_with_id(SsirModule *mod, uint32_t func_id, uint32_t block_id, const char *name) {
    wgsl_compiler_assert(mod != NULL, "ssir_block_create_with_id: mod is NULL");
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->blocks, &f->block_capacity,
                         sizeof(SsirBlock), f->block_count + 1)) {
        return 0;
    }

    SsirBlock *b = &f->blocks[f->block_count++];
    memset(b, 0, sizeof(SsirBlock));
    b->id = block_id;
    b->name = ssir_strdup(name);
    return b->id;
}

//mod nonnull
SsirBlock *ssir_get_block(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_block: mod is NULL");
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return NULL;

    for (uint32_t i = 0; i < f->block_count; i++) {
        if (f->blocks[i].id == block_id) {
            return &f->blocks[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Instruction Builder - Internal
 * ============================================================================ */

//mod nonnull
static SsirInst *ssir_add_inst(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_add_inst: mod is NULL");
    SsirBlock *b = ssir_get_block(mod, func_id, block_id);
    if (!b) return NULL;

    if (!ssir_grow_array((void **)&b->insts, &b->inst_capacity,
                         sizeof(SsirInst), b->inst_count + 1)) {
        return NULL;
    }

    SsirInst *inst = &b->insts[b->inst_count++];
    memset(inst, 0, sizeof(SsirInst));
    return inst;
}

//mod nonnull
static uint32_t ssir_emit_binary(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                 SsirOpcode op, uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_emit_binary: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = op;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = a;
    inst->operands[1] = b;
    inst->operand_count = 2;
    return inst->result;
}

//mod nonnull
static uint32_t ssir_emit_unary(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                SsirOpcode op, uint32_t type, uint32_t a) {
    wgsl_compiler_assert(mod != NULL, "ssir_emit_unary: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = op;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = a;
    inst->operand_count = 1;
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Arithmetic
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_add(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_add: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_ADD, type, a, b);
}

//mod nonnull
uint32_t ssir_build_sub(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_sub: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SUB, type, a, b);
}

//mod nonnull
uint32_t ssir_build_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_mul: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MUL, type, a, b);
}

//mod nonnull
uint32_t ssir_build_div(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_div: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_DIV, type, a, b);
}

//mod nonnull
uint32_t ssir_build_mod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_mod: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MOD, type, a, b);
}

//mod nonnull
uint32_t ssir_build_neg(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_neg: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_NEG, type, a);
}

/* ============================================================================
 * Instruction Builder - Matrix
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_mat_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_mat_mul: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MAT_MUL, type, a, b);
}

//mod nonnull
uint32_t ssir_build_mat_transpose(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t m) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_mat_transpose: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_MAT_TRANSPOSE, type, m);
}

/* ============================================================================
 * Instruction Builder - Bitwise
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_bit_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_bit_and: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_AND, type, a, b);
}

//mod nonnull
uint32_t ssir_build_bit_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_bit_or: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_OR, type, a, b);
}

//mod nonnull
uint32_t ssir_build_bit_xor(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_bit_xor: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_XOR, type, a, b);
}

//mod nonnull
uint32_t ssir_build_bit_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_bit_not: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_BIT_NOT, type, a);
}

//mod nonnull
uint32_t ssir_build_shl(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_shl: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHL, type, a, b);
}

//mod nonnull
uint32_t ssir_build_shr(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_shr: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHR, type, a, b);
}

//mod nonnull
uint32_t ssir_build_shr_logical(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_shr_logical: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHR_LOGICAL, type, a, b);
}

/* ============================================================================
 * Instruction Builder - Comparison
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_eq(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_eq: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_EQ, type, a, b);
}

//mod nonnull
uint32_t ssir_build_ne(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_ne: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_NE, type, a, b);
}

//mod nonnull
uint32_t ssir_build_lt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_lt: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_LT, type, a, b);
}

//mod nonnull
uint32_t ssir_build_le(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_le: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_LE, type, a, b);
}

//mod nonnull
uint32_t ssir_build_gt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_gt: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_GT, type, a, b);
}

//mod nonnull
uint32_t ssir_build_ge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_ge: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_GE, type, a, b);
}

/* ============================================================================
 * Instruction Builder - Logical
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_and: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_AND, type, a, b);
}

//mod nonnull
uint32_t ssir_build_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_or: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_OR, type, a, b);
}

//mod nonnull
uint32_t ssir_build_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_not: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_NOT, type, a);
}

/* ============================================================================
 * Instruction Builder - Composite
 * ============================================================================ */

//mod nonnull
//components allowed to be NULL (if count is 0)
uint32_t ssir_build_construct(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, const uint32_t *components, uint32_t count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_construct: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_CONSTRUCT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;

    /* Store in operands if fits, otherwise in extra */
    if (count <= SSIR_MAX_OPERANDS) {
        memcpy(inst->operands, components, count * sizeof(uint32_t));
        inst->operand_count = (uint8_t)count;
    } else {
        inst->operand_count = 0;
        inst->extra = (uint32_t *)SSIR_MALLOC(count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, components, count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)count;
    }
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_extract(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t composite, uint32_t index) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_extract: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_EXTRACT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = composite;
    inst->operands[1] = index;
    inst->operand_count = 2;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_insert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t composite, uint32_t value, uint32_t index) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_insert: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_INSERT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = composite;
    inst->operands[1] = value;
    inst->operands[2] = index;
    inst->operand_count = 3;
    return inst->result;
}

//mod nonnull
//indices allowed to be NULL (if index_count is 0)
uint32_t ssir_build_shuffle(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t v1, uint32_t v2,
                            const uint32_t *indices, uint32_t index_count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_shuffle: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_SHUFFLE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = v1;
    inst->operands[1] = v2;
    inst->operand_count = 2;

    if (index_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(index_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, indices, index_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)index_count;
    }
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_splat(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t type, uint32_t scalar) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_splat: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_SPLAT, type, scalar);
}

//mod nonnull
uint32_t ssir_build_extract_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t composite, uint32_t index) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_extract_dyn: mod is NULL");
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_EXTRACT_DYN, type, composite, index);
}

//mod nonnull
uint32_t ssir_build_insert_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t vector, uint32_t value, uint32_t index) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_insert_dyn: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_INSERT_DYN;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = vector;
    inst->operands[1] = value;
    inst->operands[2] = index;
    inst->operand_count = 3;
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Memory
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t ptr) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_load: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_LOAD, type, ptr);
}

//mod nonnull
void ssir_build_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                      uint32_t ptr, uint32_t value) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_store: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_STORE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = ptr;
    inst->operands[1] = value;
    inst->operand_count = 2;
}

//mod nonnull
//indices allowed to be NULL (if index_count is 0)
uint32_t ssir_build_access(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t base,
                           const uint32_t *indices, uint32_t index_count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_access: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_ACCESS;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = base;
    inst->operand_count = 1;

    if (index_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(index_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, indices, index_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)index_count;
    }
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_array_len(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t ptr) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_array_len: mod is NULL");
    uint32_t u32_type = ssir_type_u32(mod);
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_ARRAY_LEN, u32_type, ptr);
}

/* ============================================================================
 * Instruction Builder - Control Flow
 * ============================================================================ */

//mod nonnull
void ssir_build_branch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t target_block) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_branch: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BRANCH;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = target_block;
    inst->operand_count = 1;
}

//mod nonnull
void ssir_build_branch_cond_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t cond, uint32_t true_block, uint32_t false_block,
                                  uint32_t merge_block) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_branch_cond_merge: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BRANCH_COND;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = cond;
    inst->operands[1] = true_block;
    inst->operands[2] = false_block;
    inst->operands[3] = merge_block;  // 0 = no merge (for unstructured), non-0 = merge block ID
    inst->operand_count = 4;
}

//mod nonnull
void ssir_build_branch_cond(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t cond, uint32_t true_block, uint32_t false_block) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_branch_cond: mod is NULL");
    ssir_build_branch_cond_merge(mod, func_id, block_id, cond, true_block, false_block, 0);
}

//mod nonnull
void ssir_build_loop_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t merge_block, uint32_t continue_block) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_loop_merge: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_LOOP_MERGE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = merge_block;
    inst->operands[1] = continue_block;
    inst->operand_count = 2;
}

//mod nonnull
void ssir_build_selection_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t merge_block) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_selection_merge: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_SELECTION_MERGE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = merge_block;
    inst->operand_count = 1;
}

//mod nonnull
//cases allowed to be NULL (if case_count is 0)
void ssir_build_switch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t selector, uint32_t default_block,
                       const uint32_t *cases, uint32_t case_count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_switch: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_SWITCH;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = selector;
    inst->operands[1] = default_block;
    inst->operand_count = 2;

    /* cases is pairs of (value, label) */
    if (case_count > 0) {
        /* PRE: case_count <= 32767 to avoid uint16_t overflow */
        wgsl_compiler_assert(case_count <= 32767, "ssir_build_switch: case_count overflow: %u", case_count);
        inst->extra = (uint32_t *)SSIR_MALLOC(case_count * 2 * sizeof(uint32_t));
        if (!inst->extra) return;
        memcpy(inst->extra, cases, case_count * 2 * sizeof(uint32_t));
        inst->extra_count = (uint16_t)(case_count * 2);
    }
}

//mod nonnull
//incoming allowed to be NULL (if count is 0)
uint32_t ssir_build_phi(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, const uint32_t *incoming, uint32_t count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_phi: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_PHI;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operand_count = 0;

    /* incoming is pairs of (value_id, block_id) */
    if (count > 0) {
        /* PRE: count <= 32767 to avoid uint16_t overflow */
        wgsl_compiler_assert(count <= 32767, "ssir_build_phi: count overflow: %u", count);
        inst->extra = (uint32_t *)SSIR_MALLOC(count * 2 * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, incoming, count * 2 * sizeof(uint32_t));
        inst->extra_count = (uint16_t)(count * 2);
    }
    return inst->result;
}

//mod nonnull
void ssir_build_return(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t value) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_return: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_RETURN;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = value;
    inst->operand_count = 1;
}

//mod nonnull
void ssir_build_return_void(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_return_void: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_RETURN_VOID;
    inst->result = 0;
    inst->type = 0;
    inst->operand_count = 0;
}

//mod nonnull
void ssir_build_unreachable(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_unreachable: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_UNREACHABLE;
    inst->result = 0;
    inst->type = 0;
    inst->operand_count = 0;
}

/* ============================================================================
 * Instruction Builder - Call
 * ============================================================================ */

//mod nonnull
//args allowed to be NULL (if arg_count is 0)
uint32_t ssir_build_call(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t callee,
                         const uint32_t *args, uint32_t arg_count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_call: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_CALL;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = callee;
    inst->operand_count = 1;

    if (arg_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(arg_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, args, arg_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)arg_count;
    }
    return inst->result;
}

//mod nonnull
//args allowed to be NULL (if arg_count is 0)
uint32_t ssir_build_builtin(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, SsirBuiltinId builtin,
                            const uint32_t *args, uint32_t arg_count) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_builtin: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_BUILTIN;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = (uint32_t)builtin;
    inst->operand_count = 1;

    if (arg_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(arg_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, args, arg_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)arg_count;
    }
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Conversion
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_convert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_convert: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_CONVERT, type, value);
}

//mod nonnull
uint32_t ssir_build_bitcast(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_bitcast: mod is NULL");
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_BITCAST, type, value);
}

/* ============================================================================
 * Instruction Builder - Texture
 * ============================================================================ */

//mod nonnull
uint32_t ssir_build_tex_sample(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operand_count = 3;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_bias(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t bias) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_bias: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_BIAS;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = bias;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture, uint32_t sampler,
                                     uint32_t coord, uint32_t lod) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_level: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_LEVEL;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = lod;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_grad(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t ddx, uint32_t ddy) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_grad: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_GRAD;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ddx;
    inst->operands[4] = ddy;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_cmp: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_CMP;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ref;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_cmp_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                         uint32_t type, uint32_t texture, uint32_t sampler,
                                         uint32_t coord, uint32_t ref, uint32_t lod) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_cmp_level: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_CMP_LEVEL;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ref;
    inst->operands[4] = lod;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture, uint32_t sampler,
                                      uint32_t coord, uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = offset;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_bias_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                           uint32_t type, uint32_t texture, uint32_t sampler,
                                           uint32_t coord, uint32_t bias, uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_bias_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_BIAS_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = bias;
    inst->operands[4] = offset;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_level_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                            uint32_t type, uint32_t texture, uint32_t sampler,
                                            uint32_t coord, uint32_t lod, uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_level_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = lod;
    inst->operands[4] = offset;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_grad_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                           uint32_t type, uint32_t texture, uint32_t sampler,
                                           uint32_t coord, uint32_t ddx, uint32_t ddy,
                                           uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_grad_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_GRAD_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ddx;
    inst->operands[4] = ddy;
    inst->operands[5] = offset;
    inst->operand_count = 6;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_sample_cmp_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                          uint32_t type, uint32_t texture, uint32_t sampler,
                                          uint32_t coord, uint32_t ref, uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_sample_cmp_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_CMP_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ref;
    inst->operands[4] = offset;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_gather(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord, uint32_t component) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_gather: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_GATHER;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = component;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_gather_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_gather_cmp: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_GATHER_CMP;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ref;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_gather_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture, uint32_t sampler,
                                      uint32_t coord, uint32_t component, uint32_t offset) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_gather_offset: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_GATHER_OFFSET;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = component;
    inst->operands[4] = offset;
    inst->operand_count = 5;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t coord, uint32_t level) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_load: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_LOAD;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = coord;
    inst->operands[2] = level;
    inst->operand_count = 3;
    return inst->result;
}

//mod nonnull
void ssir_build_tex_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t texture, uint32_t coord, uint32_t value) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_store: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_TEX_STORE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = texture;
    inst->operands[1] = coord;
    inst->operands[2] = value;
    inst->operand_count = 3;
}

//mod nonnull
uint32_t ssir_build_tex_size(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t level) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_size: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SIZE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = level;
    inst->operand_count = 2;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_query_lod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t texture, uint32_t sampler,
                                  uint32_t coord) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_query_lod: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_QUERY_LOD;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operand_count = 3;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_query_levels(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_query_levels: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_QUERY_LEVELS;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operand_count = 1;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_tex_query_samples(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_tex_query_samples: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_QUERY_SAMPLES;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operand_count = 1;
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Sync
 * ============================================================================ */

//mod nonnull
void ssir_build_barrier(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        SsirBarrierScope scope) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_barrier: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BARRIER;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = (uint32_t)scope;
    inst->operand_count = 1;
}

//mod nonnull
uint32_t ssir_build_atomic(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, SsirAtomicOp op, uint32_t ptr,
                           uint32_t value, uint32_t comparator) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_atomic: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_ATOMIC;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = (uint32_t)op;
    inst->operands[1] = ptr;
    inst->operands[2] = value;
    inst->operands[3] = comparator;
    inst->operand_count = 4;
    return inst->result;
}

//mod nonnull
uint32_t ssir_build_atomic_ex(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, SsirAtomicOp op, uint32_t ptr,
                              uint32_t value, uint32_t comparator,
                              SsirMemoryScope scope, SsirMemorySemantics semantics) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_atomic_ex: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_ATOMIC;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = (uint32_t)op;
    inst->operands[1] = ptr;
    inst->operands[2] = value;
    inst->operands[3] = comparator;
    inst->operands[4] = (uint32_t)scope;
    inst->operands[5] = (uint32_t)semantics;
    inst->operand_count = 6;
    return inst->result;
}

//mod nonnull
void ssir_build_discard(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_discard: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_DISCARD;
    inst->result = 0;
    inst->type = 0;
    inst->operand_count = 0;
}

//mod nonnull
uint32_t ssir_build_rem(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    wgsl_compiler_assert(mod != NULL, "ssir_build_rem: mod is NULL");
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_REM;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = a;
    inst->operands[1] = b;
    inst->operand_count = 2;
    return inst->result;
}

/* ============================================================================
 * Entry Point API
 * ============================================================================ */

//mod nonnull
//name allowed to be NULL
uint32_t ssir_entry_point_create(SsirModule *mod, SsirStage stage,
                                 uint32_t func_id, const char *name) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_create: mod is NULL");
    if (!ssir_grow_array((void **)&mod->entry_points, &mod->entry_point_capacity,
                         sizeof(SsirEntryPoint), mod->entry_point_count + 1)) {
        return UINT32_MAX;
    }

    uint32_t index = mod->entry_point_count++;
    SsirEntryPoint *ep = &mod->entry_points[index];
    memset(ep, 0, sizeof(SsirEntryPoint));
    ep->stage = stage;
    ep->function = func_id;
    ep->name = ssir_strdup(name);
    ep->workgroup_size[0] = 1;
    ep->workgroup_size[1] = 1;
    ep->workgroup_size[2] = 1;
    return index;
}

//mod nonnull
SsirEntryPoint *ssir_get_entry_point(SsirModule *mod, uint32_t index) {
    wgsl_compiler_assert(mod != NULL, "ssir_get_entry_point: mod is NULL");
    if (index >= mod->entry_point_count) return NULL;
    return &mod->entry_points[index];
}

//mod nonnull
void ssir_entry_point_add_interface(SsirModule *mod, uint32_t ep_index, uint32_t global_id) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_add_interface: mod is NULL");
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (!ep) return;

    uint32_t new_count = ep->interface_count + 1;
    uint32_t *new_interface = (uint32_t *)SSIR_REALLOC(
        ep->interface, new_count * sizeof(uint32_t));
    if (!new_interface) return;
    ep->interface = new_interface;
    ep->interface[ep->interface_count++] = global_id;
}

//mod nonnull
void ssir_entry_point_set_workgroup_size(SsirModule *mod, uint32_t ep_index,
                                         uint32_t x, uint32_t y, uint32_t z) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_set_workgroup_size: mod is NULL");
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (!ep) return;
    ep->workgroup_size[0] = x;
    ep->workgroup_size[1] = y;
    ep->workgroup_size[2] = z;
}

//mod nonnull
void ssir_entry_point_set_depth_replacing(SsirModule *mod, uint32_t ep_index, bool v) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_set_depth_replacing: mod is NULL");
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (ep) ep->depth_replacing = v;
}

//mod nonnull
void ssir_entry_point_set_origin_upper_left(SsirModule *mod, uint32_t ep_index, bool v) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_set_origin_upper_left: mod is NULL");
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (ep) ep->origin_upper_left = v;
}

//mod nonnull
void ssir_entry_point_set_early_fragment_tests(SsirModule *mod, uint32_t ep_index, bool v) {
    wgsl_compiler_assert(mod != NULL, "ssir_entry_point_set_early_fragment_tests: mod is NULL");
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (ep) ep->early_fragment_tests = v;
}

/* ============================================================================
 * Use Count Analysis
 * ============================================================================ */

//f nonnull
//use_counts nonnull
void ssir_count_uses(SsirFunction *f, uint32_t *use_counts, uint32_t max_id) {
    wgsl_compiler_assert(f != NULL, "ssir_count_uses: f is NULL");
    wgsl_compiler_assert(use_counts != NULL, "ssir_count_uses: use_counts is NULL");
#define SSIR_COUNT_USE(id) do { if ((id) > 0 && (id) < max_id) use_counts[id]++; } while(0)
    for (uint32_t bi = 0; bi < f->block_count; bi++) {
        SsirBlock *blk = &f->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            SsirInst *inst = &blk->insts[ii];
            switch (inst->op) {
            /* Control flow with no value operands */
            case SSIR_OP_BRANCH:
            case SSIR_OP_LOOP_MERGE:
            case SSIR_OP_SELECTION_MERGE:
            case SSIR_OP_UNREACHABLE:
            case SSIR_OP_RETURN_VOID:
            case SSIR_OP_DISCARD:
            case SSIR_OP_BARRIER:
                break;
            case SSIR_OP_BRANCH_COND:
                SSIR_COUNT_USE(inst->operands[0]);
                break;
            case SSIR_OP_SWITCH:
                SSIR_COUNT_USE(inst->operands[0]);
                break;
            case SSIR_OP_RETURN:
                if (inst->operand_count >= 1)
                    SSIR_COUNT_USE(inst->operands[0]);
                break;
            case SSIR_OP_PHI:
                /* extra = pairs of (value, block); count values only */
                for (uint16_t pi = 0; pi < inst->extra_count; pi += 2)
                    SSIR_COUNT_USE(inst->extra[pi]);
                break;
            case SSIR_OP_BUILTIN:
            case SSIR_OP_CALL:
                /* operands[0] = builtin/func ID; rest are value args */
                for (uint8_t oi = 1; oi < inst->operand_count; oi++)
                    SSIR_COUNT_USE(inst->operands[oi]);
                break;
            case SSIR_OP_ATOMIC:
                /* operands[0]=op, [1]=ptr, [2]=val, [3]=cmp; [4+]=scope/sem */
                for (uint8_t oi = 1; oi <= 3 && oi < inst->operand_count; oi++)
                    SSIR_COUNT_USE(inst->operands[oi]);
                break;
            case SSIR_OP_EXTRACT:
            case SSIR_OP_ACCESS:
                /* operands[0] = value, rest are literal indices */
                if (inst->operand_count >= 1)
                    SSIR_COUNT_USE(inst->operands[0]);
                break;
            case SSIR_OP_INSERT:
                /* operands[0]=composite, [1]=value, [2]=index(literal) */
                if (inst->operand_count >= 1) SSIR_COUNT_USE(inst->operands[0]);
                if (inst->operand_count >= 2) SSIR_COUNT_USE(inst->operands[1]);
                break;
            case SSIR_OP_SHUFFLE:
                /* operands[0]=a, [1]=b; extra=shuffle mask (literals) */
                if (inst->operand_count >= 1) SSIR_COUNT_USE(inst->operands[0]);
                if (inst->operand_count >= 2) SSIR_COUNT_USE(inst->operands[1]);
                break;
            default:
                /* All operands are value references */
                for (uint8_t oi = 0; oi < inst->operand_count; oi++)
                    SSIR_COUNT_USE(inst->operands[oi]);
                break;
            }
        }
    }
#undef SSIR_COUNT_USE
}

//mod allowed to be NULL
void ssir_module_build_lookup(SsirModule *mod) {
    if (!mod || mod->next_id == 0) return;

    /* Free existing cache */
    if (mod->_lookup_cache) {
        SsirLookupCache *old = (SsirLookupCache *)mod->_lookup_cache;
        free(old->entries);
        free(old);
        mod->_lookup_cache = NULL;
    }

    SsirLookupCache *lc = (SsirLookupCache *)calloc(1, sizeof(*lc));
    if (!lc) return;
    lc->cap = mod->next_id;
    lc->entries = (uintptr_t *)calloc(lc->cap, sizeof(uintptr_t));
    if (!lc->entries) { free(lc); return; }

    for (uint32_t i = 0; i < mod->type_count; i++) {
        uint32_t id = mod->types[i].id;
        if (id < lc->cap) lc->entries[id] = stag_encode(&mod->types[i], STAG_TYPE);
    }
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        uint32_t id = mod->constants[i].id;
        if (id < lc->cap) lc->entries[id] = stag_encode(&mod->constants[i], STAG_CONST);
    }
    for (uint32_t i = 0; i < mod->global_count; i++) {
        uint32_t id = mod->globals[i].id;
        if (id < lc->cap) lc->entries[id] = stag_encode(&mod->globals[i], STAG_GLOBAL);
    }
    for (uint32_t i = 0; i < mod->function_count; i++) {
        uint32_t id = mod->functions[i].id;
        if (id < lc->cap) lc->entries[id] = stag_encode(&mod->functions[i], STAG_FUNC);
    }

    mod->_lookup_cache = lc;
}

/* ============================================================================
 * Validation
 * ============================================================================ */

//result nonnull
//message nonnull
static void ssir_add_validation_error(SsirValidationResult *result,
                                      SsirResult code, const char *message,
                                      uint32_t func_id, uint32_t block_id, uint32_t inst_index) {
    wgsl_compiler_assert(result != NULL, "ssir_add_validation_error: result is NULL");
    wgsl_compiler_assert(message != NULL, "ssir_add_validation_error: message is NULL");
    if (!ssir_grow_array((void **)&result->errors, &result->error_capacity,
                         sizeof(SsirValidationError), result->error_count + 1)) {
        return;
    }
    SsirValidationError *err = &result->errors[result->error_count++];
    err->code = code;
    err->message = message;
    err->func_id = func_id;
    err->block_id = block_id;
    err->inst_index = inst_index;
}

static bool ssir_is_terminator(SsirOpcode op) {
    return op == SSIR_OP_BRANCH ||
           op == SSIR_OP_BRANCH_COND ||
           op == SSIR_OP_SWITCH ||
           op == SSIR_OP_RETURN ||
           op == SSIR_OP_RETURN_VOID ||
           op == SSIR_OP_UNREACHABLE;
}

/* Resolve the type ID associated with a value ID within a function context */
//mod nonnull
//f nonnull
static uint32_t val_resolve_type(SsirModule *mod, SsirFunction *f, uint32_t id) {
    wgsl_compiler_assert(mod != NULL, "val_resolve_type: mod is NULL");
    wgsl_compiler_assert(f != NULL, "val_resolve_type: f is NULL");
    /* Instructions in the function */
    for (uint32_t bi = 0; bi < f->block_count; bi++) {
        SsirBlock *blk = &f->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            if (blk->insts[ii].result == id) return blk->insts[ii].type;
        }
    }
    /* Function parameters */
    for (uint32_t i = 0; i < f->param_count; i++) {
        if (f->params[i].id == id) return f->params[i].type;
    }
    /* Local variables */
    for (uint32_t i = 0; i < f->local_count; i++) {
        if (f->locals[i].id == id) return f->locals[i].type;
    }
    /* Constants */
    SsirConstant *c = ssir_get_constant(mod, id);
    if (c) return c->type;
    /* Globals */
    SsirGlobalVar *g = ssir_get_global(mod, id);
    if (g) return g->type;
    return 0;
}

/* Check if block 'target_id' is a successor of block 'b' */
//b nonnull
static bool block_branches_to(SsirBlock *b, uint32_t target_id) {
    wgsl_compiler_assert(b != NULL, "block_branches_to: b is NULL");
    if (b->inst_count == 0) return false;
    SsirInst *term = &b->insts[b->inst_count - 1];
    switch (term->op) {
    case SSIR_OP_BRANCH:
        return term->operands[0] == target_id;
    case SSIR_OP_BRANCH_COND:
        return term->operands[1] == target_id || term->operands[2] == target_id;
    case SSIR_OP_SWITCH:
        /* operands[1] = default block */
        if (term->operands[1] == target_id) return true;
        /* extra[] = pairs of (value, block) */
        for (uint16_t i = 1; i < term->extra_count; i += 2) {
            if (term->extra[i] == target_id) return true;
        }
        return false;
    default:
        return false;
    }
}

//mod nonnull
SsirValidationResult *ssir_validate(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_validate: mod is NULL");
    SsirValidationResult *result = (SsirValidationResult *)SSIR_MALLOC(sizeof(SsirValidationResult));
    if (!result) return NULL;
    memset(result, 0, sizeof(SsirValidationResult));
    result->valid = true;

    /* Validate each function */
    for (uint32_t fi = 0; fi < mod->function_count; fi++) {
        SsirFunction *f = &mod->functions[fi];

        /* Each function must have at least one block */
        if (f->block_count == 0) {
            ssir_add_validation_error(result, SSIR_ERROR_INVALID_BLOCK,
                "Function has no blocks", f->id, 0, 0);
            result->valid = false;
            continue;
        }

        /* Validate each block */
        for (uint32_t bi = 0; bi < f->block_count; bi++) {
            SsirBlock *b = &f->blocks[bi];

            /* Block must have instructions */
            if (b->inst_count == 0) {
                ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                    "Block has no instructions", f->id, b->id, 0);
                result->valid = false;
                continue;
            }

            /* Last instruction must be a terminator */
            SsirInst *last = &b->insts[b->inst_count - 1];
            if (!ssir_is_terminator(last->op)) {
                ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                    "Block does not end with terminator", f->id, b->id, b->inst_count - 1);
                result->valid = false;
            }

            /* Phi nodes must be at start of block */
            bool seen_non_phi = false;
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *inst = &b->insts[ii];
                if (inst->op == SSIR_OP_PHI) {
                    if (seen_non_phi) {
                        ssir_add_validation_error(result, SSIR_ERROR_PHI_PLACEMENT,
                            "Phi node not at start of block", f->id, b->id, ii);
                        result->valid = false;
                    }
                } else {
                    seen_non_phi = true;
                }

                /* No instructions after terminator */
                if (ssir_is_terminator(inst->op) && ii < b->inst_count - 1) {
                    ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                        "Instruction after terminator", f->id, b->id, ii + 1);
                    result->valid = false;
                }

                /* --- Type consistency checks --- */

                /* LOAD: operand must be pointer, pointee must match result type */
                if (inst->op == SSIR_OP_LOAD && inst->operand_count >= 1) {
                    uint32_t ptr_type_id = val_resolve_type(mod, f, inst->operands[0]);
                    SsirType *ptr_t = ssir_get_type(mod, ptr_type_id);
                    if (ptr_t && ptr_t->kind != SSIR_TYPE_PTR) {
                        ssir_add_validation_error(result, SSIR_ERROR_TYPE_MISMATCH,
                            "LOAD operand is not a pointer type", f->id, b->id, ii);
                        result->valid = false;
                    } else if (ptr_t && ptr_t->kind == SSIR_TYPE_PTR &&
                               inst->type != 0 && ptr_t->ptr.pointee != inst->type) {
                        ssir_add_validation_error(result, SSIR_ERROR_TYPE_MISMATCH,
                            "LOAD result type does not match pointer pointee type",
                            f->id, b->id, ii);
                        result->valid = false;
                    }
                }

                /* STORE: first operand must be pointer */
                if (inst->op == SSIR_OP_STORE && inst->operand_count >= 1) {
                    uint32_t ptr_type_id = val_resolve_type(mod, f, inst->operands[0]);
                    SsirType *ptr_t = ssir_get_type(mod, ptr_type_id);
                    if (ptr_t && ptr_t->kind != SSIR_TYPE_PTR) {
                        ssir_add_validation_error(result, SSIR_ERROR_ADDRESS_SPACE,
                            "STORE target is not a pointer type", f->id, b->id, ii);
                        result->valid = false;
                    }
                }

                /* Binary arithmetic: both operands should have matching types */
                if ((inst->op >= SSIR_OP_ADD && inst->op <= SSIR_OP_DIV) ||
                    inst->op == SSIR_OP_MOD || inst->op == SSIR_OP_REM) {
                    if (inst->operand_count >= 2) {
                        uint32_t t0 = val_resolve_type(mod, f, inst->operands[0]);
                        uint32_t t1 = val_resolve_type(mod, f, inst->operands[1]);
                        if (t0 != 0 && t1 != 0 && t0 != t1) {
                            ssir_add_validation_error(result, SSIR_ERROR_TYPE_MISMATCH,
                                "Binary arithmetic operands have mismatched types",
                                f->id, b->id, ii);
                            result->valid = false;
                        }
                    }
                }

                /* PHI: each incoming block must be a predecessor of this block */
                if (inst->op == SSIR_OP_PHI && inst->extra_count >= 2) {
                    for (uint16_t pi = 1; pi < inst->extra_count; pi += 2) {
                        uint32_t pred_block_id = inst->extra[pi];
                        bool found_pred = false;
                        for (uint32_t pbi = 0; pbi < f->block_count; pbi++) {
                            if (block_branches_to(&f->blocks[pbi], b->id) &&
                                f->blocks[pbi].id == pred_block_id) {
                                found_pred = true;
                                break;
                            }
                        }
                        if (!found_pred) {
                            ssir_add_validation_error(result, SSIR_ERROR_PHI_PLACEMENT,
                                "Phi incoming block is not a predecessor",
                                f->id, b->id, ii);
                            result->valid = false;
                            break;
                        }
                    }
                }
            }
        }
    }

    /* Validate entry points */
    for (uint32_t ei = 0; ei < mod->entry_point_count; ei++) {
        SsirEntryPoint *ep = &mod->entry_points[ei];

        /* Entry point must reference valid function */
        SsirFunction *f = ssir_get_function(mod, ep->function);
        if (!f) {
            ssir_add_validation_error(result, SSIR_ERROR_ENTRY_POINT,
                "Entry point references invalid function", 0, 0, 0);
            result->valid = false;
        }

        /* Compute shader must have workgroup_size > 0 */
        if (ep->stage == SSIR_STAGE_COMPUTE) {
            if (ep->workgroup_size[0] == 0 ||
                ep->workgroup_size[1] == 0 ||
                ep->workgroup_size[2] == 0) {
                ssir_add_validation_error(result, SSIR_ERROR_ENTRY_POINT,
                    "Compute shader workgroup size must be > 0", 0, 0, 0);
                result->valid = false;
            }
        }
    }

    return result;
}

//result allowed to be NULL
void ssir_validation_result_free(SsirValidationResult *result) {
    if (!result) return;
    SSIR_FREE(result->errors);
    SSIR_FREE(result);
}

/* ============================================================================
 * Debug/Utility API
 * ============================================================================ */

static const char *opcode_names[] = {
    [SSIR_OP_ADD] = "add",
    [SSIR_OP_SUB] = "sub",
    [SSIR_OP_MUL] = "mul",
    [SSIR_OP_DIV] = "div",
    [SSIR_OP_MOD] = "mod",
    [SSIR_OP_NEG] = "neg",
    [SSIR_OP_MAT_MUL] = "mat_mul",
    [SSIR_OP_MAT_TRANSPOSE] = "mat_transpose",
    [SSIR_OP_BIT_AND] = "bit_and",
    [SSIR_OP_BIT_OR] = "bit_or",
    [SSIR_OP_BIT_XOR] = "bit_xor",
    [SSIR_OP_BIT_NOT] = "bit_not",
    [SSIR_OP_SHL] = "shl",
    [SSIR_OP_SHR] = "shr",
    [SSIR_OP_SHR_LOGICAL] = "shr_logical",
    [SSIR_OP_EQ] = "eq",
    [SSIR_OP_NE] = "ne",
    [SSIR_OP_LT] = "lt",
    [SSIR_OP_LE] = "le",
    [SSIR_OP_GT] = "gt",
    [SSIR_OP_GE] = "ge",
    [SSIR_OP_AND] = "and",
    [SSIR_OP_OR] = "or",
    [SSIR_OP_NOT] = "not",
    [SSIR_OP_CONSTRUCT] = "construct",
    [SSIR_OP_EXTRACT] = "extract",
    [SSIR_OP_INSERT] = "insert",
    [SSIR_OP_SHUFFLE] = "shuffle",
    [SSIR_OP_SPLAT] = "splat",
    [SSIR_OP_EXTRACT_DYN] = "extract_dyn",
    [SSIR_OP_INSERT_DYN] = "insert_dyn",
    [SSIR_OP_LOAD] = "load",
    [SSIR_OP_STORE] = "store",
    [SSIR_OP_ACCESS] = "access",
    [SSIR_OP_ARRAY_LEN] = "array_len",
    [SSIR_OP_BRANCH] = "branch",
    [SSIR_OP_BRANCH_COND] = "branch_cond",
    [SSIR_OP_SWITCH] = "switch",
    [SSIR_OP_PHI] = "phi",
    [SSIR_OP_RETURN] = "return",
    [SSIR_OP_RETURN_VOID] = "return_void",
    [SSIR_OP_UNREACHABLE] = "unreachable",
    [SSIR_OP_CALL] = "call",
    [SSIR_OP_BUILTIN] = "builtin",
    [SSIR_OP_CONVERT] = "convert",
    [SSIR_OP_BITCAST] = "bitcast",
    [SSIR_OP_TEX_SAMPLE] = "tex_sample",
    [SSIR_OP_TEX_SAMPLE_BIAS] = "tex_sample_bias",
    [SSIR_OP_TEX_SAMPLE_LEVEL] = "tex_sample_level",
    [SSIR_OP_TEX_SAMPLE_GRAD] = "tex_sample_grad",
    [SSIR_OP_TEX_SAMPLE_CMP] = "tex_sample_cmp",
    [SSIR_OP_TEX_SAMPLE_CMP_LEVEL] = "tex_sample_cmp_level",
    [SSIR_OP_TEX_SAMPLE_OFFSET] = "tex_sample_offset",
    [SSIR_OP_TEX_SAMPLE_BIAS_OFFSET] = "tex_sample_bias_offset",
    [SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET] = "tex_sample_level_offset",
    [SSIR_OP_TEX_SAMPLE_GRAD_OFFSET] = "tex_sample_grad_offset",
    [SSIR_OP_TEX_SAMPLE_CMP_OFFSET] = "tex_sample_cmp_offset",
    [SSIR_OP_TEX_GATHER] = "tex_gather",
    [SSIR_OP_TEX_GATHER_CMP] = "tex_gather_cmp",
    [SSIR_OP_TEX_GATHER_OFFSET] = "tex_gather_offset",
    [SSIR_OP_TEX_LOAD] = "tex_load",
    [SSIR_OP_TEX_STORE] = "tex_store",
    [SSIR_OP_TEX_SIZE] = "tex_size",
    [SSIR_OP_TEX_QUERY_LOD] = "tex_query_lod",
    [SSIR_OP_TEX_QUERY_LEVELS] = "tex_query_levels",
    [SSIR_OP_TEX_QUERY_SAMPLES] = "tex_query_samples",
    [SSIR_OP_BARRIER] = "barrier",
    [SSIR_OP_ATOMIC] = "atomic",
    [SSIR_OP_LOOP_MERGE] = "loop_merge",
    [SSIR_OP_DISCARD] = "discard",
    [SSIR_OP_REM] = "rem",
    [SSIR_OP_SELECTION_MERGE] = "selection_merge",
};

const char *ssir_opcode_name(SsirOpcode op) {
    if (op >= SSIR_OP_COUNT) return "unknown";
    return opcode_names[op] ? opcode_names[op] : "unknown";
}

static const char *builtin_names[] = {
    [SSIR_BUILTIN_SIN] = "sin",
    [SSIR_BUILTIN_COS] = "cos",
    [SSIR_BUILTIN_TAN] = "tan",
    [SSIR_BUILTIN_ASIN] = "asin",
    [SSIR_BUILTIN_ACOS] = "acos",
    [SSIR_BUILTIN_ATAN] = "atan",
    [SSIR_BUILTIN_ATAN2] = "atan2",
    [SSIR_BUILTIN_SINH] = "sinh",
    [SSIR_BUILTIN_COSH] = "cosh",
    [SSIR_BUILTIN_TANH] = "tanh",
    [SSIR_BUILTIN_ASINH] = "asinh",
    [SSIR_BUILTIN_ACOSH] = "acosh",
    [SSIR_BUILTIN_ATANH] = "atanh",
    [SSIR_BUILTIN_EXP] = "exp",
    [SSIR_BUILTIN_EXP2] = "exp2",
    [SSIR_BUILTIN_LOG] = "log",
    [SSIR_BUILTIN_LOG2] = "log2",
    [SSIR_BUILTIN_POW] = "pow",
    [SSIR_BUILTIN_SQRT] = "sqrt",
    [SSIR_BUILTIN_INVERSESQRT] = "inversesqrt",
    [SSIR_BUILTIN_ABS] = "abs",
    [SSIR_BUILTIN_SIGN] = "sign",
    [SSIR_BUILTIN_FLOOR] = "floor",
    [SSIR_BUILTIN_CEIL] = "ceil",
    [SSIR_BUILTIN_ROUND] = "round",
    [SSIR_BUILTIN_TRUNC] = "trunc",
    [SSIR_BUILTIN_FRACT] = "fract",
    [SSIR_BUILTIN_MIN] = "min",
    [SSIR_BUILTIN_MAX] = "max",
    [SSIR_BUILTIN_CLAMP] = "clamp",
    [SSIR_BUILTIN_SATURATE] = "saturate",
    [SSIR_BUILTIN_MIX] = "mix",
    [SSIR_BUILTIN_STEP] = "step",
    [SSIR_BUILTIN_SMOOTHSTEP] = "smoothstep",
    [SSIR_BUILTIN_DOT] = "dot",
    [SSIR_BUILTIN_CROSS] = "cross",
    [SSIR_BUILTIN_LENGTH] = "length",
    [SSIR_BUILTIN_DISTANCE] = "distance",
    [SSIR_BUILTIN_NORMALIZE] = "normalize",
    [SSIR_BUILTIN_FACEFORWARD] = "faceforward",
    [SSIR_BUILTIN_REFLECT] = "reflect",
    [SSIR_BUILTIN_REFRACT] = "refract",
    [SSIR_BUILTIN_ALL] = "all",
    [SSIR_BUILTIN_ANY] = "any",
    [SSIR_BUILTIN_SELECT] = "select",
    [SSIR_BUILTIN_COUNTBITS] = "countbits",
    [SSIR_BUILTIN_REVERSEBITS] = "reversebits",
    [SSIR_BUILTIN_FIRSTLEADINGBIT] = "firstleadingbit",
    [SSIR_BUILTIN_FIRSTTRAILINGBIT] = "firsttrailingbit",
    [SSIR_BUILTIN_EXTRACTBITS] = "extractbits",
    [SSIR_BUILTIN_INSERTBITS] = "insertbits",
    [SSIR_BUILTIN_DPDX] = "dpdx",
    [SSIR_BUILTIN_DPDY] = "dpdy",
    [SSIR_BUILTIN_FWIDTH] = "fwidth",
    [SSIR_BUILTIN_DPDX_COARSE] = "dpdx_coarse",
    [SSIR_BUILTIN_DPDY_COARSE] = "dpdy_coarse",
    [SSIR_BUILTIN_DPDX_FINE] = "dpdx_fine",
    [SSIR_BUILTIN_DPDY_FINE] = "dpdy_fine",
    [SSIR_BUILTIN_FMA] = "fma",
    [SSIR_BUILTIN_ISINF] = "isinf",
    [SSIR_BUILTIN_ISNAN] = "isnan",
    [SSIR_BUILTIN_DEGREES] = "degrees",
    [SSIR_BUILTIN_RADIANS] = "radians",
    [SSIR_BUILTIN_MODF] = "modf",
    [SSIR_BUILTIN_FREXP] = "frexp",
    [SSIR_BUILTIN_LDEXP] = "ldexp",
    [SSIR_BUILTIN_DETERMINANT] = "determinant",
    [SSIR_BUILTIN_TRANSPOSE] = "transpose",
    [SSIR_BUILTIN_PACK4X8SNORM] = "pack4x8snorm",
    [SSIR_BUILTIN_PACK4X8UNORM] = "pack4x8unorm",
    [SSIR_BUILTIN_PACK2X16SNORM] = "pack2x16snorm",
    [SSIR_BUILTIN_PACK2X16UNORM] = "pack2x16unorm",
    [SSIR_BUILTIN_PACK2X16FLOAT] = "pack2x16float",
    [SSIR_BUILTIN_UNPACK4X8SNORM] = "unpack4x8snorm",
    [SSIR_BUILTIN_UNPACK4X8UNORM] = "unpack4x8unorm",
    [SSIR_BUILTIN_UNPACK2X16SNORM] = "unpack2x16snorm",
    [SSIR_BUILTIN_UNPACK2X16UNORM] = "unpack2x16unorm",
    [SSIR_BUILTIN_UNPACK2X16FLOAT] = "unpack2x16float",
    [SSIR_BUILTIN_SUBGROUP_BALLOT] = "subgroupBallot",
    [SSIR_BUILTIN_SUBGROUP_BROADCAST] = "subgroupBroadcast",
    [SSIR_BUILTIN_SUBGROUP_ADD] = "subgroupAdd",
    [SSIR_BUILTIN_SUBGROUP_MIN] = "subgroupMin",
    [SSIR_BUILTIN_SUBGROUP_MAX] = "subgroupMax",
    [SSIR_BUILTIN_SUBGROUP_ALL] = "subgroupAll",
    [SSIR_BUILTIN_SUBGROUP_ANY] = "subgroupAny",
    [SSIR_BUILTIN_SUBGROUP_SHUFFLE] = "subgroupShuffle",
    [SSIR_BUILTIN_SUBGROUP_PREFIX_ADD] = "subgroupPrefixAdd",
};

const char *ssir_builtin_name(SsirBuiltinId id) {
    if (id >= SSIR_BUILTIN_COUNT) return "unknown";
    return builtin_names[id] ? builtin_names[id] : "unknown";
}

static const char *type_kind_names[] = {
    [SSIR_TYPE_VOID] = "void",
    [SSIR_TYPE_BOOL] = "bool",
    [SSIR_TYPE_I32] = "i32",
    [SSIR_TYPE_U32] = "u32",
    [SSIR_TYPE_F32] = "f32",
    [SSIR_TYPE_F16] = "f16",
    [SSIR_TYPE_VEC] = "vec",
    [SSIR_TYPE_MAT] = "mat",
    [SSIR_TYPE_ARRAY] = "array",
    [SSIR_TYPE_RUNTIME_ARRAY] = "runtime_array",
    [SSIR_TYPE_STRUCT] = "struct",
    [SSIR_TYPE_PTR] = "ptr",
    [SSIR_TYPE_SAMPLER] = "sampler",
    [SSIR_TYPE_SAMPLER_COMPARISON] = "sampler_comparison",
    [SSIR_TYPE_TEXTURE] = "texture",
    [SSIR_TYPE_TEXTURE_STORAGE] = "texture_storage",
    [SSIR_TYPE_TEXTURE_DEPTH] = "texture_depth",
};

const char *ssir_type_kind_name(SsirTypeKind kind) {
    if (kind > SSIR_TYPE_TEXTURE_DEPTH) return "unknown";
    return type_kind_names[kind];
}

float ssir_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

/* ============================================================================
 * Module to String (Debug)
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StringBuilder;

//sb nonnull
//str nonnull
static void sb_append(StringBuilder *sb, const char *str) {
    wgsl_compiler_assert(sb != NULL, "sb_append: sb is NULL");
    wgsl_compiler_assert(str != NULL, "sb_append: str is NULL");
    size_t slen = strlen(str);
    if (sb->len + slen + 1 > sb->cap) {
        size_t new_cap = sb->cap ? sb->cap * 2 : 256;
        while (new_cap < sb->len + slen + 1) new_cap *= 2;
        char *new_data = (char *)SSIR_REALLOC(sb->data, new_cap);
        if (!new_data) return;
        sb->data = new_data;
        sb->cap = new_cap;
    }
    memcpy(sb->data + sb->len, str, slen);
    sb->len += slen;
    sb->data[sb->len] = '\0';
}

//sb nonnull
//fmt nonnull
static void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
    wgsl_compiler_assert(sb != NULL, "sb_appendf: sb is NULL");
    wgsl_compiler_assert(fmt != NULL, "sb_appendf: fmt is NULL");
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    sb_append(sb, buf);
}

//mod nonnull
//sb nonnull
static void ssir_type_to_string(SsirModule *mod, uint32_t type_id, StringBuilder *sb) {
    wgsl_compiler_assert(mod != NULL, "ssir_type_to_string: mod is NULL");
    wgsl_compiler_assert(sb != NULL, "ssir_type_to_string: sb is NULL");
    SsirType *t = ssir_get_type(mod, type_id);
    if (!t) {
        sb_appendf(sb, "type_%u", type_id);
        return;
    }

    switch (t->kind) {
        case SSIR_TYPE_VOID: sb_append(sb, "void"); break;
        case SSIR_TYPE_BOOL: sb_append(sb, "bool"); break;
        case SSIR_TYPE_I32: sb_append(sb, "i32"); break;
        case SSIR_TYPE_U32: sb_append(sb, "u32"); break;
        case SSIR_TYPE_F32: sb_append(sb, "f32"); break;
        case SSIR_TYPE_F16: sb_append(sb, "f16"); break;
        case SSIR_TYPE_F64: sb_append(sb, "f64"); break;
        case SSIR_TYPE_I8: sb_append(sb, "i8"); break;
        case SSIR_TYPE_U8: sb_append(sb, "u8"); break;
        case SSIR_TYPE_I16: sb_append(sb, "i16"); break;
        case SSIR_TYPE_U16: sb_append(sb, "u16"); break;
        case SSIR_TYPE_I64: sb_append(sb, "i64"); break;
        case SSIR_TYPE_U64: sb_append(sb, "u64"); break;
        case SSIR_TYPE_VEC:
            sb_append(sb, "vec");
            sb_appendf(sb, "%u<", t->vec.size);
            ssir_type_to_string(mod, t->vec.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_MAT:
            sb_appendf(sb, "mat%ux%u<", t->mat.cols, t->mat.rows);
            ssir_type_to_string(mod, t->mat.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_ARRAY:
            sb_append(sb, "array<");
            ssir_type_to_string(mod, t->array.elem, sb);
            sb_appendf(sb, ", %u>", t->array.length);
            break;
        case SSIR_TYPE_RUNTIME_ARRAY:
            sb_append(sb, "array<");
            ssir_type_to_string(mod, t->runtime_array.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_STRUCT:
            sb_appendf(sb, "struct %s", t->struc.name ? t->struc.name : "anon");
            break;
        case SSIR_TYPE_PTR: {
            static const char *space_names[] = {
                "function", "private", "workgroup", "uniform",
                "storage", "input", "output", "push_constant"
            };
            sb_append(sb, "ptr<");
            sb_append(sb, space_names[t->ptr.space]);
            sb_append(sb, ", ");
            ssir_type_to_string(mod, t->ptr.pointee, sb);
            sb_append(sb, ">");
            break;
        }
        case SSIR_TYPE_SAMPLER:
            sb_append(sb, "sampler");
            break;
        case SSIR_TYPE_SAMPLER_COMPARISON:
            sb_append(sb, "sampler_comparison");
            break;
        case SSIR_TYPE_TEXTURE:
            sb_append(sb, "texture");
            break;
        case SSIR_TYPE_TEXTURE_STORAGE:
            sb_append(sb, "texture_storage");
            break;
        case SSIR_TYPE_TEXTURE_DEPTH:
            sb_append(sb, "texture_depth");
            break;
    }
}

//mod nonnull
char *ssir_module_to_string(SsirModule *mod) {
    wgsl_compiler_assert(mod != NULL, "ssir_module_to_string: mod is NULL");
    StringBuilder sb = {0};

    sb_append(&sb, "; SSIR Module\n\n");

    /* Types */
    sb_append(&sb, "; Types\n");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        sb_appendf(&sb, "%%type_%u = ", i);
        ssir_type_to_string(mod, i, &sb);
        sb_append(&sb, "\n");
    }
    sb_append(&sb, "\n");

    /* Constants */
    if (mod->constant_count > 0) {
        sb_append(&sb, "; Constants\n");
        for (uint32_t i = 0; i < mod->constant_count; i++) {
            SsirConstant *c = &mod->constants[i];
            sb_appendf(&sb, "%%%u : ", c->id);
            ssir_type_to_string(mod, c->type, &sb);
            sb_append(&sb, " = ");
            switch (c->kind) {
                case SSIR_CONST_BOOL:
                    sb_append(&sb, c->bool_val ? "true" : "false");
                    break;
                case SSIR_CONST_I32:
                    sb_appendf(&sb, "%d", c->i32_val);
                    break;
                case SSIR_CONST_U32:
                    sb_appendf(&sb, "%uu", c->u32_val);
                    break;
                case SSIR_CONST_F32:
                    sb_appendf(&sb, "%f", c->f32_val);
                    break;
                case SSIR_CONST_F16:
                    sb_appendf(&sb, "0x%04xh", c->f16_val);
                    break;
                case SSIR_CONST_F64:
                    sb_appendf(&sb, "%f", c->f64_val);
                    break;
                case SSIR_CONST_I8:
                    sb_appendf(&sb, "%d", (int)c->i8_val);
                    break;
                case SSIR_CONST_U8:
                    sb_appendf(&sb, "%uu", (unsigned)c->u8_val);
                    break;
                case SSIR_CONST_I16:
                    sb_appendf(&sb, "%d", (int)c->i16_val);
                    break;
                case SSIR_CONST_U16:
                    sb_appendf(&sb, "%uu", (unsigned)c->u16_val);
                    break;
                case SSIR_CONST_I64:
                    sb_appendf(&sb, "%lldl", (long long)c->i64_val);
                    break;
                case SSIR_CONST_U64:
                    sb_appendf(&sb, "%lluul", (unsigned long long)c->u64_val);
                    break;
                case SSIR_CONST_COMPOSITE:
                    sb_append(&sb, "composite(");
                    for (uint32_t j = 0; j < c->composite.count; j++) {
                        if (j > 0) sb_append(&sb, ", ");
                        sb_appendf(&sb, "%%%u", c->composite.components[j]);
                    }
                    sb_append(&sb, ")");
                    break;
                case SSIR_CONST_NULL:
                    sb_append(&sb, "null");
                    break;
            }
            sb_append(&sb, "\n");
        }
        sb_append(&sb, "\n");
    }

    /* Globals */
    if (mod->global_count > 0) {
        sb_append(&sb, "; Globals\n");
        for (uint32_t i = 0; i < mod->global_count; i++) {
            SsirGlobalVar *g = &mod->globals[i];
            if (g->has_group) sb_appendf(&sb, "@group(%u) ", g->group);
            if (g->has_binding) sb_appendf(&sb, "@binding(%u) ", g->binding);
            if (g->has_location) sb_appendf(&sb, "@location(%u) ", g->location);
            if (g->builtin != SSIR_BUILTIN_NONE) sb_appendf(&sb, "@builtin(%d) ", g->builtin);
            sb_appendf(&sb, "%%%u", g->id);
            if (g->name) sb_appendf(&sb, " \"%s\"", g->name);
            sb_append(&sb, " : ");
            ssir_type_to_string(mod, g->type, &sb);
            sb_append(&sb, "\n");
        }
        sb_append(&sb, "\n");
    }

    /* Functions */
    for (uint32_t fi = 0; fi < mod->function_count; fi++) {
        SsirFunction *f = &mod->functions[fi];
        sb_appendf(&sb, "fn %%%u", f->id);
        if (f->name) sb_appendf(&sb, " \"%s\"", f->name);
        sb_append(&sb, "(");
        for (uint32_t pi = 0; pi < f->param_count; pi++) {
            if (pi > 0) sb_append(&sb, ", ");
            SsirFunctionParam *p = &f->params[pi];
            sb_appendf(&sb, "%%%u : ", p->id);
            ssir_type_to_string(mod, p->type, &sb);
        }
        sb_append(&sb, ") -> ");
        ssir_type_to_string(mod, f->return_type, &sb);
        sb_append(&sb, " {\n");

        /* Locals */
        for (uint32_t li = 0; li < f->local_count; li++) {
            SsirLocalVar *l = &f->locals[li];
            sb_appendf(&sb, "    var %%%u : ", l->id);
            ssir_type_to_string(mod, l->type, &sb);
            sb_append(&sb, "\n");
        }
        if (f->local_count > 0) sb_append(&sb, "\n");

        /* Blocks */
        for (uint32_t bi = 0; bi < f->block_count; bi++) {
            SsirBlock *b = &f->blocks[bi];
            sb_appendf(&sb, "  block_%u", b->id);
            if (b->name) sb_appendf(&sb, " \"%s\"", b->name);
            sb_append(&sb, ":\n");

            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *inst = &b->insts[ii];
                sb_append(&sb, "    ");
                if (inst->result) {
                    sb_appendf(&sb, "%%%u : ", inst->result);
                    ssir_type_to_string(mod, inst->type, &sb);
                    sb_append(&sb, " = ");
                }
                sb_append(&sb, ssir_opcode_name(inst->op));
                for (uint8_t oi = 0; oi < inst->operand_count; oi++) {
                    sb_appendf(&sb, " %%%u", inst->operands[oi]);
                }
                if (inst->extra_count > 0) {
                    sb_append(&sb, " [");
                    for (uint16_t ei = 0; ei < inst->extra_count; ei++) {
                        if (ei > 0) sb_append(&sb, ", ");
                        sb_appendf(&sb, "%u", inst->extra[ei]);
                    }
                    sb_append(&sb, "]");
                }
                sb_append(&sb, "\n");
            }
        }
        sb_append(&sb, "}\n\n");
    }

    /* Entry points */
    if (mod->entry_point_count > 0) {
        sb_append(&sb, "; Entry points\n");
        static const char *stage_names[] = { "vertex", "fragment", "compute" };
        for (uint32_t ei = 0; ei < mod->entry_point_count; ei++) {
            SsirEntryPoint *ep = &mod->entry_points[ei];
            sb_appendf(&sb, "@%s entry \"%s\" = %%%u\n",
                      stage_names[ep->stage], ep->name ? ep->name : "", ep->function);
            if (ep->stage == SSIR_STAGE_COMPUTE) {
                sb_appendf(&sb, "  workgroup_size(%u, %u, %u)\n",
                          ep->workgroup_size[0], ep->workgroup_size[1], ep->workgroup_size[2]);
            }
            if (ep->interface_count > 0) {
                sb_append(&sb, "  interface: [");
                for (uint32_t ii = 0; ii < ep->interface_count; ii++) {
                    if (ii > 0) sb_append(&sb, ", ");
                    sb_appendf(&sb, "%%%u", ep->interface[ii]);
                }
                sb_append(&sb, "]\n");
            }
        }
    }

    return sb.data;
}
