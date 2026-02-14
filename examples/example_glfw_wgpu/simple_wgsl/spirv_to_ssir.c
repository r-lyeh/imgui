/*
 * SPIR-V to SSIR VtsConverter - Implementation
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

#ifndef SPIRV_TO_SSIR_MALLOC
#define SPIRV_TO_SSIR_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef SPIRV_TO_SSIR_REALLOC
#define SPIRV_TO_SSIR_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef SPIRV_TO_SSIR_FREE
#define SPIRV_TO_SSIR_FREE(p) free((p))
#endif

#define SPV_MAGIC 0x07230203

typedef enum {
    VTS_SPV_ID_UNKNOWN = 0,
    VTS_SPV_ID_TYPE,
    VTS_SPV_ID_CONSTANT,
    VTS_SPV_ID_VARIABLE,
    VTS_SPV_ID_FUNCTION,
    VTS_SPV_ID_LABEL,
    VTS_SPV_ID_INSTRUCTION,
    VTS_SPV_ID_EXT_INST_IMPORT,
    VTS_SPV_ID_PARAM,
} VtsSpvIdKind;

typedef enum {
    VTS_SPV_TYPE_VOID = 0,
    VTS_SPV_TYPE_BOOL,
    VTS_SPV_TYPE_INT,
    VTS_SPV_TYPE_FLOAT,
    VTS_SPV_TYPE_VECTOR,
    VTS_SPV_TYPE_MATRIX,
    VTS_SPV_TYPE_ARRAY,
    VTS_SPV_TYPE_RUNTIME_ARRAY,
    VTS_SPV_TYPE_STRUCT,
    VTS_SPV_TYPE_POINTER,
    VTS_SPV_TYPE_FUNCTION,
    VTS_SPV_TYPE_IMAGE,
    VTS_SPV_TYPE_SAMPLED_IMAGE,
    VTS_SPV_TYPE_SAMPLER,
} VtsSpvTypeKind;

typedef struct {
    VtsSpvTypeKind kind;
    union {
        struct { uint32_t width; uint32_t signedness; } int_type;
        struct { uint32_t width; } float_type;
        struct { uint32_t component_type; uint32_t count; } vector;
        struct { uint32_t column_type; uint32_t columns; } matrix;
        struct { uint32_t element_type; uint32_t length_id; } array;
        struct { uint32_t element_type; } runtime_array;
        struct { uint32_t *member_types; int member_count; } struct_type;
        struct { uint32_t pointee_type; SpvStorageClass storage; } pointer;
        struct { uint32_t return_type; uint32_t *param_types; int param_count; } function;
        struct { uint32_t sampled_type; SpvDim dim; uint32_t depth; uint32_t arrayed; uint32_t ms; uint32_t sampled; SpvImageFormat format; } image;
        struct { uint32_t image_type; } sampled_image;
    };
} VtsSpvTypeInfo;

typedef struct {
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} VtsSpvDecorationEntry;

typedef struct {
    uint32_t member_index;
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} VtsSpvMemberDecoration;

typedef struct {
    VtsSpvIdKind kind;
    uint32_t id;
    char *name;
    VtsSpvTypeInfo type_info;
    uint32_t ssir_id;

    union {
        struct { uint32_t type_id; uint32_t *values; int value_count; int is_composite; int is_spec; uint32_t spec_id; } constant;
        struct { uint32_t type_id; SpvStorageClass storage_class; uint32_t initializer; } variable;
        struct { uint32_t return_type; uint32_t func_type; } function;
        struct { uint32_t type_id; SpvOp opcode; uint32_t *operands; int operand_count; } instruction;
        struct { uint32_t type_id; } param;
    };

    VtsSpvDecorationEntry *decorations;
    int decoration_count;
    VtsSpvMemberDecoration *member_decorations;
    int member_decoration_count;
    char **member_names;
    int member_name_count;
} VtsSpvIdInfo;

typedef struct {
    uint32_t label_id;
    uint32_t *instructions;
    int instruction_count;
    int instruction_cap;
    uint32_t merge_block;
    uint32_t continue_block;
    int is_loop_header;
    int is_selection_header;
} VtsSpvBasicBlock;

typedef struct {
    uint32_t id;
    char *name;
    uint32_t return_type;
    uint32_t func_type;
    uint32_t *params;
    int param_count;
    VtsSpvBasicBlock *blocks;
    int block_count;
    int block_cap;
    SpvExecutionModel exec_model;
    int is_entry_point;
    uint32_t *interface_vars;
    int interface_var_count;
    int workgroup_size[3];
    bool depth_replacing;
    bool origin_upper_left;
    bool early_fragment_tests;
    uint32_t *local_vars;
    int local_var_count;
    int local_var_cap;
} VtsSpvFunction;

typedef struct {
    uint32_t func_id;
    SpvExecutionModel exec_model;
    char *name;
    uint32_t *interface_vars;
    int interface_var_count;
} VtsPendingEntryPoint;

typedef struct {
    uint32_t func_id;
    int workgroup_size[3];
} VtsPendingWorkgroupSize;

typedef struct {
    uint32_t func_id;
    SpvExecutionMode mode;
} VtsPendingExecMode;

typedef struct {
    const uint32_t *spirv;
    size_t word_count;
    uint32_t version;
    uint32_t generator;
    uint32_t id_bound;

    VtsSpvIdInfo *ids;
    VtsSpvFunction *functions;
    int function_count;
    int function_cap;

    VtsPendingEntryPoint *pending_eps;
    int pending_ep_count;
    int pending_ep_cap;

    VtsPendingWorkgroupSize *pending_wgs;
    int pending_wg_count;
    int pending_wg_cap;

    VtsPendingExecMode *pending_exec_modes;
    int pending_exec_mode_count;
    int pending_exec_mode_cap;

    uint32_t glsl_ext_id;
    int type_depth;

    SsirModule *mod;
    const SpirvToSsirOptions *opts;
    char last_error[512];
} VtsConverter;

//c nonnull
//msg nonnull
static void vts_set_error(VtsConverter *c, const char *msg) {
    wgsl_compiler_assert(c != NULL, "vts_set_error: c is NULL");
    wgsl_compiler_assert(msg != NULL, "vts_set_error: msg is NULL");
    size_t n = strlen(msg);
    if (n >= sizeof(c->last_error)) n = sizeof(c->last_error) - 1;
    memcpy(c->last_error, msg, n);
    c->last_error[n] = 0;
}

//words nonnull
//out_words_read nonnull
static char *vts_read_string(const uint32_t *words, int word_count, int *out_words_read) {
    wgsl_compiler_assert(words != NULL, "vts_read_string: words is NULL");
    wgsl_compiler_assert(out_words_read != NULL, "vts_read_string: out_words_read is NULL");
    if (word_count <= 0) { *out_words_read = 0; return NULL; }
    int max_chars = word_count * 4;
    const char *str = (const char*)words;
    int len = 0;
    while (len < max_chars && str[len]) len++;
    *out_words_read = (len + 4) / 4;
    char *copy = (char*)SPIRV_TO_SSIR_MALLOC(len + 1);
    if (!copy) return NULL;
    memcpy(copy, str, len);
    copy[len] = 0;
    return copy;
}

//c nonnull
static void vts_add_decoration(VtsConverter *c, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "vts_add_decoration: c is NULL");
    if (target >= c->id_bound) return;
    VtsSpvIdInfo *info = &c->ids[target];
    int idx = info->decoration_count++;
    info->decorations = (VtsSpvDecorationEntry*)SPIRV_TO_SSIR_REALLOC(info->decorations, info->decoration_count * sizeof(VtsSpvDecorationEntry));
    /* PRE: realloc succeeded */
    wgsl_compiler_assert(info->decorations != NULL, "vts_add_decoration: realloc failed");
    info->decorations[idx].decoration = decor;
    info->decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->decorations[idx].literals = (uint32_t*)SPIRV_TO_SSIR_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->decorations[idx].literals = NULL;
    }
}

//c nonnull
static void vts_add_member_decoration(VtsConverter *c, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "vts_add_member_decoration: c is NULL");
    if (struct_id >= c->id_bound) return;
    VtsSpvIdInfo *info = &c->ids[struct_id];
    int idx = info->member_decoration_count++;
    info->member_decorations = (VtsSpvMemberDecoration*)SPIRV_TO_SSIR_REALLOC(info->member_decorations, info->member_decoration_count * sizeof(VtsSpvMemberDecoration));
    /* PRE: realloc succeeded */
    wgsl_compiler_assert(info->member_decorations != NULL, "vts_add_member_decoration: realloc failed");
    info->member_decorations[idx].member_index = member;
    info->member_decorations[idx].decoration = decor;
    info->member_decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->member_decorations[idx].literals = (uint32_t*)SPIRV_TO_SSIR_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->member_decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->member_decorations[idx].literals = NULL;
    }
}

//c nonnull
static int vts_has_decoration(VtsConverter *c, uint32_t id, SpvDecoration decor, uint32_t *out_value) {
    wgsl_compiler_assert(c != NULL, "vts_has_decoration: c is NULL");
    if (id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[id];
    /* PRE: decorations array valid if decoration_count > 0 */
    wgsl_compiler_assert(info->decoration_count == 0 || info->decorations != NULL, "vts_has_decoration: decorations NULL with count > 0");
    for (int i = 0; i < info->decoration_count; i++) {
        if (info->decorations[i].decoration == decor) {
            /* PRE: literals array valid if literal_count > 0 */
            wgsl_compiler_assert(info->decorations[i].literal_count == 0 || info->decorations[i].literals != NULL, "vts_has_decoration: literals NULL with count > 0");
            if (out_value && info->decorations[i].literal_count > 0) {
                *out_value = info->decorations[i].literals[0];
            }
            return 1;
        }
    }
    return 0;
}

//c nonnull
//out_offset nonnull
static int get_member_offset(VtsConverter *c, uint32_t struct_id, uint32_t member, uint32_t *out_offset) {
    wgsl_compiler_assert(c != NULL, "get_member_offset: c is NULL");
    wgsl_compiler_assert(out_offset != NULL, "get_member_offset: out_offset is NULL");
    if (struct_id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[struct_id];
    /* PRE: member_decorations array valid if count > 0 */
    wgsl_compiler_assert(info->member_decoration_count == 0 || info->member_decorations != NULL, "get_member_offset: member_decorations NULL with count > 0");
    for (int i = 0; i < info->member_decoration_count; i++) {
        if (info->member_decorations[i].member_index == member &&
            info->member_decorations[i].decoration == SpvDecorationOffset &&
            info->member_decorations[i].literal_count > 0) {
            /* PRE: literals array valid if literal_count > 0 */
            wgsl_compiler_assert(info->member_decorations[i].literals != NULL, "get_member_offset: literals NULL with count > 0");
            *out_offset = info->member_decorations[i].literals[0];
            return 1;
        }
    }
    return 0;
}

//c nonnull
static VtsSpvFunction *vts_add_function(VtsConverter *c) {
    wgsl_compiler_assert(c != NULL, "vts_add_function: c is NULL");
    if (c->function_count >= c->function_cap) {
        int ncap = c->function_cap ? c->function_cap * 2 : 8;
        c->functions = (VtsSpvFunction*)SPIRV_TO_SSIR_REALLOC(c->functions, ncap * sizeof(VtsSpvFunction));
        /* PRE: realloc succeeded */
        wgsl_compiler_assert(c->functions != NULL, "vts_add_function: realloc failed");
        c->function_cap = ncap;
    }
    VtsSpvFunction *fn = &c->functions[c->function_count++];
    memset(fn, 0, sizeof(VtsSpvFunction));
    fn->workgroup_size[0] = 1;
    fn->workgroup_size[1] = 1;
    fn->workgroup_size[2] = 1;
    return fn;
}

//fn nonnull
static VtsSpvBasicBlock *vts_add_block(VtsSpvFunction *fn, uint32_t label_id) {
    wgsl_compiler_assert(fn != NULL, "vts_add_block: fn is NULL");
    if (fn->block_count >= fn->block_cap) {
        int ncap = fn->block_cap ? fn->block_cap * 2 : 8;
        fn->blocks = (VtsSpvBasicBlock*)SPIRV_TO_SSIR_REALLOC(fn->blocks, ncap * sizeof(VtsSpvBasicBlock));
        /* PRE: realloc succeeded */
        wgsl_compiler_assert(fn->blocks != NULL, "vts_add_block: realloc failed");
        fn->block_cap = ncap;
    }
    VtsSpvBasicBlock *blk = &fn->blocks[fn->block_count++];
    memset(blk, 0, sizeof(VtsSpvBasicBlock));
    blk->label_id = label_id;
    return blk;
}

//blk nonnull
static void vts_add_block_instr(VtsSpvBasicBlock *blk, uint32_t instr_start) {
    wgsl_compiler_assert(blk != NULL, "vts_add_block_instr: blk is NULL");
    if (blk->instruction_count >= blk->instruction_cap) {
        int ncap = blk->instruction_cap ? blk->instruction_cap * 2 : 16;
        blk->instructions = (uint32_t*)SPIRV_TO_SSIR_REALLOC(blk->instructions, ncap * sizeof(uint32_t));
        /* PRE: realloc succeeded */
        wgsl_compiler_assert(blk->instructions != NULL, "vts_add_block_instr: realloc failed");
        blk->instruction_cap = ncap;
    }
    blk->instructions[blk->instruction_count++] = instr_start;
}

//fn nonnull
static void add_function_local_var(VtsSpvFunction *fn, uint32_t var_id) {
    wgsl_compiler_assert(fn != NULL, "add_function_local_var: fn is NULL");
    if (fn->local_var_count >= fn->local_var_cap) {
        int ncap = fn->local_var_cap ? fn->local_var_cap * 2 : 8;
        fn->local_vars = (uint32_t*)SPIRV_TO_SSIR_REALLOC(fn->local_vars, ncap * sizeof(uint32_t));
        /* PRE: realloc succeeded */
        wgsl_compiler_assert(fn->local_vars != NULL, "add_function_local_var: realloc failed");
        fn->local_var_cap = ncap;
    }
    fn->local_vars[fn->local_var_count++] = var_id;
}

//c nonnull
static SpirvToSsirResult parse_spirv(VtsConverter *c) {
    wgsl_compiler_assert(c != NULL, "parse_spirv: c is NULL");
    size_t pos = 5;
    VtsSpvFunction *current_fn = NULL;
    VtsSpvBasicBlock *current_block = NULL;

    while (pos < c->word_count) {
        uint32_t word0 = c->spirv[pos];
        uint16_t opcode = word0 & 0xFFFF;
        uint16_t wc = word0 >> 16;
        if (wc == 0 || pos + wc > c->word_count) {
            vts_set_error(c, "Invalid instruction");
            return SPIRV_TO_SSIR_INVALID_SPIRV;
        }

        const uint32_t *operands = &c->spirv[pos + 1];
        int operand_count = wc - 1;

        switch ((SpvOp)opcode) {
        case SpvOpName:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                int str_words;
                char *name = vts_read_string(&operands[1], operand_count - 1, &str_words);
                if (target < c->id_bound && name) {
                    c->ids[target].name = name;
                }
            }
            break;

        case SpvOpMemberName:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                int str_words;
                char *name = vts_read_string(&operands[2], operand_count - 2, &str_words);
                if (struct_id < c->id_bound && member < 4096 && name) {
                    VtsSpvIdInfo *info = &c->ids[struct_id];
                    if (member >= (uint32_t)info->member_name_count) {
                        int new_count = (int)member + 1;
                        info->member_names = (char**)SPIRV_TO_SSIR_REALLOC(info->member_names, new_count * sizeof(char*));
                        /* PRE: realloc succeeded */
                        wgsl_compiler_assert(info->member_names != NULL, "SpvOpMemberName: realloc failed");
                        for (int i = info->member_name_count; i < new_count; i++) {
                            info->member_names[i] = NULL;
                        }
                        info->member_name_count = new_count;
                    }
                    if (info->member_names[member]) SPIRV_TO_SSIR_FREE(info->member_names[member]);
                    info->member_names[member] = name;
                }
            }
            break;

        case SpvOpDecorate:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                SpvDecoration decor = (SpvDecoration)operands[1];
                vts_add_decoration(c, target, decor, &operands[2], operand_count - 2);
            }
            break;

        case SpvOpMemberDecorate:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                SpvDecoration decor = (SpvDecoration)operands[2];
                vts_add_member_decoration(c, struct_id, member, decor, &operands[3], operand_count - 3);
            }
            break;

        case SpvOpExtInstImport:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_EXT_INST_IMPORT;
                    int str_words;
                    char *name = vts_read_string(&operands[1], operand_count - 1, &str_words);
                    if (name && strcmp(name, "GLSL.std.450") == 0) {
                        c->glsl_ext_id = id;
                    }
                    SPIRV_TO_SSIR_FREE(name);
                }
            }
            break;

        case SpvOpTypeVoid:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_VOID;
                }
            }
            break;

        case SpvOpTypeBool:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_BOOL;
                }
            }
            break;

        case SpvOpTypeInt:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_INT;
                    c->ids[id].type_info.int_type.width = operands[1];
                    c->ids[id].type_info.int_type.signedness = operands[2];
                }
            }
            break;

        case SpvOpTypeFloat:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_FLOAT;
                    c->ids[id].type_info.float_type.width = operands[1];
                }
            }
            break;

        case SpvOpTypeVector:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_VECTOR;
                    c->ids[id].type_info.vector.component_type = operands[1];
                    c->ids[id].type_info.vector.count = operands[2];
                }
            }
            break;

        case SpvOpTypeMatrix:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_MATRIX;
                    c->ids[id].type_info.matrix.column_type = operands[1];
                    c->ids[id].type_info.matrix.columns = operands[2];
                }
            }
            break;

        case SpvOpTypeArray:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_ARRAY;
                    c->ids[id].type_info.array.element_type = operands[1];
                    c->ids[id].type_info.array.length_id = operands[2];
                }
            }
            break;

        case SpvOpTypeRuntimeArray:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_RUNTIME_ARRAY;
                    c->ids[id].type_info.runtime_array.element_type = operands[1];
                }
            }
            break;

        case SpvOpTypeStruct:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_STRUCT;
                    int mc = operand_count - 1;
                    c->ids[id].type_info.struct_type.member_count = mc;
                    c->ids[id].type_info.struct_type.member_types = NULL;
                    if (mc > 0) {
                        c->ids[id].type_info.struct_type.member_types = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(c->ids[id].type_info.struct_type.member_types != NULL, "SpvOpTypeStruct: malloc failed");
                        memcpy(c->ids[id].type_info.struct_type.member_types, &operands[1], mc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypePointer:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_POINTER;
                    c->ids[id].type_info.pointer.storage = (SpvStorageClass)operands[1];
                    c->ids[id].type_info.pointer.pointee_type = operands[2];
                }
            }
            break;

        case SpvOpTypeFunction:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_FUNCTION;
                    c->ids[id].type_info.function.return_type = operands[1];
                    int pc = operand_count - 2;
                    c->ids[id].type_info.function.param_count = pc;
                    c->ids[id].type_info.function.param_types = NULL;
                    if (pc > 0) {
                        c->ids[id].type_info.function.param_types = (uint32_t*)SPIRV_TO_SSIR_MALLOC(pc * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(c->ids[id].type_info.function.param_types != NULL, "SpvOpTypeFunction: malloc failed");
                        memcpy(c->ids[id].type_info.function.param_types, &operands[2], pc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypeImage:
            if (operand_count >= 8) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_IMAGE;
                    c->ids[id].type_info.image.sampled_type = operands[1];
                    c->ids[id].type_info.image.dim = (SpvDim)operands[2];
                    c->ids[id].type_info.image.depth = operands[3];
                    c->ids[id].type_info.image.arrayed = operands[4];
                    c->ids[id].type_info.image.ms = operands[5];
                    c->ids[id].type_info.image.sampled = operands[6];
                    c->ids[id].type_info.image.format = (SpvImageFormat)operands[7];
                }
            }
            break;

        case SpvOpTypeSampledImage:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_SAMPLED_IMAGE;
                    c->ids[id].type_info.sampled_image.image_type = operands[1];
                }
            }
            break;

        case SpvOpTypeSampler:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_TYPE;
                    c->ids[id].type_info.kind = VTS_SPV_TYPE_SAMPLER;
                }
            }
            break;

        case SpvOpConstant:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_CONSTANT;
                    c->ids[id].constant.type_id = type_id;
                    c->ids[id].constant.is_composite = 0;
                    c->ids[id].constant.values = NULL;
                    c->ids[id].constant.is_spec = ((SpvOp)opcode == SpvOpSpecConstant ||
                        (SpvOp)opcode == SpvOpSpecConstantTrue ||
                        (SpvOp)opcode == SpvOpSpecConstantFalse);
                    int vc = operand_count - 2;
                    c->ids[id].constant.value_count = vc;
                    if (vc > 0) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(vc * sizeof(uint32_t));
                        memcpy(c->ids[id].constant.values, &operands[2], vc * sizeof(uint32_t));
                    }
                    if ((SpvOp)opcode == SpvOpConstantTrue || (SpvOp)opcode == SpvOpSpecConstantTrue) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(sizeof(uint32_t));
                        c->ids[id].constant.values[0] = 1;
                        c->ids[id].constant.value_count = 1;
                    } else if ((SpvOp)opcode == SpvOpConstantFalse || (SpvOp)opcode == SpvOpSpecConstantFalse) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(sizeof(uint32_t));
                        c->ids[id].constant.values[0] = 0;
                        c->ids[id].constant.value_count = 1;
                    }
                }
            }
            break;

        case SpvOpConstantComposite:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_CONSTANT;
                    c->ids[id].constant.type_id = type_id;
                    c->ids[id].constant.is_composite = 1;
                    c->ids[id].constant.values = NULL;
                    int vc = operand_count - 2;
                    c->ids[id].constant.value_count = vc;
                    if (vc > 0) {
                        c->ids[id].constant.values = (uint32_t*)SPIRV_TO_SSIR_MALLOC(vc * sizeof(uint32_t));
                        memcpy(c->ids[id].constant.values, &operands[2], vc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpVariable:
            if (operand_count >= 3) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                SpvStorageClass sc = (SpvStorageClass)operands[2];
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_VARIABLE;
                    c->ids[id].variable.type_id = type_id;
                    c->ids[id].variable.storage_class = sc;
                    c->ids[id].variable.initializer = (operand_count > 3) ? operands[3] : 0;
                    if (sc == SpvStorageClassFunction && current_fn) {
                        add_function_local_var(current_fn, id);
                    }
                }
            }
            break;

        case SpvOpEntryPoint:
            if (operand_count >= 3) {
                SpvExecutionModel model = (SpvExecutionModel)operands[0];
                uint32_t fn_id = operands[1];
                int str_words;
                char *name = vts_read_string(&operands[2], operand_count - 2, &str_words);
                if (fn_id < c->id_bound) {
                    c->ids[fn_id].kind = VTS_SPV_ID_FUNCTION;
                    if (c->ids[fn_id].name) SPIRV_TO_SSIR_FREE(c->ids[fn_id].name);
                    c->ids[fn_id].name = name ? strdup(name) : NULL;
                }
                if (c->pending_ep_count >= c->pending_ep_cap) {
                    int ncap = c->pending_ep_cap ? c->pending_ep_cap * 2 : 4;
                    c->pending_eps = (VtsPendingEntryPoint*)SPIRV_TO_SSIR_REALLOC(c->pending_eps, ncap * sizeof(VtsPendingEntryPoint));
                    c->pending_ep_cap = ncap;
                }
                VtsPendingEntryPoint *ep = &c->pending_eps[c->pending_ep_count++];
                ep->func_id = fn_id;
                ep->exec_model = model;
                ep->name = name ? strdup(name) : NULL;
                int iface_start = 2 + str_words;
                int iface_count = operand_count - iface_start;
                if (iface_count > 0) {
                    ep->interface_vars = (uint32_t*)SPIRV_TO_SSIR_MALLOC(iface_count * sizeof(uint32_t));
                    memcpy(ep->interface_vars, &operands[iface_start], iface_count * sizeof(uint32_t));
                    ep->interface_var_count = iface_count;
                } else {
                    ep->interface_vars = NULL;
                    ep->interface_var_count = 0;
                }
                SPIRV_TO_SSIR_FREE(name);
            }
            break;

        case SpvOpExecutionMode:
            if (operand_count >= 2) {
                uint32_t fn_id = operands[0];
                SpvExecutionMode mode = (SpvExecutionMode)operands[1];
                if (mode == SpvExecutionModeLocalSize && operand_count >= 5) {
                    if (c->pending_wg_count >= c->pending_wg_cap) {
                        int ncap = c->pending_wg_cap ? c->pending_wg_cap * 2 : 4;
                        c->pending_wgs = (VtsPendingWorkgroupSize*)SPIRV_TO_SSIR_REALLOC(c->pending_wgs, ncap * sizeof(VtsPendingWorkgroupSize));
                        c->pending_wg_cap = ncap;
                    }
                    VtsPendingWorkgroupSize *wg = &c->pending_wgs[c->pending_wg_count++];
                    wg->func_id = fn_id;
                    wg->workgroup_size[0] = operands[2];
                    wg->workgroup_size[1] = operands[3];
                    wg->workgroup_size[2] = operands[4];
                }
                /* Store execution modes for later application */
                if (mode == SpvExecutionModeDepthReplacing ||
                    mode == SpvExecutionModeOriginUpperLeft ||
                    mode == SpvExecutionModeEarlyFragmentTests) {
                    if (c->pending_exec_mode_count >= c->pending_exec_mode_cap) {
                        int ncap = c->pending_exec_mode_cap ? c->pending_exec_mode_cap * 2 : 4;
                        c->pending_exec_modes = (VtsPendingExecMode*)SPIRV_TO_SSIR_REALLOC(
                            c->pending_exec_modes, ncap * sizeof(VtsPendingExecMode));
                        c->pending_exec_mode_cap = ncap;
                    }
                    VtsPendingExecMode *em = &c->pending_exec_modes[c->pending_exec_mode_count++];
                    em->func_id = fn_id;
                    em->mode = mode;
                }
            }
            break;

        case SpvOpFunction:
            if (operand_count >= 4) {
                uint32_t ret_type = operands[0];
                uint32_t id = operands[1];
                uint32_t func_type = operands[3];
                current_fn = vts_add_function(c);
                current_fn->id = id;
                current_fn->return_type = ret_type;
                current_fn->func_type = func_type;
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_FUNCTION;
                    c->ids[id].function.return_type = ret_type;
                    c->ids[id].function.func_type = func_type;
                    if (c->ids[id].name) {
                        current_fn->name = strdup(c->ids[id].name);
                    }
                }
                current_block = NULL;
            }
            break;

        case SpvOpFunctionParameter:
            if (current_fn && operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                int idx = current_fn->param_count++;
                current_fn->params = (uint32_t*)SPIRV_TO_SSIR_REALLOC(current_fn->params, current_fn->param_count * sizeof(uint32_t));
                /* PRE: realloc succeeded */
                wgsl_compiler_assert(current_fn->params != NULL, "SpvOpFunctionParameter: realloc failed");
                current_fn->params[idx] = id;
                if (id < c->id_bound) {
                    c->ids[id].kind = VTS_SPV_ID_PARAM;
                    c->ids[id].param.type_id = type_id;
                }
            }
            break;

        case SpvOpFunctionEnd:
            current_fn = NULL;
            current_block = NULL;
            break;

        case SpvOpLabel:
            if (current_fn && operand_count >= 1) {
                uint32_t label_id = operands[0];
                current_block = vts_add_block(current_fn, label_id);
                if (label_id < c->id_bound) {
                    c->ids[label_id].kind = VTS_SPV_ID_LABEL;
                }
            }
            break;

        case SpvOpLoopMerge:
            if (current_block && operand_count >= 2) {
                current_block->merge_block = operands[0];
                current_block->continue_block = operands[1];
                current_block->is_loop_header = 1;
            }
            break;

        case SpvOpSelectionMerge:
            if (current_block && operand_count >= 1) {
                current_block->merge_block = operands[0];
                current_block->is_selection_header = 1;
            }
            break;

        default:
            if (current_block) {
                vts_add_block_instr(current_block, pos);
            }
            if (operand_count >= 2) {
                SpvOp op = (SpvOp)opcode;
                int has_result_type = 0;
                int has_result = 0;
                switch (op) {
                case SpvOpLoad: case SpvOpAccessChain: case SpvOpInBoundsAccessChain:
                case SpvOpFAdd: case SpvOpFSub: case SpvOpFMul: case SpvOpFDiv: case SpvOpFRem: case SpvOpFMod:
                case SpvOpIAdd: case SpvOpISub: case SpvOpIMul: case SpvOpSDiv: case SpvOpUDiv: case SpvOpSRem: case SpvOpUMod: case SpvOpSMod:
                case SpvOpFNegate: case SpvOpSNegate:
                case SpvOpFOrdEqual: case SpvOpFOrdNotEqual: case SpvOpFOrdLessThan: case SpvOpFOrdGreaterThan:
                case SpvOpFOrdLessThanEqual: case SpvOpFOrdGreaterThanEqual:
                case SpvOpFUnordEqual: case SpvOpFUnordNotEqual: case SpvOpFUnordLessThan: case SpvOpFUnordGreaterThan:
                case SpvOpFUnordLessThanEqual: case SpvOpFUnordGreaterThanEqual:
                case SpvOpIEqual: case SpvOpINotEqual: case SpvOpSLessThan: case SpvOpSGreaterThan:
                case SpvOpSLessThanEqual: case SpvOpSGreaterThanEqual: case SpvOpULessThan: case SpvOpUGreaterThan:
                case SpvOpULessThanEqual: case SpvOpUGreaterThanEqual:
                case SpvOpLogicalAnd: case SpvOpLogicalOr: case SpvOpLogicalNot: case SpvOpLogicalEqual: case SpvOpLogicalNotEqual:
                case SpvOpBitwiseAnd: case SpvOpBitwiseOr: case SpvOpBitwiseXor: case SpvOpNot:
                case SpvOpShiftLeftLogical: case SpvOpShiftRightLogical: case SpvOpShiftRightArithmetic:
                case SpvOpCompositeConstruct: case SpvOpCompositeExtract: case SpvOpCompositeInsert: case SpvOpVectorShuffle:
                case SpvOpConvertFToS: case SpvOpConvertFToU: case SpvOpConvertSToF: case SpvOpConvertUToF:
                case SpvOpBitcast: case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert:
                case SpvOpSelect: case SpvOpPhi: case SpvOpDot: case SpvOpVectorTimesScalar: case SpvOpMatrixTimesScalar:
                case SpvOpVectorTimesMatrix: case SpvOpMatrixTimesVector: case SpvOpMatrixTimesMatrix:
                case SpvOpExtInst: case SpvOpImageSampleImplicitLod: case SpvOpImageSampleExplicitLod:
                case SpvOpImageSampleDrefImplicitLod: case SpvOpImageSampleDrefExplicitLod:
                case SpvOpImageGather: case SpvOpImageDrefGather:
                case SpvOpImageRead: case SpvOpImageQuerySizeLod: case SpvOpImageQuerySize:
                case SpvOpImageQueryLod: case SpvOpImageQueryLevels: case SpvOpImageQuerySamples:
                case SpvOpSampledImage: case SpvOpFunctionCall: case SpvOpTranspose:
                case SpvOpCopyObject: case SpvOpVectorExtractDynamic: case SpvOpVectorInsertDynamic:
                case SpvOpArrayLength: case SpvOpIsInf: case SpvOpIsNan:
                    has_result_type = 1;
                    has_result = 1;
                    break;
                default:
                    break;
                }
                if (has_result_type && has_result && operand_count >= 2) {
                    uint32_t type_id = operands[0];
                    uint32_t result_id = operands[1];
                    if (result_id < c->id_bound) {
                        c->ids[result_id].kind = VTS_SPV_ID_INSTRUCTION;
                        c->ids[result_id].instruction.type_id = type_id;
                        c->ids[result_id].instruction.opcode = op;
                        int remaining = operand_count - 2;
                        c->ids[result_id].instruction.operands = NULL;
                        if (remaining > 0) {
                            c->ids[result_id].instruction.operands = (uint32_t*)SPIRV_TO_SSIR_MALLOC(remaining * sizeof(uint32_t));
                            memcpy(c->ids[result_id].instruction.operands, &operands[2], remaining * sizeof(uint32_t));
                        }
                        c->ids[result_id].instruction.operand_count = remaining;
                    }
                }
            }
            break;
        }

        pos += wc;
    }

    for (int i = 0; i < c->pending_ep_count; i++) {
        VtsPendingEntryPoint *ep = &c->pending_eps[i];
        for (int j = 0; j < c->function_count; j++) {
            if (c->functions[j].id == ep->func_id) {
                c->functions[j].exec_model = ep->exec_model;
                c->functions[j].is_entry_point = 1;
                if (ep->name && !c->functions[j].name) {
                    c->functions[j].name = strdup(ep->name);
                }
                c->functions[j].interface_vars = ep->interface_vars;
                c->functions[j].interface_var_count = ep->interface_var_count;
                ep->interface_vars = NULL;
                break;
            }
        }
    }

    for (int i = 0; i < c->pending_wg_count; i++) {
        VtsPendingWorkgroupSize *wg = &c->pending_wgs[i];
        for (int j = 0; j < c->function_count; j++) {
            if (c->functions[j].id == wg->func_id) {
                c->functions[j].workgroup_size[0] = wg->workgroup_size[0];
                c->functions[j].workgroup_size[1] = wg->workgroup_size[1];
                c->functions[j].workgroup_size[2] = wg->workgroup_size[2];
                break;
            }
        }
    }

    /* Apply pending execution modes to functions */
    for (int i = 0; i < c->pending_exec_mode_count; i++) {
        VtsPendingExecMode *em = &c->pending_exec_modes[i];
        for (int j = 0; j < c->function_count; j++) {
            if (c->functions[j].id == em->func_id) {
                if (em->mode == SpvExecutionModeDepthReplacing)
                    c->functions[j].depth_replacing = true;
                else if (em->mode == SpvExecutionModeOriginUpperLeft)
                    c->functions[j].origin_upper_left = true;
                else if (em->mode == SpvExecutionModeEarlyFragmentTests)
                    c->functions[j].early_fragment_tests = true;
                break;
            }
        }
    }

    return SPIRV_TO_SSIR_SUCCESS;
}

static uint32_t convert_type(VtsConverter *c, uint32_t spv_type_id);

//c nonnull
static uint32_t convert_scalar_type(VtsConverter *c, uint32_t spv_type_id) {
    wgsl_compiler_assert(c != NULL, "convert_scalar_type: c is NULL");
    if (spv_type_id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[spv_type_id];
    if (info->kind != VTS_SPV_ID_TYPE) return 0;

    switch (info->type_info.kind) {
    case VTS_SPV_TYPE_VOID:
        return ssir_type_void(c->mod);
    case VTS_SPV_TYPE_BOOL:
        return ssir_type_bool(c->mod);
    case VTS_SPV_TYPE_INT:
        switch (info->type_info.int_type.width) {
        case 8:  return info->type_info.int_type.signedness ? ssir_type_i8(c->mod) : ssir_type_u8(c->mod);
        case 16: return info->type_info.int_type.signedness ? ssir_type_i16(c->mod) : ssir_type_u16(c->mod);
        case 32: return info->type_info.int_type.signedness ? ssir_type_i32(c->mod) : ssir_type_u32(c->mod);
        case 64: return info->type_info.int_type.signedness ? ssir_type_i64(c->mod) : ssir_type_u64(c->mod);
        default: return 0;
        }
    case VTS_SPV_TYPE_FLOAT:
        switch (info->type_info.float_type.width) {
        case 16: return ssir_type_f16(c->mod);
        case 32: return ssir_type_f32(c->mod);
        case 64: return ssir_type_f64(c->mod);
        default: return 0;
        }
    default:
        return 0;
    }
}

static SsirAddressSpace storage_class_to_addr_space(SpvStorageClass sc) {
    switch (sc) {
    case SpvStorageClassFunction: return SSIR_ADDR_FUNCTION;
    case SpvStorageClassPrivate: return SSIR_ADDR_PRIVATE;
    case SpvStorageClassWorkgroup: return SSIR_ADDR_WORKGROUP;
    case SpvStorageClassUniform: return SSIR_ADDR_UNIFORM;
    case SpvStorageClassUniformConstant: return SSIR_ADDR_UNIFORM_CONSTANT;
    case SpvStorageClassStorageBuffer: return SSIR_ADDR_STORAGE;
    case SpvStorageClassInput: return SSIR_ADDR_INPUT;
    case SpvStorageClassOutput: return SSIR_ADDR_OUTPUT;
    case SpvStorageClassPushConstant: return SSIR_ADDR_PUSH_CONSTANT;
    case SpvStorageClassPhysicalStorageBuffer: return SSIR_ADDR_PHYSICAL_STORAGE_BUFFER;
    default: return SSIR_ADDR_FUNCTION;
    }
}

static SsirTextureDim spv_dim_to_ssir(SpvDim dim, uint32_t arrayed, uint32_t ms) {
    if (ms) return arrayed ? SSIR_TEX_MULTISAMPLED_2D_ARRAY : SSIR_TEX_MULTISAMPLED_2D;
    switch (dim) {
    case SpvDim1D: return arrayed ? SSIR_TEX_1D_ARRAY : SSIR_TEX_1D;
    case SpvDim2D: return arrayed ? SSIR_TEX_2D_ARRAY : SSIR_TEX_2D;
    case SpvDim3D: return SSIR_TEX_3D;
    case SpvDimCube: return arrayed ? SSIR_TEX_CUBE_ARRAY : SSIR_TEX_CUBE;
    case SpvDimBuffer: return SSIR_TEX_BUFFER;
    default: return SSIR_TEX_2D;
    }
}

//c nonnull
static uint32_t convert_type(VtsConverter *c, uint32_t spv_type_id) {
    wgsl_compiler_assert(c != NULL, "convert_type: c is NULL");
    if (spv_type_id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[spv_type_id];
    if (info->kind != VTS_SPV_ID_TYPE) return 0;
    if (info->ssir_id) return info->ssir_id;
    if (c->type_depth > 64) return 0;
    c->type_depth++;

    uint32_t result = 0;

    switch (info->type_info.kind) {
    case VTS_SPV_TYPE_VOID:
    case VTS_SPV_TYPE_BOOL:
    case VTS_SPV_TYPE_INT:
    case VTS_SPV_TYPE_FLOAT:
        result = convert_scalar_type(c, spv_type_id);
        break;

    case VTS_SPV_TYPE_VECTOR: {
        uint32_t elem = convert_type(c, info->type_info.vector.component_type);
        result = ssir_type_vec(c->mod, elem, (uint8_t)info->type_info.vector.count);
        break;
    }

    case VTS_SPV_TYPE_MATRIX: {
        uint32_t col = convert_type(c, info->type_info.matrix.column_type);
        uint8_t rows = 0;
        if (info->type_info.matrix.column_type < c->id_bound) {
            VtsSpvIdInfo *col_info = &c->ids[info->type_info.matrix.column_type];
            /* PRE: column_type should be a vector type */
            wgsl_compiler_assert(col_info->type_info.kind == VTS_SPV_TYPE_VECTOR, "VTS_SPV_TYPE_MATRIX: column_type is not a vector");
            rows = (uint8_t)col_info->type_info.vector.count;
        }
        result = ssir_type_mat(c->mod, col, (uint8_t)info->type_info.matrix.columns, rows);
        break;
    }

    case VTS_SPV_TYPE_ARRAY: {
        uint32_t elem = convert_type(c, info->type_info.array.element_type);
        uint32_t len_id = info->type_info.array.length_id;
        uint32_t len = 0;
        if (len_id < c->id_bound && c->ids[len_id].kind == VTS_SPV_ID_CONSTANT) {
            if (c->ids[len_id].constant.value_count > 0) {
                /* PRE: values array valid if value_count > 0 */
                wgsl_compiler_assert(c->ids[len_id].constant.values != NULL, "VTS_SPV_TYPE_ARRAY: values NULL with count > 0");
                len = c->ids[len_id].constant.values[0];
            }
        }
        uint32_t stride = 0;
        vts_has_decoration(c, spv_type_id, SpvDecorationArrayStride, &stride);
        if (stride)
            result = ssir_type_array_stride(c->mod, elem, len, stride);
        else
            result = ssir_type_array(c->mod, elem, len);
        break;
    }

    case VTS_SPV_TYPE_RUNTIME_ARRAY: {
        uint32_t elem = convert_type(c, info->type_info.runtime_array.element_type);
        result = ssir_type_runtime_array(c->mod, elem);
        break;
    }

    case VTS_SPV_TYPE_STRUCT: {
        int mc = info->type_info.struct_type.member_count;
        uint32_t *members = NULL;
        uint32_t *offsets = NULL;
        const char **mnames = NULL;
        if (mc > 0) {
            members = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
            /* PRE: malloc succeeded */
            wgsl_compiler_assert(members != NULL, "VTS_SPV_TYPE_STRUCT: members malloc failed");
            offsets = (uint32_t*)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
            /* PRE: malloc succeeded */
            wgsl_compiler_assert(offsets != NULL, "VTS_SPV_TYPE_STRUCT: offsets malloc failed");
            /* PRE: member_types array valid if member_count > 0 */
            wgsl_compiler_assert(info->type_info.struct_type.member_types != NULL, "VTS_SPV_TYPE_STRUCT: member_types NULL with count > 0");
            for (int i = 0; i < mc; i++) {
                members[i] = convert_type(c, info->type_info.struct_type.member_types[i]);
                if (!get_member_offset(c, spv_type_id, i, &offsets[i])) {
                    offsets[i] = 0;
                }
            }
            /* Collect member names from OpMemberName */
            if (info->member_name_count > 0) {
                mnames = (const char **)SPIRV_TO_SSIR_MALLOC(mc * sizeof(const char *));
                /* PRE: malloc succeeded */
                wgsl_compiler_assert(mnames != NULL, "VTS_SPV_TYPE_STRUCT: mnames malloc failed");
                for (int i = 0; i < mc; i++) {
                    mnames[i] = (i < info->member_name_count) ? info->member_names[i] : NULL;
                }
            }
        }
        const char *name = info->name;
        result = ssir_type_struct_named(c->mod, name, members, mc, offsets, mnames);
        /* Parse matrix layout member decorations */
        if (mc > 0 && result) {
            SsirType *st = ssir_get_type(c->mod, result);
            if (st) {
                uint8_t *majors = NULL;
                uint32_t *mstrides = NULL;
                /* PRE: member_decorations array valid if count > 0 */
                wgsl_compiler_assert(info->member_decoration_count == 0 || info->member_decorations != NULL, "VTS_SPV_TYPE_STRUCT: member_decorations NULL with count > 0");
                for (int i = 0; i < info->member_decoration_count; i++) {
                    SpvDecoration dec = (SpvDecoration)info->member_decorations[i].decoration;
                    uint32_t mi = info->member_decorations[i].member_index;
                    if (mi >= (uint32_t)mc) continue;
                    if (dec == SpvDecorationColMajor || dec == SpvDecorationRowMajor) {
                        if (!majors) {
                            majors = (uint8_t *)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint8_t));
                            memset(majors, 0, mc * sizeof(uint8_t));
                        }
                        majors[mi] = (dec == SpvDecorationColMajor) ? 1 : 2;
                    } else if (dec == SpvDecorationMatrixStride && info->member_decorations[i].literal_count > 0) {
                        /* PRE: literals array valid if literal_count > 0 */
                        wgsl_compiler_assert(info->member_decorations[i].literals != NULL, "VTS_SPV_TYPE_STRUCT: literals NULL with count > 0");
                        if (!mstrides) {
                            mstrides = (uint32_t *)SPIRV_TO_SSIR_MALLOC(mc * sizeof(uint32_t));
                            memset(mstrides, 0, mc * sizeof(uint32_t));
                        }
                        mstrides[mi] = info->member_decorations[i].literals[0];
                    }
                }
                st->struc.matrix_major = majors;
                st->struc.matrix_strides = mstrides;
            }
        }
        SPIRV_TO_SSIR_FREE(members);
        SPIRV_TO_SSIR_FREE(offsets);
        SPIRV_TO_SSIR_FREE(mnames);
        break;
    }

    case VTS_SPV_TYPE_POINTER: {
        uint32_t pointee = convert_type(c, info->type_info.pointer.pointee_type);
        SsirAddressSpace space = storage_class_to_addr_space(info->type_info.pointer.storage);
        result = ssir_type_ptr(c->mod, pointee, space);
        break;
    }

    case VTS_SPV_TYPE_SAMPLER:
        result = ssir_type_sampler(c->mod);
        break;

    case VTS_SPV_TYPE_IMAGE: {
        SsirTextureDim dim = spv_dim_to_ssir(info->type_info.image.dim,
                                              info->type_info.image.arrayed,
                                              info->type_info.image.ms);
        if (info->type_info.image.depth == 1) {
            result = ssir_type_texture_depth(c->mod, dim);
        } else if (info->type_info.image.sampled == 2) {
            result = ssir_type_texture_storage(c->mod, dim, info->type_info.image.format, SSIR_ACCESS_READ_WRITE);
        } else {
            uint32_t sampled = convert_type(c, info->type_info.image.sampled_type);
            result = ssir_type_texture(c->mod, dim, sampled);
        }
        break;
    }

    case VTS_SPV_TYPE_SAMPLED_IMAGE: {
        result = convert_type(c, info->type_info.sampled_image.image_type);
        break;
    }

    case VTS_SPV_TYPE_FUNCTION:
        c->type_depth--;
        return 0;

    default:
        c->type_depth--;
        return 0;
    }

    info->ssir_id = result;
    c->type_depth--;
    return result;
}

//c nonnull
static uint32_t convert_constant(VtsConverter *c, uint32_t spv_const_id) {
    wgsl_compiler_assert(c != NULL, "convert_constant: c is NULL");
    if (spv_const_id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[spv_const_id];
    if (info->kind != VTS_SPV_ID_CONSTANT) return 0;
    if (info->ssir_id) return info->ssir_id;
    if (c->type_depth > 64) return 0;
    c->type_depth++;

    uint32_t type_id = convert_type(c, info->constant.type_id);
    SsirType *type = ssir_get_type(c->mod, type_id);
    if (!type) return 0;

    uint32_t result = 0;

    if (info->constant.is_composite) {
        int count = info->constant.value_count;
        uint32_t *components = NULL;
        if (count > 0) {
            /* PRE: values array valid if value_count > 0 */
            wgsl_compiler_assert(info->constant.values != NULL, "convert_constant: values NULL with count > 0");
            components = (uint32_t*)SPIRV_TO_SSIR_MALLOC(count * sizeof(uint32_t));
            for (int i = 0; i < count; i++) {
                components[i] = convert_constant(c, info->constant.values[i]);
            }
        }
        result = ssir_const_composite(c->mod, type_id, components, count);
        SPIRV_TO_SSIR_FREE(components);
    } else if (info->constant.is_spec) {
        /* Specialization constant */
        uint32_t sid = 0;
        vts_has_decoration(c, spv_const_id, SpvDecorationSpecId, &sid);
        /* PRE: values array valid if value_count > 0 */
        wgsl_compiler_assert(info->constant.value_count == 0 || info->constant.values != NULL, "convert_constant spec: values NULL with count > 0");
        switch (type->kind) {
        case SSIR_TYPE_BOOL:
            result = ssir_const_spec_bool(c->mod, info->constant.value_count > 0 && info->constant.values[0] != 0, sid);
            break;
        case SSIR_TYPE_I32:
            result = ssir_const_spec_i32(c->mod, info->constant.value_count > 0 ? (int32_t)info->constant.values[0] : 0, sid);
            break;
        case SSIR_TYPE_U32:
            result = ssir_const_spec_u32(c->mod, info->constant.value_count > 0 ? info->constant.values[0] : 0, sid);
            break;
        case SSIR_TYPE_F32: {
            float fval = 0.0f;
            if (info->constant.value_count > 0) memcpy(&fval, &info->constant.values[0], sizeof(float));
            result = ssir_const_spec_f32(c->mod, fval, sid);
            break;
        }
        default:
            c->type_depth--;
            return 0;
        }
    } else {
        /* PRE: values array valid if value_count > 0 */
        wgsl_compiler_assert(info->constant.value_count == 0 || info->constant.values != NULL, "convert_constant: values NULL with count > 0");
        switch (type->kind) {
        case SSIR_TYPE_BOOL:
            result = ssir_const_bool(c->mod, info->constant.value_count > 0 && info->constant.values[0] != 0);
            break;
        case SSIR_TYPE_I32:
            result = ssir_const_i32(c->mod, info->constant.value_count > 0 ? (int32_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_U32:
            result = ssir_const_u32(c->mod, info->constant.value_count > 0 ? info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_F32: {
            float fval = 0.0f;
            if (info->constant.value_count > 0) {
                memcpy(&fval, &info->constant.values[0], sizeof(float));
            }
            result = ssir_const_f32(c->mod, fval);
            break;
        }
        case SSIR_TYPE_F16:
            result = ssir_const_f16(c->mod, info->constant.value_count > 0 ? (uint16_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_F64: {
            double dval = 0.0;
            if (info->constant.value_count >= 2) {
                uint32_t dw[2] = { info->constant.values[0], info->constant.values[1] };
                memcpy(&dval, dw, sizeof(double));
            }
            result = ssir_const_f64(c->mod, dval);
            break;
        }
        case SSIR_TYPE_I8:
            result = ssir_const_i8(c->mod, info->constant.value_count > 0 ? (int8_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_U8:
            result = ssir_const_u8(c->mod, info->constant.value_count > 0 ? (uint8_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_I16:
            result = ssir_const_i16(c->mod, info->constant.value_count > 0 ? (int16_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_U16:
            result = ssir_const_u16(c->mod, info->constant.value_count > 0 ? (uint16_t)info->constant.values[0] : 0);
            break;
        case SSIR_TYPE_I64: {
            int64_t ival = 0;
            if (info->constant.value_count >= 2) {
                uint32_t qw[2] = { info->constant.values[0], info->constant.values[1] };
                memcpy(&ival, qw, sizeof(int64_t));
            }
            result = ssir_const_i64(c->mod, ival);
            break;
        }
        case SSIR_TYPE_U64: {
            uint64_t uval = 0;
            if (info->constant.value_count >= 2) {
                uint32_t qw[2] = { info->constant.values[0], info->constant.values[1] };
                memcpy(&uval, qw, sizeof(uint64_t));
            }
            result = ssir_const_u64(c->mod, uval);
            break;
        }
        default:
            c->type_depth--;
            return 0;
        }
    }

    /* Copy name to SSIR constant if available */
    if (result && info->name) {
        SsirConstant *sc = ssir_get_constant(c->mod, result);
        if (sc) sc->name = info->name;
    }

    info->ssir_id = result;
    c->type_depth--;
    return result;
}

static SsirBuiltinVar spv_builtin_to_ssir(SpvBuiltIn builtin) {
    switch (builtin) {
    case SpvBuiltInVertexIndex: return SSIR_BUILTIN_VERTEX_INDEX;
    case SpvBuiltInInstanceIndex: return SSIR_BUILTIN_INSTANCE_INDEX;
    case SpvBuiltInPosition: return SSIR_BUILTIN_POSITION;
    case SpvBuiltInFrontFacing: return SSIR_BUILTIN_FRONT_FACING;
    case SpvBuiltInFragDepth: return SSIR_BUILTIN_FRAG_DEPTH;
    case SpvBuiltInSampleId: return SSIR_BUILTIN_SAMPLE_INDEX;
    case SpvBuiltInSampleMask: return SSIR_BUILTIN_SAMPLE_MASK;
    case SpvBuiltInLocalInvocationId: return SSIR_BUILTIN_LOCAL_INVOCATION_ID;
    case SpvBuiltInLocalInvocationIndex: return SSIR_BUILTIN_LOCAL_INVOCATION_INDEX;
    case SpvBuiltInGlobalInvocationId: return SSIR_BUILTIN_GLOBAL_INVOCATION_ID;
    case SpvBuiltInWorkgroupId: return SSIR_BUILTIN_WORKGROUP_ID;
    case SpvBuiltInNumWorkgroups: return SSIR_BUILTIN_NUM_WORKGROUPS;
    case SpvBuiltInPointSize: return SSIR_BUILTIN_POINT_SIZE;
    case SpvBuiltInClipDistance: return SSIR_BUILTIN_CLIP_DISTANCE;
    case SpvBuiltInCullDistance: return SSIR_BUILTIN_CULL_DISTANCE;
    case SpvBuiltInLayer: return SSIR_BUILTIN_LAYER;
    case SpvBuiltInViewportIndex: return SSIR_BUILTIN_VIEWPORT_INDEX;
    case SpvBuiltInFragCoord: return SSIR_BUILTIN_FRAG_COORD;
    case SpvBuiltInHelperInvocation: return SSIR_BUILTIN_HELPER_INVOCATION;
    case SpvBuiltInPrimitiveId: return SSIR_BUILTIN_PRIMITIVE_ID;
    case SpvBuiltInBaseVertex: return SSIR_BUILTIN_BASE_VERTEX;
    case SpvBuiltInBaseInstance: return SSIR_BUILTIN_BASE_INSTANCE;
    case SpvBuiltInSubgroupSize: return SSIR_BUILTIN_SUBGROUP_SIZE;
    case SpvBuiltInSubgroupLocalInvocationId: return SSIR_BUILTIN_SUBGROUP_INVOCATION_ID;
    case SpvBuiltInSubgroupId: return SSIR_BUILTIN_SUBGROUP_ID;
    case SpvBuiltInNumSubgroups: return SSIR_BUILTIN_NUM_SUBGROUPS;
    default: return SSIR_BUILTIN_NONE;
    }
}

//c nonnull
static void convert_global_vars(VtsConverter *c) {
    wgsl_compiler_assert(c != NULL, "convert_global_vars: c is NULL");
    for (uint32_t i = 1; i < c->id_bound; i++) {
        VtsSpvIdInfo *info = &c->ids[i];
        if (info->kind != VTS_SPV_ID_VARIABLE) continue;
        if (info->variable.storage_class == SpvStorageClassFunction) continue;

        uint32_t ptr_type = convert_type(c, info->variable.type_id);
        if (!ptr_type) continue;

        const char *name = info->name;
        uint32_t gid = ssir_global_var(c->mod, name, ptr_type);
        info->ssir_id = gid;

        uint32_t group_val, binding_val, location_val, builtin_val;
        if (vts_has_decoration(c, i, SpvDecorationDescriptorSet, &group_val)) {
            ssir_global_set_group(c->mod, gid, group_val);
        }
        if (vts_has_decoration(c, i, SpvDecorationBinding, &binding_val)) {
            ssir_global_set_binding(c->mod, gid, binding_val);
        }
        if (vts_has_decoration(c, i, SpvDecorationLocation, &location_val)) {
            ssir_global_set_location(c->mod, gid, location_val);
        }
        if (vts_has_decoration(c, i, SpvDecorationBuiltIn, &builtin_val)) {
            SsirBuiltinVar bv = spv_builtin_to_ssir((SpvBuiltIn)builtin_val);
            ssir_global_set_builtin(c->mod, gid, bv);
        }
        if (vts_has_decoration(c, i, SpvDecorationFlat, NULL)) {
            ssir_global_set_interpolation(c->mod, gid, SSIR_INTERP_FLAT);
        } else if (vts_has_decoration(c, i, SpvDecorationNoPerspective, NULL)) {
            ssir_global_set_interpolation(c->mod, gid, SSIR_INTERP_LINEAR);
        }
        if (vts_has_decoration(c, i, SpvDecorationCentroid, NULL)) {
            ssir_global_set_interp_sampling(c->mod, gid, SSIR_INTERP_SAMPLING_CENTROID);
        } else if (vts_has_decoration(c, i, SpvDecorationSample, NULL)) {
            ssir_global_set_interp_sampling(c->mod, gid, SSIR_INTERP_SAMPLING_SAMPLE);
        }
        if (vts_has_decoration(c, i, SpvDecorationNonWritable, NULL)) {
            ssir_global_set_non_writable(c->mod, gid, true);
        }
        if (vts_has_decoration(c, i, SpvDecorationInvariant, NULL)) {
            ssir_global_set_invariant(c->mod, gid, true);
        }

        if (info->variable.initializer) {
            uint32_t init_id = convert_constant(c, info->variable.initializer);
            if (init_id) {
                ssir_global_set_initializer(c->mod, gid, init_id);
            }
        }
    }
}

//c nonnull
static void set_ssir_id(VtsConverter *c, uint32_t spv_id, uint32_t ssir_id) {
    wgsl_compiler_assert(c != NULL, "set_ssir_id: c is NULL");
    if (spv_id < c->id_bound) c->ids[spv_id].ssir_id = ssir_id;
}

//c nonnull
static uint32_t get_ssir_id(VtsConverter *c, uint32_t spv_id) {
    wgsl_compiler_assert(c != NULL, "get_ssir_id: c is NULL");
    if (spv_id >= c->id_bound) return 0;
    VtsSpvIdInfo *info = &c->ids[spv_id];
    if (info->ssir_id) return info->ssir_id;

    switch (info->kind) {
    case VTS_SPV_ID_TYPE:
        return convert_type(c, spv_id);
    case VTS_SPV_ID_CONSTANT:
        return convert_constant(c, spv_id);
    case VTS_SPV_ID_VARIABLE:
    case VTS_SPV_ID_INSTRUCTION:
    case VTS_SPV_ID_PARAM:
        return info->ssir_id;
    default:
        return 0;
    }
}

static SsirBuiltinId glsl_ext_to_ssir_builtin(enum GLSLstd450 glsl_op) {
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
    case GLSLstd450FAbs:
    case GLSLstd450SAbs: return SSIR_BUILTIN_ABS;
    case GLSLstd450FSign:
    case GLSLstd450SSign: return SSIR_BUILTIN_SIGN;
    case GLSLstd450Floor: return SSIR_BUILTIN_FLOOR;
    case GLSLstd450Ceil: return SSIR_BUILTIN_CEIL;
    case GLSLstd450Round: return SSIR_BUILTIN_ROUND;
    case GLSLstd450Trunc: return SSIR_BUILTIN_TRUNC;
    case GLSLstd450Fract: return SSIR_BUILTIN_FRACT;
    case GLSLstd450FMin:
    case GLSLstd450SMin:
    case GLSLstd450UMin: return SSIR_BUILTIN_MIN;
    case GLSLstd450FMax:
    case GLSLstd450SMax:
    case GLSLstd450UMax: return SSIR_BUILTIN_MAX;
    case GLSLstd450FClamp:
    case GLSLstd450SClamp:
    case GLSLstd450UClamp: return SSIR_BUILTIN_CLAMP;
    case GLSLstd450FMix: return SSIR_BUILTIN_MIX;
    case GLSLstd450Step: return SSIR_BUILTIN_STEP;
    case GLSLstd450SmoothStep: return SSIR_BUILTIN_SMOOTHSTEP;
    case GLSLstd450Length: return SSIR_BUILTIN_LENGTH;
    case GLSLstd450Distance: return SSIR_BUILTIN_DISTANCE;
    case GLSLstd450Cross: return SSIR_BUILTIN_CROSS;
    case GLSLstd450Normalize: return SSIR_BUILTIN_NORMALIZE;
    case GLSLstd450FaceForward: return SSIR_BUILTIN_FACEFORWARD;
    case GLSLstd450Reflect: return SSIR_BUILTIN_REFLECT;
    case GLSLstd450Refract: return SSIR_BUILTIN_REFRACT;
    case GLSLstd450Fma: return SSIR_BUILTIN_FMA;
    case GLSLstd450Degrees: return SSIR_BUILTIN_DEGREES;
    case GLSLstd450Radians: return SSIR_BUILTIN_RADIANS;
    case GLSLstd450Modf:
    case GLSLstd450ModfStruct: return SSIR_BUILTIN_MODF;
    case GLSLstd450Frexp:
    case GLSLstd450FrexpStruct: return SSIR_BUILTIN_FREXP;
    case GLSLstd450Ldexp: return SSIR_BUILTIN_LDEXP;
    case GLSLstd450Determinant: return SSIR_BUILTIN_DETERMINANT;
    case GLSLstd450PackSnorm4x8: return SSIR_BUILTIN_PACK4X8SNORM;
    case GLSLstd450PackUnorm4x8: return SSIR_BUILTIN_PACK4X8UNORM;
    case GLSLstd450PackSnorm2x16: return SSIR_BUILTIN_PACK2X16SNORM;
    case GLSLstd450PackUnorm2x16: return SSIR_BUILTIN_PACK2X16UNORM;
    case GLSLstd450PackHalf2x16: return SSIR_BUILTIN_PACK2X16FLOAT;
    case GLSLstd450UnpackSnorm4x8: return SSIR_BUILTIN_UNPACK4X8SNORM;
    case GLSLstd450UnpackUnorm4x8: return SSIR_BUILTIN_UNPACK4X8UNORM;
    case GLSLstd450UnpackSnorm2x16: return SSIR_BUILTIN_UNPACK2X16SNORM;
    case GLSLstd450UnpackUnorm2x16: return SSIR_BUILTIN_UNPACK2X16UNORM;
    case GLSLstd450UnpackHalf2x16: return SSIR_BUILTIN_UNPACK2X16FLOAT;
    default: return SSIR_BUILTIN_COUNT;
    }
}

//c nonnull
//fn nonnull
static void convert_function(VtsConverter *c, VtsSpvFunction *fn) {
    wgsl_compiler_assert(c != NULL, "convert_function: c is NULL");
    wgsl_compiler_assert(fn != NULL, "convert_function: fn is NULL");
    if (fn->id >= c->id_bound) return;
    uint32_t ret_type = convert_type(c, fn->return_type);
    const char *name = fn->name ? fn->name : "fn";
    uint32_t func_id = ssir_function_create(c->mod, name, ret_type);
    c->ids[fn->id].ssir_id = func_id;

    for (int i = 0; i < fn->param_count; i++) {
        uint32_t param_spv = fn->params[i];
        if (param_spv >= c->id_bound) continue;
        VtsSpvIdInfo *param_info = &c->ids[param_spv];
        uint32_t param_type = convert_type(c, param_info->param.type_id);
        const char *param_name = param_info->name;
        uint32_t param_id = ssir_function_add_param(c->mod, func_id, param_name, param_type);
        param_info->ssir_id = param_id;
    }

    for (int i = 0; i < fn->local_var_count; i++) {
        uint32_t var_spv = fn->local_vars[i];
        if (var_spv >= c->id_bound) continue;
        VtsSpvIdInfo *var_info = &c->ids[var_spv];
        uint32_t var_type = convert_type(c, var_info->variable.type_id);
        const char *var_name = var_info->name;
        uint32_t local_id = ssir_function_add_local(c->mod, func_id, var_name, var_type);
        var_info->ssir_id = local_id;
    }

    for (int bi = 0; bi < fn->block_count; bi++) {
        VtsSpvBasicBlock *blk = &fn->blocks[bi];
        const char *blk_name = NULL;
        if (blk->label_id < c->id_bound) {
            blk_name = c->ids[blk->label_id].name;
        }
        uint32_t block_id = ssir_block_create(c->mod, func_id, blk_name);
        if (blk->label_id < c->id_bound) {
            c->ids[blk->label_id].ssir_id = block_id;
        }
    }

    for (int bi = 0; bi < fn->block_count; bi++) {
        VtsSpvBasicBlock *blk = &fn->blocks[bi];
        if (blk->label_id >= c->id_bound) continue;
        uint32_t block_id = c->ids[blk->label_id].ssir_id;

        if (blk->is_loop_header) {
            uint32_t merge_id = 0, cont_id = 0;
            if (blk->merge_block < c->id_bound) merge_id = c->ids[blk->merge_block].ssir_id;
            if (blk->continue_block < c->id_bound) cont_id = c->ids[blk->continue_block].ssir_id;
            if (merge_id && cont_id) {
                ssir_build_loop_merge(c->mod, func_id, block_id, merge_id, cont_id);
            }
        } else if (blk->is_selection_header) {
            uint32_t merge_id = 0;
            if (blk->merge_block < c->id_bound) merge_id = c->ids[blk->merge_block].ssir_id;
            if (merge_id) {
                ssir_build_selection_merge(c->mod, func_id, block_id, merge_id);
            }
        }

        for (int ii = 0; ii < blk->instruction_count; ii++) {
            uint32_t inst_pos = blk->instructions[ii];
            /* PRE: inst_pos within spirv bounds */
            wgsl_compiler_assert(inst_pos < c->word_count, "convert_function: inst_pos %u >= word_count %zu", inst_pos, c->word_count);
            uint32_t word0 = c->spirv[inst_pos];
            uint16_t opcode = word0 & 0xFFFF;
            uint16_t wc = word0 >> 16;
            const uint32_t *operands = &c->spirv[inst_pos + 1];
            int operand_count = wc - 1;

            SpvOp op = (SpvOp)opcode;

            switch (op) {
            case SpvOpStore:
                if (operand_count >= 2) {
                    uint32_t ptr = get_ssir_id(c, operands[0]);
                    uint32_t val = get_ssir_id(c, operands[1]);
                    ssir_build_store(c->mod, func_id, block_id, ptr, val);
                }
                break;

            case SpvOpLoad:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t ptr = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_load(c->mod, func_id, block_id, type_id, ptr);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t base = get_ssir_id(c, operands[2]);
                    int idx_count = operand_count - 3;
                    uint32_t *indices = NULL;
                    if (idx_count > 0) {
                        indices = (uint32_t*)SPIRV_TO_SSIR_MALLOC(idx_count * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(indices != NULL, "SpvOpAccessChain: indices malloc failed");
                        for (int j = 0; j < idx_count; j++) {
                            indices[j] = get_ssir_id(c, operands[3 + j]);
                        }
                    }
                    uint32_t r = ssir_build_access(c->mod, func_id, block_id, type_id, base, indices, idx_count);
                    set_ssir_id(c, result_id, r);
                    SPIRV_TO_SSIR_FREE(indices);
                }
                break;

            case SpvOpFAdd: case SpvOpIAdd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_add(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFSub: case SpvOpISub:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_sub(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFMul: case SpvOpIMul:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mul(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFDiv: case SpvOpSDiv: case SpvOpUDiv:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_div(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFMod: case SpvOpSMod: case SpvOpUMod:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mod(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFRem: case SpvOpSRem:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_rem(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFNegate: case SpvOpSNegate:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_neg(c->mod, func_id, block_id, type_id, a);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpMatrixTimesMatrix:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mat_mul(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpMatrixTimesVector:
            case SpvOpVectorTimesMatrix:
            case SpvOpVectorTimesScalar:
            case SpvOpMatrixTimesScalar:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_mul(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpTranspose:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t m = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_mat_transpose(c->mod, func_id, block_id, type_id, m);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpIsInf:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t args[] = {a};
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_ISINF, args, 1);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpIsNan:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t args[] = {a};
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_ISNAN, args, 1);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpBitwiseAnd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_and(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpBitwiseOr:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_or(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpBitwiseXor:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_bit_xor(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpNot:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_bit_not(c->mod, func_id, block_id, type_id, a);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpShiftLeftLogical:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shl(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpShiftRightLogical:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shr_logical(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpShiftRightArithmetic:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_shr(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdEqual: case SpvOpFUnordEqual: case SpvOpIEqual: case SpvOpLogicalEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_eq(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdNotEqual: case SpvOpFUnordNotEqual: case SpvOpINotEqual: case SpvOpLogicalNotEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_ne(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdLessThan: case SpvOpFUnordLessThan: case SpvOpSLessThan: case SpvOpULessThan:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_lt(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdLessThanEqual: case SpvOpFUnordLessThanEqual: case SpvOpSLessThanEqual: case SpvOpULessThanEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_le(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdGreaterThan: case SpvOpFUnordGreaterThan: case SpvOpSGreaterThan: case SpvOpUGreaterThan:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_gt(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpFOrdGreaterThanEqual: case SpvOpFUnordGreaterThanEqual: case SpvOpSGreaterThanEqual: case SpvOpUGreaterThanEqual:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_ge(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpLogicalAnd:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_and(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpLogicalOr:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_or(c->mod, func_id, block_id, type_id, a, b);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpLogicalNot:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_not(c->mod, func_id, block_id, type_id, a);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpCompositeConstruct:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    int comp_count = operand_count - 2;
                    uint32_t *comps = NULL;
                    if (comp_count > 0) {
                        comps = (uint32_t*)SPIRV_TO_SSIR_MALLOC(comp_count * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(comps != NULL, "SpvOpCompositeConstruct: comps malloc failed");
                        for (int j = 0; j < comp_count; j++) {
                            comps[j] = get_ssir_id(c, operands[2 + j]);
                        }
                    }
                    uint32_t r = ssir_build_construct(c->mod, func_id, block_id, type_id, comps, comp_count);
                    set_ssir_id(c, result_id, r);
                    SPIRV_TO_SSIR_FREE(comps);
                }
                break;

            case SpvOpCompositeExtract:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t composite = get_ssir_id(c, operands[2]);
                    uint32_t index = operands[3];
                    uint32_t r = ssir_build_extract(c->mod, func_id, block_id, type_id, composite, index);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpCompositeInsert:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t value = get_ssir_id(c, operands[2]);
                    uint32_t composite = get_ssir_id(c, operands[3]);
                    uint32_t index = operands[4];
                    uint32_t r = ssir_build_insert(c->mod, func_id, block_id, type_id, composite, value, index);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpVectorShuffle:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t v1 = get_ssir_id(c, operands[2]);
                    uint32_t v2 = get_ssir_id(c, operands[3]);
                    int idx_count = operand_count - 4;
                    uint32_t *indices = NULL;
                    if (idx_count > 0) {
                        indices = (uint32_t*)SPIRV_TO_SSIR_MALLOC(idx_count * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(indices != NULL, "SpvOpVectorShuffle: indices malloc failed");
                        memcpy(indices, &operands[4], idx_count * sizeof(uint32_t));
                    }
                    uint32_t r = ssir_build_shuffle(c->mod, func_id, block_id, type_id, v1, v2, indices, idx_count);
                    set_ssir_id(c, result_id, r);
                    SPIRV_TO_SSIR_FREE(indices);
                }
                break;

            case SpvOpVectorExtractDynamic:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t vec = get_ssir_id(c, operands[2]);
                    uint32_t idx = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_extract_dyn(c->mod, func_id, block_id, type_id, vec, idx);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpVectorInsertDynamic:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t vec = get_ssir_id(c, operands[2]);
                    uint32_t val = get_ssir_id(c, operands[3]);
                    uint32_t idx = get_ssir_id(c, operands[4]);
                    uint32_t r = ssir_build_insert_dyn(c->mod, func_id, block_id, type_id, vec, val, idx);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpSelect:
                if (operand_count >= 5) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t cond = get_ssir_id(c, operands[2]);
                    uint32_t t = get_ssir_id(c, operands[3]);
                    uint32_t f = get_ssir_id(c, operands[4]);
                    uint32_t args[3] = { cond, t, f };
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_SELECT, args, 3);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpDot:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t a = get_ssir_id(c, operands[2]);
                    uint32_t b = get_ssir_id(c, operands[3]);
                    uint32_t args[2] = { a, b };
                    uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, SSIR_BUILTIN_DOT, args, 2);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpConvertFToS: case SpvOpConvertFToU: case SpvOpConvertSToF: case SpvOpConvertUToF:
            case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t val = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_convert(c->mod, func_id, block_id, type_id, val);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpBitcast:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t val = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_bitcast(c->mod, func_id, block_id, type_id, val);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpCopyObject:
                if (operand_count >= 3) {
                    uint32_t result_id = operands[1];
                    uint32_t src = get_ssir_id(c, operands[2]);
                    set_ssir_id(c, result_id, src);
                }
                break;

            case SpvOpPhi:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    int pair_count = (operand_count - 2) / 2;
                    uint32_t *incoming = NULL;
                    if (pair_count > 0) {
                        incoming = (uint32_t*)SPIRV_TO_SSIR_MALLOC(pair_count * 2 * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(incoming != NULL, "SpvOpPhi: incoming malloc failed");
                        for (int j = 0; j < pair_count; j++) {
                            incoming[j * 2] = get_ssir_id(c, operands[2 + j * 2]);
                            incoming[j * 2 + 1] = get_ssir_id(c, operands[3 + j * 2]);
                        }
                    }
                    uint32_t r = ssir_build_phi(c->mod, func_id, block_id, type_id, incoming, pair_count * 2);
                    set_ssir_id(c, result_id, r);
                    SPIRV_TO_SSIR_FREE(incoming);
                }
                break;

            case SpvOpBranch:
                if (operand_count >= 1) {
                    uint32_t target = get_ssir_id(c, operands[0]);
                    ssir_build_branch(c->mod, func_id, block_id, target);
                }
                break;

            case SpvOpBranchConditional:
                if (operand_count >= 3) {
                    uint32_t cond = get_ssir_id(c, operands[0]);
                    uint32_t true_block = get_ssir_id(c, operands[1]);
                    uint32_t false_block = get_ssir_id(c, operands[2]);
                    if (blk->is_selection_header && blk->merge_block) {
                        uint32_t merge = get_ssir_id(c, blk->merge_block);
                        ssir_build_branch_cond_merge(c->mod, func_id, block_id, cond, true_block, false_block, merge);
                    } else {
                        ssir_build_branch_cond(c->mod, func_id, block_id, cond, true_block, false_block);
                    }
                }
                break;

            case SpvOpReturn:
                ssir_build_return_void(c->mod, func_id, block_id);
                break;

            case SpvOpReturnValue:
                if (operand_count >= 1) {
                    uint32_t val = get_ssir_id(c, operands[0]);
                    ssir_build_return(c->mod, func_id, block_id, val);
                }
                break;

            case SpvOpUnreachable:
                ssir_build_unreachable(c->mod, func_id, block_id);
                break;

            case SpvOpKill:
                ssir_build_discard(c->mod, func_id, block_id);
                break;

            case SpvOpFunctionCall:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t callee = get_ssir_id(c, operands[2]);
                    int arg_count = operand_count - 3;
                    uint32_t *args = NULL;
                    if (arg_count > 0) {
                        args = (uint32_t*)SPIRV_TO_SSIR_MALLOC(arg_count * sizeof(uint32_t));
                        /* PRE: malloc succeeded */
                        wgsl_compiler_assert(args != NULL, "SpvOpFunctionCall: args malloc failed");
                        for (int j = 0; j < arg_count; j++) {
                            args[j] = get_ssir_id(c, operands[3 + j]);
                        }
                    }
                    uint32_t r = ssir_build_call(c->mod, func_id, block_id, type_id, callee, args, arg_count);
                    set_ssir_id(c, result_id, r);
                    SPIRV_TO_SSIR_FREE(args);
                }
                break;

            case SpvOpExtInst:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t set = operands[2];
                    uint32_t inst = operands[3];
                    if (set == c->glsl_ext_id) {
                        SsirBuiltinId bid = glsl_ext_to_ssir_builtin((enum GLSLstd450)inst);
                        if (bid != SSIR_BUILTIN_COUNT) {
                            int arg_count = operand_count - 4;
                            uint32_t *args = NULL;
                            if (arg_count > 0) {
                                args = (uint32_t*)SPIRV_TO_SSIR_MALLOC(arg_count * sizeof(uint32_t));
                                /* PRE: malloc succeeded */
                                wgsl_compiler_assert(args != NULL, "SpvOpExtInst: args malloc failed");
                                for (int j = 0; j < arg_count; j++) {
                                    args[j] = get_ssir_id(c, operands[4 + j]);
                                }
                            }
                            uint32_t r = ssir_build_builtin(c->mod, func_id, block_id, type_id, bid, args, arg_count);
                            set_ssir_id(c, result_id, r);
                            SPIRV_TO_SSIR_FREE(args);
                        }
                    }
                }
                break;

            case SpvOpArrayLength:
                if (operand_count >= 4) {
                    uint32_t result_id = operands[1];
                    uint32_t ptr = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_array_len(c->mod, func_id, block_id, ptr);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpControlBarrier:
                ssir_build_barrier(c->mod, func_id, block_id, SSIR_BARRIER_WORKGROUP);
                break;

            /* Texture operations */
            case SpvOpImageSampleImplicitLod:
            case SpvOpImageSampleExplicitLod:
            case SpvOpImageSampleDrefImplicitLod:
            case SpvOpImageSampleDrefExplicitLod:
            case SpvOpImageGather:
            case SpvOpImageDrefGather:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t si_id = operands[2];
                    uint32_t coord_id = operands[3];
                    /* Resolve OpSampledImage to get image + sampler */
                    uint32_t image_spv = 0, sampler_spv = 0;
                    if (si_id < c->id_bound &&
                        c->ids[si_id].kind == VTS_SPV_ID_INSTRUCTION &&
                        c->ids[si_id].instruction.opcode == SpvOpSampledImage &&
                        c->ids[si_id].instruction.operand_count >= 2) {
                        /* PRE: operands array valid if operand_count > 0 */
                        wgsl_compiler_assert(c->ids[si_id].instruction.operands != NULL, "SpvOpImage*: SampledImage operands NULL");
                        image_spv = c->ids[si_id].instruction.operands[0];
                        sampler_spv = c->ids[si_id].instruction.operands[1];
                    }
                    uint32_t tex = get_ssir_id(c, image_spv);
                    uint32_t samp = get_ssir_id(c, sampler_spv);
                    uint32_t coord = get_ssir_id(c, coord_id);
                    uint32_t r = 0;

                    if (op == SpvOpImageSampleImplicitLod) {
                        uint32_t mask = (operand_count > 4) ? operands[4] : 0;
                        if (mask == 0) {
                            r = ssir_build_tex_sample(c->mod, func_id, block_id, type_id, tex, samp, coord);
                        } else if (mask == 0x1 && operand_count > 5) { /* Bias */
                            uint32_t bias = get_ssir_id(c, operands[5]);
                            r = ssir_build_tex_sample_bias(c->mod, func_id, block_id, type_id, tex, samp, coord, bias);
                        } else if (mask == 0x10 && operand_count > 5) { /* ConstOffset */
                            uint32_t off = get_ssir_id(c, operands[5]);
                            r = ssir_build_tex_sample_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, off);
                        } else if (mask == (0x1|0x10) && operand_count > 6) { /* Bias|ConstOffset */
                            uint32_t bias = get_ssir_id(c, operands[5]);
                            uint32_t off = get_ssir_id(c, operands[6]);
                            r = ssir_build_tex_sample_bias_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, bias, off);
                        } else {
                            r = ssir_build_tex_sample(c->mod, func_id, block_id, type_id, tex, samp, coord);
                        }
                    } else if (op == SpvOpImageSampleExplicitLod) {
                        uint32_t mask = (operand_count > 4) ? operands[4] : 0;
                        if ((mask & 0x2) && operand_count > 5) { /* Lod */
                            uint32_t lod = get_ssir_id(c, operands[5]);
                            if (mask & 0x10 && operand_count > 6) { /* Lod|ConstOffset */
                                uint32_t off = get_ssir_id(c, operands[6]);
                                r = ssir_build_tex_sample_level_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, lod, off);
                            } else {
                                r = ssir_build_tex_sample_level(c->mod, func_id, block_id, type_id, tex, samp, coord, lod);
                            }
                        } else if ((mask & 0x4) && operand_count > 6) { /* Grad */
                            uint32_t ddx = get_ssir_id(c, operands[5]);
                            uint32_t ddy = get_ssir_id(c, operands[6]);
                            if (mask & 0x10 && operand_count > 7) { /* Grad|ConstOffset */
                                uint32_t off = get_ssir_id(c, operands[7]);
                                r = ssir_build_tex_sample_grad_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, ddx, ddy, off);
                            } else {
                                r = ssir_build_tex_sample_grad(c->mod, func_id, block_id, type_id, tex, samp, coord, ddx, ddy);
                            }
                        }
                    } else if (op == SpvOpImageSampleDrefImplicitLod && operand_count >= 5) {
                        uint32_t ref = get_ssir_id(c, operands[4]);
                        uint32_t mask = (operand_count > 5) ? operands[5] : 0;
                        if (mask == 0x10 && operand_count > 6) { /* ConstOffset */
                            uint32_t off = get_ssir_id(c, operands[6]);
                            r = ssir_build_tex_sample_cmp_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, ref, off);
                        } else {
                            r = ssir_build_tex_sample_cmp(c->mod, func_id, block_id, type_id, tex, samp, coord, ref);
                        }
                    } else if (op == SpvOpImageSampleDrefExplicitLod && operand_count >= 6) {
                        uint32_t ref = get_ssir_id(c, operands[4]);
                        uint32_t lod_mask = operands[5];
                        if ((lod_mask & 0x2) && operand_count > 6) {
                            uint32_t lod = get_ssir_id(c, operands[6]);
                            r = ssir_build_tex_sample_cmp_level(c->mod, func_id, block_id, type_id, tex, samp, coord, ref, lod);
                        } else {
                            r = ssir_build_tex_sample_cmp(c->mod, func_id, block_id, type_id, tex, samp, coord, ref);
                        }
                    } else if (op == SpvOpImageGather && operand_count >= 5) {
                        uint32_t comp = get_ssir_id(c, operands[4]);
                        uint32_t mask = (operand_count > 5) ? operands[5] : 0;
                        if (mask == 0x10 && operand_count > 6) { /* ConstOffset */
                            uint32_t off = get_ssir_id(c, operands[6]);
                            r = ssir_build_tex_gather_offset(c->mod, func_id, block_id, type_id, tex, samp, coord, comp, off);
                        } else {
                            r = ssir_build_tex_gather(c->mod, func_id, block_id, type_id, tex, samp, coord, comp);
                        }
                    } else if (op == SpvOpImageDrefGather && operand_count >= 5) {
                        uint32_t ref = get_ssir_id(c, operands[4]);
                        r = ssir_build_tex_gather_cmp(c->mod, func_id, block_id, type_id, tex, samp, coord, ref);
                    }
                    if (r) set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageRead:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t tex = get_ssir_id(c, operands[2]);
                    uint32_t coord = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_tex_load(c->mod, func_id, block_id, type_id, tex, coord, 0);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageWrite:
                if (operand_count >= 3) {
                    uint32_t tex = get_ssir_id(c, operands[0]);
                    uint32_t coord = get_ssir_id(c, operands[1]);
                    uint32_t val = get_ssir_id(c, operands[2]);
                    ssir_build_tex_store(c->mod, func_id, block_id, tex, coord, val);
                }
                break;

            case SpvOpImageQuerySizeLod:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t tex = get_ssir_id(c, operands[2]);
                    uint32_t lod = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_tex_size(c->mod, func_id, block_id, type_id, tex, lod);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageQuerySize:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t tex = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_tex_size(c->mod, func_id, block_id, type_id, tex, 0);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageQueryLod:
                if (operand_count >= 4) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    /* operands[2] is sampled image */
                    uint32_t si_id = operands[2];
                    uint32_t image_spv = 0, sampler_spv = 0;
                    if (si_id < c->id_bound &&
                        c->ids[si_id].kind == VTS_SPV_ID_INSTRUCTION &&
                        c->ids[si_id].instruction.opcode == SpvOpSampledImage &&
                        c->ids[si_id].instruction.operand_count >= 2) {
                        /* PRE: operands array valid if operand_count > 0 */
                        wgsl_compiler_assert(c->ids[si_id].instruction.operands != NULL, "SpvOpImageQueryLod: SampledImage operands NULL");
                        image_spv = c->ids[si_id].instruction.operands[0];
                        sampler_spv = c->ids[si_id].instruction.operands[1];
                    }
                    uint32_t tex = get_ssir_id(c, image_spv);
                    uint32_t samp = get_ssir_id(c, sampler_spv);
                    uint32_t coord = get_ssir_id(c, operands[3]);
                    uint32_t r = ssir_build_tex_query_lod(c->mod, func_id, block_id, type_id, tex, samp, coord);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageQueryLevels:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t tex = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_tex_query_levels(c->mod, func_id, block_id, type_id, tex);
                    set_ssir_id(c, result_id, r);
                }
                break;

            case SpvOpImageQuerySamples:
                if (operand_count >= 3) {
                    uint32_t type_id = convert_type(c, operands[0]);
                    uint32_t result_id = operands[1];
                    uint32_t tex = get_ssir_id(c, operands[2]);
                    uint32_t r = ssir_build_tex_query_samples(c->mod, func_id, block_id, type_id, tex);
                    set_ssir_id(c, result_id, r);
                }
                break;

            default:
                break;
            }
        }
    }
}

static SsirStage exec_model_to_stage(SpvExecutionModel model) {
    switch (model) {
    case SpvExecutionModelVertex: return SSIR_STAGE_VERTEX;
    case SpvExecutionModelFragment: return SSIR_STAGE_FRAGMENT;
    case SpvExecutionModelGLCompute: return SSIR_STAGE_COMPUTE;
    case SpvExecutionModelGeometry: return SSIR_STAGE_GEOMETRY;
    case SpvExecutionModelTessellationControl: return SSIR_STAGE_TESS_CONTROL;
    case SpvExecutionModelTessellationEvaluation: return SSIR_STAGE_TESS_EVAL;
    default: return SSIR_STAGE_COMPUTE;
    }
}

//c nonnull
static void convert_entry_points(VtsConverter *c) {
    wgsl_compiler_assert(c != NULL, "convert_entry_points: c is NULL");
    for (int i = 0; i < c->function_count; i++) {
        VtsSpvFunction *fn = &c->functions[i];
        if (!fn->is_entry_point) continue;

        if (fn->id >= c->id_bound) continue;
        uint32_t func_ssir_id = c->ids[fn->id].ssir_id;
        SsirStage stage = exec_model_to_stage(fn->exec_model);
        const char *name = fn->name ? fn->name : "main";

        uint32_t ep_idx = ssir_entry_point_create(c->mod, stage, func_ssir_id, name);

        for (int j = 0; j < fn->interface_var_count; j++) {
            uint32_t var_id = fn->interface_vars[j];
            if (var_id < c->id_bound) {
                uint32_t ssir_var = c->ids[var_id].ssir_id;
                if (ssir_var) {
                    ssir_entry_point_add_interface(c->mod, ep_idx, ssir_var);
                }
            }
        }

        if (stage == SSIR_STAGE_COMPUTE) {
            ssir_entry_point_set_workgroup_size(c->mod, ep_idx,
                fn->workgroup_size[0], fn->workgroup_size[1], fn->workgroup_size[2]);
        }
        if (fn->depth_replacing)
            ssir_entry_point_set_depth_replacing(c->mod, ep_idx, true);
        if (fn->origin_upper_left)
            ssir_entry_point_set_origin_upper_left(c->mod, ep_idx, true);
        if (fn->early_fragment_tests)
            ssir_entry_point_set_early_fragment_tests(c->mod, ep_idx, true);
    }
}

static void free_converter(VtsConverter *c) {
    if (!c) return;
    if (c->ids) {
        for (uint32_t i = 0; i < c->id_bound; i++) {
            SPIRV_TO_SSIR_FREE(c->ids[i].name);
            if (c->ids[i].decorations) {
                for (int j = 0; j < c->ids[i].decoration_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].decorations[j].literals);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].decorations);
            }
            if (c->ids[i].member_decorations) {
                for (int j = 0; j < c->ids[i].member_decoration_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].member_decorations[j].literals);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].member_decorations);
            }
            if (c->ids[i].member_names) {
                for (int j = 0; j < c->ids[i].member_name_count; j++) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].member_names[j]);
                }
                SPIRV_TO_SSIR_FREE(c->ids[i].member_names);
            }
            if (c->ids[i].kind == VTS_SPV_ID_TYPE) {
                if (c->ids[i].type_info.kind == VTS_SPV_TYPE_STRUCT && c->ids[i].type_info.struct_type.member_types) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].type_info.struct_type.member_types);
                }
                if (c->ids[i].type_info.kind == VTS_SPV_TYPE_FUNCTION && c->ids[i].type_info.function.param_types) {
                    SPIRV_TO_SSIR_FREE(c->ids[i].type_info.function.param_types);
                }
            }
            if (c->ids[i].kind == VTS_SPV_ID_CONSTANT && c->ids[i].constant.values) {
                SPIRV_TO_SSIR_FREE(c->ids[i].constant.values);
            }
            if (c->ids[i].kind == VTS_SPV_ID_INSTRUCTION && c->ids[i].instruction.operands) {
                SPIRV_TO_SSIR_FREE(c->ids[i].instruction.operands);
            }
        }
        SPIRV_TO_SSIR_FREE(c->ids);
    }
    if (c->functions) {
        for (int i = 0; i < c->function_count; i++) {
            SPIRV_TO_SSIR_FREE(c->functions[i].name);
            SPIRV_TO_SSIR_FREE(c->functions[i].params);
            SPIRV_TO_SSIR_FREE(c->functions[i].interface_vars);
            SPIRV_TO_SSIR_FREE(c->functions[i].local_vars);
            for (int j = 0; j < c->functions[i].block_count; j++) {
                SPIRV_TO_SSIR_FREE(c->functions[i].blocks[j].instructions);
            }
            SPIRV_TO_SSIR_FREE(c->functions[i].blocks);
        }
        SPIRV_TO_SSIR_FREE(c->functions);
    }
    if (c->pending_eps) {
        for (int i = 0; i < c->pending_ep_count; i++) {
            SPIRV_TO_SSIR_FREE(c->pending_eps[i].name);
            SPIRV_TO_SSIR_FREE(c->pending_eps[i].interface_vars);
        }
        SPIRV_TO_SSIR_FREE(c->pending_eps);
    }
    SPIRV_TO_SSIR_FREE(c->pending_wgs);
    SPIRV_TO_SSIR_FREE(c->pending_exec_modes);
}

SpirvToSsirResult spirv_to_ssir(
    const uint32_t *spirv,
    size_t word_count,
    const SpirvToSsirOptions *opts,
    SsirModule **out_module,
    char **out_error
) {
    if (!spirv || word_count < 5 || !out_module) {
        if (out_error) *out_error = strdup("Invalid input");
        return SPIRV_TO_SSIR_INVALID_SPIRV;
    }

    if (spirv[0] != SPV_MAGIC) {
        if (out_error) *out_error = strdup("Invalid SPIR-V magic number");
        return SPIRV_TO_SSIR_INVALID_SPIRV;
    }

    VtsConverter c;
    memset(&c, 0, sizeof(c));
    c.spirv = spirv;
    c.word_count = word_count;
    c.version = spirv[1];
    c.generator = spirv[2];
    c.id_bound = spirv[3];
    c.opts = opts;

    /* Sanity-check id_bound: must not exceed the number of words in the module
       (each ID definition requires at least one word), and cap to avoid
       allocating gigabytes from a crafted header. */
    if (c.id_bound == 0 || c.id_bound > word_count) {
        if (out_error) *out_error = strdup("Invalid SPIR-V: id_bound out of range");
        return SPIRV_TO_SSIR_INVALID_SPIRV;
    }

    c.ids = (VtsSpvIdInfo*)SPIRV_TO_SSIR_MALLOC(c.id_bound * sizeof(VtsSpvIdInfo));
    if (!c.ids) {
        if (out_error) *out_error = strdup("Out of memory");
        return SPIRV_TO_SSIR_INTERNAL_ERROR;
    }
    memset(c.ids, 0, c.id_bound * sizeof(VtsSpvIdInfo));
    for (uint32_t i = 0; i < c.id_bound; i++) {
        c.ids[i].id = i;
    }

    c.mod = ssir_module_create();
    if (!c.mod) {
        free_converter(&c);
        if (out_error) *out_error = strdup("Failed to create SSIR module");
        return SPIRV_TO_SSIR_INTERNAL_ERROR;
    }

    SpirvToSsirResult result = parse_spirv(&c);
    if (result != SPIRV_TO_SSIR_SUCCESS) {
        ssir_module_destroy(c.mod);
        if (out_error) *out_error = strdup(c.last_error);
        free_converter(&c);
        return result;
    }

    for (uint32_t i = 1; i < c.id_bound; i++) {
        if (c.ids[i].kind == VTS_SPV_ID_TYPE) {
            convert_type(&c, i);
        }
    }

    for (uint32_t i = 1; i < c.id_bound; i++) {
        if (c.ids[i].kind == VTS_SPV_ID_CONSTANT) {
            convert_constant(&c, i);
        }
    }

    convert_global_vars(&c);

    for (int i = 0; i < c.function_count; i++) {
        convert_function(&c, &c.functions[i]);
    }

    convert_entry_points(&c);

    *out_module = c.mod;
    ssir_module_build_lookup(c.mod);
    free_converter(&c);
    return SPIRV_TO_SSIR_SUCCESS;
}

const char *spirv_to_ssir_result_string(SpirvToSsirResult result) {
    switch (result) {
    case SPIRV_TO_SSIR_SUCCESS: return "Success";
    case SPIRV_TO_SSIR_INVALID_SPIRV: return "Invalid SPIR-V";
    case SPIRV_TO_SSIR_UNSUPPORTED_FEATURE: return "Unsupported feature";
    case SPIRV_TO_SSIR_INTERNAL_ERROR: return "Internal error";
    default: return "Unknown error";
    }
}

void spirv_to_ssir_free(void *p) {
    SPIRV_TO_SSIR_FREE(p);
}
