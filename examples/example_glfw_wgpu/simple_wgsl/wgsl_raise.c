#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "simple_wgsl.h"
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

#ifndef WGSL_MALLOC
#define WGSL_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef WGSL_REALLOC
#define WGSL_REALLOC(P, SZ) realloc((P), (SZ))
#endif
#ifndef WGSL_FREE
#define WGSL_FREE(P) free((P))
#endif

#define SPV_MAGIC 0x07230203

typedef enum {
    SPV_ID_UNKNOWN = 0,
    SPV_ID_TYPE,
    SPV_ID_CONSTANT,
    SPV_ID_VARIABLE,
    SPV_ID_FUNCTION,
    SPV_ID_LABEL,
    SPV_ID_INSTRUCTION,
    SPV_ID_EXT_INST_IMPORT
} SpvIdKind;

typedef enum {
    SPV_TYPE_VOID = 0,
    SPV_TYPE_BOOL,
    SPV_TYPE_INT,
    SPV_TYPE_FLOAT,
    SPV_TYPE_VECTOR,
    SPV_TYPE_MATRIX,
    SPV_TYPE_ARRAY,
    SPV_TYPE_RUNTIME_ARRAY,
    SPV_TYPE_STRUCT,
    SPV_TYPE_POINTER,
    SPV_TYPE_FUNCTION,
    SPV_TYPE_IMAGE,
    SPV_TYPE_SAMPLED_IMAGE,
    SPV_TYPE_SAMPLER
} SpvTypeKind;

typedef struct {
    SpvTypeKind kind;
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
} SpvTypeInfo;

typedef struct {
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SpvDecorationEntry;

typedef struct {
    uint32_t member_index;
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SpvMemberDecoration;

typedef struct {
    SpvIdKind kind;
    uint32_t id;
    char *name;

    SpvTypeInfo type_info;

    union {
        struct { uint32_t type_id; uint32_t value[4]; int value_count; } constant;
        struct { uint32_t type_id; SpvStorageClass storage_class; uint32_t initializer; } variable;
        struct { uint32_t return_type; uint32_t func_type; } function;
        struct { uint32_t type_id; SpvOp opcode; uint32_t *operands; int operand_count; } instruction;
    };

    SpvDecorationEntry *decorations;
    int decoration_count;
    SpvMemberDecoration *member_decorations;
    int member_decoration_count;
    char **member_names;
    int member_name_count;
} SpvIdInfo;

typedef struct {
    uint32_t label_id;
    uint32_t *instructions;
    int instruction_count;
    int instruction_cap;
    uint32_t merge_block;
    uint32_t continue_block;
    int is_loop_header;
    int is_selection_header;
    int emitted;
} SpvBasicBlock;

typedef struct {
    uint32_t id;
    char *name;
    uint32_t return_type;
    uint32_t func_type;
    uint32_t *params;
    int param_count;
    SpvBasicBlock *blocks;
    int block_count;
    int block_cap;
    SpvExecutionModel exec_model;
    int is_entry_point;
    uint32_t *interface_vars;
    int interface_var_count;
    int workgroup_size[3];
} SpvFunction;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int indent;
} StringBuffer;

typedef struct {
    uint32_t func_id;
    SpvExecutionModel exec_model;
    char *name;
    uint32_t *interface_vars;
    int interface_var_count;
} PendingEntryPoint;

typedef struct {
    uint32_t func_id;
    int workgroup_size[3];
} PendingWorkgroupSize;

struct WgslRaiser {
    const uint32_t *spirv;
    size_t word_count;

    uint32_t version;
    uint32_t generator;
    uint32_t id_bound;

    SpvIdInfo *ids;
    SpvFunction *functions;
    int function_count;
    int function_cap;

    PendingEntryPoint *pending_eps;
    int pending_ep_count;
    int pending_ep_cap;

    PendingWorkgroupSize *pending_wgs;
    int pending_wg_count;
    int pending_wg_cap;

    uint32_t glsl_ext_id;

    int var_counter;
    int emit_depth;
    int emit_calls;
    int type_depth;

    char *output;
    char last_error[256];
    StringBuffer sb;
};

//r nonnull
//msg nonnull
static void wr_set_error(WgslRaiser *r, const char *msg) {
    wgsl_compiler_assert(r != NULL, "wr_set_error: r is NULL");
    wgsl_compiler_assert(msg != NULL, "wr_set_error: msg is NULL");
    size_t n = strlen(msg);
    if (n >= sizeof(r->last_error)) n = sizeof(r->last_error) - 1;
    memcpy(r->last_error, msg, n);
    r->last_error[n] = 0;
}

//sb nonnull
static void sb_init(StringBuffer *sb) {
    wgsl_compiler_assert(sb != NULL, "sb_init: sb is NULL");
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->indent = 0;
}

//sb nonnull
static void sb_free(StringBuffer *sb) {
    wgsl_compiler_assert(sb != NULL, "sb_free: sb is NULL");
    WGSL_FREE(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}

//sb nonnull
static int sb_reserve(StringBuffer *sb, size_t need) {
    wgsl_compiler_assert(sb != NULL, "sb_reserve: sb is NULL");
    if (sb->len + need + 1 <= sb->cap) return 1;
    size_t ncap = sb->cap ? sb->cap : 256;
    while (ncap < sb->len + need + 1) ncap *= 2;
    char *nd = (char*)WGSL_REALLOC(sb->data, ncap);
    if (!nd) return 0;
    sb->data = nd;
    sb->cap = ncap;
    return 1;
}

//sb nonnull
//s nonnull
static void wr_sb_append(StringBuffer *sb, const char *s) {
    wgsl_compiler_assert(sb != NULL, "wr_sb_append: sb is NULL");
    wgsl_compiler_assert(s != NULL, "wr_sb_append: s is NULL");
    size_t n = strlen(s);
    if (!sb_reserve(sb, n)) return;
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = 0;
}

//sb nonnull
//fmt nonnull
static void wr_sb_appendf(StringBuffer *sb, const char *fmt, ...) {
    wgsl_compiler_assert(sb != NULL, "wr_sb_appendf: sb is NULL");
    wgsl_compiler_assert(fmt != NULL, "wr_sb_appendf: fmt is NULL");
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    wr_sb_append(sb, buf);
}

//sb nonnull
static void sb_indent(StringBuffer *sb) {
    wgsl_compiler_assert(sb != NULL, "sb_indent: sb is NULL");
    for (int i = 0; i < sb->indent; i++) wr_sb_append(sb, "    ");
}

//sb nonnull
static void sb_newline(StringBuffer *sb) {
    wgsl_compiler_assert(sb != NULL, "sb_newline: sb is NULL");
    wr_sb_append(sb, "\n");
}

//words nonnull
//out_words_read nonnull
static const char *read_string(const uint32_t *words, int word_count, int *out_words_read) {
    wgsl_compiler_assert(words != NULL, "read_string: words is NULL");
    wgsl_compiler_assert(out_words_read != NULL, "read_string: out_words_read is NULL");
    if (word_count <= 0) { *out_words_read = 0; return ""; }
    int max_chars = word_count * 4;
    const char *str = (const char*)words;
    int len = 0;
    while (len < max_chars && str[len]) len++;
    *out_words_read = (len + 4) / 4;
    char *copy = (char*)WGSL_MALLOC(len + 1);
    if (!copy) return "";
    memcpy(copy, str, len);
    copy[len] = 0;
    return copy;
}

WgslRaiser *wgsl_raise_create(const uint32_t *spirv, size_t word_count) {
    if (!spirv || word_count < 5) return NULL;
    if (spirv[0] != SPV_MAGIC) return NULL;

    WgslRaiser *r = (WgslRaiser*)WGSL_MALLOC(sizeof(WgslRaiser));
    if (!r) return NULL;

    r->spirv = spirv;
    r->word_count = word_count;
    r->version = spirv[1];
    r->generator = spirv[2];
    r->id_bound = spirv[3];

    /* Sanity-check id_bound against actual module size to prevent
       huge allocations from crafted headers. */
    if (r->id_bound == 0 || r->id_bound > word_count) {
        WGSL_FREE(r);
        return NULL;
    }

    r->ids = (SpvIdInfo*)WGSL_MALLOC(r->id_bound * sizeof(SpvIdInfo));
    if (!r->ids) {
        WGSL_FREE(r);
        return NULL;
    }
    memset(r->ids, 0, r->id_bound * sizeof(SpvIdInfo));
    for (uint32_t i = 0; i < r->id_bound; i++) {
        r->ids[i].id = i;
    }

    r->functions = NULL;
    r->function_count = 0;
    r->function_cap = 0;
    r->pending_eps = NULL;
    r->pending_ep_count = 0;
    r->pending_ep_cap = 0;
    r->pending_wgs = NULL;
    r->pending_wg_count = 0;
    r->pending_wg_cap = 0;
    r->glsl_ext_id = 0;
    r->output = NULL;
    r->last_error[0] = 0;
    sb_init(&r->sb);

    return r;
}

void wgsl_raise_destroy(WgslRaiser *r) {
    if (!r) return;
    if (r->ids) {
        for (uint32_t i = 0; i < r->id_bound; i++) {
            if (r->ids[i].name) WGSL_FREE(r->ids[i].name);
            if (r->ids[i].decorations) {
                for (int j = 0; j < r->ids[i].decoration_count; j++) {
                    WGSL_FREE(r->ids[i].decorations[j].literals);
                }
                WGSL_FREE(r->ids[i].decorations);
            }
            if (r->ids[i].member_decorations) {
                for (int j = 0; j < r->ids[i].member_decoration_count; j++) {
                    WGSL_FREE(r->ids[i].member_decorations[j].literals);
                }
                WGSL_FREE(r->ids[i].member_decorations);
            }
            if (r->ids[i].member_names) {
                for (int j = 0; j < r->ids[i].member_name_count; j++) {
                    WGSL_FREE(r->ids[i].member_names[j]);
                }
                WGSL_FREE(r->ids[i].member_names);
            }
            if (r->ids[i].kind == SPV_ID_TYPE) {
                if (r->ids[i].type_info.kind == SPV_TYPE_STRUCT && r->ids[i].type_info.struct_type.member_types) {
                    WGSL_FREE(r->ids[i].type_info.struct_type.member_types);
                }
                if (r->ids[i].type_info.kind == SPV_TYPE_FUNCTION && r->ids[i].type_info.function.param_types) {
                    WGSL_FREE(r->ids[i].type_info.function.param_types);
                }
            }
            if (r->ids[i].kind == SPV_ID_INSTRUCTION && r->ids[i].instruction.operands) {
                WGSL_FREE(r->ids[i].instruction.operands);
            }
        }
        WGSL_FREE(r->ids);
    }
    if (r->functions) {
        for (int i = 0; i < r->function_count; i++) {
            WGSL_FREE(r->functions[i].name);
            WGSL_FREE(r->functions[i].params);
            WGSL_FREE(r->functions[i].interface_vars);
            for (int j = 0; j < r->functions[i].block_count; j++) {
                WGSL_FREE(r->functions[i].blocks[j].instructions);
            }
            WGSL_FREE(r->functions[i].blocks);
        }
        WGSL_FREE(r->functions);
    }
    if (r->pending_eps) {
        for (int i = 0; i < r->pending_ep_count; i++) {
            WGSL_FREE(r->pending_eps[i].name);
            WGSL_FREE(r->pending_eps[i].interface_vars);
        }
        WGSL_FREE(r->pending_eps);
    }
    WGSL_FREE(r->pending_wgs);
    WGSL_FREE(r->output);
    sb_free(&r->sb);
    WGSL_FREE(r);
}

//r nonnull
static void add_decoration(WgslRaiser *r, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(r != NULL, "add_decoration: r is NULL");
    if (target >= r->id_bound) return;
    SpvIdInfo *info = &r->ids[target];
    int idx = info->decoration_count++;
    info->decorations = (SpvDecorationEntry*)WGSL_REALLOC(info->decorations, info->decoration_count * sizeof(SpvDecorationEntry));
    // PRE: realloc succeeded
    wgsl_compiler_assert(info->decorations != NULL, "add_decoration: realloc failed");
    info->decorations[idx].decoration = decor;
    info->decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->decorations[idx].literals = (uint32_t*)WGSL_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->decorations[idx].literals = NULL;
    }
}

//r nonnull
static void add_member_decoration(WgslRaiser *r, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(r != NULL, "add_member_decoration: r is NULL");
    if (struct_id >= r->id_bound) return;
    SpvIdInfo *info = &r->ids[struct_id];
    int idx = info->member_decoration_count++;
    info->member_decorations = (SpvMemberDecoration*)WGSL_REALLOC(info->member_decorations, info->member_decoration_count * sizeof(SpvMemberDecoration));
    // PRE: realloc succeeded
    wgsl_compiler_assert(info->member_decorations != NULL, "add_member_decoration: realloc failed");
    info->member_decorations[idx].member_index = member;
    info->member_decorations[idx].decoration = decor;
    info->member_decorations[idx].literal_count = lit_count;
    if (lit_count > 0) {
        info->member_decorations[idx].literals = (uint32_t*)WGSL_MALLOC(lit_count * sizeof(uint32_t));
        memcpy(info->member_decorations[idx].literals, literals, lit_count * sizeof(uint32_t));
    } else {
        info->member_decorations[idx].literals = NULL;
    }
}

//r nonnull
static int has_decoration(WgslRaiser *r, uint32_t id, SpvDecoration decor, uint32_t *out_value) {
    wgsl_compiler_assert(r != NULL, "has_decoration: r is NULL");
    if (id >= r->id_bound) return 0;
    SpvIdInfo *info = &r->ids[id];
    // PRE: decorations array valid if decoration_count > 0
    wgsl_compiler_assert(info->decoration_count == 0 || info->decorations != NULL, "has_decoration: decorations NULL with count > 0");
    for (int i = 0; i < info->decoration_count; i++) {
        if (info->decorations[i].decoration == decor) {
            // PRE: literals array valid if literal_count > 0
            wgsl_compiler_assert(info->decorations[i].literal_count == 0 || info->decorations[i].literals != NULL, "has_decoration: literals NULL with count > 0");
            if (out_value && info->decorations[i].literal_count > 0) {
                *out_value = info->decorations[i].literals[0];
            }
            return 1;
        }
    }
    return 0;
}

//r nonnull
static SpvFunction *add_function(WgslRaiser *r) {
    wgsl_compiler_assert(r != NULL, "add_function: r is NULL");
    if (r->function_count >= r->function_cap) {
        int ncap = r->function_cap ? r->function_cap * 2 : 8;
        r->functions = (SpvFunction*)WGSL_REALLOC(r->functions, ncap * sizeof(SpvFunction));
        r->function_cap = ncap;
    }
    SpvFunction *fn = &r->functions[r->function_count++];
    memset(fn, 0, sizeof(SpvFunction));
    fn->workgroup_size[0] = 1;
    fn->workgroup_size[1] = 1;
    fn->workgroup_size[2] = 1;
    return fn;
}

//fn nonnull
static SpvBasicBlock *add_block(SpvFunction *fn, uint32_t label_id) {
    wgsl_compiler_assert(fn != NULL, "add_block: fn is NULL");
    if (fn->block_count >= fn->block_cap) {
        int ncap = fn->block_cap ? fn->block_cap * 2 : 8;
        fn->blocks = (SpvBasicBlock*)WGSL_REALLOC(fn->blocks, ncap * sizeof(SpvBasicBlock));
        fn->block_cap = ncap;
    }
    SpvBasicBlock *blk = &fn->blocks[fn->block_count++];
    memset(blk, 0, sizeof(SpvBasicBlock));
    blk->label_id = label_id;
    return blk;
}

//blk nonnull
static void add_block_instr(SpvBasicBlock *blk, uint32_t instr_start) {
    wgsl_compiler_assert(blk != NULL, "add_block_instr: blk is NULL");
    if (blk->instruction_count >= blk->instruction_cap) {
        int ncap = blk->instruction_cap ? blk->instruction_cap * 2 : 16;
        blk->instructions = (uint32_t*)WGSL_REALLOC(blk->instructions, ncap * sizeof(uint32_t));
        blk->instruction_cap = ncap;
    }
    blk->instructions[blk->instruction_count++] = instr_start;
}

WgslRaiseResult wgsl_raise_parse(WgslRaiser *r) {
    if (!r) return WGSL_RAISE_INVALID_SPIRV;

    size_t pos = 5;
    SpvFunction *current_fn = NULL;
    SpvBasicBlock *current_block = NULL;

    while (pos < r->word_count) {
        uint32_t word0 = r->spirv[pos];
        uint16_t opcode = word0 & 0xFFFF;
        uint16_t wc = word0 >> 16;
        if (wc == 0 || pos + wc > r->word_count) {
            wr_set_error(r, "Invalid instruction");
            return WGSL_RAISE_INVALID_SPIRV;
        }

        const uint32_t *operands = &r->spirv[pos + 1];
        int operand_count = wc - 1;

        switch ((SpvOp)opcode) {
        case SpvOpName:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                int str_words;
                const char *name = read_string(&operands[1], operand_count - 1, &str_words);
                if (target < r->id_bound) {
                    r->ids[target].name = (char*)name;
                }
            }
            break;

        case SpvOpMemberName:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                int str_words;
                const char *name = read_string(&operands[2], operand_count - 2, &str_words);
                if (struct_id < r->id_bound && member < 4096) {
                    SpvIdInfo *info = &r->ids[struct_id];
                    if (member >= (uint32_t)info->member_name_count) {
                        int new_count = (int)member + 1;
                        info->member_names = (char**)WGSL_REALLOC(info->member_names, new_count * sizeof(char*));
                        // PRE: realloc succeeded
                        wgsl_compiler_assert(info->member_names != NULL, "OpMemberName: realloc failed");
                        for (int i = info->member_name_count; i < new_count; i++) {
                            info->member_names[i] = NULL;
                        }
                        info->member_name_count = new_count;
                    }
                    // PRE: member_names array valid after realloc
                    wgsl_compiler_assert(info->member_names != NULL, "OpMemberName: member_names NULL");
                    if (info->member_names[member]) WGSL_FREE(info->member_names[member]);
                    info->member_names[member] = (char*)name;
                } else {
                    WGSL_FREE((void*)name);
                }
            }
            break;

        case SpvOpDecorate:
            if (operand_count >= 2) {
                uint32_t target = operands[0];
                SpvDecoration decor = (SpvDecoration)operands[1];
                add_decoration(r, target, decor, &operands[2], operand_count - 2);
            }
            break;

        case SpvOpMemberDecorate:
            if (operand_count >= 3) {
                uint32_t struct_id = operands[0];
                uint32_t member = operands[1];
                SpvDecoration decor = (SpvDecoration)operands[2];
                add_member_decoration(r, struct_id, member, decor, &operands[3], operand_count - 3);
            }
            break;

        case SpvOpExtInstImport:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_EXT_INST_IMPORT;
                    int str_words;
                    const char *name = read_string(&operands[1], operand_count - 1, &str_words);
                    if (strcmp(name, "GLSL.std.450") == 0) {
                        r->glsl_ext_id = id;
                    }
                    WGSL_FREE((void*)name);
                }
            }
            break;

        case SpvOpTypeVoid:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_VOID;
                }
            }
            break;

        case SpvOpTypeBool:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_BOOL;
                }
            }
            break;

        case SpvOpTypeInt:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_INT;
                    r->ids[id].type_info.int_type.width = operands[1];
                    r->ids[id].type_info.int_type.signedness = operands[2];
                }
            }
            break;

        case SpvOpTypeFloat:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_FLOAT;
                    r->ids[id].type_info.float_type.width = operands[1];
                }
            }
            break;

        case SpvOpTypeVector:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_VECTOR;
                    r->ids[id].type_info.vector.component_type = operands[1];
                    r->ids[id].type_info.vector.count = operands[2];
                }
            }
            break;

        case SpvOpTypeMatrix:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_MATRIX;
                    r->ids[id].type_info.matrix.column_type = operands[1];
                    r->ids[id].type_info.matrix.columns = operands[2];
                }
            }
            break;

        case SpvOpTypeArray:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_ARRAY;
                    r->ids[id].type_info.array.element_type = operands[1];
                    r->ids[id].type_info.array.length_id = operands[2];
                }
            }
            break;

        case SpvOpTypeRuntimeArray:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_RUNTIME_ARRAY;
                    r->ids[id].type_info.runtime_array.element_type = operands[1];
                }
            }
            break;

        case SpvOpTypeStruct:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_STRUCT;
                    int mc = operand_count - 1;
                    r->ids[id].type_info.struct_type.member_count = mc;
                    if (mc > 0) {
                        r->ids[id].type_info.struct_type.member_types = (uint32_t*)WGSL_MALLOC(mc * sizeof(uint32_t));
                        memcpy(r->ids[id].type_info.struct_type.member_types, &operands[1], mc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypePointer:
            if (operand_count >= 3) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_POINTER;
                    r->ids[id].type_info.pointer.storage = (SpvStorageClass)operands[1];
                    r->ids[id].type_info.pointer.pointee_type = operands[2];
                }
            }
            break;

        case SpvOpTypeFunction:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_FUNCTION;
                    r->ids[id].type_info.function.return_type = operands[1];
                    int pc = operand_count - 2;
                    r->ids[id].type_info.function.param_count = pc;
                    if (pc > 0) {
                        r->ids[id].type_info.function.param_types = (uint32_t*)WGSL_MALLOC(pc * sizeof(uint32_t));
                        memcpy(r->ids[id].type_info.function.param_types, &operands[2], pc * sizeof(uint32_t));
                    }
                }
            }
            break;

        case SpvOpTypeImage:
            if (operand_count >= 8) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_IMAGE;
                    r->ids[id].type_info.image.sampled_type = operands[1];
                    r->ids[id].type_info.image.dim = (SpvDim)operands[2];
                    r->ids[id].type_info.image.depth = operands[3];
                    r->ids[id].type_info.image.arrayed = operands[4];
                    r->ids[id].type_info.image.ms = operands[5];
                    r->ids[id].type_info.image.sampled = operands[6];
                    r->ids[id].type_info.image.format = (SpvImageFormat)operands[7];
                }
            }
            break;

        case SpvOpTypeSampledImage:
            if (operand_count >= 2) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_SAMPLED_IMAGE;
                    r->ids[id].type_info.sampled_image.image_type = operands[1];
                }
            }
            break;

        case SpvOpTypeSampler:
            if (operand_count >= 1) {
                uint32_t id = operands[0];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_TYPE;
                    r->ids[id].type_info.kind = SPV_TYPE_SAMPLER;
                }
            }
            break;

        case SpvOpConstant:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_CONSTANT;
                    r->ids[id].constant.type_id = type_id;
                    int vc = operand_count - 2;
                    r->ids[id].constant.value_count = vc;
                    for (int i = 0; i < vc && i < 4; i++) {
                        r->ids[id].constant.value[i] = operands[2 + i];
                    }
                    if ((SpvOp)opcode == SpvOpConstantTrue) {
                        r->ids[id].constant.value[0] = 1;
                        r->ids[id].constant.value_count = 1;
                    } else if ((SpvOp)opcode == SpvOpConstantFalse) {
                        r->ids[id].constant.value[0] = 0;
                        r->ids[id].constant.value_count = 1;
                    }
                }
            }
            break;

        case SpvOpConstantComposite:
            if (operand_count >= 2) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_CONSTANT;
                    r->ids[id].constant.type_id = type_id;
                    int vc = operand_count - 2;
                    r->ids[id].constant.value_count = vc;
                    for (int i = 0; i < vc && i < 4; i++) {
                        r->ids[id].constant.value[i] = operands[2 + i];
                    }
                }
            }
            break;

        case SpvOpVariable:
            if (operand_count >= 3) {
                uint32_t type_id = operands[0];
                uint32_t id = operands[1];
                SpvStorageClass sc = (SpvStorageClass)operands[2];
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_VARIABLE;
                    r->ids[id].variable.type_id = type_id;
                    r->ids[id].variable.storage_class = sc;
                    r->ids[id].variable.initializer = (operand_count > 3) ? operands[3] : 0;
                }
            }
            break;

        case SpvOpEntryPoint:
            if (operand_count >= 3) {
                SpvExecutionModel model = (SpvExecutionModel)operands[0];
                uint32_t fn_id = operands[1];
                int str_words;
                const char *name = read_string(&operands[2], operand_count - 2, &str_words);
                if (fn_id < r->id_bound) {
                    r->ids[fn_id].kind = SPV_ID_FUNCTION;
                    if (r->ids[fn_id].name) WGSL_FREE(r->ids[fn_id].name);
                    r->ids[fn_id].name = (char*)WGSL_MALLOC(strlen(name) + 1);
                    strcpy(r->ids[fn_id].name, name);
                }
                if (r->pending_ep_count >= r->pending_ep_cap) {
                    int ncap = r->pending_ep_cap ? r->pending_ep_cap * 2 : 4;
                    r->pending_eps = (PendingEntryPoint*)WGSL_REALLOC(r->pending_eps, ncap * sizeof(PendingEntryPoint));
                    r->pending_ep_cap = ncap;
                }
                PendingEntryPoint *ep = &r->pending_eps[r->pending_ep_count++];
                ep->func_id = fn_id;
                ep->exec_model = model;
                ep->name = (char*)WGSL_MALLOC(strlen(name) + 1);
                strcpy(ep->name, name);
                int iface_start = 2 + str_words;
                int iface_count = operand_count - iface_start;
                if (iface_count > 0) {
                    ep->interface_vars = (uint32_t*)WGSL_MALLOC(iface_count * sizeof(uint32_t));
                    memcpy(ep->interface_vars, &operands[iface_start], iface_count * sizeof(uint32_t));
                    ep->interface_var_count = iface_count;
                } else {
                    ep->interface_vars = NULL;
                    ep->interface_var_count = 0;
                }
                WGSL_FREE((void*)name);
            }
            break;

        case SpvOpExecutionMode:
            if (operand_count >= 2) {
                uint32_t fn_id = operands[0];
                SpvExecutionMode mode = (SpvExecutionMode)operands[1];
                if (mode == SpvExecutionModeLocalSize && operand_count >= 5) {
                    if (r->pending_wg_count >= r->pending_wg_cap) {
                        int ncap = r->pending_wg_cap ? r->pending_wg_cap * 2 : 4;
                        r->pending_wgs = (PendingWorkgroupSize*)WGSL_REALLOC(r->pending_wgs, ncap * sizeof(PendingWorkgroupSize));
                        r->pending_wg_cap = ncap;
                    }
                    PendingWorkgroupSize *wg = &r->pending_wgs[r->pending_wg_count++];
                    wg->func_id = fn_id;
                    wg->workgroup_size[0] = operands[2];
                    wg->workgroup_size[1] = operands[3];
                    wg->workgroup_size[2] = operands[4];
                }
            }
            break;

        case SpvOpFunction:
            if (operand_count >= 4) {
                uint32_t ret_type = operands[0];
                uint32_t id = operands[1];
                uint32_t func_type = operands[3];
                current_fn = add_function(r);
                current_fn->id = id;
                current_fn->return_type = ret_type;
                current_fn->func_type = func_type;
                if (id < r->id_bound) {
                    r->ids[id].kind = SPV_ID_FUNCTION;
                    r->ids[id].function.return_type = ret_type;
                    r->ids[id].function.func_type = func_type;
                    if (r->ids[id].name) {
                        current_fn->name = (char*)WGSL_MALLOC(strlen(r->ids[id].name) + 1);
                        strcpy(current_fn->name, r->ids[id].name);
                    }
                }
                current_block = NULL;
            }
            break;

        case SpvOpFunctionParameter:
            if (current_fn && operand_count >= 2) {
                uint32_t id = operands[1];
                int idx = current_fn->param_count++;
                current_fn->params = (uint32_t*)WGSL_REALLOC(current_fn->params, current_fn->param_count * sizeof(uint32_t));
                current_fn->params[idx] = id;
            }
            break;

        case SpvOpFunctionEnd:
            current_fn = NULL;
            current_block = NULL;
            break;

        case SpvOpLabel:
            if (current_fn && operand_count >= 1) {
                uint32_t label_id = operands[0];
                current_block = add_block(current_fn, label_id);
                if (label_id < r->id_bound) {
                    r->ids[label_id].kind = SPV_ID_LABEL;
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
                add_block_instr(current_block, pos);
            }
            if (operand_count >= 2) {
                SpvOp op = (SpvOp)opcode;
                int has_result_type = 0;
                int has_result = 0;
                switch (op) {
                case SpvOpLoad: case SpvOpAccessChain: case SpvOpInBoundsAccessChain:
                case SpvOpFAdd: case SpvOpFSub: case SpvOpFMul: case SpvOpFDiv: case SpvOpFRem:
                case SpvOpIAdd: case SpvOpISub: case SpvOpIMul: case SpvOpSDiv: case SpvOpUDiv: case SpvOpSRem: case SpvOpUMod:
                case SpvOpFNegate: case SpvOpSNegate:
                case SpvOpFOrdEqual: case SpvOpFOrdNotEqual: case SpvOpFOrdLessThan: case SpvOpFOrdGreaterThan:
                case SpvOpFOrdLessThanEqual: case SpvOpFOrdGreaterThanEqual:
                case SpvOpIEqual: case SpvOpINotEqual: case SpvOpSLessThan: case SpvOpSGreaterThan:
                case SpvOpSLessThanEqual: case SpvOpSGreaterThanEqual: case SpvOpULessThan: case SpvOpUGreaterThan:
                case SpvOpULessThanEqual: case SpvOpUGreaterThanEqual:
                case SpvOpLogicalAnd: case SpvOpLogicalOr: case SpvOpLogicalNot: case SpvOpLogicalEqual: case SpvOpLogicalNotEqual:
                case SpvOpBitwiseAnd: case SpvOpBitwiseOr: case SpvOpBitwiseXor: case SpvOpNot:
                case SpvOpShiftLeftLogical: case SpvOpShiftRightLogical: case SpvOpShiftRightArithmetic:
                case SpvOpCompositeConstruct: case SpvOpCompositeExtract: case SpvOpVectorShuffle:
                case SpvOpConvertFToS: case SpvOpConvertFToU: case SpvOpConvertSToF: case SpvOpConvertUToF:
                case SpvOpBitcast: case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert:
                case SpvOpSelect: case SpvOpPhi: case SpvOpDot: case SpvOpVectorTimesScalar: case SpvOpMatrixTimesScalar:
                case SpvOpVectorTimesMatrix: case SpvOpMatrixTimesVector: case SpvOpMatrixTimesMatrix:
                case SpvOpExtInst: case SpvOpImageSampleImplicitLod: case SpvOpImageSampleExplicitLod:
                case SpvOpSampledImage:
                    has_result_type = 1;
                    has_result = 1;
                    break;
                default:
                    break;
                }
                if (has_result_type && has_result && operand_count >= 2) {
                    uint32_t type_id = operands[0];
                    uint32_t result_id = operands[1];
                    if (result_id < r->id_bound) {
                        r->ids[result_id].kind = SPV_ID_INSTRUCTION;
                        r->ids[result_id].instruction.type_id = type_id;
                        r->ids[result_id].instruction.opcode = op;
                        int op_count = operand_count - 2;
                        r->ids[result_id].instruction.operand_count = op_count;
                        if (op_count > 0) {
                            r->ids[result_id].instruction.operands = (uint32_t*)WGSL_MALLOC(op_count * sizeof(uint32_t));
                            memcpy(r->ids[result_id].instruction.operands, &operands[2], op_count * sizeof(uint32_t));
                        }
                    }
                }
            }
            break;
        }
        pos += wc;
    }

    for (int i = 0; i < r->function_count; i++) {
        uint32_t fn_id = r->functions[i].id;
        if (fn_id < r->id_bound && r->ids[fn_id].name && !r->functions[i].name) {
            r->functions[i].name = (char*)WGSL_MALLOC(strlen(r->ids[fn_id].name) + 1);
            strcpy(r->functions[i].name, r->ids[fn_id].name);
        }
    }

    for (int i = 0; i < r->pending_ep_count; i++) {
        PendingEntryPoint *ep = &r->pending_eps[i];
        for (int j = 0; j < r->function_count; j++) {
            if (r->functions[j].id == ep->func_id) {
                r->functions[j].is_entry_point = 1;
                r->functions[j].exec_model = ep->exec_model;
                if (r->functions[j].name) WGSL_FREE(r->functions[j].name);
                r->functions[j].name = (char*)WGSL_MALLOC(strlen(ep->name) + 1);
                strcpy(r->functions[j].name, ep->name);
                if (ep->interface_var_count > 0) {
                    r->functions[j].interface_vars = (uint32_t*)WGSL_MALLOC(ep->interface_var_count * sizeof(uint32_t));
                    memcpy(r->functions[j].interface_vars, ep->interface_vars, ep->interface_var_count * sizeof(uint32_t));
                    r->functions[j].interface_var_count = ep->interface_var_count;
                }
                break;
            }
        }
    }

    for (int i = 0; i < r->pending_wg_count; i++) {
        PendingWorkgroupSize *wg = &r->pending_wgs[i];
        for (int j = 0; j < r->function_count; j++) {
            if (r->functions[j].id == wg->func_id) {
                r->functions[j].workgroup_size[0] = wg->workgroup_size[0];
                r->functions[j].workgroup_size[1] = wg->workgroup_size[1];
                r->functions[j].workgroup_size[2] = wg->workgroup_size[2];
                break;
            }
        }
    }

    return WGSL_RAISE_SUCCESS;
}

WgslRaiseResult wgsl_raise_analyze(WgslRaiser *r) {
    if (!r) return WGSL_RAISE_INVALID_SPIRV;
    return WGSL_RAISE_SUCCESS;
}

//r nonnull
static const char *type_to_wgsl(WgslRaiser *r, uint32_t type_id) {
    wgsl_compiler_assert(r != NULL, "type_to_wgsl: r is NULL");
    if (type_id >= r->id_bound) return "unknown";
    SpvIdInfo *info = &r->ids[type_id];
    if (info->kind != SPV_ID_TYPE) return "unknown";
    if (r->type_depth > 64) return "unknown";
    r->type_depth++;

    static char buf[256];
    const char *result = "unknown";

    switch (info->type_info.kind) {
    case SPV_TYPE_VOID: result = "void"; break;
    case SPV_TYPE_BOOL: result = "bool"; break;
    case SPV_TYPE_INT:
        if (info->type_info.int_type.width == 32)
            result = info->type_info.int_type.signedness ? "i32" : "u32";
        else
            result = "i32";
        break;
    case SPV_TYPE_FLOAT:
        if (info->type_info.float_type.width == 16) result = "f16";
        else if (info->type_info.float_type.width == 32) result = "f32";
        else result = "f32";
        break;
    case SPV_TYPE_VECTOR: {
        uint32_t comp = info->type_info.vector.component_type;
        uint32_t cnt = info->type_info.vector.count;
        const char *ct = type_to_wgsl(r, comp);
        if (strcmp(ct, "f32") == 0) {
            snprintf(buf, sizeof(buf), "vec%u<f32>", cnt);
        } else if (strcmp(ct, "i32") == 0) {
            snprintf(buf, sizeof(buf), "vec%u<i32>", cnt);
        } else if (strcmp(ct, "u32") == 0) {
            snprintf(buf, sizeof(buf), "vec%u<u32>", cnt);
        } else if (strcmp(ct, "bool") == 0) {
            snprintf(buf, sizeof(buf), "vec%u<bool>", cnt);
        } else {
            snprintf(buf, sizeof(buf), "vec%u<%s>", cnt, ct);
        }
        result = buf;
        break;
    }
    case SPV_TYPE_MATRIX: {
        uint32_t col_type = info->type_info.matrix.column_type;
        uint32_t cols = info->type_info.matrix.columns;
        if (col_type < r->id_bound && r->ids[col_type].kind == SPV_ID_TYPE &&
            r->ids[col_type].type_info.kind == SPV_TYPE_VECTOR) {
            uint32_t rows = r->ids[col_type].type_info.vector.count;
            uint32_t elem = r->ids[col_type].type_info.vector.component_type;
            const char *et = type_to_wgsl(r, elem);
            if (strcmp(et, "f32") == 0) {
                snprintf(buf, sizeof(buf), "mat%ux%u<f32>", cols, rows);
            } else {
                snprintf(buf, sizeof(buf), "mat%ux%u<%s>", cols, rows, et);
            }
            result = buf;
        } else {
            result = "mat4x4<f32>";
        }
        break;
    }
    case SPV_TYPE_ARRAY: {
        uint32_t elem = info->type_info.array.element_type;
        uint32_t len_id = info->type_info.array.length_id;
        const char *et = type_to_wgsl(r, elem);
        if (len_id < r->id_bound && r->ids[len_id].kind == SPV_ID_CONSTANT) {
            uint32_t len = r->ids[len_id].constant.value[0];
            snprintf(buf, sizeof(buf), "array<%s, %u>", et, len);
        } else {
            snprintf(buf, sizeof(buf), "array<%s>", et);
        }
        result = buf;
        break;
    }
    case SPV_TYPE_RUNTIME_ARRAY: {
        uint32_t elem = info->type_info.runtime_array.element_type;
        const char *et = type_to_wgsl(r, elem);
        snprintf(buf, sizeof(buf), "array<%s>", et);
        result = buf;
        break;
    }
    case SPV_TYPE_STRUCT:
        result = info->name ? info->name : "UnnamedStruct";
        break;
    case SPV_TYPE_POINTER:
        result = type_to_wgsl(r, info->type_info.pointer.pointee_type);
        break;
    case SPV_TYPE_IMAGE: {
        SpvDim dim = info->type_info.image.dim;
        if (dim == SpvDim2D && info->type_info.image.sampled == 1)
            result = "texture_2d<f32>";
        else if (dim == SpvDim2D)
            result = "texture_storage_2d<rgba8unorm, write>";
        else
            result = "texture_2d<f32>";
        break;
    }
    case SPV_TYPE_SAMPLED_IMAGE:
        result = type_to_wgsl(r, info->type_info.sampled_image.image_type);
        break;
    case SPV_TYPE_SAMPLER:
        result = "sampler";
        break;
    default:
        break;
    }
    r->type_depth--;
    return result;
}

static const char *get_builtin_name(SpvBuiltIn builtin) {
    switch (builtin) {
    case SpvBuiltInPosition: return "position";
    case SpvBuiltInVertexIndex: return "vertex_index";
    case SpvBuiltInInstanceIndex: return "instance_index";
    case SpvBuiltInFrontFacing: return "front_facing";
    case SpvBuiltInFragCoord: return "position";
    case SpvBuiltInFragDepth: return "frag_depth";
    case SpvBuiltInGlobalInvocationId: return "global_invocation_id";
    case SpvBuiltInLocalInvocationId: return "local_invocation_id";
    case SpvBuiltInLocalInvocationIndex: return "local_invocation_index";
    case SpvBuiltInWorkgroupId: return "workgroup_id";
    case SpvBuiltInNumWorkgroups: return "num_workgroups";
    default: return "unknown_builtin";
    }
}

//r nonnull
static void emit_struct_decl(WgslRaiser *r, uint32_t type_id) {
    wgsl_compiler_assert(r != NULL, "emit_struct_decl: r is NULL");
    if (type_id >= r->id_bound) return;
    SpvIdInfo *info = &r->ids[type_id];
    if (info->kind != SPV_ID_TYPE || info->type_info.kind != SPV_TYPE_STRUCT) return;

    const char *name = info->name ? info->name : "UnnamedStruct";
    wr_sb_appendf(&r->sb, "struct %s {\n", name);
    r->sb.indent++;

    for (int i = 0; i < info->type_info.struct_type.member_count; i++) {
        uint32_t member_type = info->type_info.struct_type.member_types[i];

        sb_indent(&r->sb);

        // PRE: member_decorations array valid if count > 0
        wgsl_compiler_assert(info->member_decoration_count == 0 || info->member_decorations != NULL, "emit_struct_type: member_decorations NULL with count > 0");
        for (int j = 0; j < info->member_decoration_count; j++) {
            if (info->member_decorations[j].member_index == (uint32_t)i) {
                SpvDecoration d = info->member_decorations[j].decoration;
                // PRE: literals array valid if literal_count > 0
                wgsl_compiler_assert(info->member_decorations[j].literal_count == 0 || info->member_decorations[j].literals != NULL, "emit_struct_type: literals NULL with count > 0");
                if (d == SpvDecorationLocation && info->member_decorations[j].literal_count > 0) {
                    wr_sb_appendf(&r->sb, "@location(%u) ", info->member_decorations[j].literals[0]);
                } else if (d == SpvDecorationBuiltIn && info->member_decorations[j].literal_count > 0) {
                    wr_sb_appendf(&r->sb, "@builtin(%s) ", get_builtin_name((SpvBuiltIn)info->member_decorations[j].literals[0]));
                }
            }
        }

        const char *member_name = NULL;
        if (info->member_names && i < info->member_name_count && info->member_names[i]) {
            member_name = info->member_names[i];
        }
        if (member_name) {
            wr_sb_appendf(&r->sb, "%s: %s,\n", member_name, type_to_wgsl(r, member_type));
        } else {
            wr_sb_appendf(&r->sb, "member%d: %s,\n", i, type_to_wgsl(r, member_type));
        }
    }

    r->sb.indent--;
    wr_sb_append(&r->sb, "};\n\n");
}

//r nonnull
static void emit_global_var(WgslRaiser *r, uint32_t var_id) {
    wgsl_compiler_assert(r != NULL, "emit_global_var: r is NULL");
    if (var_id >= r->id_bound) return;
    SpvIdInfo *info = &r->ids[var_id];
    if (info->kind != SPV_ID_VARIABLE) return;

    SpvStorageClass sc = info->variable.storage_class;
    if (sc == SpvStorageClassFunction) return;

    uint32_t group = 0, binding = 0, location = 0;
    int has_group = has_decoration(r, var_id, SpvDecorationDescriptorSet, &group);
    int has_binding = has_decoration(r, var_id, SpvDecorationBinding, &binding);
    int has_location = has_decoration(r, var_id, SpvDecorationLocation, &location);
    uint32_t builtin_val = 0;
    int has_builtin = has_decoration(r, var_id, SpvDecorationBuiltIn, &builtin_val);

    if (sc == SpvStorageClassInput || sc == SpvStorageClassOutput) {
        return;
    }

    if (has_group) wr_sb_appendf(&r->sb, "@group(%u) ", group);
    if (has_binding) wr_sb_appendf(&r->sb, "@binding(%u) ", binding);

    uint32_t ptr_type = info->variable.type_id;
    uint32_t pointee_type = 0;
    if (ptr_type < r->id_bound && r->ids[ptr_type].kind == SPV_ID_TYPE &&
        r->ids[ptr_type].type_info.kind == SPV_TYPE_POINTER) {
        pointee_type = r->ids[ptr_type].type_info.pointer.pointee_type;
    }

    const char *type_str = type_to_wgsl(r, pointee_type);
    if (!info->name) {
        char buf[32];
        snprintf(buf, sizeof(buf), "_var%d", r->var_counter++);
        info->name = (char*)WGSL_MALLOC(strlen(buf) + 1);
        strcpy(info->name, buf);
    }
    const char *name = info->name;

    if (sc == SpvStorageClassUniform) {
        wr_sb_appendf(&r->sb, "var<uniform> %s: %s;\n", name, type_str);
    } else if (sc == SpvStorageClassStorageBuffer) {
        wr_sb_appendf(&r->sb, "var<storage, read_write> %s: %s;\n", name, type_str);
    } else if (sc == SpvStorageClassUniformConstant) {
        if (r->ids[pointee_type].kind == SPV_ID_TYPE) {
            SpvTypeKind tk = r->ids[pointee_type].type_info.kind;
            if (tk == SPV_TYPE_IMAGE || tk == SPV_TYPE_SAMPLED_IMAGE) {
                wr_sb_appendf(&r->sb, "var %s: %s;\n", name, type_str);
            } else if (tk == SPV_TYPE_SAMPLER) {
                wr_sb_appendf(&r->sb, "var %s: sampler;\n", name);
            } else {
                wr_sb_appendf(&r->sb, "var %s: %s;\n", name, type_str);
            }
        } else {
            wr_sb_appendf(&r->sb, "var %s: %s;\n", name, type_str);
        }
    } else if (sc == SpvStorageClassPrivate) {
        wr_sb_appendf(&r->sb, "var<private> %s: %s;\n", name, type_str);
    } else if (sc == SpvStorageClassWorkgroup) {
        wr_sb_appendf(&r->sb, "var<workgroup> %s: %s;\n", name, type_str);
    }
}

//r nonnull
static const char *get_id_name(WgslRaiser *r, uint32_t id) {
    wgsl_compiler_assert(r != NULL, "get_id_name: r is NULL");
    static char buf[64];
    if (id >= r->id_bound) {
        snprintf(buf, sizeof(buf), "_id%u", id);
        return buf;
    }
    if (r->ids[id].name) return r->ids[id].name;
    snprintf(buf, sizeof(buf), "_id%u", id);
    return buf;
}

static void emit_expression(WgslRaiser *r, uint32_t id, StringBuffer *out);

//r nonnull
//out nonnull
static void emit_constant(WgslRaiser *r, uint32_t id, StringBuffer *out) {
    wgsl_compiler_assert(r != NULL, "emit_constant: r is NULL");
    wgsl_compiler_assert(out != NULL, "emit_constant: out is NULL");
    if (id >= r->id_bound) { wr_sb_append(out, "0"); return; }
    SpvIdInfo *info = &r->ids[id];
    if (info->kind != SPV_ID_CONSTANT) { wr_sb_append(out, "0"); return; }

    uint32_t type_id = info->constant.type_id;
    if (type_id >= r->id_bound) { wr_sb_append(out, "0"); return; }
    SpvIdInfo *type_info = &r->ids[type_id];

    if (type_info->kind == SPV_ID_TYPE) {
        SpvTypeKind tk = type_info->type_info.kind;
        if (tk == SPV_TYPE_BOOL) {
            wr_sb_append(out, info->constant.value[0] ? "true" : "false");
        } else if (tk == SPV_TYPE_INT) {
            if (type_info->type_info.int_type.signedness) {
                wr_sb_appendf(out, "%di", (int32_t)info->constant.value[0]);
            } else {
                wr_sb_appendf(out, "%uu", info->constant.value[0]);
            }
        } else if (tk == SPV_TYPE_FLOAT) {
            union { uint32_t u; float f; } conv;
            conv.u = info->constant.value[0];
            if (conv.f >= -2147483648.0f && conv.f <= 2147483520.0f && conv.f == (float)(int)conv.f) {
                wr_sb_appendf(out, "%.1f", conv.f);
            } else {
                wr_sb_appendf(out, "%g", conv.f);
            }
        } else if (tk == SPV_TYPE_VECTOR) {
            const char *t = type_to_wgsl(r, type_id);
            wr_sb_appendf(out, "%s(", t);
            for (int i = 0; i < info->constant.value_count; i++) {
                if (i > 0) wr_sb_append(out, ", ");
                emit_expression(r, info->constant.value[i], out);
            }
            wr_sb_append(out, ")");
        } else {
            wr_sb_appendf(out, "%u", info->constant.value[0]);
        }
    } else {
        wr_sb_appendf(out, "%u", info->constant.value[0]);
    }
}

static const char *glsl_extinst_name(uint32_t inst) {
    switch (inst) {
    case GLSLstd450Sin: return "sin";
    case GLSLstd450Cos: return "cos";
    case GLSLstd450Tan: return "tan";
    case GLSLstd450Asin: return "asin";
    case GLSLstd450Acos: return "acos";
    case GLSLstd450Atan: return "atan";
    case GLSLstd450Sinh: return "sinh";
    case GLSLstd450Cosh: return "cosh";
    case GLSLstd450Tanh: return "tanh";
    case GLSLstd450Pow: return "pow";
    case GLSLstd450Exp: return "exp";
    case GLSLstd450Log: return "log";
    case GLSLstd450Exp2: return "exp2";
    case GLSLstd450Log2: return "log2";
    case GLSLstd450Sqrt: return "sqrt";
    case GLSLstd450InverseSqrt: return "inverseSqrt";
    case GLSLstd450FAbs: return "abs";
    case GLSLstd450SAbs: return "abs";
    case GLSLstd450FSign: return "sign";
    case GLSLstd450SSign: return "sign";
    case GLSLstd450Floor: return "floor";
    case GLSLstd450Ceil: return "ceil";
    case GLSLstd450Round: return "round";
    case GLSLstd450Trunc: return "trunc";
    case GLSLstd450Fract: return "fract";
    case GLSLstd450FMin: return "min";
    case GLSLstd450SMin: return "min";
    case GLSLstd450UMin: return "min";
    case GLSLstd450FMax: return "max";
    case GLSLstd450SMax: return "max";
    case GLSLstd450UMax: return "max";
    case GLSLstd450FClamp: return "clamp";
    case GLSLstd450SClamp: return "clamp";
    case GLSLstd450UClamp: return "clamp";
    case GLSLstd450FMix: return "mix";
    case GLSLstd450Step: return "step";
    case GLSLstd450SmoothStep: return "smoothstep";
    case GLSLstd450Length: return "length";
    case GLSLstd450Distance: return "distance";
    case GLSLstd450Cross: return "cross";
    case GLSLstd450Normalize: return "normalize";
    case GLSLstd450Reflect: return "reflect";
    case GLSLstd450Refract: return "refract";
    case GLSLstd450Determinant: return "determinant";
    case GLSLstd450MatrixInverse: return "inverse";
    default: return "unknown";
    }
}

//r nonnull
//out nonnull
static void emit_expression(WgslRaiser *r, uint32_t id, StringBuffer *out) {
    wgsl_compiler_assert(r != NULL, "emit_expression: r is NULL");
    wgsl_compiler_assert(out != NULL, "emit_expression: out is NULL");
    if (id >= r->id_bound) { wr_sb_appendf(out, "_id%u", id); return; }
    if (r->emit_depth > 256 || r->emit_calls > 100000 || out->len > 4*1024*1024) {
        wr_sb_append(out, "/*deep*/0"); return;
    }
    r->emit_depth++;
    r->emit_calls++;
    SpvIdInfo *info = &r->ids[id];

    switch (info->kind) {
    case SPV_ID_CONSTANT:
        emit_constant(r, id, out);
        break;
    case SPV_ID_VARIABLE:
        wr_sb_append(out, get_id_name(r, id));
        break;
    case SPV_ID_INSTRUCTION: {
        SpvOp op = info->instruction.opcode;
        uint32_t *ops = info->instruction.operands;
        int op_count = info->instruction.operand_count;

        switch (op) {
        case SpvOpLoad:
            if (op_count >= 1) emit_expression(r, ops[0], out);
            break;
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain:
            if (op_count >= 1) {
                emit_expression(r, ops[0], out);
                for (int i = 1; i < op_count; i++) {
                    uint32_t idx_id = ops[i];
                    if (idx_id < r->id_bound && r->ids[idx_id].kind == SPV_ID_CONSTANT) {
                        uint32_t idx = r->ids[idx_id].constant.value[0];
                        uint32_t base_id = ops[0];
                        if (base_id < r->id_bound && r->ids[base_id].kind == SPV_ID_VARIABLE) {
                            uint32_t ptr_type = r->ids[base_id].variable.type_id;
                            if (ptr_type < r->id_bound && r->ids[ptr_type].kind == SPV_ID_TYPE &&
                                r->ids[ptr_type].type_info.kind == SPV_TYPE_POINTER) {
                                uint32_t pointee = r->ids[ptr_type].type_info.pointer.pointee_type;
                                if (pointee < r->id_bound && r->ids[pointee].kind == SPV_ID_TYPE &&
                                    r->ids[pointee].type_info.kind == SPV_TYPE_STRUCT) {
                                    wr_sb_appendf(out, ".member%u", idx);
                                    continue;
                                }
                            }
                        }
                        wr_sb_appendf(out, "[%u]", idx);
                    } else {
                        wr_sb_append(out, "[");
                        emit_expression(r, idx_id, out);
                        wr_sb_append(out, "]");
                    }
                }
            }
            break;
        case SpvOpFAdd: case SpvOpIAdd:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " + ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFSub: case SpvOpISub:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " - ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFMul: case SpvOpIMul:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " * ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFDiv: case SpvOpSDiv: case SpvOpUDiv:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " / ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFRem: case SpvOpSRem: case SpvOpUMod:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " % ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFNegate: case SpvOpSNegate:
            if (op_count >= 1) {
                wr_sb_append(out, "(-");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdEqual: case SpvOpIEqual: case SpvOpLogicalEqual:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " == ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdNotEqual: case SpvOpINotEqual: case SpvOpLogicalNotEqual:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " != ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdLessThan: case SpvOpSLessThan: case SpvOpULessThan:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " < ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdGreaterThan: case SpvOpSGreaterThan: case SpvOpUGreaterThan:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " > ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdLessThanEqual: case SpvOpSLessThanEqual: case SpvOpULessThanEqual:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " <= ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpFOrdGreaterThanEqual: case SpvOpSGreaterThanEqual: case SpvOpUGreaterThanEqual:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " >= ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpLogicalAnd:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " && ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpLogicalOr:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " || ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpLogicalNot:
            if (op_count >= 1) {
                wr_sb_append(out, "(!");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpBitwiseAnd:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " & ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpBitwiseOr:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " | ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpBitwiseXor:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " ^ ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpNot:
            if (op_count >= 1) {
                wr_sb_append(out, "(~");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpShiftLeftLogical:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " << ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpShiftRightLogical: case SpvOpShiftRightArithmetic:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " >> ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpCompositeConstruct: {
            const char *t = type_to_wgsl(r, info->instruction.type_id);
            wr_sb_appendf(out, "%s(", t);
            for (int i = 0; i < op_count; i++) {
                if (i > 0) wr_sb_append(out, ", ");
                emit_expression(r, ops[i], out);
            }
            wr_sb_append(out, ")");
            break;
        }
        case SpvOpCompositeExtract:
            if (op_count >= 2) {
                emit_expression(r, ops[0], out);
                for (int i = 1; i < op_count; i++) {
                    uint32_t idx = ops[i];
                    const char *swiz = "xyzw";
                    if (idx < 4) {
                        wr_sb_appendf(out, ".%c", swiz[idx]);
                    } else {
                        wr_sb_appendf(out, "[%u]", idx);
                    }
                }
            }
            break;
        case SpvOpVectorShuffle:
            if (op_count >= 2) {
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ".");
                for (int i = 2; i < op_count; i++) {
                    uint32_t idx = ops[i];
                    const char *swiz = "xyzw";
                    if (idx < 4) {
                        wr_sb_appendf(out, "%c", swiz[idx]);
                    }
                }
            }
            break;
        case SpvOpConvertFToS:
            if (op_count >= 1) {
                wr_sb_append(out, "i32(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpConvertFToU:
            if (op_count >= 1) {
                wr_sb_append(out, "u32(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpConvertSToF: case SpvOpConvertUToF:
            if (op_count >= 1) {
                wr_sb_append(out, "f32(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpBitcast: case SpvOpFConvert: case SpvOpSConvert: case SpvOpUConvert: {
            const char *t = type_to_wgsl(r, info->instruction.type_id);
            if (op_count >= 1) {
                wr_sb_appendf(out, "%s(", t);
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        }
        case SpvOpSelect:
            if (op_count >= 3) {
                wr_sb_append(out, "select(");
                emit_expression(r, ops[2], out);
                wr_sb_append(out, ", ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ", ");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpDot:
            if (op_count >= 2) {
                wr_sb_append(out, "dot(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, ", ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpVectorTimesScalar:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " * ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpMatrixTimesScalar:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " * ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpVectorTimesMatrix:
        case SpvOpMatrixTimesVector:
        case SpvOpMatrixTimesMatrix:
            if (op_count >= 2) {
                wr_sb_append(out, "(");
                emit_expression(r, ops[0], out);
                wr_sb_append(out, " * ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpExtInst:
            if (op_count >= 2) {
                uint32_t inst = ops[1];
                const char *fn = glsl_extinst_name(inst);
                wr_sb_appendf(out, "%s(", fn);
                for (int i = 2; i < op_count; i++) {
                    if (i > 2) wr_sb_append(out, ", ");
                    emit_expression(r, ops[i], out);
                }
                wr_sb_append(out, ")");
            }
            break;
        case SpvOpSampledImage:
            if (op_count >= 1) {
                emit_expression(r, ops[0], out);
            }
            break;
        case SpvOpImageSampleImplicitLod:
        case SpvOpImageSampleExplicitLod:
            if (op_count >= 2) {
                wr_sb_append(out, "textureSample(");
                uint32_t sampled_img = ops[0];
                if (sampled_img < r->id_bound && r->ids[sampled_img].kind == SPV_ID_INSTRUCTION &&
                    r->ids[sampled_img].instruction.opcode == SpvOpSampledImage) {
                    uint32_t *si_ops = r->ids[sampled_img].instruction.operands;
                    // PRE: operands array valid for SampledImage instruction
                    wgsl_compiler_assert(si_ops != NULL, "SpvOpImageSampleExplicitLod: SampledImage operands NULL");
                    // PRE: SampledImage requires at least 2 operands
                    wgsl_compiler_assert(r->ids[sampled_img].instruction.operand_count >= 2, "SpvOpImageSampleExplicitLod: SampledImage operand_count < 2");
                    emit_expression(r, si_ops[0], out);
                    wr_sb_append(out, ", ");
                    emit_expression(r, si_ops[1], out);
                } else {
                    emit_expression(r, ops[0], out);
                    wr_sb_append(out, ", sampler_placeholder");
                }
                wr_sb_append(out, ", ");
                emit_expression(r, ops[1], out);
                wr_sb_append(out, ")");
            }
            break;
        default:
            wr_sb_appendf(out, "_id%u", id);
            break;
        }
        break;
    }
    default:
        wr_sb_append(out, get_id_name(r, id));
        break;
    }
    r->emit_depth--;
}

//fn nonnull
static SpvBasicBlock *find_block(SpvFunction *fn, uint32_t label_id) {
    wgsl_compiler_assert(fn != NULL, "find_block: fn is NULL");
    for (int i = 0; i < fn->block_count; i++) {
        if (fn->blocks[i].label_id == label_id) return &fn->blocks[i];
    }
    return NULL;
}

//r nonnull
//fn nonnull
//blk nonnull
static void emit_block(WgslRaiser *r, SpvFunction *fn, SpvBasicBlock *blk) {
    wgsl_compiler_assert(r != NULL, "emit_block: r is NULL");
    wgsl_compiler_assert(fn != NULL, "emit_block: fn is NULL");
    wgsl_compiler_assert(blk != NULL, "emit_block: blk is NULL");
    if (blk->emitted || r->sb.len > 4*1024*1024) return;
    blk->emitted = 1;

    // Handle loop header - emit as while loop
    if (blk->is_loop_header) {
        // Find the condition from the condition block
        for (int i = 0; i < blk->instruction_count; i++) {
            uint32_t instr_pos = blk->instructions[i];
            uint32_t word0 = r->spirv[instr_pos];
            uint16_t opcode = word0 & 0xFFFF;
            uint16_t wc = word0 >> 16;
            const uint32_t *operands = &r->spirv[instr_pos + 1];
            int operand_count = wc - 1;

            if ((SpvOp)opcode == SpvOpBranch && operand_count >= 1) {
                uint32_t cond_label = operands[0];
                SpvBasicBlock *cond_blk = find_block(fn, cond_label);
                if (cond_blk && !cond_blk->emitted) {
                    cond_blk->emitted = 1;
                    // Find the conditional branch in the condition block
                    for (int j = 0; j < cond_blk->instruction_count; j++) {
                        uint32_t cinstr_pos = cond_blk->instructions[j];
                        uint32_t cword0 = r->spirv[cinstr_pos];
                        uint16_t copcode = cword0 & 0xFFFF;
                        uint16_t cwc = cword0 >> 16;
                        const uint32_t *coperands = &r->spirv[cinstr_pos + 1];
                        int coperand_count = cwc - 1;

                        if ((SpvOp)copcode == SpvOpBranchConditional && coperand_count >= 3) {
                            // Emit: while (cond) { ... }
                            sb_indent(&r->sb);
                            wr_sb_append(&r->sb, "while (");
                            emit_expression(r, coperands[0], &r->sb);
                            wr_sb_append(&r->sb, ") {\n");
                            r->sb.indent++;

                            // Emit body block (true branch)
                            SpvBasicBlock *body_blk = find_block(fn, coperands[1]);
                            if (body_blk) emit_block(r, fn, body_blk);

                            // Emit continue block (loop increment)
                            SpvBasicBlock *cont_blk = find_block(fn, blk->continue_block);
                            if (cont_blk) emit_block(r, fn, cont_blk);

                            r->sb.indent--;
                            sb_indent(&r->sb);
                            wr_sb_append(&r->sb, "}\n");
                        }
                    }
                }
            }
        }

        // Continue to merge block
        SpvBasicBlock *merge_blk = find_block(fn, blk->merge_block);
        if (merge_blk) emit_block(r, fn, merge_blk);
        return;
    }

    for (int i = 0; i < blk->instruction_count; i++) {
        uint32_t instr_pos = blk->instructions[i];
        uint32_t word0 = r->spirv[instr_pos];
        uint16_t opcode = word0 & 0xFFFF;
        uint16_t wc = word0 >> 16;
        const uint32_t *operands = &r->spirv[instr_pos + 1];
        int operand_count = wc - 1;

        switch ((SpvOp)opcode) {
        case SpvOpStore:
            if (operand_count >= 2) {
                sb_indent(&r->sb);
                emit_expression(r, operands[0], &r->sb);
                wr_sb_append(&r->sb, " = ");
                emit_expression(r, operands[1], &r->sb);
                wr_sb_append(&r->sb, ";\n");
            }
            break;
        case SpvOpReturn:
            sb_indent(&r->sb);
            wr_sb_append(&r->sb, "return;\n");
            break;
        case SpvOpReturnValue:
            if (operand_count >= 1) {
                sb_indent(&r->sb);
                wr_sb_append(&r->sb, "return ");
                emit_expression(r, operands[0], &r->sb);
                wr_sb_append(&r->sb, ";\n");
            }
            break;
        case SpvOpBranch:
            break;
        case SpvOpBranchConditional:
            if (operand_count >= 3) {
                sb_indent(&r->sb);
                wr_sb_append(&r->sb, "if (");
                emit_expression(r, operands[0], &r->sb);
                wr_sb_append(&r->sb, ") {\n");
                r->sb.indent++;
                for (int j = 0; j < fn->block_count; j++) {
                    if (fn->blocks[j].label_id == operands[1]) {
                        emit_block(r, fn, &fn->blocks[j]);
                        break;
                    }
                }
                r->sb.indent--;
                sb_indent(&r->sb);
                wr_sb_append(&r->sb, "} else {\n");
                r->sb.indent++;
                for (int j = 0; j < fn->block_count; j++) {
                    if (fn->blocks[j].label_id == operands[2]) {
                        emit_block(r, fn, &fn->blocks[j]);
                        break;
                    }
                }
                r->sb.indent--;
                sb_indent(&r->sb);
                wr_sb_append(&r->sb, "}\n");
            }
            break;
        case SpvOpVariable:
            if (operand_count >= 3) {
                uint32_t type_id = operands[0];
                uint32_t var_id = operands[1];
                SpvStorageClass sc = (SpvStorageClass)operands[2];
                if (sc == SpvStorageClassFunction) {
                    uint32_t pointee = 0;
                    if (type_id < r->id_bound && r->ids[type_id].kind == SPV_ID_TYPE &&
                        r->ids[type_id].type_info.kind == SPV_TYPE_POINTER) {
                        pointee = r->ids[type_id].type_info.pointer.pointee_type;
                    }
                    const char *type_str = type_to_wgsl(r, pointee);
                    const char *name = get_id_name(r, var_id);
                    sb_indent(&r->sb);
                    wr_sb_appendf(&r->sb, "var %s: %s;\n", name, type_str);
                }
            }
            break;
        default:
            break;
        }
    }
}

//r nonnull
//fn nonnull
static void emit_function(WgslRaiser *r, SpvFunction *fn) {
    wgsl_compiler_assert(r != NULL, "emit_function: r is NULL");
    wgsl_compiler_assert(fn != NULL, "emit_function: fn is NULL");
    const char *name = fn->name ? fn->name : "_fn";

    if (fn->is_entry_point) {
        switch (fn->exec_model) {
        case SpvExecutionModelVertex:
            wr_sb_append(&r->sb, "@vertex ");
            break;
        case SpvExecutionModelFragment:
            wr_sb_append(&r->sb, "@fragment ");
            break;
        case SpvExecutionModelGLCompute:
            wr_sb_appendf(&r->sb, "@compute @workgroup_size(%d, %d, %d) ",
                       fn->workgroup_size[0], fn->workgroup_size[1], fn->workgroup_size[2]);
            break;
        default:
            break;
        }
    }

    wr_sb_appendf(&r->sb, "fn %s(", name);

    int first_param = 1;
    for (int i = 0; i < fn->interface_var_count; i++) {
        uint32_t var_id = fn->interface_vars[i];
        if (var_id >= r->id_bound) continue;
        SpvIdInfo *info = &r->ids[var_id];
        if (info->kind != SPV_ID_VARIABLE) continue;
        if (info->variable.storage_class != SpvStorageClassInput) continue;

        uint32_t ptr_type = info->variable.type_id;
        uint32_t pointee = 0;
        if (ptr_type < r->id_bound && r->ids[ptr_type].kind == SPV_ID_TYPE &&
            r->ids[ptr_type].type_info.kind == SPV_TYPE_POINTER) {
            pointee = r->ids[ptr_type].type_info.pointer.pointee_type;
        }

        if (!first_param) wr_sb_append(&r->sb, ", ");
        first_param = 0;

        uint32_t loc = 0, builtin_val = 0;
        int has_loc = has_decoration(r, var_id, SpvDecorationLocation, &loc);
        int has_builtin = has_decoration(r, var_id, SpvDecorationBuiltIn, &builtin_val);

        if (has_loc) {
            wr_sb_appendf(&r->sb, "@location(%u) ", loc);
        } else if (has_builtin) {
            wr_sb_appendf(&r->sb, "@builtin(%s) ", get_builtin_name((SpvBuiltIn)builtin_val));
        }

        if (!info->name) {
            info->name = (char*)WGSL_MALLOC(16);
            snprintf(info->name, 16, "_in%d", r->var_counter++);
        }
        const char *pname = info->name;
        const char *ptype = type_to_wgsl(r, pointee);
        wr_sb_appendf(&r->sb, "%s: %s", pname, ptype);
    }

    wr_sb_append(&r->sb, ")");

    int has_output = 0;
    for (int i = 0; i < fn->interface_var_count; i++) {
        uint32_t var_id = fn->interface_vars[i];
        if (var_id >= r->id_bound) continue;
        SpvIdInfo *info = &r->ids[var_id];
        if (info->kind != SPV_ID_VARIABLE) continue;
        if (info->variable.storage_class != SpvStorageClassOutput) continue;

        uint32_t ptr_type = info->variable.type_id;
        uint32_t pointee = 0;
        if (ptr_type < r->id_bound && r->ids[ptr_type].kind == SPV_ID_TYPE &&
            r->ids[ptr_type].type_info.kind == SPV_TYPE_POINTER) {
            pointee = r->ids[ptr_type].type_info.pointer.pointee_type;
        }

        wr_sb_append(&r->sb, " -> ");

        uint32_t loc = 0, builtin_val = 0;
        int has_loc = has_decoration(r, var_id, SpvDecorationLocation, &loc);
        int has_builtin = has_decoration(r, var_id, SpvDecorationBuiltIn, &builtin_val);

        if (has_loc) {
            wr_sb_appendf(&r->sb, "@location(%u) ", loc);
        } else if (has_builtin) {
            wr_sb_appendf(&r->sb, "@builtin(%s) ", get_builtin_name((SpvBuiltIn)builtin_val));
        }

        wr_sb_append(&r->sb, type_to_wgsl(r, pointee));
        has_output = 1;
        break;
    }

    if (!has_output && fn->return_type < r->id_bound) {
        SpvIdInfo *ret_info = &r->ids[fn->return_type];
        if (ret_info->kind == SPV_ID_TYPE && ret_info->type_info.kind != SPV_TYPE_VOID) {
            wr_sb_append(&r->sb, " -> ");
            wr_sb_append(&r->sb, type_to_wgsl(r, fn->return_type));
        }
    }

    wr_sb_append(&r->sb, " {\n");
    r->sb.indent++;

    // Emit function-local variable declarations
    for (uint32_t i = 0; i < r->id_bound; i++) {
        SpvIdInfo *info = &r->ids[i];
        if (info->kind == SPV_ID_VARIABLE &&
            info->variable.storage_class == SpvStorageClassFunction) {
            uint32_t ptr_type = info->variable.type_id;
            uint32_t pointee = 0;
            if (ptr_type < r->id_bound && r->ids[ptr_type].kind == SPV_ID_TYPE &&
                r->ids[ptr_type].type_info.kind == SPV_TYPE_POINTER) {
                pointee = r->ids[ptr_type].type_info.pointer.pointee_type;
            }
            const char *type_str = type_to_wgsl(r, pointee);
            const char *vname = get_id_name(r, i);
            sb_indent(&r->sb);
            wr_sb_appendf(&r->sb, "var %s: %s;\n", vname, type_str);
        }
    }

    // Reset emitted flags
    for (int i = 0; i < fn->block_count; i++) {
        fn->blocks[i].emitted = 0;
    }

    for (int i = 0; i < fn->block_count; i++) {
        emit_block(r, fn, &fn->blocks[i]);
    }

    r->sb.indent--;
    wr_sb_append(&r->sb, "}\n\n");
}

const char *wgsl_raise_emit(WgslRaiser *r, const WgslRaiseOptions *options) {
    if (!r) return NULL;

    sb_free(&r->sb);
    sb_init(&r->sb);

    for (uint32_t i = 0; i < r->id_bound; i++) {
        if (r->ids[i].kind == SPV_ID_TYPE && r->ids[i].type_info.kind == SPV_TYPE_STRUCT) {
            int is_interface_struct = 0;
            for (int j = 0; j < r->ids[i].member_decoration_count; j++) {
                SpvDecoration d = r->ids[i].member_decorations[j].decoration;
                if (d == SpvDecorationLocation || d == SpvDecorationBuiltIn) {
                    is_interface_struct = 1;
                    break;
                }
            }
            if (!is_interface_struct) {
                emit_struct_decl(r, i);
            }
        }
    }

    for (uint32_t i = 0; i < r->id_bound; i++) {
        if (r->ids[i].kind == SPV_ID_VARIABLE) {
            emit_global_var(r, i);
        }
    }
    sb_newline(&r->sb);

    for (int i = 0; i < r->function_count; i++) {
        emit_function(r, &r->functions[i]);
    }

    if (r->output) WGSL_FREE(r->output);
    r->output = (char*)WGSL_MALLOC(r->sb.len + 1);
    if (r->output) {
        memcpy(r->output, r->sb.data, r->sb.len);
        r->output[r->sb.len] = 0;
    }

    return r->output;
}

WgslRaiseResult wgsl_raise_to_wgsl(
    const uint32_t *spirv,
    size_t word_count,
    const WgslRaiseOptions *options,
    char **out_wgsl,
    char **out_error
) {
    if (out_wgsl) *out_wgsl = NULL;
    if (out_error) *out_error = NULL;

    WgslRaiser *r = wgsl_raise_create(spirv, word_count);
    if (!r) {
        if (out_error) {
            *out_error = (char*)WGSL_MALLOC(32);
            strcpy(*out_error, "Failed to create raiser");
        }
        return WGSL_RAISE_INVALID_SPIRV;
    }

    WgslRaiseResult res = wgsl_raise_parse(r);
    if (res != WGSL_RAISE_SUCCESS) {
        if (out_error && r->last_error[0]) {
            *out_error = (char*)WGSL_MALLOC(strlen(r->last_error) + 1);
            strcpy(*out_error, r->last_error);
        }
        wgsl_raise_destroy(r);
        return res;
    }

    res = wgsl_raise_analyze(r);
    if (res != WGSL_RAISE_SUCCESS) {
        if (out_error && r->last_error[0]) {
            *out_error = (char*)WGSL_MALLOC(strlen(r->last_error) + 1);
            strcpy(*out_error, r->last_error);
        }
        wgsl_raise_destroy(r);
        return res;
    }

    const char *wgsl = wgsl_raise_emit(r, options);
    if (!wgsl) {
        wgsl_raise_destroy(r);
        return WGSL_RAISE_INTERNAL_ERROR;
    }

    if (out_wgsl) {
        *out_wgsl = (char*)WGSL_MALLOC(strlen(wgsl) + 1);
        strcpy(*out_wgsl, wgsl);
    }

    wgsl_raise_destroy(r);
    return WGSL_RAISE_SUCCESS;
}

int wgsl_raise_entry_point_count(WgslRaiser *r) {
    if (!r) return 0;
    int count = 0;
    for (int i = 0; i < r->function_count; i++) {
        if (r->functions[i].is_entry_point) count++;
    }
    return count;
}

const char *wgsl_raise_entry_point_name(WgslRaiser *r, int index) {
    if (!r) return NULL;
    int count = 0;
    for (int i = 0; i < r->function_count; i++) {
        if (r->functions[i].is_entry_point) {
            if (count == index) return r->functions[i].name;
            count++;
        }
    }
    return NULL;
}

void wgsl_raise_free(void *p) {
    WGSL_FREE(p);
}
