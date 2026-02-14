/*
 * simple_wgsl.h - Unified Header
 *
 * A WGSL (WebGPU Shading Language) compiler library implementing:
 *   WGSL Source <-> AST <-> Resolver -> SSIR <-> SPIR-V
 *
 * Include this single header to access all functionality.
 */

#ifndef SIMPLE_WGSL_H
#define SIMPLE_WGSL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory Allocation Macros (Customizable)
 * ============================================================================ */

#ifndef NODE_ALLOC
#define NODE_ALLOC(T) (T *)calloc(1, sizeof(T))
#endif
#ifndef NODE_MALLOC
#define NODE_MALLOC(SZ) calloc(1, (SZ))
#endif
#ifndef NODE_REALLOC
#define NODE_REALLOC(P, SZ) realloc((P), (SZ))
#endif
#ifndef NODE_FREE
#define NODE_FREE(P) free((P))
#endif

#ifndef SSIR_MALLOC
#define SSIR_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef SSIR_REALLOC
#define SSIR_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef SSIR_FREE
#define SSIR_FREE(p) free((p))
#endif

/* ============================================================================
 * Compiler Assert Macro (libc-independent portable trap)
 * ============================================================================ */

#ifndef WGSL_COMPILER_TRAP
#if defined(__GNUC__) || defined(__clang__)
#define WGSL_COMPILER_TRAP() __builtin_trap()
#elif defined(_MSC_VER)
#define WGSL_COMPILER_TRAP() __debugbreak()
#else
#define WGSL_COMPILER_TRAP() (*(volatile int*)0 = 0)
#endif
#endif

#ifndef wgsl_compiler_assert
#define wgsl_compiler_assert(cond, fmt, ...) do { \
    if (!(cond)) { WGSL_COMPILER_TRAP(); } \
} while(0)
#endif

/* ============================================================================
 * WGSL PARSER
 * ============================================================================ */

typedef enum WgslNodeType {
    WGSL_NODE_PROGRAM = 1,
    WGSL_NODE_STRUCT,
    WGSL_NODE_STRUCT_FIELD,
    WGSL_NODE_GLOBAL_VAR,
    WGSL_NODE_FUNCTION,
    WGSL_NODE_PARAM,
    WGSL_NODE_TYPE,
    WGSL_NODE_ATTRIBUTE,
    WGSL_NODE_BLOCK,
    WGSL_NODE_VAR_DECL,
    WGSL_NODE_RETURN,
    WGSL_NODE_EXPR_STMT,
    WGSL_NODE_IF,
    WGSL_NODE_WHILE,
    WGSL_NODE_FOR,
    WGSL_NODE_IDENT,
    WGSL_NODE_LITERAL,
    WGSL_NODE_BINARY,
    WGSL_NODE_ASSIGN,
    WGSL_NODE_CALL,
    WGSL_NODE_MEMBER,
    WGSL_NODE_INDEX,
    WGSL_NODE_UNARY,
    WGSL_NODE_TERNARY,
    WGSL_NODE_DO_WHILE,
    WGSL_NODE_SWITCH,
    WGSL_NODE_CASE,
    WGSL_NODE_BREAK,
    WGSL_NODE_CONTINUE,
    WGSL_NODE_DISCARD
} WgslNodeType;

typedef struct WgslAstNode WgslAstNode;

typedef struct Attribute {
    char *name;
    int arg_count;
    WgslAstNode **args;
} Attribute;

typedef struct TypeNode {
    char *name;
    int type_arg_count;
    WgslAstNode **type_args;
    int expr_arg_count;
    WgslAstNode **expr_args;
} TypeNode;

typedef struct StructField {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    WgslAstNode *type;
} StructField;

typedef struct StructDecl {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    int field_count;
    WgslAstNode **fields;
} StructDecl;

typedef struct GlobalVar {
    int attr_count;
    WgslAstNode **attrs;
    char *address_space;
    char *name;
    WgslAstNode *type;
} GlobalVar;

typedef struct Param {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    WgslAstNode *type;
} Param;

typedef struct Block {
    int stmt_count;
    WgslAstNode **stmts;
} Block;

typedef enum {
    WGSL_DECL_VAR = 0,
    WGSL_DECL_LET,
    WGSL_DECL_CONST,
    WGSL_DECL_OVERRIDE,
} WgslDeclKind;

typedef struct VarDecl {
    char *name;
    WgslAstNode *type;
    WgslAstNode *init;
    WgslDeclKind kind;
    int attr_count;
    WgslAstNode **attrs;
} VarDecl;

typedef struct ReturnNode {
    WgslAstNode *expr;
} ReturnNode;

typedef struct ExprStmt {
    WgslAstNode *expr;
} ExprStmt;

typedef struct Function {
    int attr_count;
    WgslAstNode **attrs;
    char *name;
    int param_count;
    WgslAstNode **params;
    int ret_attr_count;
    WgslAstNode **ret_attrs;
    WgslAstNode *return_type;
    WgslAstNode *body;
} Function;

typedef struct IfStmt {
    WgslAstNode *cond;
    WgslAstNode *then_branch;
    WgslAstNode *else_branch;
} IfStmt;

typedef struct WhileStmt {
    WgslAstNode *cond;
    WgslAstNode *body;
} WhileStmt;

typedef struct ForStmt {
    WgslAstNode *init;
    WgslAstNode *cond;
    WgslAstNode *cont;
    WgslAstNode *body;
} ForStmt;

typedef struct Ident {
    char *name;
} Ident;

typedef enum WgslLiteralKind { WGSL_LIT_INT, WGSL_LIT_FLOAT } WgslLiteralKind;

typedef struct Literal {
    WgslLiteralKind kind;
    char *lexeme;
} Literal;

typedef struct Binary {
    char *op;
    WgslAstNode *left;
    WgslAstNode *right;
} Binary;

typedef struct Assign {
    char *op; /* "=" or "+=" etc. NULL treated as "=" */
    WgslAstNode *lhs;
    WgslAstNode *rhs;
} Assign;

typedef struct Call {
    WgslAstNode *callee;
    int arg_count;
    WgslAstNode **args;
} Call;

typedef struct Member {
    WgslAstNode *object;
    char *member;
} Member;

typedef struct Index {
    WgslAstNode *object;
    WgslAstNode *index;
} Index;

typedef struct Unary {
    char *op;
    int is_postfix;
    WgslAstNode *expr;
} Unary;

typedef struct Ternary {
    WgslAstNode *cond;
    WgslAstNode *then_expr;
    WgslAstNode *else_expr;
} Ternary;

typedef struct DoWhileStmt {
    WgslAstNode *body;
    WgslAstNode *cond;
} DoWhileStmt;

typedef struct SwitchStmt {
    WgslAstNode *expr;
    int case_count;
    WgslAstNode **cases;
} SwitchStmt;

typedef struct CaseClause {
    WgslAstNode *expr; /* NULL for default */
    int stmt_count;
    WgslAstNode **stmts;
} CaseClause;

typedef struct Program {
    int decl_count;
    WgslAstNode **decls;
} Program;

struct WgslAstNode {
    WgslNodeType type;
    int line;
    int col;
    union {
        Program program;
        Attribute attribute;
        TypeNode type_node;
        StructDecl struct_decl;
        StructField struct_field;
        GlobalVar global_var;
        Function function;
        Param param;
        Block block;
        VarDecl var_decl;
        ReturnNode return_stmt;
        ExprStmt expr_stmt;
        IfStmt if_stmt;
        WhileStmt while_stmt;
        ForStmt for_stmt;
        Ident ident;
        Literal literal;
        Binary binary;
        Assign assign;
        Call call;
        Member member;
        Index index;
        Unary unary;
        Ternary ternary;
        DoWhileStmt do_while_stmt;
        SwitchStmt switch_stmt;
        CaseClause case_clause;
    };
};

WgslAstNode *wgsl_parse(const char *source);
void wgsl_free_ast(WgslAstNode *node);
const char *wgsl_node_type_name(WgslNodeType t);
void wgsl_debug_print(const WgslAstNode *node, int indent);

/* Shader Stage (also used by resolver, declared early for glsl_parse) */
typedef enum WgslStage{
    WGSL_STAGE_UNKNOWN = 0,
    WGSL_STAGE_VERTEX,
    WGSL_STAGE_FRAGMENT,
    WGSL_STAGE_COMPUTE
} WgslStage;

/* GLSL Parser */
WgslAstNode *glsl_parse(const char *source, WgslStage stage);

/* ============================================================================
 * WGSL RESOLVER
 * ============================================================================ */

typedef enum {
    WGSL_SYM_GLOBAL = 1,
    WGSL_SYM_PARAM,
    WGSL_SYM_LOCAL
} WgslSymbolKind;

typedef struct {
    int id;
    WgslSymbolKind kind;
    const char *name;
    int has_group;
    int group_index;
    int has_binding;
    int binding_index;
    int has_min_binding_size;
    int min_binding_size;
    const WgslAstNode* decl_node;
    const WgslAstNode* function_node;
} WgslSymbolInfo;

typedef enum {
    WGSL_NUM_UNKNOWN = 0,
    WGSL_NUM_F32,
    WGSL_NUM_I32,
    WGSL_NUM_U32,
    WGSL_NUM_F16,
    WGSL_NUM_BOOL
} WgslNumericType;

typedef struct WgslVertexSlot{
    int location;
    int component_count;
    WgslNumericType numeric_type;
    int byte_size;
} WgslVertexSlot;

typedef struct WgslFragmentOutput{
    int location;
    int component_count;
    WgslNumericType numeric_type;
} WgslFragmentOutput;

typedef struct WgslResolverEntrypoint{
    const char *name;
    WgslStage stage;
    const WgslAstNode *function_node;
} WgslResolverEntrypoint;

typedef struct WgslResolver WgslResolver;

WgslResolver *wgsl_resolver_build(const WgslAstNode *program);
void wgsl_resolver_free(WgslResolver *r);

const WgslSymbolInfo *wgsl_resolver_all_symbols(const WgslResolver *r, int *out_count);
const WgslSymbolInfo *wgsl_resolver_globals(const WgslResolver *r, int *out_count);
const WgslSymbolInfo *wgsl_resolver_binding_vars(const WgslResolver *r, int *out_count);

int wgsl_resolver_ident_symbol_id(const WgslResolver *r, const WgslAstNode *ident_node);

int wgsl_resolver_vertex_inputs(const WgslResolver *r, const char *vertex_entry_name, WgslVertexSlot **out_slots);

const WgslResolverEntrypoint *wgsl_resolver_entrypoints(const WgslResolver *r, int *out_count);

const WgslSymbolInfo *wgsl_resolver_entrypoint_globals(const WgslResolver *r, const char *entry_name, int *out_count);
const WgslSymbolInfo *wgsl_resolver_entrypoint_binding_vars(const WgslResolver *r, const char *entry_name, int *out_count);

int wgsl_resolver_fragment_outputs(const WgslResolver *r, const char *fragment_entry_name, WgslFragmentOutput **frag_outputs);

void wgsl_resolve_free(void *p);

/* ============================================================================
 * WGSL LOWER (WGSL -> SPIR-V)
 * ============================================================================ */

typedef struct WgslLower WgslLower;

typedef enum {
    WGSL_LOWER_OK = 0,
    WGSL_LOWER_ERR_INVALID_INPUT,
    WGSL_LOWER_ERR_UNSUPPORTED,
    WGSL_LOWER_ERR_INTERNAL,
    WGSL_LOWER_ERR_OOM
} WgslLowerResult;

typedef enum {
    WGSL_LOWER_ENV_VULKAN_1_1 = 1,
    WGSL_LOWER_ENV_VULKAN_1_2,
    WGSL_LOWER_ENV_VULKAN_1_3,
    WGSL_LOWER_ENV_WEBGPU
} WgslLowerEnv;

typedef enum {
    WGSL_LOWER_PACK_DEFAULT = 0,
    WGSL_LOWER_PACK_STD430,
    WGSL_LOWER_PACK_STD140
} WgslLowerPacking;

typedef struct {
    uint32_t spirv_version;
    WgslLowerEnv env;
    WgslLowerPacking packing;
    int enable_debug_names;
    int enable_line_info;
    int zero_initialize_vars;
    int relax_block_layout;
    int use_khr_shader_draw_parameters;
    uint32_t id_bound_hint;
} WgslLowerOptions;

typedef struct {
    const char *name;
    WgslStage stage;
    uint32_t function_id;
    uint32_t interface_count;
    const uint32_t *interface_ids;
} WgslLowerEntrypointInfo;

typedef struct {
    uint32_t capability_count;
    const uint32_t *capabilities;
    uint32_t extension_count;
    const char *const *extensions;
} WgslLowerModuleFeatures;

WgslLower *wgsl_lower_create(const WgslAstNode *program,
                             const WgslResolver *resolver,
                             const WgslLowerOptions *opts);

void wgsl_lower_destroy(WgslLower *lower);

WgslLowerResult wgsl_lower_emit_spirv(const WgslAstNode *program,
                                      const WgslResolver *resolver,
                                      const WgslLowerOptions *opts,
                                      uint32_t **out_words,
                                      size_t *out_word_count);

WgslLowerResult wgsl_lower_serialize(const WgslLower *lower,
                                     uint32_t **out_words,
                                     size_t *out_word_count);

WgslLowerResult wgsl_lower_serialize_into(const WgslLower *lower,
                                          uint32_t *out_words,
                                          size_t max_words,
                                          size_t *out_written);

const char *wgsl_lower_last_error(const WgslLower *lower);

const WgslLowerModuleFeatures *wgsl_lower_module_features(const WgslLower *lower);

const WgslLowerEntrypointInfo *wgsl_lower_entrypoints(const WgslLower *lower, int *out_count);

uint32_t wgsl_lower_node_result_id(const WgslLower *lower, const WgslAstNode *node);

uint32_t wgsl_lower_symbol_result_id(const WgslLower *lower, int symbol_id);

void wgsl_lower_free(void *p);

struct SsirModule;
const struct SsirModule *wgsl_lower_get_ssir(const WgslLower *lower);

/* ============================================================================
 * WGSL RAISE (SPIR-V -> WGSL)
 * ============================================================================ */

typedef struct WgslRaiser WgslRaiser;

typedef struct WgslRaiseOptions {
    int emit_debug_comments;
    int preserve_names;
    int inline_constants;
} WgslRaiseOptions;

typedef enum WgslRaiseResult {
    WGSL_RAISE_SUCCESS = 0,
    WGSL_RAISE_INVALID_SPIRV,
    WGSL_RAISE_UNSUPPORTED_FEATURE,
    WGSL_RAISE_INTERNAL_ERROR
} WgslRaiseResult;

WgslRaiseResult wgsl_raise_to_wgsl(
    const uint32_t *spirv,
    size_t word_count,
    const WgslRaiseOptions *options,
    char **out_wgsl,
    char **out_error
);

WgslRaiser *wgsl_raise_create(const uint32_t *spirv, size_t word_count);
WgslRaiseResult wgsl_raise_parse(WgslRaiser *r);
WgslRaiseResult wgsl_raise_analyze(WgslRaiser *r);
const char *wgsl_raise_emit(WgslRaiser *r, const WgslRaiseOptions *options);
void wgsl_raise_destroy(WgslRaiser *r);

int wgsl_raise_entry_point_count(WgslRaiser *r);
const char *wgsl_raise_entry_point_name(WgslRaiser *r, int index);

void wgsl_raise_free(void *p);

/* ============================================================================
 * SSIR (Simple Shader IR)
 * ============================================================================ */

typedef struct SsirModule SsirModule;
typedef struct SsirType SsirType;
typedef struct SsirInst SsirInst;
typedef struct SsirFunction SsirFunction;
typedef struct SsirBlock SsirBlock;
typedef struct SsirConstant SsirConstant;
typedef struct SsirGlobalVar SsirGlobalVar;
typedef struct SsirEntryPoint SsirEntryPoint;

/* Type System */

typedef enum SsirTypeKind {
    SSIR_TYPE_VOID,
    SSIR_TYPE_BOOL,
    SSIR_TYPE_I32,
    SSIR_TYPE_U32,
    SSIR_TYPE_F32,
    SSIR_TYPE_F16,
    SSIR_TYPE_F64,
    SSIR_TYPE_I8,
    SSIR_TYPE_U8,
    SSIR_TYPE_I16,
    SSIR_TYPE_U16,
    SSIR_TYPE_I64,
    SSIR_TYPE_U64,
    SSIR_TYPE_VEC,
    SSIR_TYPE_MAT,
    SSIR_TYPE_ARRAY,
    SSIR_TYPE_RUNTIME_ARRAY,
    SSIR_TYPE_STRUCT,
    SSIR_TYPE_PTR,
    SSIR_TYPE_SAMPLER,
    SSIR_TYPE_SAMPLER_COMPARISON,
    SSIR_TYPE_TEXTURE,
    SSIR_TYPE_TEXTURE_STORAGE,
    SSIR_TYPE_TEXTURE_DEPTH,
} SsirTypeKind;

typedef enum SsirAddressSpace {
    SSIR_ADDR_FUNCTION,
    SSIR_ADDR_PRIVATE,
    SSIR_ADDR_WORKGROUP,
    SSIR_ADDR_UNIFORM,
    SSIR_ADDR_UNIFORM_CONSTANT,
    SSIR_ADDR_STORAGE,
    SSIR_ADDR_INPUT,
    SSIR_ADDR_OUTPUT,
    SSIR_ADDR_PUSH_CONSTANT,
    SSIR_ADDR_PHYSICAL_STORAGE_BUFFER,
} SsirAddressSpace;

typedef enum SsirLayoutRule {
    SSIR_LAYOUT_NONE,
    SSIR_LAYOUT_STD140,
    SSIR_LAYOUT_STD430,
    SSIR_LAYOUT_SCALAR,
} SsirLayoutRule;

typedef enum SsirClipSpaceConvention {
    SSIR_CLIP_SPACE_VULKAN,   /* Y-down, Z [0,1] */
    SSIR_CLIP_SPACE_OPENGL,   /* Y-up, Z [-1,1] */
    SSIR_CLIP_SPACE_DIRECTX,  /* Y-up, Z [0,1] */
    SSIR_CLIP_SPACE_METAL,    /* Y-up, Z [0,1] (same as DirectX) */
} SsirClipSpaceConvention;

typedef enum SsirTextureDim {
    SSIR_TEX_1D,
    SSIR_TEX_2D,
    SSIR_TEX_3D,
    SSIR_TEX_CUBE,
    SSIR_TEX_2D_ARRAY,
    SSIR_TEX_CUBE_ARRAY,
    SSIR_TEX_MULTISAMPLED_2D,
    SSIR_TEX_1D_ARRAY,
    SSIR_TEX_BUFFER,
    SSIR_TEX_MULTISAMPLED_2D_ARRAY,
} SsirTextureDim;

typedef enum SsirAccessMode {
    SSIR_ACCESS_READ,
    SSIR_ACCESS_WRITE,
    SSIR_ACCESS_READ_WRITE,
} SsirAccessMode;

typedef enum SsirInterpolation {
    SSIR_INTERP_NONE,
    SSIR_INTERP_PERSPECTIVE,
    SSIR_INTERP_LINEAR,
    SSIR_INTERP_FLAT,
} SsirInterpolation;

typedef enum SsirInterpolationSampling {
    SSIR_INTERP_SAMPLING_NONE,
    SSIR_INTERP_SAMPLING_CENTER,
    SSIR_INTERP_SAMPLING_CENTROID,
    SSIR_INTERP_SAMPLING_SAMPLE,
} SsirInterpolationSampling;

typedef enum SsirBuiltinVar {
    SSIR_BUILTIN_NONE = 0,
    SSIR_BUILTIN_VERTEX_INDEX,
    SSIR_BUILTIN_INSTANCE_INDEX,
    SSIR_BUILTIN_POSITION,
    SSIR_BUILTIN_FRONT_FACING,
    SSIR_BUILTIN_FRAG_DEPTH,
    SSIR_BUILTIN_SAMPLE_INDEX,
    SSIR_BUILTIN_SAMPLE_MASK,
    SSIR_BUILTIN_LOCAL_INVOCATION_ID,
    SSIR_BUILTIN_LOCAL_INVOCATION_INDEX,
    SSIR_BUILTIN_GLOBAL_INVOCATION_ID,
    SSIR_BUILTIN_WORKGROUP_ID,
    SSIR_BUILTIN_NUM_WORKGROUPS,
    SSIR_BUILTIN_POINT_SIZE,
    SSIR_BUILTIN_CLIP_DISTANCE,
    SSIR_BUILTIN_CULL_DISTANCE,
    SSIR_BUILTIN_LAYER,
    SSIR_BUILTIN_VIEWPORT_INDEX,
    SSIR_BUILTIN_FRAG_COORD,
    SSIR_BUILTIN_HELPER_INVOCATION,
    SSIR_BUILTIN_PRIMITIVE_ID,
    SSIR_BUILTIN_BASE_VERTEX,
    SSIR_BUILTIN_BASE_INSTANCE,
    SSIR_BUILTIN_SUBGROUP_SIZE,
    SSIR_BUILTIN_SUBGROUP_INVOCATION_ID,
    SSIR_BUILTIN_SUBGROUP_ID,
    SSIR_BUILTIN_NUM_SUBGROUPS,
} SsirBuiltinVar;

typedef enum SsirStage {
    SSIR_STAGE_VERTEX,
    SSIR_STAGE_FRAGMENT,
    SSIR_STAGE_COMPUTE,
    SSIR_STAGE_GEOMETRY,
    SSIR_STAGE_TESS_CONTROL,
    SSIR_STAGE_TESS_EVAL,
} SsirStage;

struct SsirType {
    uint32_t id;
    SsirTypeKind kind;
    union {
        struct {
            uint32_t elem;
            uint8_t size;
        } vec;
        struct {
            uint32_t elem;
            uint8_t cols;
            uint8_t rows;
        } mat;
        struct {
            uint32_t elem;
            uint32_t length;
            uint32_t stride;
        } array;
        struct {
            uint32_t elem;
        } runtime_array;
        struct {
            const char *name;
            uint32_t *members;
            uint32_t member_count;
            uint32_t *offsets;
            const char **member_names;
            uint8_t *matrix_major;    /* 0=unset, 1=col-major, 2=row-major */
            uint32_t *matrix_strides; /* 0=unset, else MatrixStride value */
            SsirLayoutRule layout_rule;
        } struc;
        struct {
            uint32_t pointee;
            SsirAddressSpace space;
        } ptr;
        struct {
            SsirTextureDim dim;
            uint32_t sampled_type;
        } texture;
        struct {
            SsirTextureDim dim;
            uint32_t format;
            SsirAccessMode access;
        } texture_storage;
        struct {
            SsirTextureDim dim;
        } texture_depth;
    };
};

/* Instructions */

typedef enum SsirOpcode {
    SSIR_OP_ADD,
    SSIR_OP_SUB,
    SSIR_OP_MUL,
    SSIR_OP_DIV,
    SSIR_OP_MOD,
    SSIR_OP_NEG,
    SSIR_OP_MAT_MUL,
    SSIR_OP_MAT_TRANSPOSE,
    SSIR_OP_BIT_AND,
    SSIR_OP_BIT_OR,
    SSIR_OP_BIT_XOR,
    SSIR_OP_BIT_NOT,
    SSIR_OP_SHL,
    SSIR_OP_SHR,
    SSIR_OP_SHR_LOGICAL,
    SSIR_OP_EQ,
    SSIR_OP_NE,
    SSIR_OP_LT,
    SSIR_OP_LE,
    SSIR_OP_GT,
    SSIR_OP_GE,
    SSIR_OP_AND,
    SSIR_OP_OR,
    SSIR_OP_NOT,
    SSIR_OP_CONSTRUCT,
    SSIR_OP_EXTRACT,
    SSIR_OP_INSERT,
    SSIR_OP_SHUFFLE,
    SSIR_OP_SPLAT,
    SSIR_OP_EXTRACT_DYN,
    SSIR_OP_INSERT_DYN,
    SSIR_OP_LOAD,
    SSIR_OP_STORE,
    SSIR_OP_ACCESS,
    SSIR_OP_ARRAY_LEN,
    SSIR_OP_BRANCH,
    SSIR_OP_BRANCH_COND,
    SSIR_OP_SWITCH,
    SSIR_OP_PHI,
    SSIR_OP_RETURN,
    SSIR_OP_RETURN_VOID,
    SSIR_OP_UNREACHABLE,
    SSIR_OP_CALL,
    SSIR_OP_BUILTIN,
    SSIR_OP_CONVERT,
    SSIR_OP_BITCAST,
    SSIR_OP_TEX_SAMPLE,
    SSIR_OP_TEX_SAMPLE_BIAS,
    SSIR_OP_TEX_SAMPLE_LEVEL,
    SSIR_OP_TEX_SAMPLE_GRAD,
    SSIR_OP_TEX_SAMPLE_CMP,
    SSIR_OP_TEX_SAMPLE_CMP_LEVEL,
    SSIR_OP_TEX_SAMPLE_OFFSET,
    SSIR_OP_TEX_SAMPLE_BIAS_OFFSET,
    SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET,
    SSIR_OP_TEX_SAMPLE_GRAD_OFFSET,
    SSIR_OP_TEX_SAMPLE_CMP_OFFSET,
    SSIR_OP_TEX_GATHER,
    SSIR_OP_TEX_GATHER_CMP,
    SSIR_OP_TEX_GATHER_OFFSET,
    SSIR_OP_TEX_LOAD,
    SSIR_OP_TEX_STORE,
    SSIR_OP_TEX_SIZE,
    SSIR_OP_TEX_QUERY_LOD,
    SSIR_OP_TEX_QUERY_LEVELS,
    SSIR_OP_TEX_QUERY_SAMPLES,
    SSIR_OP_BARRIER,
    SSIR_OP_ATOMIC,
    SSIR_OP_LOOP_MERGE,
    SSIR_OP_DISCARD,
    SSIR_OP_REM,
    SSIR_OP_SELECTION_MERGE,
    SSIR_OP_COUNT
} SsirOpcode;

typedef enum SsirBuiltinId {
    SSIR_BUILTIN_SIN,
    SSIR_BUILTIN_COS,
    SSIR_BUILTIN_TAN,
    SSIR_BUILTIN_ASIN,
    SSIR_BUILTIN_ACOS,
    SSIR_BUILTIN_ATAN,
    SSIR_BUILTIN_ATAN2,
    SSIR_BUILTIN_SINH,
    SSIR_BUILTIN_COSH,
    SSIR_BUILTIN_TANH,
    SSIR_BUILTIN_ASINH,
    SSIR_BUILTIN_ACOSH,
    SSIR_BUILTIN_ATANH,
    SSIR_BUILTIN_EXP,
    SSIR_BUILTIN_EXP2,
    SSIR_BUILTIN_LOG,
    SSIR_BUILTIN_LOG2,
    SSIR_BUILTIN_POW,
    SSIR_BUILTIN_SQRT,
    SSIR_BUILTIN_INVERSESQRT,
    SSIR_BUILTIN_ABS,
    SSIR_BUILTIN_SIGN,
    SSIR_BUILTIN_FLOOR,
    SSIR_BUILTIN_CEIL,
    SSIR_BUILTIN_ROUND,
    SSIR_BUILTIN_TRUNC,
    SSIR_BUILTIN_FRACT,
    SSIR_BUILTIN_MIN,
    SSIR_BUILTIN_MAX,
    SSIR_BUILTIN_CLAMP,
    SSIR_BUILTIN_SATURATE,
    SSIR_BUILTIN_MIX,
    SSIR_BUILTIN_STEP,
    SSIR_BUILTIN_SMOOTHSTEP,
    SSIR_BUILTIN_DOT,
    SSIR_BUILTIN_CROSS,
    SSIR_BUILTIN_LENGTH,
    SSIR_BUILTIN_DISTANCE,
    SSIR_BUILTIN_NORMALIZE,
    SSIR_BUILTIN_FACEFORWARD,
    SSIR_BUILTIN_REFLECT,
    SSIR_BUILTIN_REFRACT,
    SSIR_BUILTIN_ALL,
    SSIR_BUILTIN_ANY,
    SSIR_BUILTIN_SELECT,
    SSIR_BUILTIN_COUNTBITS,
    SSIR_BUILTIN_REVERSEBITS,
    SSIR_BUILTIN_FIRSTLEADINGBIT,
    SSIR_BUILTIN_FIRSTTRAILINGBIT,
    SSIR_BUILTIN_EXTRACTBITS,
    SSIR_BUILTIN_INSERTBITS,
    SSIR_BUILTIN_DPDX,
    SSIR_BUILTIN_DPDY,
    SSIR_BUILTIN_FWIDTH,
    SSIR_BUILTIN_DPDX_COARSE,
    SSIR_BUILTIN_DPDY_COARSE,
    SSIR_BUILTIN_DPDX_FINE,
    SSIR_BUILTIN_DPDY_FINE,
    SSIR_BUILTIN_FMA,
    SSIR_BUILTIN_ISINF,
    SSIR_BUILTIN_ISNAN,
    SSIR_BUILTIN_DEGREES,
    SSIR_BUILTIN_RADIANS,
    SSIR_BUILTIN_MODF,
    SSIR_BUILTIN_FREXP,
    SSIR_BUILTIN_LDEXP,
    SSIR_BUILTIN_DETERMINANT,
    SSIR_BUILTIN_TRANSPOSE,
    SSIR_BUILTIN_PACK4X8SNORM,
    SSIR_BUILTIN_PACK4X8UNORM,
    SSIR_BUILTIN_PACK2X16SNORM,
    SSIR_BUILTIN_PACK2X16UNORM,
    SSIR_BUILTIN_PACK2X16FLOAT,
    SSIR_BUILTIN_UNPACK4X8SNORM,
    SSIR_BUILTIN_UNPACK4X8UNORM,
    SSIR_BUILTIN_UNPACK2X16SNORM,
    SSIR_BUILTIN_UNPACK2X16UNORM,
    SSIR_BUILTIN_UNPACK2X16FLOAT,
    /* Subgroup operations */
    SSIR_BUILTIN_SUBGROUP_BALLOT,
    SSIR_BUILTIN_SUBGROUP_BROADCAST,
    SSIR_BUILTIN_SUBGROUP_ADD,
    SSIR_BUILTIN_SUBGROUP_MIN,
    SSIR_BUILTIN_SUBGROUP_MAX,
    SSIR_BUILTIN_SUBGROUP_ALL,
    SSIR_BUILTIN_SUBGROUP_ANY,
    SSIR_BUILTIN_SUBGROUP_SHUFFLE,
    SSIR_BUILTIN_SUBGROUP_PREFIX_ADD,
    SSIR_BUILTIN_COUNT
} SsirBuiltinId;

typedef enum SsirAtomicOp {
    SSIR_ATOMIC_LOAD,
    SSIR_ATOMIC_STORE,
    SSIR_ATOMIC_ADD,
    SSIR_ATOMIC_SUB,
    SSIR_ATOMIC_MAX,
    SSIR_ATOMIC_MIN,
    SSIR_ATOMIC_AND,
    SSIR_ATOMIC_OR,
    SSIR_ATOMIC_XOR,
    SSIR_ATOMIC_EXCHANGE,
    SSIR_ATOMIC_COMPARE_EXCHANGE,
} SsirAtomicOp;

typedef enum SsirMemoryScope {
    SSIR_SCOPE_DEVICE,
    SSIR_SCOPE_WORKGROUP,
    SSIR_SCOPE_SUBGROUP,
    SSIR_SCOPE_INVOCATION,
} SsirMemoryScope;

typedef enum SsirMemorySemantics {
    SSIR_SEMANTICS_RELAXED,
    SSIR_SEMANTICS_ACQUIRE,
    SSIR_SEMANTICS_RELEASE,
    SSIR_SEMANTICS_ACQUIRE_RELEASE,
    SSIR_SEMANTICS_SEQ_CST,
} SsirMemorySemantics;

typedef enum SsirBarrierScope {
    SSIR_BARRIER_WORKGROUP,
    SSIR_BARRIER_STORAGE,
    SSIR_BARRIER_SUBGROUP,
    SSIR_BARRIER_IMAGE,
} SsirBarrierScope;

#define SSIR_MAX_OPERANDS 8

struct SsirInst {
    SsirOpcode op;
    uint32_t result;
    uint32_t type;
    uint32_t operands[SSIR_MAX_OPERANDS];
    uint8_t operand_count;
    uint32_t *extra;
    uint16_t extra_count;
};

/* Constants */

typedef enum SsirConstantKind {
    SSIR_CONST_BOOL,
    SSIR_CONST_I32,
    SSIR_CONST_U32,
    SSIR_CONST_F32,
    SSIR_CONST_F16,
    SSIR_CONST_F64,
    SSIR_CONST_I8,
    SSIR_CONST_U8,
    SSIR_CONST_I16,
    SSIR_CONST_U16,
    SSIR_CONST_I64,
    SSIR_CONST_U64,
    SSIR_CONST_COMPOSITE,
    SSIR_CONST_NULL,
} SsirConstantKind;

struct SsirConstant {
    uint32_t id;
    uint32_t type;
    SsirConstantKind kind;
    const char *name;
    bool is_specialization;
    uint32_t spec_id;
    union {
        bool bool_val;
        int32_t i32_val;
        uint32_t u32_val;
        float f32_val;
        uint16_t f16_val;
        double f64_val;
        int8_t i8_val;
        uint8_t u8_val;
        int16_t i16_val;
        uint16_t u16_val;
        int64_t i64_val;
        uint64_t u64_val;
        struct {
            uint32_t *components;
            uint32_t count;
        } composite;
    };
};

/* Basic Blocks */

struct SsirBlock {
    uint32_t id;
    const char *name;
    SsirInst *insts;
    uint32_t inst_count;
    uint32_t inst_capacity;
};

/* Functions */

typedef struct SsirFunctionParam {
    uint32_t id;
    uint32_t type;
    const char *name;
} SsirFunctionParam;

typedef struct SsirLocalVar {
    uint32_t id;
    uint32_t type;
    const char *name;
    bool has_initializer;
    uint32_t initializer;
} SsirLocalVar;

struct SsirFunction {
    uint32_t id;
    const char *name;
    uint32_t return_type;
    SsirFunctionParam *params;
    uint32_t param_count;
    SsirLocalVar *locals;
    uint32_t local_count;
    uint32_t local_capacity;
    SsirBlock *blocks;
    uint32_t block_count;
    uint32_t block_capacity;
};

/* Global Variables */

struct SsirGlobalVar {
    uint32_t id;
    const char *name;
    uint32_t type;
    bool has_group;
    uint32_t group;
    bool has_binding;
    uint32_t binding;
    bool has_location;
    uint32_t location;
    SsirBuiltinVar builtin;
    SsirInterpolation interp;
    SsirInterpolationSampling interp_sampling;
    bool non_writable;
    bool invariant;
    bool has_initializer;
    uint32_t initializer;
};

/* Entry Points */

struct SsirEntryPoint {
    SsirStage stage;
    uint32_t function;
    const char *name;
    uint32_t *interface;
    uint32_t interface_count;
    uint32_t workgroup_size[3];
    bool depth_replacing;
    bool origin_upper_left;
    bool early_fragment_tests;
};

/* Module */

typedef struct SsirNameEntry {
    uint32_t id;
    const char *name;
} SsirNameEntry;

struct SsirModule {
    SsirType *types;
    uint32_t type_count;
    uint32_t type_capacity;
    SsirConstant *constants;
    uint32_t constant_count;
    uint32_t constant_capacity;
    SsirGlobalVar *globals;
    uint32_t global_count;
    uint32_t global_capacity;
    SsirFunction *functions;
    uint32_t function_count;
    uint32_t function_capacity;
    SsirEntryPoint *entry_points;
    uint32_t entry_point_count;
    uint32_t entry_point_capacity;
    SsirNameEntry *names;
    uint32_t name_count;
    uint32_t name_capacity;
    uint32_t next_id;
    SsirClipSpaceConvention clip_space;
    void *_lookup_cache; /* Internal: ID lookup acceleration, do not access directly */
};

/* Result Codes */

typedef enum SsirResult {
    SSIR_OK = 0,
    SSIR_ERROR_OUT_OF_MEMORY,
    SSIR_ERROR_INVALID_TYPE,
    SSIR_ERROR_INVALID_ID,
    SSIR_ERROR_INVALID_OPERAND,
    SSIR_ERROR_TYPE_MISMATCH,
    SSIR_ERROR_INVALID_BLOCK,
    SSIR_ERROR_INVALID_FUNCTION,
    SSIR_ERROR_SSA_VIOLATION,
    SSIR_ERROR_TERMINATOR_MISSING,
    SSIR_ERROR_PHI_PLACEMENT,
    SSIR_ERROR_ADDRESS_SPACE,
    SSIR_ERROR_ENTRY_POINT,
} SsirResult;

/* Module API */

SsirModule *ssir_module_create(void);
void ssir_module_destroy(SsirModule *mod);
uint32_t ssir_module_alloc_id(SsirModule *mod);
void ssir_set_name(SsirModule *mod, uint32_t id, const char *name);
void ssir_module_set_clip_space(SsirModule *mod, SsirClipSpaceConvention convention);

/* Type API */

uint32_t ssir_type_void(SsirModule *mod);
uint32_t ssir_type_bool(SsirModule *mod);
uint32_t ssir_type_i32(SsirModule *mod);
uint32_t ssir_type_u32(SsirModule *mod);
uint32_t ssir_type_f32(SsirModule *mod);
uint32_t ssir_type_f16(SsirModule *mod);
uint32_t ssir_type_f64(SsirModule *mod);
uint32_t ssir_type_i8(SsirModule *mod);
uint32_t ssir_type_u8(SsirModule *mod);
uint32_t ssir_type_i16(SsirModule *mod);
uint32_t ssir_type_u16(SsirModule *mod);
uint32_t ssir_type_i64(SsirModule *mod);
uint32_t ssir_type_u64(SsirModule *mod);

uint32_t ssir_type_vec(SsirModule *mod, uint32_t elem_type, uint8_t size);
uint32_t ssir_type_mat(SsirModule *mod, uint32_t col_type, uint8_t cols, uint8_t rows);
uint32_t ssir_type_array(SsirModule *mod, uint32_t elem_type, uint32_t length);
uint32_t ssir_type_array_stride(SsirModule *mod, uint32_t elem_type, uint32_t length, uint32_t stride);
uint32_t ssir_type_runtime_array(SsirModule *mod, uint32_t elem_type);
uint32_t ssir_type_struct(SsirModule *mod, const char *name,
                          const uint32_t *members, uint32_t member_count,
                          const uint32_t *offsets);
uint32_t ssir_type_struct_named(SsirModule *mod, const char *name,
                                const uint32_t *members, uint32_t member_count,
                                const uint32_t *offsets,
                                const char *const *member_names);
uint32_t ssir_type_ptr(SsirModule *mod, uint32_t pointee_type, SsirAddressSpace space);

uint32_t ssir_type_sampler(SsirModule *mod);
uint32_t ssir_type_sampler_comparison(SsirModule *mod);
uint32_t ssir_type_texture(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type);
uint32_t ssir_type_texture_storage(SsirModule *mod, SsirTextureDim dim,
                                   uint32_t format, SsirAccessMode access);
uint32_t ssir_type_texture_depth(SsirModule *mod, SsirTextureDim dim);

SsirType *ssir_get_type(SsirModule *mod, uint32_t type_id);

bool ssir_type_is_scalar(const SsirType *t);
bool ssir_type_is_integer(const SsirType *t);
bool ssir_type_is_signed(const SsirType *t);
bool ssir_type_is_unsigned(const SsirType *t);
bool ssir_type_is_float(const SsirType *t);
bool ssir_type_is_bool(const SsirType *t);
bool ssir_type_is_vector(const SsirType *t);
bool ssir_type_is_matrix(const SsirType *t);
bool ssir_type_is_numeric(const SsirType *t);

uint32_t ssir_type_scalar_of(SsirModule *mod, uint32_t type_id);

/* Constant API */

uint32_t ssir_const_bool(SsirModule *mod, bool val);
uint32_t ssir_const_i32(SsirModule *mod, int32_t val);
uint32_t ssir_const_u32(SsirModule *mod, uint32_t val);
uint32_t ssir_const_f32(SsirModule *mod, float val);
uint32_t ssir_const_f16(SsirModule *mod, uint16_t val);
uint32_t ssir_const_f64(SsirModule *mod, double val);
uint32_t ssir_const_i8(SsirModule *mod, int8_t val);
uint32_t ssir_const_u8(SsirModule *mod, uint8_t val);
uint32_t ssir_const_i16(SsirModule *mod, int16_t val);
uint32_t ssir_const_u16(SsirModule *mod, uint16_t val);
uint32_t ssir_const_i64(SsirModule *mod, int64_t val);
uint32_t ssir_const_u64(SsirModule *mod, uint64_t val);
uint32_t ssir_const_composite(SsirModule *mod, uint32_t type_id,
                              const uint32_t *components, uint32_t count);
uint32_t ssir_const_null(SsirModule *mod, uint32_t type_id);
uint32_t ssir_const_spec_bool(SsirModule *mod, bool default_val, uint32_t spec_id);
uint32_t ssir_const_spec_i32(SsirModule *mod, int32_t default_val, uint32_t spec_id);
uint32_t ssir_const_spec_u32(SsirModule *mod, uint32_t default_val, uint32_t spec_id);
uint32_t ssir_const_spec_f32(SsirModule *mod, float default_val, uint32_t spec_id);

SsirConstant *ssir_get_constant(SsirModule *mod, uint32_t const_id);

/* Global Variable API */

uint32_t ssir_global_var(SsirModule *mod, const char *name, uint32_t ptr_type);
SsirGlobalVar *ssir_get_global(SsirModule *mod, uint32_t global_id);

void ssir_global_set_group(SsirModule *mod, uint32_t global_id, uint32_t group);
void ssir_global_set_binding(SsirModule *mod, uint32_t global_id, uint32_t binding);
void ssir_global_set_location(SsirModule *mod, uint32_t global_id, uint32_t location);
void ssir_global_set_builtin(SsirModule *mod, uint32_t global_id, SsirBuiltinVar builtin);
void ssir_global_set_interpolation(SsirModule *mod, uint32_t global_id, SsirInterpolation interp);
void ssir_global_set_interp_sampling(SsirModule *mod, uint32_t global_id, SsirInterpolationSampling sampling);
void ssir_global_set_non_writable(SsirModule *mod, uint32_t global_id, bool non_writable);
void ssir_global_set_invariant(SsirModule *mod, uint32_t global_id, bool invariant);
void ssir_global_set_initializer(SsirModule *mod, uint32_t global_id, uint32_t const_id);

/* Function API */

uint32_t ssir_function_create(SsirModule *mod, const char *name, uint32_t return_type);
SsirFunction *ssir_get_function(SsirModule *mod, uint32_t func_id);

uint32_t ssir_function_add_param(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t type);

uint32_t ssir_function_add_local(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t ptr_type);

/* Block API */

uint32_t ssir_block_create(SsirModule *mod, uint32_t func_id, const char *name);
uint32_t ssir_block_create_with_id(SsirModule *mod, uint32_t func_id, uint32_t block_id, const char *name);
SsirBlock *ssir_get_block(SsirModule *mod, uint32_t func_id, uint32_t block_id);

/* Instruction Builder API */

uint32_t ssir_build_add(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_sub(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_div(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_neg(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a);

uint32_t ssir_build_mat_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_mat_transpose(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t m);

uint32_t ssir_build_bit_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_xor(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_bit_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a);
uint32_t ssir_build_shl(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_shr(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_shr_logical(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t a, uint32_t b);

uint32_t ssir_build_eq(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_ne(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_lt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_le(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_gt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_ge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);

uint32_t ssir_build_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b);
uint32_t ssir_build_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a);

uint32_t ssir_build_construct(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, const uint32_t *components, uint32_t count);
uint32_t ssir_build_extract(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t composite, uint32_t index);
uint32_t ssir_build_insert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t composite, uint32_t value, uint32_t index);
uint32_t ssir_build_shuffle(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t v1, uint32_t v2,
                            const uint32_t *indices, uint32_t index_count);
uint32_t ssir_build_splat(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t type, uint32_t scalar);
uint32_t ssir_build_extract_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t composite, uint32_t index);
uint32_t ssir_build_insert_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t vector, uint32_t value, uint32_t index);

uint32_t ssir_build_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t ptr);
void ssir_build_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                      uint32_t ptr, uint32_t value);
uint32_t ssir_build_access(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t base,
                           const uint32_t *indices, uint32_t index_count);
uint32_t ssir_build_array_len(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t ptr);

void ssir_build_branch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t target_block);
void ssir_build_branch_cond(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t cond, uint32_t true_block, uint32_t false_block);
void ssir_build_branch_cond_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t cond, uint32_t true_block, uint32_t false_block,
                                  uint32_t merge_block);
void ssir_build_loop_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t merge_block, uint32_t continue_block);
void ssir_build_selection_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t merge_block);
void ssir_build_switch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t selector, uint32_t default_block,
                       const uint32_t *cases, uint32_t case_count);
uint32_t ssir_build_phi(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, const uint32_t *incoming, uint32_t count);
void ssir_build_return(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t value);
void ssir_build_return_void(SsirModule *mod, uint32_t func_id, uint32_t block_id);
void ssir_build_unreachable(SsirModule *mod, uint32_t func_id, uint32_t block_id);

uint32_t ssir_build_call(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t callee,
                         const uint32_t *args, uint32_t arg_count);
uint32_t ssir_build_builtin(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, SsirBuiltinId builtin,
                            const uint32_t *args, uint32_t arg_count);

uint32_t ssir_build_convert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value);
uint32_t ssir_build_bitcast(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value);

uint32_t ssir_build_tex_sample(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord);
uint32_t ssir_build_tex_sample_bias(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t bias);
uint32_t ssir_build_tex_sample_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture, uint32_t sampler,
                                     uint32_t coord, uint32_t lod);
uint32_t ssir_build_tex_sample_grad(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t ddx, uint32_t ddy);
uint32_t ssir_build_tex_sample_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref);
uint32_t ssir_build_tex_sample_cmp_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                         uint32_t type, uint32_t texture, uint32_t sampler,
                                         uint32_t coord, uint32_t ref, uint32_t lod);
uint32_t ssir_build_tex_sample_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture, uint32_t sampler,
                                      uint32_t coord, uint32_t offset);
uint32_t ssir_build_tex_sample_bias_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                           uint32_t type, uint32_t texture, uint32_t sampler,
                                           uint32_t coord, uint32_t bias, uint32_t offset);
uint32_t ssir_build_tex_sample_level_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                            uint32_t type, uint32_t texture, uint32_t sampler,
                                            uint32_t coord, uint32_t lod, uint32_t offset);
uint32_t ssir_build_tex_sample_grad_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                           uint32_t type, uint32_t texture, uint32_t sampler,
                                           uint32_t coord, uint32_t ddx, uint32_t ddy,
                                           uint32_t offset);
uint32_t ssir_build_tex_sample_cmp_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                          uint32_t type, uint32_t texture, uint32_t sampler,
                                          uint32_t coord, uint32_t ref, uint32_t offset);
uint32_t ssir_build_tex_gather(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord, uint32_t component);
uint32_t ssir_build_tex_gather_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref);
uint32_t ssir_build_tex_gather_offset(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture, uint32_t sampler,
                                      uint32_t coord, uint32_t component, uint32_t offset);
uint32_t ssir_build_tex_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t coord, uint32_t level);
void ssir_build_tex_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t texture, uint32_t coord, uint32_t value);
uint32_t ssir_build_tex_size(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t level);
uint32_t ssir_build_tex_query_lod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t texture, uint32_t sampler,
                                  uint32_t coord);
uint32_t ssir_build_tex_query_levels(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture);
uint32_t ssir_build_tex_query_samples(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                      uint32_t type, uint32_t texture);

void ssir_build_barrier(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        SsirBarrierScope scope);
uint32_t ssir_build_atomic(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, SsirAtomicOp op, uint32_t ptr,
                           uint32_t value, uint32_t comparator);
uint32_t ssir_build_atomic_ex(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, SsirAtomicOp op, uint32_t ptr,
                              uint32_t value, uint32_t comparator,
                              SsirMemoryScope scope, SsirMemorySemantics semantics);
void ssir_build_discard(SsirModule *mod, uint32_t func_id, uint32_t block_id);
uint32_t ssir_build_rem(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b);

/* Entry Point API */

uint32_t ssir_entry_point_create(SsirModule *mod, SsirStage stage,
                                 uint32_t func_id, const char *name);
SsirEntryPoint *ssir_get_entry_point(SsirModule *mod, uint32_t index);

void ssir_entry_point_add_interface(SsirModule *mod, uint32_t ep_index, uint32_t global_id);
void ssir_entry_point_set_workgroup_size(SsirModule *mod, uint32_t ep_index,
                                         uint32_t x, uint32_t y, uint32_t z);
void ssir_entry_point_set_depth_replacing(SsirModule *mod, uint32_t ep_index, bool v);
void ssir_entry_point_set_origin_upper_left(SsirModule *mod, uint32_t ep_index, bool v);
void ssir_entry_point_set_early_fragment_tests(SsirModule *mod, uint32_t ep_index, bool v);

/* Validation API */

typedef struct SsirValidationError {
    SsirResult code;
    const char *message;
    uint32_t func_id;
    uint32_t block_id;
    uint32_t inst_index;
} SsirValidationError;

typedef struct SsirValidationResult {
    bool valid;
    SsirValidationError *errors;
    uint32_t error_count;
    uint32_t error_capacity;
} SsirValidationResult;

SsirValidationResult *ssir_validate(SsirModule *mod);
void ssir_validation_result_free(SsirValidationResult *result);

/* Analysis API */

void ssir_count_uses(SsirFunction *f, uint32_t *use_counts, uint32_t max_id);
void ssir_module_build_lookup(SsirModule *mod);

/* Debug/Utility API */

const char *ssir_opcode_name(SsirOpcode op);
const char *ssir_builtin_name(SsirBuiltinId id);
const char *ssir_type_kind_name(SsirTypeKind kind);
char *ssir_module_to_string(SsirModule *mod);
float ssir_f16_to_f32(uint16_t h);

/* ============================================================================
 * SSIR TO SPIR-V
 * ============================================================================ */

typedef struct SsirToSpirvOptions {
    uint32_t spirv_version;
    int enable_debug_names;
    int enable_line_info;
} SsirToSpirvOptions;

typedef enum SsirToSpirvResult {
    SSIR_TO_SPIRV_OK = 0,
    SSIR_TO_SPIRV_ERR_INVALID_INPUT,
    SSIR_TO_SPIRV_ERR_UNSUPPORTED,
    SSIR_TO_SPIRV_ERR_INTERNAL,
    SSIR_TO_SPIRV_ERR_OOM,
} SsirToSpirvResult;

SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
                                 const SsirToSpirvOptions *opts,
                                 uint32_t **out_words,
                                 size_t *out_count);

void ssir_to_spirv_free(void *p);

const char *ssir_to_spirv_result_string(SsirToSpirvResult result);

/* ============================================================================
 * SSIR TO WGSL
 * ============================================================================ */

typedef enum SsirToWgslResult {
    SSIR_TO_WGSL_OK = 0,
    SSIR_TO_WGSL_ERR_INVALID_INPUT,
    SSIR_TO_WGSL_ERR_UNSUPPORTED,
    SSIR_TO_WGSL_ERR_INTERNAL,
    SSIR_TO_WGSL_ERR_OOM,
} SsirToWgslResult;

typedef struct SsirToWgslOptions {
    int preserve_names;
} SsirToWgslOptions;

SsirToWgslResult ssir_to_wgsl(const SsirModule *mod,
                              const SsirToWgslOptions *opts,
                              char **out_wgsl,
                              char **out_error);

void ssir_to_wgsl_free(void *p);

const char *ssir_to_wgsl_result_string(SsirToWgslResult result);

/* ============================================================================
 * SSIR TO GLSL
 * ============================================================================ */

typedef enum SsirToGlslResult {
    SSIR_TO_GLSL_OK = 0,
    SSIR_TO_GLSL_ERR_INVALID_INPUT,
    SSIR_TO_GLSL_ERR_UNSUPPORTED,
    SSIR_TO_GLSL_ERR_INTERNAL,
    SSIR_TO_GLSL_ERR_OOM,
} SsirToGlslResult;

typedef struct SsirToGlslOptions {
    int preserve_names;
    int target_opengl; /* suppress Vulkan-only qualifiers (set = N) */
} SsirToGlslOptions;

SsirToGlslResult ssir_to_glsl(const SsirModule *mod,
                               SsirStage stage,
                               const SsirToGlslOptions *opts,
                               char **out_glsl,
                               char **out_error);

void ssir_to_glsl_free(void *p);

const char *ssir_to_glsl_result_string(SsirToGlslResult result);

/* ============================================================================
 * SSIR TO MSL (Metal Shading Language)
 * ============================================================================ */

typedef enum SsirToMslResult {
    SSIR_TO_MSL_OK = 0,
    SSIR_TO_MSL_ERR_INVALID_INPUT,
    SSIR_TO_MSL_ERR_UNSUPPORTED,
    SSIR_TO_MSL_ERR_INTERNAL,
    SSIR_TO_MSL_ERR_OOM,
} SsirToMslResult;

typedef struct SsirToMslOptions {
    int preserve_names;
} SsirToMslOptions;

SsirToMslResult ssir_to_msl(const SsirModule *mod,
                              const SsirToMslOptions *opts,
                              char **out_msl,
                              char **out_error);

void ssir_to_msl_free(void *p);

const char *ssir_to_msl_result_string(SsirToMslResult result);

/* ============================================================================
 * SPIR-V TO SSIR
 * ============================================================================ */

typedef enum {
    SPIRV_TO_SSIR_SUCCESS = 0,
    SPIRV_TO_SSIR_INVALID_SPIRV,
    SPIRV_TO_SSIR_UNSUPPORTED_FEATURE,
    SPIRV_TO_SSIR_INTERNAL_ERROR
} SpirvToSsirResult;

typedef struct {
    bool preserve_names;
    bool preserve_locations;
} SpirvToSsirOptions;

SpirvToSsirResult spirv_to_ssir(
    const uint32_t *spirv,
    size_t word_count,
    const SpirvToSsirOptions *opts,
    SsirModule **out_module,
    char **out_error
);

const char *spirv_to_ssir_result_string(SpirvToSsirResult result);

void spirv_to_ssir_free(void *p);

/* ============================================================================
 * SSIR TO HLSL
 * ============================================================================ */

typedef enum SsirToHlslResult {
    SSIR_TO_HLSL_OK = 0,
    SSIR_TO_HLSL_ERR_INVALID_INPUT,
    SSIR_TO_HLSL_ERR_UNSUPPORTED,
    SSIR_TO_HLSL_ERR_INTERNAL,
    SSIR_TO_HLSL_ERR_OOM,
} SsirToHlslResult;

typedef struct SsirToHlslOptions {
    int preserve_names;
    int shader_model_major; // e.g., 6
    int shader_model_minor; // e.g., 0
} SsirToHlslOptions;

SsirToHlslResult ssir_to_hlsl(const SsirModule *mod,
                              SsirStage stage,
                              const SsirToHlslOptions *opts,
                              char **out_hlsl,
                              char **out_error);

void ssir_to_hlsl_free(void *p);

const char *ssir_to_hlsl_result_string(SsirToHlslResult result);

/* ============================================================================
 * MSL TO SSIR (Metal Shading Language Parser)
 * ============================================================================ */

typedef enum {
    MSL_TO_SSIR_OK = 0,
    MSL_TO_SSIR_PARSE_ERROR,
    MSL_TO_SSIR_TYPE_ERROR,
    MSL_TO_SSIR_UNSUPPORTED,
} MslToSsirResult;

typedef struct {
    int preserve_names;
} MslToSsirOptions;

MslToSsirResult msl_to_ssir(const char *msl_source, const MslToSsirOptions *opts,
                              SsirModule **out_module, char **out_error);

void msl_to_ssir_free(char *str);

const char *msl_to_ssir_result_string(MslToSsirResult r);

#ifdef __cplusplus
}
#endif

#endif /* SIMPLE_WGSL_H */
