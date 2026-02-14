# Technical Reference

Complete technical documentation for simple_wgsl: architecture, intermediate representation, API surface, and internal design.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Compilation Pipeline](#compilation-pipeline)
  - [Forward Path (Source to Binary)](#forward-path-source-to-binary)
  - [Reverse Path (Binary to Source)](#reverse-path-binary-to-source)
  - [Cross-Compilation Path](#cross-compilation-path)
- [AST (Abstract Syntax Tree)](#ast-abstract-syntax-tree)
  - [Node Types](#node-types)
  - [AST Structure](#ast-structure)
  - [Source Location Tracking](#source-location-tracking)
- [Resolver](#resolver)
  - [Symbol Table](#symbol-table)
  - [Binding Extraction](#binding-extraction)
  - [Entry Point Detection](#entry-point-detection)
  - [Vertex Input Reflection](#vertex-input-reflection)
  - [Fragment Output Reflection](#fragment-output-reflection)
- [SSIR (Simple Shader Intermediate Representation)](#ssir-simple-shader-intermediate-representation)
  - [Design Principles](#design-principles)
  - [Module Structure](#module-structure)
  - [Type System](#type-system)
  - [Constants](#constants)
  - [Global Variables](#global-variables)
  - [Functions and Blocks](#functions-and-blocks)
  - [Instruction Set](#instruction-set)
  - [Built-in Functions](#built-in-functions)
  - [Entry Points](#entry-points)
  - [Validation](#validation)
- [Lowering (AST to SPIR-V)](#lowering-ast-to-spirv)
  - [Lowering Options](#lowering-options)
  - [Section-Buffered Emission](#section-buffered-emission)
  - [Two-Phase API](#two-phase-api)
  - [Introspection](#introspection)
- [Raising (SPIR-V to WGSL)](#raising-spirv-to-wgsl)
  - [Raise Options](#raise-options)
  - [Incremental API](#incremental-api)
- [SSIR Emitters](#ssir-emitters)
  - [SSIR to SPIR-V](#ssir-to-spirv)
  - [SSIR to WGSL](#ssir-to-wgsl)
  - [SSIR to GLSL](#ssir-to-glsl)
  - [SSIR to MSL](#ssir-to-msl)
  - [SSIR to HLSL](#ssir-to-hlsl)
- [SPIR-V to SSIR Deserialization](#spirv-to-ssir-deserialization)
- [MSL to SSIR Parser](#msl-to-ssir-parser)
- [Memory Management](#memory-management)
- [Language Mapping Tables](#language-mapping-tables)
  - [WGSL to GLSL Mapping](#wgsl-to-glsl-mapping)
- [Error Handling](#error-handling)

---

## Architecture Overview

Simple WGSL is organized around a central intermediate representation called SSIR. All source languages converge to SSIR, and all output formats diverge from it.

```
                        ┌─────────────┐
    WGSL source ──────► │  wgsl_parse  │──► AST ──► Resolver ──► Lowering ──┐
    GLSL source ──────► │  glsl_parse  │──► AST ──────────────────────────┘ │
                        └─────────────┘                                     ▼
    MSL source  ──────► msl_to_ssir ────────────────────────────────► SsirModule
    SPIR-V binary ───► spirv_to_ssir ──────────────────────────────►     │
                                                                         │
                        ┌────────────────────────────────────────────────┘
                        │
                        ├──► ssir_to_spirv ──► SPIR-V binary
                        ├──► ssir_to_wgsl  ──► WGSL source
                        ├──► ssir_to_glsl  ──► GLSL 450 source
                        ├──► ssir_to_msl   ──► Metal Shading Language
                        └──► ssir_to_hlsl  ──► HLSL source
```

Key architectural decisions:

- **Shared AST for WGSL and GLSL**: Both parsers produce the same `WgslAstNode` tree, so the resolver and lowering stages are shared between both input languages.
- **MSL bypasses the AST**: The MSL parser produces an `SsirModule` directly rather than going through the shared AST, because MSL's C++-based syntax differs too significantly from WGSL/GLSL.
- **SPIR-V is both input and output**: SPIR-V can be deserialized into SSIR for decompilation, and SSIR can be serialized back to SPIR-V for compilation.
- **Pure C99 core**: The library itself uses no C++ features. Tests use C++17 for Google Test and RAII convenience.

---

## Compilation Pipeline

### Forward Path (Source to Binary)

The forward path compiles WGSL or GLSL source code to SPIR-V binary:

```
Source (WGSL/GLSL)  ──►  AST (WgslAstNode*)  ──►  Resolver  ──►  SPIR-V (uint32_t[])
                     parse                    resolve         lower
```

**Stage 1: Parsing** -- Hand-written recursive-descent parsers with operator-precedence climbing tokenize and parse the input into a shared AST. Both WGSL and GLSL produce the same `WgslAstNode` tree structure. The parser tracks line and column numbers for every node.

**Stage 2: Resolution** -- The resolver walks the AST and builds a symbol table with scope tracking. It extracts `@group`/`@binding` attributes for resource binding reflection, identifies entry points and their shader stages, and resolves identifier references to their declaration sites.

**Stage 3: Lowering** -- The lowering pass walks the resolved AST, builds an SSIR module internally, then serializes to SPIR-V. It uses section-buffered emission: SPIR-V words are accumulated into separate buffers (capabilities, extensions, types, globals, functions, etc.) and concatenated at the end in the order mandated by the SPIR-V specification.

### Reverse Path (Binary to Source)

The reverse path decompiles SPIR-V binary back to WGSL source:

```
SPIR-V (uint32_t[])  ──►  SsirModule  ──►  WGSL source (char*)
                     spirv_to_ssir     wgsl_raise / ssir_to_wgsl
```

The `wgsl_raise_to_wgsl()` convenience function wraps this two-step process. The incremental `WgslRaiser` API exposes each phase separately for more control.

### Cross-Compilation Path

To cross-compile between languages, chain the appropriate parser and emitter through SSIR:

```
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_glsl ──► GLSL 450
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_msl  ──► MSL
WGSL ──► parse ──► resolve ──► lower ──► SSIR ──► ssir_to_hlsl ──► HLSL
MSL  ──► msl_to_ssir ──► SSIR ──► ssir_to_spirv ──► SPIR-V
SPIR-V ──► spirv_to_ssir ──► SSIR ──► ssir_to_glsl ──► GLSL 450
```

---

## AST (Abstract Syntax Tree)

### Node Types

Every AST node is a `WgslAstNode`, a tagged union discriminated by `WgslNodeType`:

| Node Type | Enum | Description |
|-----------|------|-------------|
| Program | `WGSL_NODE_PROGRAM` | Root node, contains top-level declarations |
| Struct | `WGSL_NODE_STRUCT` | Struct type declaration |
| Struct Field | `WGSL_NODE_STRUCT_FIELD` | Individual field within a struct |
| Global Variable | `WGSL_NODE_GLOBAL_VAR` | Module-scope `var<>` declaration |
| Function | `WGSL_NODE_FUNCTION` | Function declaration with params, return type, and body |
| Parameter | `WGSL_NODE_PARAM` | Function parameter |
| Type | `WGSL_NODE_TYPE` | Type reference (e.g., `vec4<f32>`) |
| Attribute | `WGSL_NODE_ATTRIBUTE` | Attribute (e.g., `@group(0)`, `@vertex`) |
| Block | `WGSL_NODE_BLOCK` | Brace-delimited statement list `{ ... }` |
| Var Declaration | `WGSL_NODE_VAR_DECL` | Local `var`, `let`, or `const` declaration |
| Return | `WGSL_NODE_RETURN` | Return statement |
| Expression Statement | `WGSL_NODE_EXPR_STMT` | Expression used as a statement |
| If | `WGSL_NODE_IF` | If/else statement |
| While | `WGSL_NODE_WHILE` | While loop |
| For | `WGSL_NODE_FOR` | For loop (init; cond; cont) |
| Do-While | `WGSL_NODE_DO_WHILE` | Do-while loop (GLSL) |
| Switch | `WGSL_NODE_SWITCH` | Switch statement |
| Case | `WGSL_NODE_CASE` | Case clause (NULL expr = default) |
| Break | `WGSL_NODE_BREAK` | Break statement |
| Continue | `WGSL_NODE_CONTINUE` | Continue statement |
| Discard | `WGSL_NODE_DISCARD` | Fragment discard |
| Identifier | `WGSL_NODE_IDENT` | Variable/function reference |
| Literal | `WGSL_NODE_LITERAL` | Integer or float literal |
| Binary | `WGSL_NODE_BINARY` | Binary operator expression |
| Assign | `WGSL_NODE_ASSIGN` | Assignment (`=`, `+=`, etc.) |
| Call | `WGSL_NODE_CALL` | Function or type constructor call |
| Member | `WGSL_NODE_MEMBER` | Member access (`object.field`) |
| Index | `WGSL_NODE_INDEX` | Subscript (`object[index]`) |
| Unary | `WGSL_NODE_UNARY` | Unary operator (prefix or postfix) |
| Ternary | `WGSL_NODE_TERNARY` | Ternary conditional (GLSL `? :`) |

### AST Structure

```c
struct WgslAstNode {
    WgslNodeType type;     // discriminator
    int line;              // source line (1-based)
    int col;               // source column (1-based)
    union {
        Program program;
        Function function;
        StructDecl struct_decl;
        // ... one variant per node type
    };
};
```

Variable declarations distinguish between three kinds via `WgslDeclKind`:

| Kind | Enum | Semantics |
|------|------|-----------|
| `var` | `WGSL_DECL_VAR` | Mutable variable |
| `let` | `WGSL_DECL_LET` | Immutable binding |
| `const` | `WGSL_DECL_CONST` | Compile-time constant |

### Source Location Tracking

Every AST node stores its source `line` and `col` (both 1-based). The lowering pass can optionally emit `OpLine` instructions in SPIR-V for debugger source mapping when `enable_line_info` is set.

---

## Resolver

The resolver performs semantic analysis on the AST, producing a `WgslResolver` that stores symbol information, scope chains, and resource binding data.

### Symbol Table

```c
typedef struct {
    int id;                           // unique symbol ID
    WgslSymbolKind kind;              // GLOBAL, PARAM, or LOCAL
    const char *name;                 // identifier name
    int has_group, group_index;       // @group(N)
    int has_binding, binding_index;   // @binding(N)
    int has_min_binding_size;         // @min_binding_size(N)
    int min_binding_size;
    const WgslAstNode *decl_node;    // declaration AST node
    const WgslAstNode *function_node; // enclosing function (NULL for globals)
} WgslSymbolInfo;
```

Query functions:

| Function | Returns |
|----------|---------|
| `wgsl_resolver_all_symbols(r, &count)` | Every symbol in the program |
| `wgsl_resolver_globals(r, &count)` | Only module-scope symbols |
| `wgsl_resolver_binding_vars(r, &count)` | Only symbols with `@group`/`@binding` |
| `wgsl_resolver_ident_symbol_id(r, ident_node)` | Symbol ID for a specific identifier AST node |
| `wgsl_resolver_entrypoint_globals(r, "main", &count)` | Globals used by a specific entry point |
| `wgsl_resolver_entrypoint_binding_vars(r, "main", &count)` | Binding vars for a specific entry point |

### Binding Extraction

The resolver extracts `@group(G)` and `@binding(B)` from global variable attributes. This data is available both for the whole program and per entry point, enabling descriptor set layout generation.

### Entry Point Detection

```c
typedef struct {
    const char *name;                   // function name
    WgslStage stage;                    // VERTEX, FRAGMENT, or COMPUTE
    const WgslAstNode *function_node;   // AST node
} WgslResolverEntrypoint;
```

Query with `wgsl_resolver_entrypoints(r, &count)`. Shader stage is determined from `@vertex`, `@fragment`, or `@compute` attributes.

### Vertex Input Reflection

```c
typedef struct {
    int location;
    int component_count;    // 1-4
    WgslNumericType numeric_type;  // F32, I32, U32, F16, BOOL
    int byte_size;
} WgslVertexSlot;
```

Call `wgsl_resolver_vertex_inputs(r, "main", &slots)` to get the vertex input layout for a vertex entry point. This is sufficient to construct `VkVertexInputAttributeDescription` arrays.

### Fragment Output Reflection

```c
typedef struct {
    int location;
    int component_count;
    WgslNumericType numeric_type;
} WgslFragmentOutput;
```

Call `wgsl_resolver_fragment_outputs(r, "main", &outputs)` to get fragment output locations and formats.

---

## SSIR (Simple Shader Intermediate Representation)

SSIR is a low-level, language-independent intermediate representation that serves as the canonical representation of shader semantics. Every conversion in the library flows through SSIR.

### Design Principles

- **ID-based references**: All types, values, constants, and variables use unique `uint32_t` IDs instead of pointers. This enables serialization, cross-references, and avoids pointer invalidation when arrays grow.
- **SSA form**: Every instruction result gets a unique ID. PHI nodes handle control-flow merges.
- **Explicit control flow**: Basic blocks with explicit branch/return terminators rather than implicit tree structure.
- **Complete type system**: Scalars, vectors, matrices, arrays, structs, pointers with address spaces, textures, and samplers.
- **No language bias**: SSIR does not privilege any source or target language. Mappings to SPIR-V, WGSL, GLSL, MSL, and HLSL are all done by separate emitters.

### Module Structure

```c
struct SsirModule {
    SsirType      *types;           // type array
    uint32_t       type_count;
    SsirConstant  *constants;       // constant array
    uint32_t       constant_count;
    SsirGlobalVar *globals;         // global variable array
    uint32_t       global_count;
    SsirFunction  *functions;       // function array
    uint32_t       function_count;
    SsirEntryPoint *entry_points;   // entry point array
    uint32_t       entry_point_count;
    SsirNameEntry  *names;          // debug name table
    uint32_t       name_count;
    uint32_t       next_id;         // next available ID
    SsirClipSpaceConvention clip_space;  // coordinate convention
};
```

```
SsirModule
├── Types[]          (type declarations: scalars, vectors, matrices, structs, pointers, textures, ...)
├── Constants[]      (compile-time values: bool, i32, u32, f32, f16, f64, composite, null, ...)
├── GlobalVars[]     (module-scope variables with bindings, locations, builtins)
├── Functions[]
│   ├── Params[]     (function parameters)
│   ├── Locals[]     (function-scope pointer variables)
│   └── Blocks[]     (basic blocks)
│       └── Insts[]  (SSA instructions)
├── EntryPoints[]    (shader stage entry points with interface variables)
└── Names[]          (debug name table)
```

### Type System

All types are stored in the module's type array and referenced by ID.

| Type Kind | Enum | Description |
|-----------|------|-------------|
| Void | `SSIR_TYPE_VOID` | Void type (function returns) |
| Bool | `SSIR_TYPE_BOOL` | Boolean |
| I32 | `SSIR_TYPE_I32` | Signed 32-bit integer |
| U32 | `SSIR_TYPE_U32` | Unsigned 32-bit integer |
| F32 | `SSIR_TYPE_F32` | 32-bit float |
| F16 | `SSIR_TYPE_F16` | 16-bit float (half) |
| F64 | `SSIR_TYPE_F64` | 64-bit float (double) |
| I8/U8/I16/U16/I64/U64 | `SSIR_TYPE_*` | Extended integer types |
| Vector | `SSIR_TYPE_VEC` | Vector (elem type + size 2/3/4) |
| Matrix | `SSIR_TYPE_MAT` | Matrix (column type + cols + rows) |
| Array | `SSIR_TYPE_ARRAY` | Fixed-length array (elem type + length + stride) |
| Runtime Array | `SSIR_TYPE_RUNTIME_ARRAY` | Variable-length array (storage buffers) |
| Struct | `SSIR_TYPE_STRUCT` | Struct with member types, offsets, names, layout rule |
| Pointer | `SSIR_TYPE_PTR` | Pointer (pointee type + address space) |
| Sampler | `SSIR_TYPE_SAMPLER` | Sampler |
| Comparison Sampler | `SSIR_TYPE_SAMPLER_COMPARISON` | Comparison sampler (depth) |
| Texture | `SSIR_TYPE_TEXTURE` | Sampled texture (dim + sampled type) |
| Storage Texture | `SSIR_TYPE_TEXTURE_STORAGE` | Storage texture (dim + format + access) |
| Depth Texture | `SSIR_TYPE_TEXTURE_DEPTH` | Depth texture (dim) |

**Address Spaces**:

| Space | Enum | Description |
|-------|------|-------------|
| Function | `SSIR_ADDR_FUNCTION` | Function-local variables |
| Private | `SSIR_ADDR_PRIVATE` | Module-scope private variables |
| Workgroup | `SSIR_ADDR_WORKGROUP` | Shared within workgroup |
| Uniform | `SSIR_ADDR_UNIFORM` | Uniform buffer |
| Uniform Constant | `SSIR_ADDR_UNIFORM_CONSTANT` | Textures and samplers |
| Storage | `SSIR_ADDR_STORAGE` | Storage buffer |
| Input | `SSIR_ADDR_INPUT` | Shader stage input |
| Output | `SSIR_ADDR_OUTPUT` | Shader stage output |
| Push Constant | `SSIR_ADDR_PUSH_CONSTANT` | Vulkan push constants |
| Physical Storage Buffer | `SSIR_ADDR_PHYSICAL_STORAGE_BUFFER` | Buffer device address |

**Layout Rules** (for struct member offsets):

| Rule | Enum |
|------|------|
| None | `SSIR_LAYOUT_NONE` |
| std140 | `SSIR_LAYOUT_STD140` |
| std430 | `SSIR_LAYOUT_STD430` |
| Scalar | `SSIR_LAYOUT_SCALAR` |

**Texture Dimensions**:

`SSIR_TEX_1D`, `SSIR_TEX_2D`, `SSIR_TEX_3D`, `SSIR_TEX_CUBE`, `SSIR_TEX_2D_ARRAY`, `SSIR_TEX_CUBE_ARRAY`, `SSIR_TEX_MULTISAMPLED_2D`, `SSIR_TEX_1D_ARRAY`, `SSIR_TEX_BUFFER`, `SSIR_TEX_MULTISAMPLED_2D_ARRAY`

**Clip Space Conventions**:

| Convention | Y Direction | Z Range | Enum |
|-----------|-------------|---------|------|
| Vulkan | Y-down | [0, 1] | `SSIR_CLIP_SPACE_VULKAN` |
| OpenGL | Y-up | [-1, 1] | `SSIR_CLIP_SPACE_OPENGL` |
| DirectX | Y-up | [0, 1] | `SSIR_CLIP_SPACE_DIRECTX` |
| Metal | Y-up | [0, 1] | `SSIR_CLIP_SPACE_METAL` |

### Constants

```c
struct SsirConstant {
    uint32_t id;
    uint32_t type;
    SsirConstantKind kind;       // BOOL, I32, U32, F32, F16, F64, I8, U8, ..., COMPOSITE, NULL
    const char *name;
    bool is_specialization;      // true for specialization constants
    uint32_t spec_id;            // specialization ID
    union { ... };               // kind-specific value
};
```

Supported constant kinds: `SSIR_CONST_BOOL`, `SSIR_CONST_I32`, `SSIR_CONST_U32`, `SSIR_CONST_F32`, `SSIR_CONST_F16`, `SSIR_CONST_F64`, `SSIR_CONST_I8`, `SSIR_CONST_U8`, `SSIR_CONST_I16`, `SSIR_CONST_U16`, `SSIR_CONST_I64`, `SSIR_CONST_U64`, `SSIR_CONST_COMPOSITE`, `SSIR_CONST_NULL`.

Specialization constants are supported with `ssir_const_spec_*()` functions that set `is_specialization = true` and store the `spec_id`.

### Global Variables

```c
struct SsirGlobalVar {
    uint32_t id;
    const char *name;
    uint32_t type;           // must be a pointer type
    bool has_group;          uint32_t group;
    bool has_binding;        uint32_t binding;
    bool has_location;       uint32_t location;
    SsirBuiltinVar builtin; // NONE, VERTEX_INDEX, POSITION, FRAG_DEPTH, ...
    SsirInterpolation interp;
    SsirInterpolationSampling interp_sampling;
    bool non_writable;
    bool invariant;
    bool has_initializer;    uint32_t initializer;
};
```

Global variables represent all module-scope declarations: uniform/storage buffers (`@group`/`@binding`), shader inputs/outputs (`@location`), and built-in variables (`@builtin`).

**Supported built-in variables**: `VERTEX_INDEX`, `INSTANCE_INDEX`, `POSITION`, `FRONT_FACING`, `FRAG_DEPTH`, `SAMPLE_INDEX`, `SAMPLE_MASK`, `LOCAL_INVOCATION_ID`, `LOCAL_INVOCATION_INDEX`, `GLOBAL_INVOCATION_ID`, `WORKGROUP_ID`, `NUM_WORKGROUPS`, `POINT_SIZE`, `CLIP_DISTANCE`, `CULL_DISTANCE`, `LAYER`, `VIEWPORT_INDEX`, `FRAG_COORD`, `HELPER_INVOCATION`, `PRIMITIVE_ID`, `BASE_VERTEX`, `BASE_INSTANCE`, `SUBGROUP_SIZE`, `SUBGROUP_INVOCATION_ID`, `SUBGROUP_ID`, `NUM_SUBGROUPS`

**Interpolation modes**: `PERSPECTIVE`, `LINEAR`, `FLAT`

**Interpolation sampling**: `CENTER`, `CENTROID`, `SAMPLE`

### Functions and Blocks

```c
struct SsirFunction {
    uint32_t id;
    const char *name;
    uint32_t return_type;
    SsirFunctionParam *params;    uint32_t param_count;
    SsirLocalVar *locals;         uint32_t local_count;
    SsirBlock *blocks;            uint32_t block_count;
};

struct SsirBlock {
    uint32_t id;
    const char *name;
    SsirInst *insts;              uint32_t inst_count;
};

struct SsirInst {
    SsirOpcode op;
    uint32_t result;              // result ID (0 if none)
    uint32_t type;                // result type ID (0 if none)
    uint32_t operands[SSIR_MAX_OPERANDS];  // up to 8 inline operands
    uint8_t operand_count;
    uint32_t *extra;              // overflow operands (e.g., phi sources)
    uint16_t extra_count;
};
```

Local variables are stored separately from instructions. They represent `OpVariable` with function storage class. Each local has a pointer type and an optional initializer.

### Instruction Set

SSIR supports 70+ opcodes organized into categories:

**Arithmetic**:
`ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `REM`, `NEG`

**Matrix**:
`MAT_MUL`, `MAT_TRANSPOSE`

**Bitwise**:
`BIT_AND`, `BIT_OR`, `BIT_XOR`, `BIT_NOT`, `SHL`, `SHR`, `SHR_LOGICAL`

**Comparison**:
`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`

**Logical**:
`AND`, `OR`, `NOT`

**Composite**:
`CONSTRUCT`, `EXTRACT`, `INSERT`, `SHUFFLE`, `SPLAT`, `EXTRACT_DYN`, `INSERT_DYN`

**Memory**:
`LOAD`, `STORE`, `ACCESS` (pointer access chain), `ARRAY_LEN`

**Control Flow**:
`BRANCH`, `BRANCH_COND`, `SWITCH`, `PHI`, `RETURN`, `RETURN_VOID`, `UNREACHABLE`, `LOOP_MERGE`, `SELECTION_MERGE`

**Invocation**:
`CALL`, `BUILTIN`, `CONVERT`, `BITCAST`

**Texture**:
`TEX_SAMPLE`, `TEX_SAMPLE_BIAS`, `TEX_SAMPLE_LEVEL`, `TEX_SAMPLE_GRAD`, `TEX_SAMPLE_CMP`, `TEX_SAMPLE_CMP_LEVEL`, `TEX_SAMPLE_OFFSET`, `TEX_SAMPLE_BIAS_OFFSET`, `TEX_SAMPLE_LEVEL_OFFSET`, `TEX_SAMPLE_GRAD_OFFSET`, `TEX_SAMPLE_CMP_OFFSET`, `TEX_GATHER`, `TEX_GATHER_CMP`, `TEX_GATHER_OFFSET`, `TEX_LOAD`, `TEX_STORE`, `TEX_SIZE`, `TEX_QUERY_LOD`, `TEX_QUERY_LEVELS`, `TEX_QUERY_SAMPLES`

**Synchronization**:
`BARRIER`, `ATOMIC`, `DISCARD`

### Built-in Functions

72+ built-in functions callable via `SSIR_OP_BUILTIN`:

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Exponential**: `exp`, `exp2`, `log`, `log2`, `pow`, `sqrt`, `inverseSqrt`

**Numeric**: `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`, `fract`, `min`, `max`, `clamp`, `saturate`, `mix`, `step`, `smoothstep`, `fma`

**Geometry**: `dot`, `cross`, `length`, `distance`, `normalize`, `faceForward`, `reflect`, `refract`

**Logic**: `all`, `any`, `select`

**Bit Manipulation**: `countBits`, `reverseBits`, `firstLeadingBit`, `firstTrailingBit`, `extractBits`, `insertBits`

**Derivative**: `dpdx`, `dpdy`, `fwidth`, `dpdxCoarse`, `dpdyCoarse`, `dpdxFine`, `dpdyFine`

**Floating Point**: `isInf`, `isNan`, `degrees`, `radians`, `modf`, `frexp`, `ldexp`

**Matrix**: `determinant`, `transpose`

**Packing**: `pack4x8snorm`, `pack4x8unorm`, `pack2x16snorm`, `pack2x16unorm`, `pack2x16float`, `unpack4x8snorm`, `unpack4x8unorm`, `unpack2x16snorm`, `unpack2x16unorm`, `unpack2x16float`

**Subgroup**: `subgroupBallot`, `subgroupBroadcast`, `subgroupAdd`, `subgroupMin`, `subgroupMax`, `subgroupAll`, `subgroupAny`, `subgroupShuffle`, `subgroupPrefixAdd`

### Entry Points

```c
struct SsirEntryPoint {
    SsirStage stage;              // VERTEX, FRAGMENT, COMPUTE, GEOMETRY, TESS_CONTROL, TESS_EVAL
    uint32_t function;            // function ID
    const char *name;
    uint32_t *interface;          // interface variable IDs
    uint32_t interface_count;
    uint32_t workgroup_size[3];   // compute workgroup dimensions
    bool depth_replacing;         // writes gl_FragDepth
    bool origin_upper_left;       // fragment coord origin
    bool early_fragment_tests;    // early depth/stencil
};
```

### Validation

The SSIR validation API checks module consistency:

```c
SsirValidationResult *ssir_validate(SsirModule *mod);
```

Validates:
- Type references exist and are valid
- SSA property: no ID is defined twice
- Every block ends with a terminator (branch, return, unreachable)
- PHI nodes are only at the start of blocks
- Operand types match instruction requirements
- Address space consistency for pointer operations

Error codes: `SSIR_OK`, `SSIR_ERROR_OUT_OF_MEMORY`, `SSIR_ERROR_INVALID_TYPE`, `SSIR_ERROR_INVALID_ID`, `SSIR_ERROR_INVALID_OPERAND`, `SSIR_ERROR_TYPE_MISMATCH`, `SSIR_ERROR_INVALID_BLOCK`, `SSIR_ERROR_INVALID_FUNCTION`, `SSIR_ERROR_SSA_VIOLATION`, `SSIR_ERROR_TERMINATOR_MISSING`, `SSIR_ERROR_PHI_PLACEMENT`, `SSIR_ERROR_ADDRESS_SPACE`, `SSIR_ERROR_ENTRY_POINT`

---

## Lowering (AST to SPIR-V)

### Lowering Options

```c
typedef struct {
    uint32_t spirv_version;              // e.g., 0x00010500 for SPIR-V 1.5
    WgslLowerEnv env;                    // VULKAN_1_1, VULKAN_1_2, VULKAN_1_3, WEBGPU
    WgslLowerPacking packing;            // DEFAULT, STD430, STD140
    int enable_debug_names;              // emit OpName/OpMemberName
    int enable_line_info;                // emit OpLine for source mapping
    int zero_initialize_vars;            // zero-init all variables
    int relax_block_layout;              // relaxed block layout rules
    int use_khr_shader_draw_parameters;  // KHR_shader_draw_parameters extension
    uint32_t id_bound_hint;              // pre-allocate ID space
} WgslLowerOptions;
```

**Target environments**:

| Environment | Enum | Description |
|------------|------|-------------|
| Vulkan 1.1 | `WGSL_LOWER_ENV_VULKAN_1_1` | Minimum Vulkan support |
| Vulkan 1.2 | `WGSL_LOWER_ENV_VULKAN_1_2` | Recommended default |
| Vulkan 1.3 | `WGSL_LOWER_ENV_VULKAN_1_3` | Latest Vulkan features |
| WebGPU | `WGSL_LOWER_ENV_WEBGPU` | WebGPU-compatible SPIR-V |

### Section-Buffered Emission

Internally, lowering accumulates SPIR-V words into separate section buffers:

1. Capabilities
2. Extensions
3. ExtInstImport
4. Memory model
5. Entry points
6. Execution modes
7. Debug info (names, lines)
8. Annotations (decorations)
9. Type declarations
10. Global variables
11. Function declarations
12. Function bodies

These are concatenated in SPIR-V spec order at serialization time.

### Two-Phase API

The lowering API supports both one-shot and two-phase usage:

**One-shot** (parse + resolve + lower + serialize in one call):
```c
WgslLowerResult wgsl_lower_emit_spirv(program, resolver, &opts, &spirv, &word_count);
```

**Two-phase** (create lower context, inspect, then serialize):
```c
WgslLower *lower = wgsl_lower_create(program, resolver, &opts);
// ... inspect SSIR, query entry points, etc.
WgslLowerResult res = wgsl_lower_serialize(lower, &spirv, &word_count);
wgsl_lower_destroy(lower);
```

The two-phase API also provides `wgsl_lower_serialize_into()` for writing into a pre-allocated buffer.

### Introspection

After lowering, you can query:

| Function | Returns |
|----------|---------|
| `wgsl_lower_get_ssir(lower)` | The internal SSIR module (read-only) |
| `wgsl_lower_entrypoints(lower, &count)` | Entry point info with SPIR-V function IDs and interface variable IDs |
| `wgsl_lower_module_features(lower)` | Required capabilities and extensions |
| `wgsl_lower_last_error(lower)` | Last error message string |
| `wgsl_lower_node_result_id(lower, node)` | SPIR-V result ID for an AST node |
| `wgsl_lower_symbol_result_id(lower, sym_id)` | SPIR-V result ID for a resolver symbol |

---

## Raising (SPIR-V to WGSL)

### Raise Options

```c
typedef struct {
    int emit_debug_comments;  // add comments with SPIR-V IDs
    int preserve_names;       // use OpName debug info for identifiers
    int inline_constants;     // inline constant values instead of declaring them
} WgslRaiseOptions;
```

### Incremental API

The raiser can be used as a single convenience call or incrementally:

**One-shot**:
```c
WgslRaiseResult wgsl_raise_to_wgsl(spirv, word_count, &opts, &wgsl_out, &error);
```

**Incremental**:
```c
WgslRaiser *r = wgsl_raise_create(spirv, word_count);
wgsl_raise_parse(r);                          // deserialize SPIR-V
wgsl_raise_analyze(r);                        // reconstruct control flow
int n = wgsl_raise_entry_point_count(r);      // inspect before emitting
const char *name = wgsl_raise_entry_point_name(r, 0);
const char *wgsl = wgsl_raise_emit(r, &opts); // emit WGSL text
wgsl_raise_destroy(r);
```

---

## SSIR Emitters

### SSIR to SPIR-V

```c
SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
                                 const SsirToSpirvOptions *opts,
                                 uint32_t **out_words,
                                 size_t *out_count);
```

Options: `spirv_version`, `enable_debug_names`, `enable_line_info`.

### SSIR to WGSL

```c
SsirToWgslResult ssir_to_wgsl(const SsirModule *mod,
                               const SsirToWgslOptions *opts,
                               char **out_wgsl,
                               char **out_error);
```

Options: `preserve_names` (use debug names for identifiers).

### SSIR to GLSL

```c
SsirToGlslResult ssir_to_glsl(const SsirModule *mod,
                                SsirStage stage,
                                const SsirToGlslOptions *opts,
                                char **out_glsl,
                                char **out_error);
```

Options: `preserve_names`, `target_opengl` (suppress Vulkan-only qualifiers like `set = N`).

Requires a `stage` parameter because GLSL programs are per-stage (unlike WGSL modules which can contain multiple entry points).

### SSIR to MSL

```c
SsirToMslResult ssir_to_msl(const SsirModule *mod,
                              const SsirToMslOptions *opts,
                              char **out_msl,
                              char **out_error);
```

Options: `preserve_names`.

### SSIR to HLSL

```c
SsirToHlslResult ssir_to_hlsl(const SsirModule *mod,
                                SsirStage stage,
                                const SsirToHlslOptions *opts,
                                char **out_hlsl,
                                char **out_error);
```

Options: `preserve_names`, `shader_model_major`, `shader_model_minor`.

---

## SPIR-V to SSIR Deserialization

```c
SpirvToSsirResult spirv_to_ssir(const uint32_t *spirv,
                                 size_t word_count,
                                 const SpirvToSsirOptions *opts,
                                 SsirModule **out_module,
                                 char **out_error);
```

Options: `preserve_names` (import OpName debug info), `preserve_locations` (keep location decorations).

Parses a SPIR-V binary into a full `SsirModule`. Handles:
- All standard SPIR-V type instructions
- Constants and specialization constants
- Global variable decorations (group, binding, location, builtin, interpolation)
- Function bodies with full control flow
- Texture sampling and storage operations
- Atomic operations and barriers

---

## MSL to SSIR Parser

```c
MslToSsirResult msl_to_ssir(const char *msl_source,
                              const MslToSsirOptions *opts,
                              SsirModule **out_module,
                              char **out_error);
```

Parses Metal Shading Language source directly into an `SsirModule`, bypassing the AST. Options: `preserve_names`.

---

## Memory Management

All memory allocation in simple_wgsl goes through overridable macros. Define these before including `simple_wgsl.h` to use custom allocators:

**AST allocators** (used by parsers):
```c
#define NODE_ALLOC(T)      my_alloc(sizeof(T))   // typed allocation
#define NODE_MALLOC(SZ)    my_alloc(SZ)           // raw allocation
#define NODE_REALLOC(P,SZ) my_realloc(P, SZ)      // reallocation
#define NODE_FREE(P)       my_free(P)              // deallocation
```

**SSIR allocators** (used by SSIR module and all emitters):
```c
#define SSIR_MALLOC(sz)     my_alloc(sz)
#define SSIR_REALLOC(p,sz)  my_realloc(p, sz)
#define SSIR_FREE(p)        my_free(p)
```

Default: `calloc`/`realloc`/`free` from the C standard library. Note that `NODE_ALLOC` and `NODE_MALLOC` use `calloc` (zero-initialized) by default.

**Ownership rules**:
- `wgsl_parse()` returns an AST owned by the caller. Free with `wgsl_free_ast()`.
- `wgsl_resolver_build()` returns a resolver owned by the caller. Free with `wgsl_resolver_free()`.
- `wgsl_lower_create()` returns a lower context that owns its internal SSIR. Free with `wgsl_lower_destroy()`.
- `wgsl_lower_emit_spirv()` allocates the output SPIR-V buffer. Free with `wgsl_lower_free()`.
- `spirv_to_ssir()` allocates the output module. Free with `ssir_module_destroy()`.
- All `ssir_to_*()` output strings/buffers are freed with their respective `*_free()` functions.
- `wgsl_raise_to_wgsl()` output is freed with `wgsl_raise_free()`.

---

## Language Mapping Tables

### WGSL to GLSL Mapping

| WGSL / SSIR | GLSL 450 |
|-------------|----------|
| `f32` / `i32` / `u32` | `float` / `int` / `uint` |
| `vec4<f32>` | `vec4` |
| `vec3<i32>` | `ivec3` |
| `vec2<u32>` | `uvec2` |
| `mat4x4<f32>` | `mat4` |
| `mat3x3<f32>` | `mat3` |
| `@location(N) param` | `layout(location = N) in/out` |
| `@builtin(position)` (vertex out) | `gl_Position` |
| `@builtin(position)` (fragment in) | `gl_FragCoord` |
| `@builtin(vertex_index)` | `gl_VertexIndex` |
| `@builtin(instance_index)` | `gl_InstanceIndex` |
| `@builtin(frag_depth)` | `gl_FragDepth` |
| `@builtin(front_facing)` | `gl_FrontFacing` |
| `@builtin(global_invocation_id)` | `gl_GlobalInvocationID` |
| `@builtin(local_invocation_id)` | `gl_LocalInvocationID` |
| `@builtin(workgroup_id)` | `gl_WorkGroupID` |
| `@group(G) @binding(B) var<uniform>` | `layout(std140, set = G, binding = B) uniform` |
| `@group(G) @binding(B) var<storage>` | `layout(std430, set = G, binding = B) buffer` |
| `@workgroup_size(X,Y,Z)` | `layout(local_size_x=X, ...) in;` |
| Entry point function | `void main()` |
| `inverseSqrt` | `inversesqrt` |
| `countOneBits` | `bitCount` |
| `dpdx` / `dpdy` | `dFdx` / `dFdy` |
| `textureSample(t, s, c)` | `texture(t, c)` |
| `textureLoad(t, c, l)` | `texelFetch(t, c, l)` |

---

## Error Handling

Every major API function returns a result enum. The pattern is consistent across all modules:

```c
// Check the result code
if (result != XXX_OK) {
    // Get the error message (if the API provides one)
    const char *msg = xxx_last_error(context);
    fprintf(stderr, "Error: %s\n", msg);
}
```

Result enums follow the naming convention `<Module>Result` with values `*_OK = 0` for success and specific error codes for different failure modes.

All emitters provide a `*_result_string()` function that converts a result code to a human-readable string:

```c
ssir_to_spirv_result_string(result)
ssir_to_wgsl_result_string(result)
ssir_to_glsl_result_string(result)
ssir_to_msl_result_string(result)
ssir_to_hlsl_result_string(result)
spirv_to_ssir_result_string(result)
msl_to_ssir_result_string(result)
```
