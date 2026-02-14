# Tutorial

Step-by-step guides for every major use case of simple_wgsl.

---

## Table of Contents

- [Setup](#setup)
- [1. Parsing WGSL Source](#1-parsing-wgsl-source)
  - [Parsing and Inspecting the AST](#parsing-and-inspecting-the-ast)
  - [Walking the AST Manually](#walking-the-ast-manually)
- [2. Parsing GLSL Source](#2-parsing-glsl-source)
- [3. Resolving Symbols and Bindings](#3-resolving-symbols-and-bindings)
  - [Querying the Symbol Table](#querying-the-symbol-table)
  - [Extracting Resource Bindings](#extracting-resource-bindings)
  - [Enumerating Entry Points](#enumerating-entry-points)
  - [Vertex Input Reflection](#vertex-input-reflection)
  - [Fragment Output Reflection](#fragment-output-reflection)
  - [Per-Entry-Point Resource Queries](#per-entry-point-resource-queries)
- [4. Compiling to SPIR-V](#4-compiling-to-spirv)
  - [One-Shot Compilation](#one-shot-compilation)
  - [Two-Phase Compilation with Inspection](#two-phase-compilation-with-inspection)
  - [Writing SPIR-V to a File](#writing-spirv-to-a-file)
  - [Choosing a Target Environment](#choosing-a-target-environment)
- [5. Decompiling SPIR-V to WGSL](#5-decompiling-spirv-to-wgsl)
  - [One-Shot Decompilation](#one-shot-decompilation)
  - [Incremental Decompilation](#incremental-decompilation)
- [6. Cross-Compiling Between Languages](#6-cross-compiling-between-languages)
  - [WGSL to GLSL 450](#wgsl-to-glsl-450)
  - [WGSL to MSL (Metal)](#wgsl-to-msl-metal)
  - [WGSL to HLSL](#wgsl-to-hlsl)
  - [SPIR-V to GLSL](#spirv-to-glsl)
  - [MSL to SPIR-V](#msl-to-spirv)
- [7. Working with SSIR Directly](#7-working-with-ssir-directly)
  - [Inspecting SSIR After Lowering](#inspecting-ssir-after-lowering)
  - [Building an SSIR Module from Scratch](#building-an-ssir-module-from-scratch)
  - [Validating an SSIR Module](#validating-an-ssir-module)
  - [Printing SSIR as Text](#printing-ssir-as-text)
- [8. Full Round-Trip: WGSL to SPIR-V to WGSL](#8-full-round-trip-wgsl-to-spirv-to-wgsl)
- [9. Full Round-Trip: WGSL to GLSL to SPIR-V](#9-full-round-trip-wgsl-to-glsl-to-spirv)
- [10. Custom Memory Allocators](#10-custom-memory-allocators)
- [11. Error Handling Patterns](#11-error-handling-patterns)
- [12. Real-World Examples](#12-real-world-examples)
  - [Compute Shader: Double Array Values](#compute-shader-double-array-values)
  - [Vertex + Fragment Pipeline](#vertex--fragment-pipeline)
  - [Texture Sampling Shader](#texture-sampling-shader)

---

## Setup

Include the single public header. Link against the `wgsl_compiler` static library (or compile the `.c` files directly).

```c
#include "simple_wgsl.h"
#include <stdio.h>
#include <stdlib.h>
```

All examples below assume these includes. Every function shown compiles and runs as a standalone program when linked with `wgsl_compiler`.

---

## 1. Parsing WGSL Source

### Parsing and Inspecting the AST

The simplest operation: parse WGSL source into an AST, print it, and free it.

```c
int main(void) {
    const char *source =
        "struct Uniforms {\n"
        "    mvp: mat4x4<f32>,\n"
        "};\n"
        "\n"
        "@group(0) @binding(0) var<uniform> uniforms: Uniforms;\n"
        "\n"
        "@vertex\n"
        "fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n"
        "    return uniforms.mvp * vec4<f32>(pos, 1.0);\n"
        "}\n";

    WgslAstNode *ast = wgsl_parse(source);
    if (!ast) {
        fprintf(stderr, "Parse error\n");
        return 1;
    }

    // Print the AST tree for debugging
    wgsl_debug_print(ast, 0);

    // Check the number of top-level declarations
    printf("Top-level declarations: %d\n", ast->program.decl_count);

    wgsl_free_ast(ast);
    return 0;
}
```

`wgsl_parse()` returns `NULL` on parse error. On success, the returned `WgslAstNode*` has type `WGSL_NODE_PROGRAM` and contains an array of top-level declarations (structs, functions, global variables).

### Walking the AST Manually

You can traverse the AST yourself to extract information:

```c
void list_functions(const WgslAstNode *program) {
    for (int i = 0; i < program->program.decl_count; i++) {
        const WgslAstNode *decl = program->program.decls[i];

        if (decl->type == WGSL_NODE_FUNCTION) {
            printf("Function: %s (line %d)\n", decl->function.name, decl->line);
            printf("  Parameters: %d\n", decl->function.param_count);

            // Check for shader stage attributes
            for (int a = 0; a < decl->function.attr_count; a++) {
                const WgslAstNode *attr = decl->function.attrs[a];
                printf("  Attribute: @%s\n", attr->attribute.name);
            }

            if (decl->function.return_type) {
                printf("  Return type: %s\n", decl->function.return_type->type_node.name);
            }
        }

        if (decl->type == WGSL_NODE_STRUCT) {
            printf("Struct: %s (%d fields)\n",
                   decl->struct_decl.name,
                   decl->struct_decl.field_count);
        }

        if (decl->type == WGSL_NODE_GLOBAL_VAR) {
            printf("Global var: %s (address space: %s)\n",
                   decl->global_var.name,
                   decl->global_var.address_space ? decl->global_var.address_space : "default");
        }
    }
}
```

---

## 2. Parsing GLSL Source

GLSL parsing uses the same AST types as WGSL. You must specify the shader stage because GLSL doesn't have stage attributes in the source.

```c
int main(void) {
    const char *glsl_source =
        "#version 450\n"
        "\n"
        "layout(location = 0) in vec3 inPosition;\n"
        "layout(location = 1) in vec3 inColor;\n"
        "layout(location = 0) out vec3 fragColor;\n"
        "\n"
        "layout(set = 0, binding = 0) uniform UBO {\n"
        "    mat4 mvp;\n"
        "} ubo;\n"
        "\n"
        "void main() {\n"
        "    gl_Position = ubo.mvp * vec4(inPosition, 1.0);\n"
        "    fragColor = inColor;\n"
        "}\n";

    WgslAstNode *ast = glsl_parse(glsl_source, WGSL_STAGE_VERTEX);
    if (!ast) {
        fprintf(stderr, "GLSL parse error\n");
        return 1;
    }

    wgsl_debug_print(ast, 0);
    wgsl_free_ast(ast);
    return 0;
}
```

Because `glsl_parse()` produces the same `WgslAstNode*` tree, you can pass it through the same resolver and lowering pipeline as WGSL.

---

## 3. Resolving Symbols and Bindings

The resolver performs semantic analysis on a parsed AST, building a symbol table and extracting resource binding information.

### Querying the Symbol Table

```c
int main(void) {
    const char *source =
        "@group(0) @binding(0) var<uniform> transform: mat4x4<f32>;\n"
        "@group(0) @binding(1) var<storage, read> data: array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    var local_val: f32 = data[gid.x];\n"
        "}\n";

    WgslAstNode *ast = wgsl_parse(source);
    WgslResolver *r = wgsl_resolver_build(ast);

    // Get all symbols (globals + params + locals)
    int count;
    const WgslSymbolInfo *syms = wgsl_resolver_all_symbols(r, &count);
    for (int i = 0; i < count; i++) {
        const char *kind_str = "?";
        switch (syms[i].kind) {
            case WGSL_SYM_GLOBAL: kind_str = "global"; break;
            case WGSL_SYM_PARAM:  kind_str = "param";  break;
            case WGSL_SYM_LOCAL:  kind_str = "local";  break;
        }
        printf("Symbol [%d]: %s (%s)\n", syms[i].id, syms[i].name, kind_str);
    }

    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
    return 0;
}
```

### Extracting Resource Bindings

```c
void print_bindings(const WgslResolver *r) {
    int count;
    const WgslSymbolInfo *bindings = wgsl_resolver_binding_vars(r, &count);
    printf("Resource bindings (%d):\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %s: group=%d, binding=%d\n",
               bindings[i].name,
               bindings[i].group_index,
               bindings[i].binding_index);
    }
}
```

This is exactly the information you need to create Vulkan descriptor set layouts or WebGPU bind group layouts.

### Enumerating Entry Points

```c
void print_entry_points(const WgslResolver *r) {
    int count;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &count);
    for (int i = 0; i < count; i++) {
        const char *stage_name = "unknown";
        switch (eps[i].stage) {
            case WGSL_STAGE_VERTEX:   stage_name = "vertex";   break;
            case WGSL_STAGE_FRAGMENT: stage_name = "fragment"; break;
            case WGSL_STAGE_COMPUTE:  stage_name = "compute";  break;
            default: break;
        }
        printf("Entry point: %s (%s)\n", eps[i].name, stage_name);
    }
}
```

### Vertex Input Reflection

Useful for constructing `VkVertexInputAttributeDescription` arrays:

```c
void print_vertex_inputs(const WgslResolver *r, const char *entry_name) {
    WgslVertexSlot *slots;
    int count = wgsl_resolver_vertex_inputs(r, entry_name, &slots);

    printf("Vertex inputs for '%s' (%d):\n", entry_name, count);
    for (int i = 0; i < count; i++) {
        printf("  location=%d components=%d bytes=%d\n",
               slots[i].location,
               slots[i].component_count,
               slots[i].byte_size);
    }
    wgsl_resolve_free(slots);
}
```

### Fragment Output Reflection

```c
void print_fragment_outputs(const WgslResolver *r, const char *entry_name) {
    WgslFragmentOutput *outputs;
    int count = wgsl_resolver_fragment_outputs(r, entry_name, &outputs);

    printf("Fragment outputs for '%s' (%d):\n", entry_name, count);
    for (int i = 0; i < count; i++) {
        printf("  location=%d components=%d\n",
               outputs[i].location,
               outputs[i].component_count);
    }
    wgsl_resolve_free(outputs);
}
```

### Per-Entry-Point Resource Queries

When a module has multiple entry points, you often want to know which resources each entry point actually uses:

```c
void print_entry_resources(const WgslResolver *r) {
    int ep_count;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &ep_count);

    for (int i = 0; i < ep_count; i++) {
        printf("Entry point '%s' uses:\n", eps[i].name);

        int bind_count;
        const WgslSymbolInfo *binds =
            wgsl_resolver_entrypoint_binding_vars(r, eps[i].name, &bind_count);

        for (int j = 0; j < bind_count; j++) {
            printf("  group=%d binding=%d (%s)\n",
                   binds[j].group_index,
                   binds[j].binding_index,
                   binds[j].name);
        }
    }
}
```

---

## 4. Compiling to SPIR-V

### One-Shot Compilation

The simplest way to compile WGSL to SPIR-V:

```c
int main(void) {
    const char *source =
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    data[gid.x] = data[gid.x] * 2.0;\n"
        "}\n";

    // 1. Parse
    WgslAstNode *ast = wgsl_parse(source);
    if (!ast) { fprintf(stderr, "Parse error\n"); return 1; }

    // 2. Resolve
    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) { fprintf(stderr, "Resolve error\n"); wgsl_free_ast(ast); return 1; }

    // 3. Lower to SPIR-V
    WgslLowerOptions opts = {0};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    opts.enable_debug_names = 1;

    uint32_t *spirv;
    size_t word_count;
    WgslLowerResult res = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);

    if (res == WGSL_LOWER_OK) {
        printf("Success: %zu SPIR-V words (%zu bytes)\n", word_count, word_count * 4);
        // Use spirv/word_count with Vulkan, WebGPU, etc.
    } else {
        fprintf(stderr, "Lower error: %d\n", res);
    }

    // 4. Cleanup
    wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

### Two-Phase Compilation with Inspection

Use the two-phase API to inspect the SSIR module and entry points before serializing:

```c
int main(void) {
    const char *source = /* ... */;
    WgslAstNode *ast = wgsl_parse(source);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    WgslLowerOptions opts = {0};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    opts.enable_debug_names = 1;

    // Create the lower context (does all compilation work)
    WgslLower *lower = wgsl_lower_create(ast, resolver, &opts);
    if (!lower) {
        fprintf(stderr, "Lower error: %s\n", "creation failed");
        return 1;
    }

    // Inspect: entry points
    int ep_count;
    const WgslLowerEntrypointInfo *eps = wgsl_lower_entrypoints(lower, &ep_count);
    for (int i = 0; i < ep_count; i++) {
        printf("Entry: %s (function_id=%u, %u interface vars)\n",
               eps[i].name, eps[i].function_id, eps[i].interface_count);
    }

    // Inspect: capabilities and extensions
    const WgslLowerModuleFeatures *features = wgsl_lower_module_features(lower);
    printf("Capabilities: %u, Extensions: %u\n",
           features->capability_count, features->extension_count);

    // Inspect: the internal SSIR module
    const SsirModule *ssir = wgsl_lower_get_ssir(lower);
    printf("SSIR: %u types, %u constants, %u globals, %u functions\n",
           ssir->type_count, ssir->constant_count,
           ssir->global_count, ssir->function_count);

    // Serialize to SPIR-V
    uint32_t *spirv;
    size_t word_count;
    WgslLowerResult res = wgsl_lower_serialize(lower, &spirv, &word_count);

    // ... use spirv ...

    wgsl_lower_free(spirv);
    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

### Writing SPIR-V to a File

```c
void write_spirv(const char *path, const uint32_t *spirv, size_t word_count) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); return; }
    fwrite(spirv, sizeof(uint32_t), word_count, f);
    fclose(f);
    printf("Wrote %zu bytes to %s\n", word_count * 4, path);
}
```

The output is a standard SPIR-V binary that can be validated with `spirv-val`, disassembled with `spirv-dis`, or loaded directly by Vulkan's `vkCreateShaderModule`.

### Choosing a Target Environment

```c
WgslLowerOptions opts = {0};

// For Vulkan 1.2 applications (recommended default)
opts.env = WGSL_LOWER_ENV_VULKAN_1_2;

// For Vulkan 1.3 (enables additional features)
opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

// For WebGPU (Dawn, wgpu)
opts.env = WGSL_LOWER_ENV_WEBGPU;

// Struct packing
opts.packing = WGSL_LOWER_PACK_STD430;  // or STD140

// Debug info
opts.enable_debug_names = 1;  // OpName for better debugging
opts.enable_line_info = 1;    // OpLine for source mapping

// Safety
opts.zero_initialize_vars = 1;  // zero-init all variables
```

---

## 5. Decompiling SPIR-V to WGSL

### One-Shot Decompilation

```c
int main(void) {
    // Assume spirv/word_count were loaded from a .spv file
    uint32_t *spirv = /* ... */;
    size_t word_count = /* ... */;

    WgslRaiseOptions opts = {0};
    opts.preserve_names = 1;       // use OpName debug info
    opts.inline_constants = 1;     // inline constant values
    opts.emit_debug_comments = 0;  // don't add SPIR-V ID comments

    char *wgsl_out = NULL;
    char *error = NULL;
    WgslRaiseResult res = wgsl_raise_to_wgsl(spirv, word_count, &opts, &wgsl_out, &error);

    if (res == WGSL_RAISE_SUCCESS) {
        printf("Decompiled WGSL:\n%s\n", wgsl_out);
    } else {
        fprintf(stderr, "Raise error: %s\n", error ? error : "unknown");
    }

    wgsl_raise_free(wgsl_out);
    wgsl_raise_free(error);
    return 0;
}
```

### Incremental Decompilation

The incremental API lets you inspect the SPIR-V before emitting WGSL:

```c
int main(void) {
    uint32_t *spirv = /* ... */;
    size_t word_count = /* ... */;

    WgslRaiser *raiser = wgsl_raise_create(spirv, word_count);
    if (!raiser) { fprintf(stderr, "Failed to create raiser\n"); return 1; }

    // Phase 1: Parse the SPIR-V binary
    WgslRaiseResult res = wgsl_raise_parse(raiser);
    if (res != WGSL_RAISE_SUCCESS) { wgsl_raise_destroy(raiser); return 1; }

    // Phase 2: Analyze (reconstruct control flow, etc.)
    res = wgsl_raise_analyze(raiser);
    if (res != WGSL_RAISE_SUCCESS) { wgsl_raise_destroy(raiser); return 1; }

    // Inspect entry points before emitting
    int ep_count = wgsl_raise_entry_point_count(raiser);
    printf("Found %d entry point(s):\n", ep_count);
    for (int i = 0; i < ep_count; i++) {
        printf("  %s\n", wgsl_raise_entry_point_name(raiser, i));
    }

    // Phase 3: Emit WGSL
    WgslRaiseOptions opts = {0};
    opts.preserve_names = 1;
    const char *wgsl = wgsl_raise_emit(raiser, &opts);
    printf("WGSL:\n%s\n", wgsl);

    wgsl_raise_destroy(raiser);
    return 0;
}
```

---

## 6. Cross-Compiling Between Languages

### WGSL to GLSL 450

Compile WGSL source to GLSL 450 via the SSIR path:

```c
int main(void) {
    const char *wgsl_source =
        "@vertex\n"
        "fn main(@location(0) pos: vec3<f32>,\n"
        "        @location(1) color: vec3<f32>)\n"
        "     -> @builtin(position) vec4<f32> {\n"
        "    return vec4<f32>(pos, 1.0);\n"
        "}\n";

    // 1. Parse and resolve
    WgslAstNode *ast = wgsl_parse(wgsl_source);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    // 2. Lower to get SSIR
    WgslLowerOptions lower_opts = {0};
    lower_opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    lower_opts.enable_debug_names = 1;

    WgslLower *lower = wgsl_lower_create(ast, resolver, &lower_opts);
    const SsirModule *ssir = wgsl_lower_get_ssir(lower);

    // 3. Emit GLSL from SSIR
    SsirToGlslOptions glsl_opts = {0};
    glsl_opts.preserve_names = 1;

    char *glsl_out = NULL;
    char *error = NULL;
    SsirToGlslResult res = ssir_to_glsl(ssir, SSIR_STAGE_VERTEX, &glsl_opts,
                                          &glsl_out, &error);

    if (res == SSIR_TO_GLSL_OK) {
        printf("GLSL 450:\n%s\n", glsl_out);
    } else {
        fprintf(stderr, "Error: %s\n", error ? error : ssir_to_glsl_result_string(res));
    }

    ssir_to_glsl_free(glsl_out);
    ssir_to_glsl_free(error);
    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

Note: `ssir_to_glsl` requires a `stage` parameter because GLSL programs are per-stage.

### WGSL to MSL (Metal)

Same pattern, but using `ssir_to_msl`:

```c
    // After getting ssir from wgsl_lower_get_ssir(lower):

    SsirToMslOptions msl_opts = {0};
    msl_opts.preserve_names = 1;

    char *msl_out = NULL;
    char *error = NULL;
    SsirToMslResult res = ssir_to_msl(ssir, &msl_opts, &msl_out, &error);

    if (res == SSIR_TO_MSL_OK) {
        printf("MSL:\n%s\n", msl_out);
    }

    ssir_to_msl_free(msl_out);
    ssir_to_msl_free(error);
```

### WGSL to HLSL

```c
    // After getting ssir from wgsl_lower_get_ssir(lower):

    SsirToHlslOptions hlsl_opts = {0};
    hlsl_opts.preserve_names = 1;
    hlsl_opts.shader_model_major = 6;
    hlsl_opts.shader_model_minor = 0;

    char *hlsl_out = NULL;
    char *error = NULL;
    SsirToHlslResult res = ssir_to_hlsl(ssir, SSIR_STAGE_VERTEX, &hlsl_opts,
                                          &hlsl_out, &error);

    if (res == SSIR_TO_HLSL_OK) {
        printf("HLSL:\n%s\n", hlsl_out);
    }

    ssir_to_hlsl_free(hlsl_out);
    ssir_to_hlsl_free(error);
```

### SPIR-V to GLSL

Decompile a SPIR-V binary to GLSL 450:

```c
int main(void) {
    uint32_t *spirv = /* ... */;
    size_t word_count = /* ... */;

    // 1. Deserialize SPIR-V into SSIR
    SpirvToSsirOptions spv_opts = {0};
    spv_opts.preserve_names = true;
    spv_opts.preserve_locations = true;

    SsirModule *module = NULL;
    char *error = NULL;
    SpirvToSsirResult res = spirv_to_ssir(spirv, word_count, &spv_opts, &module, &error);

    if (res != SPIRV_TO_SSIR_SUCCESS) {
        fprintf(stderr, "SPIR-V error: %s\n", error ? error : "unknown");
        spirv_to_ssir_free(error);
        return 1;
    }

    // 2. Emit GLSL from SSIR
    SsirToGlslOptions glsl_opts = {0};
    glsl_opts.preserve_names = 1;

    char *glsl_out = NULL;
    char *glsl_error = NULL;
    ssir_to_glsl(module, SSIR_STAGE_VERTEX, &glsl_opts, &glsl_out, &glsl_error);

    printf("GLSL:\n%s\n", glsl_out);

    ssir_to_glsl_free(glsl_out);
    ssir_to_glsl_free(glsl_error);
    ssir_module_destroy(module);
    spirv_to_ssir_free(error);
    return 0;
}
```

### MSL to SPIR-V

Parse MSL source, convert to SSIR, then serialize to SPIR-V:

```c
int main(void) {
    const char *msl_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "kernel void add_arrays(device float *a [[buffer(0)]],\n"
        "                       device float *b [[buffer(1)]],\n"
        "                       device float *result [[buffer(2)]],\n"
        "                       uint idx [[thread_position_in_grid]]) {\n"
        "    result[idx] = a[idx] + b[idx];\n"
        "}\n";

    // 1. Parse MSL to SSIR
    MslToSsirOptions msl_opts = {0};
    msl_opts.preserve_names = 1;

    SsirModule *module = NULL;
    char *error = NULL;
    MslToSsirResult res = msl_to_ssir(msl_source, &msl_opts, &module, &error);

    if (res != MSL_TO_SSIR_OK) {
        fprintf(stderr, "MSL error: %s\n", error ? error : msl_to_ssir_result_string(res));
        msl_to_ssir_free(error);
        return 1;
    }

    // 2. Serialize SSIR to SPIR-V
    SsirToSpirvOptions spv_opts = {0};
    spv_opts.enable_debug_names = 1;

    uint32_t *spirv = NULL;
    size_t word_count = 0;
    SsirToSpirvResult spv_res = ssir_to_spirv(module, &spv_opts, &spirv, &word_count);

    if (spv_res == SSIR_TO_SPIRV_OK) {
        printf("Generated %zu SPIR-V words\n", word_count);
    }

    ssir_to_spirv_free(spirv);
    ssir_module_destroy(module);
    msl_to_ssir_free(error);
    return 0;
}
```

---

## 7. Working with SSIR Directly

### Inspecting SSIR After Lowering

After compiling WGSL, you can inspect the SSIR module:

```c
void inspect_ssir(const SsirModule *mod) {
    printf("Module: %u types, %u constants, %u globals, %u functions, %u entry points\n",
           mod->type_count, mod->constant_count, mod->global_count,
           mod->function_count, mod->entry_point_count);

    // Iterate types
    for (uint32_t i = 0; i < mod->type_count; i++) {
        const SsirType *t = &mod->types[i];
        printf("  Type[%u] (id=%u): %s", i, t->id, ssir_type_kind_name(t->kind));
        if (t->kind == SSIR_TYPE_VEC) {
            printf(" <%u x %u>", t->vec.elem, t->vec.size);
        }
        if (t->kind == SSIR_TYPE_STRUCT) {
            printf(" '%s' (%u members)", t->struc.name ? t->struc.name : "<anon>",
                   t->struc.member_count);
        }
        printf("\n");
    }

    // Iterate functions
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const SsirFunction *fn = &mod->functions[i];
        printf("  Function[%u] '%s': %u params, %u locals, %u blocks\n",
               i, fn->name ? fn->name : "<unnamed>",
               fn->param_count, fn->local_count, fn->block_count);

        for (uint32_t b = 0; b < fn->block_count; b++) {
            const SsirBlock *blk = &fn->blocks[b];
            printf("    Block[%u] (id=%u): %u instructions\n",
                   b, blk->id, blk->inst_count);

            for (uint32_t j = 0; j < blk->inst_count; j++) {
                const SsirInst *inst = &blk->insts[j];
                printf("      %%%u = %s", inst->result, ssir_opcode_name(inst->op));
                for (uint8_t k = 0; k < inst->operand_count; k++) {
                    printf(" %%%u", inst->operands[k]);
                }
                printf("\n");
            }
        }
    }
}
```

### Building an SSIR Module from Scratch

You can build an SSIR module programmatically without parsing any source language. This is useful for code generation, IR manipulation, or creating test cases.

This example builds a compute shader that writes `42.0` to a storage buffer:

```c
int main(void) {
    SsirModule *mod = ssir_module_create();

    // --- Types ---
    uint32_t void_t   = ssir_type_void(mod);
    uint32_t f32_t    = ssir_type_f32(mod);
    uint32_t u32_t    = ssir_type_u32(mod);
    uint32_t vec3u_t  = ssir_type_vec(mod, u32_t, 3);
    uint32_t rt_arr_t = ssir_type_runtime_array(mod, f32_t);

    // Struct wrapping the runtime array (for storage buffer)
    uint32_t members[] = { rt_arr_t };
    uint32_t offsets[] = { 0 };
    uint32_t struct_t = ssir_type_struct(mod, "Data", members, 1, offsets);

    // Pointer types
    uint32_t ptr_struct_t = ssir_type_ptr(mod, struct_t, SSIR_ADDR_STORAGE);
    uint32_t ptr_f32_t    = ssir_type_ptr(mod, f32_t, SSIR_ADDR_STORAGE);
    uint32_t ptr_vec3u_t  = ssir_type_ptr(mod, vec3u_t, SSIR_ADDR_INPUT);

    // --- Constants ---
    uint32_t const_0   = ssir_const_u32(mod, 0);
    uint32_t const_42  = ssir_const_f32(mod, 42.0f);

    // --- Globals ---
    // Storage buffer: @group(0) @binding(0) var<storage, read_write> data: Data
    uint32_t data_var = ssir_global_var(mod, "data", ptr_struct_t);
    ssir_global_set_group(mod, data_var, 0);
    ssir_global_set_binding(mod, data_var, 0);

    // Built-in input: @builtin(global_invocation_id)
    uint32_t gid_var = ssir_global_var(mod, "gid", ptr_vec3u_t);
    ssir_global_set_builtin(mod, gid_var, SSIR_BUILTIN_GLOBAL_INVOCATION_ID);

    // --- Function ---
    uint32_t func = ssir_function_create(mod, "main", void_t);
    uint32_t entry_block = ssir_block_create(mod, func, "entry");

    // Load gid.x
    uint32_t gid_val = ssir_build_load(mod, func, entry_block, vec3u_t, gid_var);
    uint32_t gid_x = ssir_build_extract(mod, func, entry_block, u32_t, gid_val, 0);

    // Access data.arr[gid.x]
    uint32_t indices[] = { const_0, gid_x };
    uint32_t elem_ptr = ssir_build_access(mod, func, entry_block, ptr_f32_t,
                                           data_var, indices, 2);

    // Store 42.0
    ssir_build_store(mod, func, entry_block, elem_ptr, const_42);

    // Return
    ssir_build_return_void(mod, func, entry_block);

    // --- Entry Point ---
    uint32_t ep = ssir_entry_point_create(mod, SSIR_STAGE_COMPUTE, func, "main");
    ssir_entry_point_add_interface(mod, ep, data_var);
    ssir_entry_point_add_interface(mod, ep, gid_var);
    ssir_entry_point_set_workgroup_size(mod, ep, 64, 1, 1);

    // --- Emit SPIR-V ---
    SsirToSpirvOptions opts = {0};
    opts.enable_debug_names = 1;

    uint32_t *spirv;
    size_t word_count;
    SsirToSpirvResult res = ssir_to_spirv(mod, &opts, &spirv, &word_count);

    if (res == SSIR_TO_SPIRV_OK) {
        printf("Generated %zu SPIR-V words from hand-built SSIR\n", word_count);
    }

    ssir_to_spirv_free(spirv);
    ssir_module_destroy(mod);
    return 0;
}
```

### Validating an SSIR Module

Run validation on a module to check for structural errors:

```c
void check_ssir(SsirModule *mod) {
    SsirValidationResult *result = ssir_validate(mod);

    if (result->valid) {
        printf("SSIR module is valid\n");
    } else {
        printf("SSIR validation errors (%u):\n", result->error_count);
        for (uint32_t i = 0; i < result->error_count; i++) {
            SsirValidationError *e = &result->errors[i];
            printf("  [func=%u block=%u inst=%u] %s\n",
                   e->func_id, e->block_id, e->inst_index, e->message);
        }
    }

    ssir_validation_result_free(result);
}
```

### Printing SSIR as Text

Get a human-readable dump of the entire SSIR module:

```c
void dump_ssir(SsirModule *mod) {
    char *text = ssir_module_to_string(mod);
    if (text) {
        printf("%s\n", text);
        free(text);
    }
}
```

This produces output similar to SPIR-V disassembly, showing all types, constants, globals, functions, blocks, and instructions with their IDs.

---

## 8. Full Round-Trip: WGSL to SPIR-V to WGSL

Compile WGSL to SPIR-V, then decompile back to WGSL:

```c
int main(void) {
    const char *original =
        "@group(0) @binding(0) var<storage, read_write> buf: array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    buf[gid.x] = buf[gid.x] + 1.0;\n"
        "}\n";

    printf("Original WGSL:\n%s\n", original);

    // Forward: WGSL -> SPIR-V
    WgslAstNode *ast = wgsl_parse(original);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    WgslLowerOptions lower_opts = {0};
    lower_opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    lower_opts.enable_debug_names = 1;

    uint32_t *spirv;
    size_t word_count;
    wgsl_lower_emit_spirv(ast, resolver, &lower_opts, &spirv, &word_count);

    // Reverse: SPIR-V -> WGSL
    WgslRaiseOptions raise_opts = {0};
    raise_opts.preserve_names = 1;

    char *roundtripped = NULL;
    char *error = NULL;
    wgsl_raise_to_wgsl(spirv, word_count, &raise_opts, &roundtripped, &error);

    printf("Round-tripped WGSL:\n%s\n", roundtripped);

    wgsl_raise_free(roundtripped);
    wgsl_raise_free(error);
    wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

The round-tripped WGSL will be semantically equivalent but may differ syntactically (different variable names, expanded type constructors, etc.).

---

## 9. Full Round-Trip: WGSL to GLSL to SPIR-V

Validate the GLSL cross-compilation by compiling back to SPIR-V:

```c
int main(void) {
    const char *wgsl_source =
        "@vertex\n"
        "fn main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n"
        "    return vec4<f32>(pos, 1.0);\n"
        "}\n";

    // Step 1: WGSL -> SSIR
    WgslAstNode *ast1 = wgsl_parse(wgsl_source);
    WgslResolver *res1 = wgsl_resolver_build(ast1);
    WgslLowerOptions opts1 = {0};
    opts1.env = WGSL_LOWER_ENV_VULKAN_1_2;
    opts1.enable_debug_names = 1;
    WgslLower *lower1 = wgsl_lower_create(ast1, res1, &opts1);
    const SsirModule *ssir1 = wgsl_lower_get_ssir(lower1);

    // Step 2: SSIR -> GLSL
    SsirToGlslOptions glsl_opts = {0};
    glsl_opts.preserve_names = 1;
    char *glsl = NULL, *err1 = NULL;
    ssir_to_glsl(ssir1, SSIR_STAGE_VERTEX, &glsl_opts, &glsl, &err1);
    printf("GLSL:\n%s\n", glsl);

    // Step 3: GLSL -> AST -> SPIR-V (validates the GLSL is compilable)
    WgslAstNode *ast2 = glsl_parse(glsl, WGSL_STAGE_VERTEX);
    WgslResolver *res2 = wgsl_resolver_build(ast2);
    WgslLowerOptions opts2 = {0};
    opts2.env = WGSL_LOWER_ENV_VULKAN_1_2;
    uint32_t *spirv;
    size_t word_count;
    WgslLowerResult lr = wgsl_lower_emit_spirv(ast2, res2, &opts2, &spirv, &word_count);

    if (lr == WGSL_LOWER_OK) {
        printf("Round-trip successful: WGSL -> GLSL -> SPIR-V (%zu words)\n", word_count);
    }

    // Cleanup
    wgsl_lower_free(spirv);
    wgsl_resolver_free(res2);
    wgsl_free_ast(ast2);
    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(err1);
    wgsl_lower_destroy(lower1);
    wgsl_resolver_free(res1);
    wgsl_free_ast(ast1);
    return 0;
}
```

---

## 10. Custom Memory Allocators

To embed simple_wgsl in an environment with custom memory management (game engines, WASM, embedded), define allocator macros before including the header:

```c
// my_allocator.h
void *my_arena_alloc(size_t size);
void *my_arena_realloc(void *ptr, size_t size);
void  my_arena_free(void *ptr);
```

```c
// my_shader_compiler.c
#include "my_allocator.h"

// Override AST allocators
#define NODE_ALLOC(T)       (T *)my_arena_alloc(sizeof(T))
#define NODE_MALLOC(SZ)     my_arena_alloc(SZ)
#define NODE_REALLOC(P, SZ) my_arena_realloc(P, SZ)
#define NODE_FREE(P)        my_arena_free(P)

// Override SSIR allocators
#define SSIR_MALLOC(sz)     my_arena_alloc(sz)
#define SSIR_REALLOC(p, sz) my_arena_realloc(p, sz)
#define SSIR_FREE(p)        my_arena_free(p)

#include "simple_wgsl.h"

// Now all library allocations go through your allocator
```

The macros must be defined before `#include "simple_wgsl.h"` in every translation unit that includes it. `NODE_ALLOC` and `NODE_MALLOC` default to `calloc` (zero-initialized memory), so your replacement should also zero-initialize for safety.

---

## 11. Error Handling Patterns

Every API follows a consistent pattern: call a function, check the result enum, and optionally retrieve an error message.

```c
// Pattern: one-shot with error output
char *error = NULL;
SsirToWgslResult res = ssir_to_wgsl(mod, &opts, &output, &error);
if (res != SSIR_TO_WGSL_OK) {
    fprintf(stderr, "Failed: %s (code: %s)\n",
            error ? error : "unknown",
            ssir_to_wgsl_result_string(res));
    ssir_to_wgsl_free(error);
    return;
}

// Pattern: lower context with last_error
WgslLower *lower = wgsl_lower_create(ast, resolver, &opts);
if (!lower) {
    fprintf(stderr, "Lowering failed\n");
    return;
}
WgslLowerResult res = wgsl_lower_serialize(lower, &spirv, &word_count);
if (res != WGSL_LOWER_OK) {
    fprintf(stderr, "Serialize failed: %s\n", wgsl_lower_last_error(lower));
}

// Pattern: checking for NULL returns
WgslAstNode *ast = wgsl_parse(source);
if (!ast) {
    fprintf(stderr, "Parse failed\n");
    return;
}
```

All `*_result_string()` functions convert result codes to human-readable strings:

```c
ssir_to_spirv_result_string(SSIR_TO_SPIRV_ERR_UNSUPPORTED)  // "unsupported"
spirv_to_ssir_result_string(SPIRV_TO_SSIR_INVALID_SPIRV)    // "invalid SPIR-V"
```

---

## 12. Real-World Examples

### Compute Shader: Double Array Values

A complete compute shader that doubles every element of a storage buffer:

```c
int main(void) {
    const char *source =
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    data[gid.x] = data[gid.x] * 2.0;\n"
        "}\n";

    WgslAstNode *ast = wgsl_parse(source);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    // Print bindings for descriptor set layout creation
    int bind_count;
    const WgslSymbolInfo *bindings = wgsl_resolver_binding_vars(resolver, &bind_count);
    for (int i = 0; i < bind_count; i++) {
        printf("Binding: %s at group=%d binding=%d\n",
               bindings[i].name, bindings[i].group_index, bindings[i].binding_index);
    }

    // Compile to SPIR-V
    WgslLowerOptions opts = {0};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_2;

    uint32_t *spirv;
    size_t word_count;
    wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);
    printf("SPIR-V: %zu words\n", word_count);

    // The spirv pointer can now be passed directly to vkCreateShaderModule

    wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

### Vertex + Fragment Pipeline

Compile a module with both vertex and fragment entry points:

```c
int main(void) {
    const char *source =
        "struct VertexOutput {\n"
        "    @builtin(position) pos: vec4<f32>,\n"
        "    @location(0) color: vec3<f32>,\n"
        "};\n"
        "\n"
        "@vertex\n"
        "fn vs_main(@location(0) position: vec3<f32>,\n"
        "           @location(1) color: vec3<f32>) -> VertexOutput {\n"
        "    var out: VertexOutput;\n"
        "    out.pos = vec4<f32>(position, 1.0);\n"
        "    out.color = color;\n"
        "    return out;\n"
        "}\n"
        "\n"
        "@fragment\n"
        "fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(color, 1.0);\n"
        "}\n";

    WgslAstNode *ast = wgsl_parse(source);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    // Enumerate entry points
    int ep_count;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(resolver, &ep_count);
    for (int i = 0; i < ep_count; i++) {
        printf("Entry: %s (stage=%d)\n", eps[i].name, eps[i].stage);
    }

    // Get vertex input layout for pipeline creation
    WgslVertexSlot *slots;
    int slot_count = wgsl_resolver_vertex_inputs(resolver, "vs_main", &slots);
    printf("Vertex inputs: %d\n", slot_count);
    for (int i = 0; i < slot_count; i++) {
        printf("  loc=%d components=%d bytes=%d\n",
               slots[i].location, slots[i].component_count, slots[i].byte_size);
    }
    wgsl_resolve_free(slots);

    // Get fragment outputs
    WgslFragmentOutput *frag_outs;
    int frag_count = wgsl_resolver_fragment_outputs(resolver, "fs_main", &frag_outs);
    printf("Fragment outputs: %d\n", frag_count);
    wgsl_resolve_free(frag_outs);

    // Compile to SPIR-V (contains both entry points)
    WgslLowerOptions opts = {0};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    opts.enable_debug_names = 1;

    uint32_t *spirv;
    size_t word_count;
    wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);
    printf("SPIR-V: %zu words (both stages in one module)\n", word_count);

    wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```

### Texture Sampling Shader

A fragment shader that samples a texture:

```c
int main(void) {
    const char *source =
        "@group(0) @binding(0) var t: texture_2d<f32>;\n"
        "@group(0) @binding(1) var s: sampler;\n"
        "\n"
        "@fragment\n"
        "fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
        "    return textureSample(t, s, uv);\n"
        "}\n";

    WgslAstNode *ast = wgsl_parse(source);
    WgslResolver *resolver = wgsl_resolver_build(ast);

    // Show the two bindings (texture + sampler)
    int count;
    const WgslSymbolInfo *bindings = wgsl_resolver_binding_vars(resolver, &count);
    for (int i = 0; i < count; i++) {
        printf("Binding: %s (group=%d, binding=%d)\n",
               bindings[i].name, bindings[i].group_index, bindings[i].binding_index);
    }

    // Compile to SPIR-V
    WgslLowerOptions opts = {0};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_2;
    opts.enable_debug_names = 1;

    uint32_t *spirv;
    size_t word_count;
    WgslLowerResult res = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);
    printf("Result: %s (%zu words)\n",
           res == WGSL_LOWER_OK ? "OK" : "ERROR", word_count);

    // Cross-compile to GLSL to verify the mapping
    WgslLower *lower = wgsl_lower_create(ast, resolver, &opts);
    const SsirModule *ssir = wgsl_lower_get_ssir(lower);

    SsirToGlslOptions glsl_opts = {0};
    glsl_opts.preserve_names = 1;
    char *glsl = NULL, *error = NULL;
    ssir_to_glsl(ssir, SSIR_STAGE_FRAGMENT, &glsl_opts, &glsl, &error);

    if (glsl) {
        printf("\nEquivalent GLSL:\n%s\n", glsl);
    }

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
    wgsl_lower_destroy(lower);
    wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return 0;
}
```
