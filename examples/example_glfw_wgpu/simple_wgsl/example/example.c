/*
 * Example: WGSL -> SSIR
 *
 * Demonstrates parsing WGSL source and viewing the corresponding SSIR.
 */

#include "simple_wgsl.h"
#include <stdio.h>
#include <stdlib.h>

static void print_ssir_for_wgsl(const char *name, const char *wgsl_source) {
    printf("=== %s ===\n", name);
    printf("WGSL:\n%s\n", wgsl_source);

    WgslAstNode *ast = wgsl_parse(wgsl_source);
    if (!ast) {
        printf("Parse error!\n\n");
        return;
    }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        printf("Resolve error!\n\n");
        wgsl_free_ast(ast);
        return;
    }

    WgslLowerOptions opts = {0};
    opts.enable_debug_names = 1;
    WgslLower *lower = wgsl_lower_create(ast, resolver, &opts);
    if (!lower) {
        printf("Lower error!\n\n");
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        return;
    }

    const SsirModule *ssir = wgsl_lower_get_ssir(lower);
    if (ssir) {
        char *ssir_str = ssir_module_to_string((SsirModule *)ssir);
        if (ssir_str) {
            printf("SSIR:\n%s\n", ssir_str);
            free(ssir_str);
        }
    }

    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    printf("\n");
}

int main(void) {
    /* Example 1: Simple compute shader that doubles values */
    const char *compute_double =
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    data[gid.x] = data[gid.x] * 2.0;\n"
        "}\n";

    /* Example 2: Simple add function */
    const char *add_function =
        "fn add(a: f32, b: f32) -> f32 {\n"
        "    return a + b;\n"
        "}\n"
        "\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "}\n";

    /* Example 3: Vertex shader with position output */
    const char *vertex_shader =
        "struct VertexOutput {\n"
        "    @builtin(position) pos: vec4<f32>,\n"
        "};\n"
        "\n"
        "@vertex\n"
        "fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {\n"
        "    var out: VertexOutput;\n"
        "    out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);\n"
        "    return out;\n"
        "}\n";

    /* Example 4: Arithmetic operations */
    const char *arithmetic =
        "@group(0) @binding(0) var<storage, read_write> result: f32;\n"
        "\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "    var x: f32 = 10.0;\n"
        "    var y: f32 = 3.0;\n"
        "    result = (x + y) * (x - y);\n"
        "}\n";

    /* Example 5: Control flow with if statement */
    const char *control_flow =
        "@group(0) @binding(0) var<storage, read_write> out: i32;\n"
        "\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "    var x: i32 = 5;\n"
        "    if (x > 0) {\n"
        "        out = 1;\n"
        "    } else {\n"
        "        out = -1;\n"
        "    }\n"
        "}\n";

    print_ssir_for_wgsl("Compute: Double Array Values", compute_double);
    print_ssir_for_wgsl("Simple Add Function", add_function);
    print_ssir_for_wgsl("Vertex Shader", vertex_shader);
    print_ssir_for_wgsl("Arithmetic Operations", arithmetic);
    print_ssir_for_wgsl("Control Flow (If/Else)", control_flow);

    return 0;
}
