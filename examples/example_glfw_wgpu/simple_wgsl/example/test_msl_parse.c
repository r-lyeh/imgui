#include "simple_wgsl.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("Testing MSL parser...\n");
    fflush(stdout);

    const char *msl =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "kernel void main0(\n"
        "    device float* data [[buffer(0)]],\n"
        "    uint3 gid [[thread_position_in_grid]]\n"
        ") {\n"
        "    data[gid.x] = (data[gid.x] * 2.0);\n"
        "}\n";

    printf("Input MSL:\n%s\n", msl);
    fflush(stdout);

    SsirModule *mod = NULL;
    char *error = NULL;
    MslToSsirOptions opts = {0};
    opts.preserve_names = 1;

    printf("Calling msl_to_ssir...\n");
    fflush(stdout);

    MslToSsirResult result = msl_to_ssir(msl, &opts, &mod, &error);

    printf("Result: %d (%s)\n", result, msl_to_ssir_result_string(result));
    if (error) printf("Error: %s\n", error);
    fflush(stdout);

    if (result == MSL_TO_SSIR_OK && mod) {
        char *dump = ssir_module_to_string(mod);
        if (dump) {
            printf("SSIR:\n%s\n", dump);
            free(dump);
        }

        /* Convert back to MSL */
        char *msl2 = NULL, *msl_err = NULL;
        SsirToMslOptions msl_opts = {0};
        msl_opts.preserve_names = 1;
        ssir_to_msl(mod, &msl_opts, &msl2, &msl_err);
        if (msl2) {
            printf("Round-tripped MSL:\n%s\n", msl2);
            ssir_to_msl_free(msl2);
        }
        if (msl_err) ssir_to_msl_free(msl_err);

        ssir_module_destroy(mod);
    }

    msl_to_ssir_free(error);

    /* Test 2: Vertex + Fragment shader */
    printf("\n=== Test 2: Vertex + Fragment ===\n");
    fflush(stdout);

    const char *wgsl2 =
        "struct VertexOutput {\n"
        "    @builtin(position) pos: vec4<f32>,\n"
        "    @location(0) color: vec3<f32>,\n"
        "};\n"
        "\n"
        "@vertex\n"
        "fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {\n"
        "    var out: VertexOutput;\n"
        "    var x: f32 = 0.0;\n"
        "    var y: f32 = 0.5;\n"
        "    var r: f32 = 1.0;\n"
        "    var g: f32 = 0.0;\n"
        "    var b: f32 = 0.0;\n"
        "    if (vid == 1u) {\n"
        "        x = -0.5; y = -0.5;\n"
        "        r = 0.0; g = 1.0; b = 0.0;\n"
        "    }\n"
        "    if (vid == 2u) {\n"
        "        x = 0.5; y = -0.5;\n"
        "        r = 0.0; g = 0.0; b = 1.0;\n"
        "    }\n"
        "    out.pos = vec4<f32>(x, y, 0.0, 1.0);\n"
        "    out.color = vec3<f32>(r, g, b);\n"
        "    return out;\n"
        "}\n"
        "\n"
        "@fragment\n"
        "fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(color, 1.0);\n"
        "}\n";

    /* First, convert WGSL → MSL */
    WgslAstNode *ast = wgsl_parse(wgsl2);
    if (!ast) { fprintf(stderr, "WGSL parse failed\n"); return 1; }
    WgslResolver *resolver = wgsl_resolver_build(ast);
    WgslLowerOptions lower_opts = {0};
    lower_opts.enable_debug_names = 1;
    WgslLower *lower = wgsl_lower_create(ast, resolver, &lower_opts);
    const SsirModule *ssir = wgsl_lower_get_ssir(lower);

    char *msl_vf = NULL, *msl_err = NULL;
    SsirToMslOptions msl_opts2 = {0};
    msl_opts2.preserve_names = 1;
    ssir_to_msl(ssir, &msl_opts2, &msl_vf, &msl_err);
    ssir_to_msl_free(msl_err);
    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (!msl_vf) { printf("FAIL: WGSL to MSL conversion failed\n"); return 1; }
    printf("Original MSL:\n%s\n", msl_vf);
    fflush(stdout);

    /* Parse the MSL back to SSIR */
    SsirModule *mod2 = NULL;
    char *error2 = NULL;
    printf("Parsing MSL...\n");
    fflush(stdout);
    MslToSsirResult r2 = msl_to_ssir(msl_vf, &opts, &mod2, &error2);
    printf("Result: %d (%s)\n", r2, msl_to_ssir_result_string(r2));
    if (error2) printf("Error: %s\n", error2);
    fflush(stdout);

    if (r2 == MSL_TO_SSIR_OK && mod2) {
        /* Round-trip back to MSL */
        char *msl_rt = NULL, *rt_err = NULL;
        SsirToMslOptions rt_opts = {0};
        rt_opts.preserve_names = 1;
        ssir_to_msl(mod2, &rt_opts, &msl_rt, &rt_err);
        if (msl_rt) {
            printf("Round-tripped MSL:\n%s\n", msl_rt);
            ssir_to_msl_free(msl_rt);
        }
        ssir_to_msl_free(rt_err);
        ssir_module_destroy(mod2);
    }

    msl_to_ssir_free(error2);
    ssir_to_msl_free(msl_vf);

    printf("Done.\n");
    return (result == MSL_TO_SSIR_OK) ? 0 : 1;
}
