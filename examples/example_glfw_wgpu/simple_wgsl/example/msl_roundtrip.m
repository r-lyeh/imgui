/*
 * MSL Round-Trip Test
 *
 * Tests: WGSL -> SSIR -> MSL -> parse MSL -> SSIR -> MSL -> Metal compile
 * Verifies the MSL parser can reconstruct a valid SSIR module.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "simple_wgsl.h"
#include <stdio.h>
#include <string.h>

static char *wgsl_to_msl(const char *wgsl_source) {
    WgslAstNode *ast = wgsl_parse(wgsl_source);
    if (!ast) { fprintf(stderr, "WGSL parse error\n"); return NULL; }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); return NULL; }

    WgslLowerOptions lower_opts = {0};
    lower_opts.enable_debug_names = 1;
    WgslLower *lower = wgsl_lower_create(ast, resolver, &lower_opts);
    if (!lower) { wgsl_resolver_free(resolver); wgsl_free_ast(ast); return NULL; }

    const SsirModule *ssir = wgsl_lower_get_ssir(lower);
    if (!ssir) { wgsl_lower_destroy(lower); wgsl_resolver_free(resolver); wgsl_free_ast(ast); return NULL; }

    char *msl_source = NULL, *error = NULL;
    SsirToMslOptions msl_opts = {0};
    msl_opts.preserve_names = 1;
    SsirToMslResult result = ssir_to_msl(ssir, &msl_opts, &msl_source, &error);
    ssir_to_msl_free(error);
    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    if (result != SSIR_TO_MSL_OK) return NULL;
    return msl_source;
}

static int test_roundtrip(const char *name, const char *wgsl_source, id<MTLDevice> device) {
    printf("--- %s ---\n", name);

    /* Step 1: WGSL -> MSL (original pipeline) */
    char *msl1 = wgsl_to_msl(wgsl_source);
    if (!msl1) { printf("FAIL: WGSL -> MSL failed\n\n"); return 0; }
    printf("Original MSL:\n%s\n", msl1);

    /* Step 2: Parse MSL back to SSIR */
    SsirModule *parsed_ssir = NULL;
    char *parse_error = NULL;
    MslToSsirOptions parse_opts = {0};
    parse_opts.preserve_names = 1;
    MslToSsirResult pr = msl_to_ssir(msl1, &parse_opts, &parsed_ssir, &parse_error);
    if (pr != MSL_TO_SSIR_OK) {
        printf("FAIL: MSL parse error: %s\n\n", parse_error ? parse_error : "unknown");
        msl_to_ssir_free(parse_error);
        ssir_to_msl_free(msl1);
        return 0;
    }
    msl_to_ssir_free(parse_error);
    printf("MSL parsed to SSIR successfully.\n");

    /* Step 3: SSIR dump */
    char *dump = ssir_module_to_string(parsed_ssir);
    if (dump) {
        printf("SSIR dump:\n%s\n", dump);
        free(dump);
    }

    /* Step 4: SSIR -> MSL again */
    char *msl2 = NULL, *msl2_error = NULL;
    SsirToMslOptions msl2_opts = {0};
    msl2_opts.preserve_names = 1;
    SsirToMslResult mr = ssir_to_msl(parsed_ssir, &msl2_opts, &msl2, &msl2_error);
    ssir_module_destroy(parsed_ssir);
    if (mr != SSIR_TO_MSL_OK) {
        printf("FAIL: SSIR -> MSL (round 2) failed: %s\n\n", msl2_error ? msl2_error : "unknown");
        ssir_to_msl_free(msl2_error);
        ssir_to_msl_free(msl1);
        return 0;
    }
    ssir_to_msl_free(msl2_error);
    printf("Round-tripped MSL:\n%s\n", msl2);

    /* Step 5: Compile round-tripped MSL with Metal */
    NSError *nsError = nil;
    NSString *mslString = [NSString stringWithUTF8String:msl2];
    id<MTLLibrary> library = [device newLibraryWithSource:mslString options:nil error:&nsError];
    if (!library) {
        printf("FAIL: Metal compilation of round-tripped MSL failed: %s\n\n",
               [[nsError localizedDescription] UTF8String]);
        ssir_to_msl_free(msl1);
        ssir_to_msl_free(msl2);
        return 0;
    }

    printf("Metal compilation successful! Functions:");
    for (NSString *fname in [library functionNames])
        printf(" %s", [fname UTF8String]);
    printf("\n");

    printf("PASS\n\n");
    ssir_to_msl_free(msl1);
    ssir_to_msl_free(msl2);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }
        printf("Metal device: %s\n\n", [[device name] UTF8String]);

        int pass = 0, total = 0;

        /* Test 1: Compute shader */
        {
            const char *wgsl =
                "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
                "\n"
                "@compute @workgroup_size(64)\n"
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
                "    data[gid.x] = data[gid.x] * 2.0;\n"
                "}\n";
            total++;
            pass += test_roundtrip("Compute Shader (data doubler)", wgsl, device);
        }

        /* Test 2: Vertex + Fragment shader */
        {
            const char *wgsl =
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
            total++;
            pass += test_roundtrip("Vertex + Fragment Shader (triangle)", wgsl, device);
        }

        printf("=== Results: %d/%d passed ===\n", pass, total);
        return (pass == total) ? 0 : 1;
    }
}
