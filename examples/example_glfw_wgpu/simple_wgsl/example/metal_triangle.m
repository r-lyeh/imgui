/*
 * Metal Triangle Render Example
 *
 * Demonstrates WGSL -> MSL conversion for a vertex + fragment shader pipeline.
 * Renders a colored triangle off-screen and verifies the result.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "simple_wgsl.h"
#include <stdio.h>

static char *wgsl_to_msl(const char *wgsl_source) {
    WgslAstNode *ast = wgsl_parse(wgsl_source);
    if (!ast) {
        fprintf(stderr, "WGSL parse error\n");
        return NULL;
    }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        fprintf(stderr, "WGSL resolve error\n");
        wgsl_free_ast(ast);
        return NULL;
    }

    WgslLowerOptions lower_opts = {0};
    lower_opts.enable_debug_names = 1;
    WgslLower *lower = wgsl_lower_create(ast, resolver, &lower_opts);
    if (!lower) {
        fprintf(stderr, "WGSL lower error\n");
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        return NULL;
    }

    const SsirModule *ssir = wgsl_lower_get_ssir(lower);
    if (!ssir) {
        fprintf(stderr, "Failed to get SSIR module\n");
        wgsl_lower_destroy(lower);
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        return NULL;
    }

    char *msl_source = NULL;
    char *error = NULL;
    SsirToMslOptions msl_opts = {0};
    msl_opts.preserve_names = 1;

    SsirToMslResult result = ssir_to_msl(ssir, &msl_opts, &msl_source, &error);
    if (result != SSIR_TO_MSL_OK) {
        fprintf(stderr, "SSIR to MSL error: %s\n", error ? error : ssir_to_msl_result_string(result));
        ssir_to_msl_free(error);
        wgsl_lower_destroy(lower);
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        return NULL;
    }

    ssir_to_msl_free(error);
    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    return msl_source;
}

static id<MTLFunction> find_function(id<MTLLibrary> lib, NSString *preferred, MTLFunctionType type) {
    id<MTLFunction> fn = [lib newFunctionWithName:preferred];
    if (fn) return fn;

    /* Search through available functions */
    for (NSString *name in [lib functionNames]) {
        fn = [lib newFunctionWithName:name];
        if (fn && [fn functionType] == type) return fn;
    }
    return nil;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        /*
         * WGSL shader with vertex and fragment entry points.
         * The vertex shader generates a triangle from vertex_index.
         * The fragment shader outputs solid red.
         */
        const char *wgsl_source =
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

        printf("=== WGSL to Metal Triangle Render Example ===\n\n");
        printf("WGSL source:\n%s\n", wgsl_source);

        /* Convert WGSL -> MSL */
        char *msl_source = wgsl_to_msl(wgsl_source);
        if (!msl_source) return 1;

        printf("Generated MSL:\n%s\n", msl_source);

        /* Create Metal device */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device available\n");
            ssir_to_msl_free(msl_source);
            return 1;
        }
        printf("Metal device: %s\n\n", [[device name] UTF8String]);

        /* Compile MSL source */
        NSError *nsError = nil;
        NSString *mslString = [NSString stringWithUTF8String:msl_source];
        id<MTLLibrary> library = [device newLibraryWithSource:mslString
                                                      options:nil
                                                        error:&nsError];
        if (!library) {
            fprintf(stderr, "MSL compilation failed: %s\n",
                    [[nsError localizedDescription] UTF8String]);
            ssir_to_msl_free(msl_source);
            return 1;
        }
        printf("MSL compiled successfully.\n");
        printf("Available functions:");
        for (NSString *name in [library functionNames])
            printf(" %s", [name UTF8String]);
        printf("\n\n");

        /* Find vertex and fragment functions */
        id<MTLFunction> vertexFn = find_function(library, @"vs_main", MTLFunctionTypeVertex);
        id<MTLFunction> fragmentFn = find_function(library, @"fs_main", MTLFunctionTypeFragment);

        if (!vertexFn || !fragmentFn) {
            fprintf(stderr, "Could not find vertex/fragment functions\n");
            ssir_to_msl_free(msl_source);
            return 1;
        }

        /* Create render pipeline */
        MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDesc.vertexFunction = vertexFn;
        pipelineDesc.fragmentFunction = fragmentFn;
        pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

        id<MTLRenderPipelineState> pipeline =
            [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&nsError];
        if (!pipeline) {
            fprintf(stderr, "Render pipeline creation failed: %s\n",
                    [[nsError localizedDescription] UTF8String]);
            ssir_to_msl_free(msl_source);
            return 1;
        }
        printf("Render pipeline created.\n");

        /* Create off-screen render target */
        const int W = 64, H = 64;
        MTLTextureDescriptor *texDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                              width:W
                                                             height:H
                                                          mipmapped:NO];
        texDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        texDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> renderTarget = [device newTextureWithDescriptor:texDesc];

        /* Render */
        MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        passDesc.colorAttachments[0].texture = renderTarget;
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder =
            [cmdBuf renderCommandEncoderWithDescriptor:passDesc];

        [encoder setRenderPipelineState:pipeline];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Read back center pixel */
        uint8_t pixel[4] = {0};
        [renderTarget getBytes:pixel
                   bytesPerRow:W * 4
                    fromRegion:MTLRegionMake2D(W / 2, H / 2, 1, 1)
                   mipmapLevel:0];

        printf("\nCenter pixel RGBA: (%u, %u, %u, %u)\n", pixel[0], pixel[1], pixel[2], pixel[3]);

        /* The center of the triangle should have some color (not black) */
        int pass = (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) && pixel[3] > 0;
        printf("%s\n", pass ? "PASS: Triangle rendered (non-black pixel at center)." :
                              "FAIL: Center pixel is black (triangle may not have rendered).");

        ssir_to_msl_free(msl_source);
        return pass ? 0 : 1;
    }
}
