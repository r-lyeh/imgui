/*
 * Metal Compute Example
 *
 * Demonstrates the full pipeline: WGSL -> SSIR -> MSL -> Metal GPU execution.
 * Parses a WGSL compute shader, converts to MSL, compiles with Metal,
 * dispatches a compute kernel, and reads back the results.
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
        fprintf(stderr, "WGSL lower error: %s\n", wgsl_lower_last_error(lower));
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        /* WGSL compute shader: doubles each element in a storage buffer */
        const char *wgsl_source =
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
            "\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            "    data[gid.x] = data[gid.x] * 2.0;\n"
            "}\n";

        printf("=== WGSL to Metal Compute Example ===\n\n");
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

        /* Find the kernel function ("main" is renamed to "main0" in MSL) */
        id<MTLFunction> function = [library newFunctionWithName:@"main0"];
        if (!function) {
            function = [library newFunctionWithName:@"main"];
        }
        if (!function) {
            function = [library newFunctionWithName:@"compute_main"];
        }
        if (!function) {
            /* List available functions */
            NSArray *names = [library functionNames];
            fprintf(stderr, "Could not find entry function. Available: ");
            for (NSString *n in names) fprintf(stderr, "%s ", [n UTF8String]);
            fprintf(stderr, "\n");

            /* Try the first available function */
            if ([names count] > 0) {
                function = [library newFunctionWithName:[names objectAtIndex:0]];
                printf("Using function: %s\n", [[names objectAtIndex:0] UTF8String]);
            }
        }
        if (!function) {
            fprintf(stderr, "No compute function found\n");
            ssir_to_msl_free(msl_source);
            return 1;
        }

        /* Create compute pipeline */
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&nsError];
        if (!pipeline) {
            fprintf(stderr, "Pipeline creation failed: %s\n",
                    [[nsError localizedDescription] UTF8String]);
            ssir_to_msl_free(msl_source);
            return 1;
        }

        /* Create buffer with test data */
        const int N = 256;
        float inputData[N];
        for (int i = 0; i < N; i++) inputData[i] = (float)i;

        id<MTLBuffer> buffer = [device newBufferWithBytes:inputData
                                                   length:sizeof(inputData)
                                                  options:MTLResourceStorageModeShared];

        /* Execute compute kernel */
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];

        MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Read back and verify */
        float *result = (float *)[buffer contents];
        printf("\nResults (first 8 elements):\n");
        int pass = 1;
        for (int i = 0; i < 8; i++) {
            float expected = (float)i * 2.0f;
            printf("  data[%d] = %.1f (expected %.1f) %s\n",
                   i, result[i], expected,
                   (result[i] == expected) ? "OK" : "FAIL");
            if (result[i] != expected) pass = 0;
        }
        printf("\n%s\n", pass ? "PASS: All values doubled correctly." :
                                "FAIL: Some values incorrect.");

        ssir_to_msl_free(msl_source);
        return pass ? 0 : 1;
    }
}
