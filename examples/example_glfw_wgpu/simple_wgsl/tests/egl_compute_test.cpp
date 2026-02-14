#ifdef WGSL_HAS_EGL

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <cstring>

#include "egl_compute_harness.h"
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

/* ---- helpers ---- */

namespace {

struct GlslResult {
    bool success;
    std::string glsl;
    std::string error;
};

GlslResult WgslToOpenGlsl(const char* wgsl_source) {
    GlslResult r;
    r.success = false;

    WgslAstNode* ast = wgsl_parse(wgsl_source);
    if (!ast) { r.error = "WGSL parse failed"; return r; }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); r.error = "Resolve failed"; return r; }

    WgslLowerOptions lopts = {};
    lopts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    lopts.enable_debug_names = 1;

    WgslLower* lower = wgsl_lower_create(ast, resolver, &lopts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    if (!lower) { r.error = "Lower failed"; return r; }

    const SsirModule* ssir = wgsl_lower_get_ssir(lower);
    if (!ssir) { wgsl_lower_destroy(lower); r.error = "No SSIR"; return r; }

    char* glsl = nullptr;
    char* glsl_err = nullptr;
    SsirToGlslOptions gopts = {};
    gopts.preserve_names = 1;
    gopts.target_opengl = 1;

    SsirToGlslResult gres = ssir_to_glsl(ssir, SSIR_STAGE_COMPUTE, &gopts, &glsl, &glsl_err);
    wgsl_lower_destroy(lower);

    if (gres != SSIR_TO_GLSL_OK) {
        r.error = glsl_err ? glsl_err : "GLSL emit failed";
        if (glsl) ssir_to_glsl_free(glsl);
        if (glsl_err) ssir_to_glsl_free(glsl_err);
        return r;
    }

    r.glsl = glsl;
    ssir_to_glsl_free(glsl);
    if (glsl_err) ssir_to_glsl_free(glsl_err);
    r.success = true;
    return r;
}

} // namespace

/* ---- test fixture ---- */

class EGLComputeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<egl_compute::Context>();
        } catch (const std::exception& e) {
            skip_reason_ = std::string("EGL/OpenGL not available: ") + e.what();
        }
    }
    static void TearDownTestSuite() { ctx_.reset(); }
    void SetUp() override {
        if (!ctx_) GTEST_SKIP() << skip_reason_;
    }

    static std::unique_ptr<egl_compute::Context> ctx_;
    static std::string skip_reason_;
};

std::unique_ptr<egl_compute::Context> EGLComputeTest::ctx_;
std::string EGLComputeTest::skip_reason_ = "EGL context not initialized";

/* ---- tests ---- */

TEST_F(EGLComputeTest, SimpleCompute) {
    auto g = WgslToOpenGlsl(R"(
        @compute @workgroup_size(1) fn main() {}
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, StorageBufferCopy) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = input.data[id.x];
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, ArithmeticOps) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> a: Buf;
        @group(0) @binding(1) var<storage, read> b: Buf;
        @group(0) @binding(2) var<storage, read_write> result: Buf;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let i = id.x;
            result.data[i] = a.data[i] + b.data[i] * 2.0 - 1.0;
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, IntegerArithmetic) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<i32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let i = id.x;
            let v = input.data[i];
            output.data[i] = v * 3 + 7;
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, UniformBuffer) {
    auto g = WgslToOpenGlsl(R"(
        struct Params { scale: f32, offset: f32 };
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read_write> output: Buf;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            output.data[id.x] = params.scale + params.offset;
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, MathBuiltins) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let i = id.x;
            let v = input.data[i];
            output.data[i] = sin(v) + cos(v) + sqrt(abs(v));
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, Conditional) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read> input: Buf;
        @group(0) @binding(1) var<storage, read_write> output: Buf;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let i = id.x;
            if (input.data[i] > 0.0) {
                output.data[i] = 1.0;
            } else {
                output.data[i] = 0.0;
            }
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLComputeTest, Loop) {
    auto g = WgslToOpenGlsl(R"(
        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read_write> buf: Buf;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var sum: f32 = 0.0;
            for (var i: u32 = 0u; i < 10u; i = i + 1u) {
                sum = sum + f32(i);
            }
            buf.data[id.x] = sum;
        }
    )");
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileComputeShader(g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

#endif // WGSL_HAS_EGL
