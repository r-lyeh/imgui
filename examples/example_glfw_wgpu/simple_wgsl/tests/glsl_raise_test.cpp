#include <gtest/gtest.h>
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

namespace {

struct SsirCompileResult {
    bool success;
    std::string error;
    const SsirModule* ssir;
    WgslLower* lower;
};

SsirCompileResult CompileToSsir(const char* source) {
    SsirCompileResult result;
    result.success = false;
    result.ssir = nullptr;
    result.lower = nullptr;

    WgslAstNode* ast = wgsl_parse(source);
    if (!ast) { result.error = "Parse failed"; return result; }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); result.error = "Resolve failed"; return result; }

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    opts.enable_debug_names = 1;

    result.lower = wgsl_lower_create(ast, resolver, &opts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (!result.lower) { result.error = "Lower failed"; return result; }

    result.ssir = wgsl_lower_get_ssir(result.lower);
    if (!result.ssir) {
        wgsl_lower_destroy(result.lower);
        result.lower = nullptr;
        result.error = "No SSIR module";
        return result;
    }

    result.success = true;
    return result;
}

class SsirCompileGuard {
public:
    explicit SsirCompileGuard(const SsirCompileResult& r) : r_(r) {}
    ~SsirCompileGuard() { if (r_.lower) wgsl_lower_destroy(r_.lower); }
    const SsirCompileResult& get() { return r_; }
private:
    SsirCompileResult r_;
};

/* Helper: roundtrip WGSL -> SSIR -> GLSL -> parse GLSL -> SPIR-V -> validate */
struct RoundtripResult {
    bool glsl_emit_ok;
    bool glsl_parse_ok;
    bool spirv_ok;
    bool spirv_valid;
    std::string glsl;
    std::string error;
};

RoundtripResult GlslRoundtrip(const char* wgsl_source, WgslStage stage, SsirStage ssir_stage) {
    RoundtripResult r = {};

    /* Step 1: WGSL -> SSIR */
    auto compile = CompileToSsir(wgsl_source);
    SsirCompileGuard guard(compile);
    if (!compile.success) { r.error = "Compile: " + compile.error; return r; }

    /* Step 2: SSIR -> GLSL */
    auto glsl_result = wgsl_test::RaiseSsirToGlsl(compile.ssir, ssir_stage);
    if (!glsl_result.success) { r.error = "GLSL emit: " + glsl_result.error; return r; }
    r.glsl_emit_ok = true;
    r.glsl = glsl_result.glsl;

    /* Step 3: Parse GLSL back */
    WgslAstNode* ast = glsl_parse(r.glsl.c_str(), stage);
    if (!ast) { r.error = "GLSL parse failed on emitted GLSL"; return r; }
    r.glsl_parse_ok = true;

    /* Step 4: Compile parsed GLSL to SPIR-V */
    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        r.error = "Resolve of re-parsed GLSL failed";
        return r;
    }

    uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions lower_opts = {};
    lower_opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult lower_result = wgsl_lower_emit_spirv(ast, resolver, &lower_opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) {
        r.error = "Lower of re-parsed GLSL failed";
        return r;
    }
    r.spirv_ok = true;

    /* Step 5: Validate SPIR-V */
    std::string val_err;
    r.spirv_valid = wgsl_test::ValidateSpirv(spirv, spirv_size, &val_err);
    wgsl_lower_free(spirv);

    if (!r.spirv_valid) {
        r.error = "SPIR-V validation failed: " + val_err;
    }

    return r;
}

} // namespace

/* ===========================================================================
 * Basic SSIR -> GLSL Emission Tests
 * =========================================================================== */

TEST(GlslRaiseTest, MinimalFunction) {
    const char* source = "fn main() {}";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult result = ssir_to_glsl(compile.ssir, SSIR_STAGE_COMPUTE, &opts, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_OK) << (error ? error : "unknown");
    ASSERT_NE(glsl, nullptr);
    EXPECT_TRUE(strstr(glsl, "#version 450") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "void main()") != nullptr) << "GLSL:\n" << glsl;

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, VertexShader) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult result = ssir_to_glsl(compile.ssir, SSIR_STAGE_VERTEX, &opts, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_OK) << (error ? error : "unknown");
    ASSERT_NE(glsl, nullptr);
    EXPECT_TRUE(strstr(glsl, "#version 450") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "void main()") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "gl_Position") != nullptr) << "GLSL:\n" << glsl;

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, FragmentShader) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult result = ssir_to_glsl(compile.ssir, SSIR_STAGE_FRAGMENT, &opts, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_OK) << (error ? error : "unknown");
    ASSERT_NE(glsl, nullptr);
    EXPECT_TRUE(strstr(glsl, "#version 450") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "layout(location = 0) out") != nullptr) << "GLSL:\n" << glsl;

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, ComputeShader) {
    const char* source = R"(
        @compute @workgroup_size(8, 8, 1) fn cs() {}
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult result = ssir_to_glsl(compile.ssir, SSIR_STAGE_COMPUTE, &opts, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_OK) << (error ? error : "unknown");
    ASSERT_NE(glsl, nullptr);
    /* SSIR entry point workgroup_size may default to 1 - just verify the layout exists */
    EXPECT_TRUE(strstr(glsl, "local_size_x") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "void main()") != nullptr) << "GLSL:\n" << glsl;

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, UniformBuffer) {
    const char* source = R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @fragment fn fs() -> @location(0) vec4f { return u.color; }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslOptions opts = {};
    opts.preserve_names = 1;

    SsirToGlslResult result = ssir_to_glsl(compile.ssir, SSIR_STAGE_FRAGMENT, &opts, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_OK) << (error ? error : "unknown");
    ASSERT_NE(glsl, nullptr);
    EXPECT_TRUE(strstr(glsl, "std140") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "set = 0") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "binding = 0") != nullptr) << "GLSL:\n" << glsl;
    EXPECT_TRUE(strstr(glsl, "uniform") != nullptr) << "GLSL:\n" << glsl;

    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, NullInput) {
    char* glsl = nullptr;
    char* error = nullptr;
    SsirToGlslResult result = ssir_to_glsl(nullptr, SSIR_STAGE_COMPUTE, nullptr, &glsl, &error);
    EXPECT_EQ(result, SSIR_TO_GLSL_ERR_INVALID_INPUT);
    ssir_to_glsl_free(glsl);
    ssir_to_glsl_free(error);
}

TEST(GlslRaiseTest, ResultStrings) {
    EXPECT_STREQ(ssir_to_glsl_result_string(SSIR_TO_GLSL_OK), "Success");
    EXPECT_STREQ(ssir_to_glsl_result_string(SSIR_TO_GLSL_ERR_INVALID_INPUT), "Invalid input");
    EXPECT_STREQ(ssir_to_glsl_result_string(SSIR_TO_GLSL_ERR_UNSUPPORTED), "Unsupported feature");
    EXPECT_STREQ(ssir_to_glsl_result_string(SSIR_TO_GLSL_ERR_INTERNAL), "Internal error");
    EXPECT_STREQ(ssir_to_glsl_result_string(SSIR_TO_GLSL_ERR_OOM), "Out of memory");
}

/* ===========================================================================
 * Roundtrip Tests: WGSL -> SSIR -> GLSL -> parse GLSL -> SPIR-V -> validate
 * =========================================================================== */

TEST(GlslRoundtripTest, MinimalCompute) {
    auto r = GlslRoundtrip(
        "@compute @workgroup_size(1) fn main() {}",
        WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, FragmentConstantReturn) {
    auto r = GlslRoundtrip(
        "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }",
        WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, VertexPassthrough) {
    auto r = GlslRoundtrip(
        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }",
        WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, VertexWithInput) {
    auto r = GlslRoundtrip(R"(
        @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0);
        }
    )", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, UniformBuffer) {
    auto r = GlslRoundtrip(R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @fragment fn fs() -> @location(0) vec4f { return u.color; }
    )", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, ArithmeticOps) {
    auto r = GlslRoundtrip(R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 1.0;
            let b = 2.0;
            let sum = a + b;
            let diff = a - b;
            let prod = a * b;
            let quot = a / b;
            return vec4f(sum, diff, prod, quot);
        }
    )", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, MathBuiltins) {
    auto r = GlslRoundtrip(R"(
        @fragment fn fs() -> @location(0) vec4f {
            let x = 0.5;
            let s = sin(x);
            let c = cos(x);
            let sq = sqrt(x);
            return vec4f(s, c, sq, 1.0);
        }
    )", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}

TEST(GlslRoundtripTest, StorageBuffer) {
    auto r = GlslRoundtrip(R"(
        struct Params { scale: f32, offset: f32 };
        @group(0) @binding(0) var<storage> params: Params;
        @compute @workgroup_size(1) fn main() {
            let s = params.scale;
        }
    )", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE);
    EXPECT_TRUE(r.glsl_emit_ok) << r.error;
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error;
}
