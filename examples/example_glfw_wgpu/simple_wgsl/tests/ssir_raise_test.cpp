#include <gtest/gtest.h>
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

namespace {

class SsirGuard {
public:
    explicit SsirGuard(SsirModule* m) : m_(m) {}
    ~SsirGuard() { if (m_) ssir_module_destroy(m_); }
    SsirModule* get() { return m_; }
private:
    SsirModule* m_;
};

struct SsirCompileResult {
    bool success;
    std::string error;
    const SsirModule* ssir;  // Valid only while lower context is alive
    WgslLower* lower;        // Caller must destroy
};

// Compile WGSL to SSIR using the lowering context
SsirCompileResult CompileToSsir(const char* source) {
    SsirCompileResult result;
    result.success = false;
    result.ssir = nullptr;
    result.lower = nullptr;

    WgslAstNode* ast = wgsl_parse(source);
    if (!ast) {
        result.error = "Parse failed";
        return result;
    }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        result.error = "Resolve failed";
        return result;
    }

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    opts.enable_debug_names = 1;

    result.lower = wgsl_lower_create(ast, resolver, &opts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (!result.lower) {
        result.error = "Lower failed";
        return result;
    }

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

// RAII wrapper for SsirCompileResult
class SsirCompileGuard {
public:
    explicit SsirCompileGuard(const SsirCompileResult& r) : r_(r) {}
    ~SsirCompileGuard() { if (r_.lower) wgsl_lower_destroy(r_.lower); }
    const SsirCompileResult& get() { return r_; }
private:
    SsirCompileResult r_;
};

} // namespace

// Test minimal empty function
TEST(SsirRaiseTest, MinimalFunction) {
    const char* source = "fn main() {}";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "main") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test vertex shader with builtin position output
TEST(SsirRaiseTest, VertexShader) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "position") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test fragment shader with location output
TEST(SsirRaiseTest, FragmentShader) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test compute shader with workgroup size
TEST(SsirRaiseTest, ComputeShader) {
    const char* source = R"(
        @compute @workgroup_size(8, 8, 1) fn cs() {}
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test uniform buffer with struct
TEST(SsirRaiseTest, UniformBuffer) {
    const char* source = R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @fragment fn fs() -> @location(0) vec4f { return u.color; }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@group(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@binding(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "var<uniform>") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test arithmetic operations
TEST(SsirRaiseTest, ArithmeticOps) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 1.0;
            let b = 2.0;
            let sum = a + b;
            let diff = a - b;
            let prod = a * b;
            let quot = a / b;
            return vec4f(sum, diff, prod, quot);
        }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    // Just verify it produces valid output with vec4
    EXPECT_TRUE(strstr(wgsl, "vec4") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test math builtin functions
TEST(SsirRaiseTest, MathBuiltins) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let x = 0.5;
            let s = sin(x);
            let c = cos(x);
            let sq = sqrt(x);
            return vec4f(s, c, sq, 1.0);
        }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "sin") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "cos") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "sqrt") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test vertex input
TEST(SsirRaiseTest, VertexInput) {
    const char* source = R"(
        @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0);
        }
    )";
    auto compile = CompileToSsir(source);
    SsirCompileGuard guard(compile);
    ASSERT_TRUE(compile.success) << compile.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@builtin(position)") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test null input error
TEST(SsirRaiseTest, NullInput) {
    char* wgsl = nullptr;
    char* error = nullptr;

    SsirToWgslResult result = ssir_to_wgsl(nullptr, nullptr, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_ERR_INVALID_INPUT);
    EXPECT_NE(error, nullptr);

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

// Test result string function
TEST(SsirRaiseTest, ResultStrings) {
    EXPECT_STREQ(ssir_to_wgsl_result_string(SSIR_TO_WGSL_OK), "Success");
    EXPECT_STREQ(ssir_to_wgsl_result_string(SSIR_TO_WGSL_ERR_INVALID_INPUT), "Invalid input");
    EXPECT_STREQ(ssir_to_wgsl_result_string(SSIR_TO_WGSL_ERR_UNSUPPORTED), "Unsupported feature");
    EXPECT_STREQ(ssir_to_wgsl_result_string(SSIR_TO_WGSL_ERR_INTERNAL), "Internal error");
    EXPECT_STREQ(ssir_to_wgsl_result_string(SSIR_TO_WGSL_ERR_OOM), "Out of memory");
}
