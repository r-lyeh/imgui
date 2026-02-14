#include <gtest/gtest.h>
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

namespace {

class SsirModuleGuard {
public:
    explicit SsirModuleGuard(SsirModule* m) : m_(m) {}
    ~SsirModuleGuard() { if (m_) ssir_module_destroy(m_); }
    SsirModule* get() { return m_; }
private:
    SsirModule* m_;
};

struct SpirvToSsirTestResult {
    bool success;
    std::string error;
    SsirModule* ssir;
};

SpirvToSsirTestResult ConvertSpirvToSsir(const std::vector<uint32_t>& spirv) {
    SpirvToSsirTestResult result;
    result.success = false;
    result.ssir = nullptr;

    char* error = nullptr;
    SpirvToSsirOptions opts = {};
    opts.preserve_names = true;
    opts.preserve_locations = true;

    SpirvToSsirResult res = spirv_to_ssir(
        spirv.data(), spirv.size(), &opts, &result.ssir, &error);

    if (res != SPIRV_TO_SSIR_SUCCESS) {
        result.error = error ? error : "Conversion failed";
        spirv_to_ssir_free(error);
        return result;
    }

    result.success = true;
    return result;
}

} // namespace

TEST(SpirvToSsir, InvalidInput) {
    SsirModule* mod = nullptr;
    char* error = nullptr;

    SpirvToSsirResult result = spirv_to_ssir(nullptr, 0, nullptr, &mod, &error);
    EXPECT_EQ(result, SPIRV_TO_SSIR_INVALID_SPIRV);
    EXPECT_NE(error, nullptr);
    spirv_to_ssir_free(error);
}

TEST(SpirvToSsir, InvalidMagic) {
    uint32_t bad_spirv[] = { 0x12345678, 0, 0, 0, 0 };
    SsirModule* mod = nullptr;
    char* error = nullptr;

    SpirvToSsirResult result = spirv_to_ssir(bad_spirv, 5, nullptr, &mod, &error);
    EXPECT_EQ(result, SPIRV_TO_SSIR_INVALID_SPIRV);
    spirv_to_ssir_free(error);
}

TEST(SpirvToSsir, ResultStrings) {
    EXPECT_STREQ(spirv_to_ssir_result_string(SPIRV_TO_SSIR_SUCCESS), "Success");
    EXPECT_STREQ(spirv_to_ssir_result_string(SPIRV_TO_SSIR_INVALID_SPIRV), "Invalid SPIR-V");
    EXPECT_STREQ(spirv_to_ssir_result_string(SPIRV_TO_SSIR_UNSUPPORTED_FEATURE), "Unsupported feature");
    EXPECT_STREQ(spirv_to_ssir_result_string(SPIRV_TO_SSIR_INTERNAL_ERROR), "Internal error");
}

TEST(SpirvToSsir, MinimalFunction) {
    const char* source = "fn main() {}";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;
    ASSERT_NE(convert.ssir, nullptr);

    EXPECT_GT(convert.ssir->function_count, 0u);
}

TEST(SpirvToSsir, VertexShaderRoundtrip) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "position") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, FragmentShaderRoundtrip) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, ComputeShaderRoundtrip) {
    const char* source = R"(
        @compute @workgroup_size(8, 8, 1) fn cs() {}
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, UniformBuffer) {
    const char* source = R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @fragment fn fs() -> @location(0) vec4f { return u.color; }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@group(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@binding(0)") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, ArithmeticOps) {
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
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, MathBuiltins) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let x = 0.5;
            let s = sin(x);
            let c = cos(x);
            let sq = sqrt(x);
            return vec4f(s, c, sq, 1.0);
        }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "sin") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "cos") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "sqrt") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, VertexInput) {
    const char* source = R"(
        @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0);
        }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;

    SsirToWgslResult result = ssir_to_wgsl(convert.ssir, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
}

TEST(SpirvToSsir, ControlFlow) {
    const char* source = R"(
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {
            var result: f32;
            if (x > 0.5) {
                result = 1.0;
            } else {
                result = 0.0;
            }
            return vec4f(result);
        }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    EXPECT_GT(convert.ssir->function_count, 0u);
}

TEST(SpirvToSsir, FullRoundtrip) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";

    auto compile1 = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile1.success) << compile1.error;

    auto convert = ConvertSpirvToSsir(compile1.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    uint32_t* spirv2 = nullptr;
    size_t spirv2_count = 0;
    SsirToSpirvOptions spirv_opts = {};
    spirv_opts.enable_debug_names = 1;

    SsirToSpirvResult spirv_result = ssir_to_spirv(convert.ssir, &spirv_opts, &spirv2, &spirv2_count);
    ASSERT_EQ(spirv_result, SSIR_TO_SPIRV_OK);

    std::string error;
    bool valid = wgsl_test::ValidateSpirv(spirv2, spirv2_count, &error);
    EXPECT_TRUE(valid) << "SPIR-V validation failed: " << error;

    ssir_to_spirv_free(spirv2);
}

TEST(SpirvToSsir, TypeReconstruction) {
    const char* source = R"(
        struct MyStruct {
            a: f32,
            b: vec2f,
            c: vec4f,
        };
        @group(0) @binding(0) var<uniform> u: MyStruct;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(u.a); }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    EXPECT_GT(convert.ssir->type_count, 0u);
    EXPECT_GT(convert.ssir->global_count, 0u);
}

TEST(SpirvToSsir, Constants) {
    const char* source = R"(
        const PI = 3.14159;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(PI); }
    )";
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    auto convert = ConvertSpirvToSsir(compile.spirv);
    SsirModuleGuard guard(convert.ssir);
    ASSERT_TRUE(convert.success) << convert.error;

    EXPECT_GT(convert.ssir->constant_count, 0u);
}
