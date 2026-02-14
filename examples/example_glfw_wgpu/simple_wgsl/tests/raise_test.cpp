#include <gtest/gtest.h>
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

class RaiseGuard {
public:
    explicit RaiseGuard(WgslRaiser* r) : r_(r) {}
    ~RaiseGuard() { if (r_) wgsl_raise_destroy(r_); }
    WgslRaiser* get() { return r_; }
private:
    WgslRaiser* r_;
};

TEST(RaiseTest, InvalidSpirv) {
    uint32_t bad_spirv[] = { 0x12345678, 0, 0, 0, 0 };
    WgslRaiser* r = wgsl_raise_create(bad_spirv, 5);
    EXPECT_EQ(r, nullptr);
}

TEST(RaiseTest, NullInput) {
    WgslRaiser* r = wgsl_raise_create(nullptr, 0);
    EXPECT_EQ(r, nullptr);
}

TEST(RaiseTest, MinimalFunction) {
    const char* source = "fn main() {}";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "main") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, VertexShader) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "vs") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "position") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, FragmentShader) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "fs") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, ComputeShader) {
    const char* source = R"(
        @compute @workgroup_size(8, 8, 1) fn cs() {}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, UniformBuffer) {
    const char* source = R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @fragment fn fs() -> @location(0) vec4f { return u.color; }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@group(0)") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "@binding(0)") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "var<uniform>") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, TextureSampler) {
    const char* source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(tex, samp, vec2f(0.5, 0.5));
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "texture_2d") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "sampler") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, ArithmeticOperations) {
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
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, EntryPointCount) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    RaiseGuard raiser(wgsl_raise_create(result.spirv.data(), result.spirv.size()));
    ASSERT_NE(raiser.get(), nullptr);

    EXPECT_EQ(wgsl_raise_parse(raiser.get()), WGSL_RAISE_SUCCESS);
    EXPECT_EQ(wgsl_raise_entry_point_count(raiser.get()), 1);

    const char* name0 = wgsl_raise_entry_point_name(raiser.get(), 0);
    EXPECT_NE(name0, nullptr);
    EXPECT_STREQ(name0, "vs");
}

TEST(RaiseTest, VertexInput) {
    const char* source = R"(
        @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "@builtin(position)") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}

TEST(RaiseTest, MathFunctions) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let x = 0.5;
            let s = sin(x);
            let c = cos(x);
            let sq = sqrt(x);
            return vec4f(s, c, sq, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(result.success) << "Compile failed: " << result.error;

    char* wgsl = nullptr;
    char* error = nullptr;
    WgslRaiseResult raise_result = wgsl_raise_to_wgsl(
        result.spirv.data(), result.spirv.size(), nullptr, &wgsl, &error);

    EXPECT_EQ(raise_result, WGSL_RAISE_SUCCESS) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "sin") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "cos") != nullptr);
    EXPECT_TRUE(strstr(wgsl, "sqrt") != nullptr);

    wgsl_raise_free(wgsl);
    wgsl_raise_free(error);
}
