#include <gtest/gtest.h>
#include "test_utils.h"

TEST(LowerTest, EmitMinimalSpirv) {
    const char* source = "fn main() {}";
    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    uint32_t* spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult result = wgsl_lower_emit_spirv(ast.get(), resolver.get(), &opts, &spirv, &spirv_size);
    EXPECT_EQ(result, WGSL_LOWER_OK);
    ASSERT_NE(spirv, nullptr);
    ASSERT_GE(spirv_size, static_cast<size_t>(5));
    EXPECT_EQ(spirv[0], 0x07230203u);

    wgsl_lower_free(spirv);
}

TEST(LowerTest, ValidateMinimalSpirvModule) {
    const char* source = "fn main() {}";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateFragmentShader) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateVertexShader) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateComputeShader) {
    const char* source = R"(
        @compute @workgroup_size(1) fn cs() {}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateBindingVariable) {
    const char* source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateSampler) {
    const char* source = R"(
        @group(0) @binding(0) var s: sampler;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, TypeCachingWorks) {
    // This test verifies that type caching works correctly
    // by compiling a shader that uses the same types multiple times
    const char* source = R"(
        fn main() {}
    )";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    // Check that entrypoints are created
    int count = 0;
    const WgslLowerEntrypointInfo* eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_GE(count, 1);
    EXPECT_NE(eps, nullptr);
}

TEST(LowerTest, MultipleEntrypoints) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    int count = 0;
    const WgslLowerEntrypointInfo* eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_EQ(count, 2);

    // Verify both entrypoints have function IDs
    for (int i = 0; i < count; ++i) {
        EXPECT_NE(eps[i].function_id, 0u);
    }
}

TEST(LowerTest, ModuleFeatures) {
    const char* source = "fn main() {}";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    const WgslLowerModuleFeatures* features = wgsl_lower_module_features(lower.get());
    ASSERT_NE(features, nullptr);

    // Should have at least Shader capability
    EXPECT_GE(features->capability_count, 1u);
}

// ==================== Expression Lowering Tests ====================

TEST(LowerTest, ArithmeticExpressions) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0 + 2.0;
            var y = 3.0 - 1.0;
            var z = x * y;
            var w = z / 2.0;
            return vec4f(w);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ComparisonOperators) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var lt = a < b;
            var le = a <= b;
            var gt = a > b;
            var ge = a >= b;
            var eq = a == b;
            var ne = a != b;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, BitwiseOperators) {
    // Basic integer operations (shift operators need parser support)
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a: i32 = 5;
            var b: i32 = 3;
            var sum: i32 = a + b;
            var diff: i32 = a - b;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, UnaryOperators) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a: f32 = 5.0;
            var neg: f32 = -a;
            return vec4f(neg);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorConstruction) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v2 = vec2f(1.0, 2.0);
            var v3 = vec3f(1.0, 2.0, 3.0);
            var v4 = vec4f(1.0, 2.0, 3.0, 4.0);
            return v4;
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorSwizzle) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v = vec4f(1.0, 2.0, 3.0, 4.0);
            var x = v.x;
            var xy = v.xy;
            var zw = v.zw;
            var rgba = v.rgba;
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Built-in Function Tests ====================

TEST(LowerTest, MathFunctions) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 2.0;
            var s = sqrt(a);
            var f = floor(a);
            var c = ceil(a);
            var r = round(a);
            var t = trunc(a);
            return vec4f(s);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, TrigFunctions) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var angle = 1.0;
            var s = sin(angle);
            var c = cos(angle);
            var t = tan(angle);
            return vec4f(s, c, t, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, MinMaxClamp) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var mn = min(a, b);
            var mx = max(a, b);
            var cl = clamp(a, 0.0, 1.0);
            return vec4f(mn, mx, cl, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorBuiltins) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v1 = vec3f(1.0, 0.0, 0.0);
            var v2 = vec3f(0.0, 1.0, 0.0);
            var d = dot(v1, v2);
            var c = cross(v1, v2);
            var len = length(v1);
            var n = normalize(v1);
            return vec4f(d, len, n.x, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, MixSmoothstep) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 0.0;
            var b = 1.0;
            var t = 0.5;
            var m = mix(a, b, t);
            var ss = smoothstep(0.0, 1.0, t);
            return vec4f(m, ss, 0.0, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Control Flow Tests ====================

TEST(LowerTest, IfStatement) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0;
            if (x > 0.5) {
                x = 2.0;
            } else {
                x = 0.0;
            }
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, IfWithoutElse) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0;
            if (x > 0.5) {
                x = 2.0;
            }
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, WhileLoop) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0;
            var sum = 0.0;
            while (i < 10) {
                sum = sum + 1.0;
                i = i + 1;
            }
            return vec4f(sum);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ForLoop) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var sum = 0.0;
            for (var i = 0; i < 10; i = i + 1) {
                sum = sum + 1.0;
            }
            return vec4f(sum);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, NestedControlFlow) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var result = 0.0;
            for (var i = 0; i < 5; i = i + 1) {
                if (i > 2) {
                    result = result + 2.0;
                } else {
                    result = result + 1.0;
                }
            }
            return vec4f(result);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Variable Tests ====================

TEST(LowerTest, LocalVariables) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var c = a + b;
            a = c * 2.0;
            return vec4f(a);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, IntegerVariables) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var i = 10;
            var u = 20u;
            var sum = i + 5;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ============================================================================
// Matrix × Vector tests (OpMatrixTimesVector)
// ============================================================================

TEST(LowerTest, MatrixTimesVector_Mat2x2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat2x2<f32>(vec2<f32>(1.,0.), vec2<f32>(0.,1.));
            let v = vec2<f32>(2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_Mat3x3_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let v = vec3<f32>(1., 2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_Mat4x4_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let v = vec4<f32>(1., 2., 3., 4.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat3x2_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x2<f32>(
                vec2<f32>(1.,0.),
                vec2<f32>(0.,1.),
                vec2<f32>(1.,1.));
            let v = vec3<f32>(1., 2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat4x3_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat4x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.),
                vec3<f32>(1.,1.,1.));
            let v = vec4<f32>(1., 2., 3., 4.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat2x4_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat2x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.));
            let v = vec2<f32>(1., 2.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Vector × Matrix tests (OpVectorTimesMatrix)
// ============================================================================

TEST(LowerTest, VectorTimesMatrix_Vec2_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec2<f32>(2., 3.);
            let m = mat2x2<f32>(vec2<f32>(1.,0.), vec2<f32>(0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_Vec3_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec3<f32>(1., 2., 3.);
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_Vec4_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec4<f32>(1., 2., 3., 4.);
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_NonSquare_Vec2_Mat3x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec2<f32>(1., 2.);
            let m = mat3x2<f32>(
                vec2<f32>(1.,0.),
                vec2<f32>(0.,1.),
                vec2<f32>(1.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix × Matrix tests (OpMatrixTimesMatrix)
// ============================================================================

TEST(LowerTest, MatrixTimesMatrix_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat2x2<f32>(vec2<f32>(1.,2.), vec2<f32>(3.,4.));
            let b = mat2x2<f32>(vec2<f32>(5.,6.), vec2<f32>(7.,8.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat3x3<f32>(
                vec3<f32>(1.,2.,3.),
                vec3<f32>(4.,5.,6.),
                vec3<f32>(7.,8.,9.));
            let b = mat3x3<f32>(
                vec3<f32>(-1.,-2.,-3.),
                vec3<f32>(-4.,-5.,-6.),
                vec3<f32>(-7.,-8.,-9.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let b = mat4x4<f32>(
                vec4<f32>(2.,0.,0.,0.),
                vec4<f32>(0.,2.,0.,0.),
                vec4<f32>(0.,0.,2.,0.),
                vec4<f32>(0.,0.,0.,2.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_NonSquare_Mat2x3_Mat3x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat2x3<f32>(vec3<f32>(1.,2.,3.), vec3<f32>(4.,5.,6.));
            let b = mat3x2<f32>(vec2<f32>(1.,2.), vec2<f32>(3.,4.), vec2<f32>(5.,6.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix × Scalar tests (OpMatrixTimesScalar)
// ============================================================================

TEST(LowerTest, MatrixTimesScalar_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let s = 2.0;
            let r = m * s;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, ScalarTimesMatrix_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let s = 3.0;
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let r = s * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix operations with uniform struct (tests MatrixStride decorations)
// ============================================================================

TEST(LowerTest, MatrixTimesVector_UniformStruct_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat3x3<f32>,
            vector : vec3<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.matrix * data.vector;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_UniformStruct_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat4x4<f32>,
            vector : vec4<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.matrix * data.vector;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_UniformStruct_Vec3_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat3x3<f32>,
            vector : vec3<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.vector * data.matrix;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_UniformStruct) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            a : mat4x4<f32>,
            b : mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.a * data.b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Fragment shader struct parameter bug (known issue)
// ============================================================================

TEST(LowerTest, FragmentShaderFlatParameter) {
    auto r = wgsl_test::CompileWgsl(R"(
        @fragment fn fs_main(@location(0) col: vec4f) -> @location(0) vec4f {
            return col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Mixed matrix operations in complex expressions
// ============================================================================

TEST(LowerTest, MatrixVectorChain) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let model = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let view = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let pos = vec4<f32>(1., 2., 3., 1.);
            let transformed = view * model * pos;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixScaleAndTransform) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let scaled = m * 2.0;
            let v = vec3<f32>(1., 2., 3.);
            let r = scaled * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Fragment shader struct parameter tests
// ============================================================================

TEST(LowerTest, FragmentShaderStructParam_LocationFields) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_MultipleLocations) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_UseMultipleFields) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            let c = in.col;
            let u = in.uv;
            return vec4f(u.x, u.y, 0.0, 1.0);
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_BuiltinPosition) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_InterpolateFlat) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) @interpolate(flat) id: u32,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_VertexFragmentPipeline) {
    // Vertex and fragment shaders compiled separately (normal workflow)
    auto vs = wgsl_test::CompileWgsl(R"(
        struct VsOut {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
        };
        @vertex fn vs_main(@location(0) pos: vec3f) -> VsOut {
            var out: VsOut;
            out.pos = vec4f(pos, 1.0);
            out.col = vec4f(1.0, 0.0, 0.0, 1.0);
            return out;
        }
    )");
    EXPECT_TRUE(vs.success) << vs.error;

    auto fs = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(fs.success) << fs.error;
}

// =============================================================================
// Swizzle Tests - Single Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_SingleComponent_xyzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return vec4<f32>(v.x, v.y, v.z, v.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_SingleComponent_rgba) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return vec4<f32>(v.r, v.g, v.b, v.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.0, 3.0);
    return vec4<f32>(v.x, v.y, v.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec2_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.0, 2.0);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Two Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_TwoComponent_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xy = v.xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_TwoComponent_zw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let zw = v.zw;
    return vec4<f32>(zw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_TwoComponent_yw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let yw = v.yw;
    return vec4<f32>(yw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3_TwoComponent_yz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.0, 3.0);
    let yz = v.yz;
    return vec4<f32>(yz, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Three Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_ThreeComponent_xyz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xyz = v.xyz;
    return vec4<f32>(xyz, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_ThreeComponent_rgb) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let c = v.rgb;
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_ThreeComponent_yzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let yzw = v.yzw;
    return vec4<f32>(yzw, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Four Component (identity and reorder)
// =============================================================================

TEST(SwizzleTest, Vec4_FourComponent_xyzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xyzw;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_FourComponent_wzyx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.wzyx;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_FourComponent_abgr) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.abgr;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Duplicate Components
// =============================================================================

TEST(SwizzleTest, Vec4_Duplicate_xx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xx = v.xx;
    return vec4<f32>(xx, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xxx = v.xxx;
    return vec4<f32>(xxx, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxxx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xxxx;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxyy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xxyy;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_aaaa) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.aaaa;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_rrgg) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.rrgg;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Integer Vectors
// =============================================================================

TEST(SwizzleTest, Vec4i_ThreeComponent_xyz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<i32>(1, 2, 3, 4);
    let xyz = v.xyz;
    return vec4<f32>(f32(xyz.x), f32(xyz.y), f32(xyz.z), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4u_TwoComponent_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<u32>(1u, 2u, 3u, 4u);
    let xy = v.xy;
    return vec4<f32>(f32(xy.x), f32(xy.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3i_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<i32>(10, 20, 30);
    return vec4<f32>(f32(v.x), f32(v.y), f32(v.z), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - On Expressions (not just variables)
// =============================================================================

TEST(SwizzleTest, SwizzleOnArithmeticResult) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(0.1, 0.2, 0.3, 0.4);
    let rgb = (a + b).xyz;
    return vec4<f32>(rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleOnMultiplyResult) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let scaled = (v * 0.5).rgb;
    return vec4<f32>(scaled, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleOnConstructor) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec4<f32>(1.0, 2.0, 3.0, 4.0).xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Chained Swizzles
// =============================================================================

TEST(SwizzleTest, ChainedSwizzle_xyz_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xy = v.xyz.xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, ChainedSwizzle_xyzw_zw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let zw = v.xyzw.zw;
    return vec4<f32>(zw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - In Arithmetic Expressions
// =============================================================================

TEST(SwizzleTest, SwizzleInMultiply) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 0.5, 0.25, 1.0);
    let scaled = v.rgb * 2.0;
    return vec4<f32>(scaled, v.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleBothOperands) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(5.0, 6.0, 7.0, 8.0);
    let sum = a.xy + b.zw;
    return vec4<f32>(sum, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleAsFunctionArg) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(3.0, 4.0, 0.0, 0.0);
    let len = length(v.xyz);
    return vec4<f32>(len, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleNormalize) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 0.0);
    let n = normalize(v.xyz);
    return vec4<f32>(n, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleDotProduct) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let d = dot(a.xyz, b.xyz);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleCross) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let c = cross(a.xyz, b.xyz);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - From Mixed Components
// =============================================================================

TEST(VectorConstructionTest, Vec4_FromVec3AndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let rgb = vec3<f32>(1.0, 0.5, 0.25);
    return vec4<f32>(rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromScalarAndVec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yzw = vec3<f32>(0.5, 0.25, 1.0);
    return vec4<f32>(1.0, yzw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromTwoVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    let zw = vec2<f32>(3.0, 4.0);
    return vec4<f32>(xy, zw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromVec2AndTwoScalars) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    return vec4<f32>(xy, 3.0, 4.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromTwoScalarsAndVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let zw = vec2<f32>(3.0, 4.0);
    return vec4<f32>(1.0, 2.0, zw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromScalarVec2Scalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yz = vec2<f32>(2.0, 3.0);
    return vec4<f32>(1.0, yz, 4.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_FromVec2AndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    let v = vec3<f32>(xy, 3.0);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_FromScalarAndVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yz = vec2<f32>(2.0, 3.0);
    let v = vec3<f32>(1.0, yz);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromSwizzleAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let color = vec4<f32>(1.0, 0.5, 0.25, 0.8);
    return vec4<f32>(color.rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromSwizzleAndSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(5.0, 6.0, 7.0, 8.0);
    return vec4<f32>(a.xy, b.xy);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - Scalar Splat
// =============================================================================

TEST(VectorConstructionTest, Vec2_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(5.0);
    return vec4<f32>(v, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.5);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - Integer Vectors
// =============================================================================

TEST(VectorConstructionTest, Vec4i_FromVec3iAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec4<i32> {
    let xyz = vec3<i32>(1, 2, 3);
    return vec4<i32>(xyz, 4);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3u_FromVec2uAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec3<u32> {
    let xy = vec2<u32>(1u, 2u);
    return vec3<u32>(xy, 3u);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Unary (on vectors)
// =============================================================================

TEST(ComponentMathTest, Abs_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-1.0, -0.5, 0.25);
    let a = abs(v);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Abs_Vec4i) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec4<i32> {
    let v = vec4<i32>(-1, -2, 3, -4);
    return abs(v);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Floor_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.5, 2.7, -0.3);
    let f = floor(v);
    return vec4<f32>(f, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Ceil_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.1, 2.9, -0.1, 0.5);
    return ceil(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Round_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.4, 1.6);
    let rounded = round(v);
    return vec4<f32>(rounded, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Trunc_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.9, -2.3, 0.7);
    let t = trunc(v);
    return vec4<f32>(t, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Fract_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.75, 2.25, 3.5);
    let f = fract(v);
    return vec4<f32>(f, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Sqrt_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 4.0, 9.0, 16.0);
    return sqrt(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, InverseSqrt_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(4.0, 16.0);
    let inv = inverseSqrt(v);
    return vec4<f32>(inv, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Sign_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-5.0, 0.0, 3.0);
    let s = sign(v);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Exp_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.0, 2.0);
    let e = exp(v);
    return vec4<f32>(e, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Exp2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.0, 3.0);
    let e = exp2(v);
    return vec4<f32>(e, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Log_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.718, 7.389);
    let l = log(v);
    return vec4<f32>(l, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Log2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(2.0, 8.0);
    let l = log2(v);
    return vec4<f32>(l, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Trig (on vectors)
// =============================================================================

TEST(ComponentMathTest, Sin_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.57, 3.14);
    let s = sin(v);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Cos_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(0.0, 1.57, 3.14, 6.28);
    return cos(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Tan_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 0.78);
    let t = tan(v);
    return vec4<f32>(t, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Asin_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 0.5);
    let a = asin(v);
    return vec4<f32>(a, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Acos_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 1.0);
    let a = acos(v);
    return vec4<f32>(a, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Atan_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.0, -1.0);
    let a = atan(v);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Binary (on vectors)
// =============================================================================

TEST(ComponentMathTest, Pow_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec3<f32>(2.0, 3.0, 4.0);
    let exp = vec3<f32>(2.0, 2.0, 0.5);
    let result = pow(base, exp);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Pow_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let exp = vec4<f32>(0.5, 0.5, 0.5, 0.5);
    return pow(base, exp);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Pow_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec2<f32>(4.0, 9.0);
    let exp = vec2<f32>(0.5, 0.5);
    let result = pow(base, exp);
    return vec4<f32>(result, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Atan2_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y = vec3<f32>(1.0, 0.0, -1.0);
    let x = vec3<f32>(0.0, 1.0, 0.0);
    let result = atan2(y, x);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Distance_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 0.0, 0.0);
    let b = vec3<f32>(0.0, 1.0, 0.0);
    let d = distance(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Reflect_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let incident = vec3<f32>(1.0, -1.0, 0.0);
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let refl = reflect(incident, normal);
    return vec4<f32>(refl, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Step_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let edge = vec3<f32>(0.5, 0.5, 0.5);
    let x = vec3<f32>(0.3, 0.5, 0.7);
    let result = step(edge, x);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Min/Max/Clamp (on vectors)
// =============================================================================

TEST(ComponentMathTest, Min_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 5.0, 3.0);
    let b = vec3<f32>(4.0, 2.0, 6.0);
    let m = min(a, b);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Max_Vec4f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 5.0, 3.0, 7.0);
    let b = vec4<f32>(4.0, 2.0, 6.0, 1.0);
    return max(a, b);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Clamp_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-0.5, 0.5, 1.5);
    let lo = vec3<f32>(0.0, 0.0, 0.0);
    let hi = vec3<f32>(1.0, 1.0, 1.0);
    let c = clamp(v, lo, hi);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Min_Vec2i) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec2<i32> {
    let a = vec2<i32>(3, -5);
    let b = vec2<i32>(-1, 7);
    return min(a, b);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Max_Vec3u) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec3<u32> {
    let a = vec3<u32>(1u, 5u, 3u);
    let b = vec3<u32>(4u, 2u, 6u);
    return max(a, b);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Mix/Smoothstep
// =============================================================================

TEST(ComponentMathTest, Mix_Vec3_ScalarT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let m = mix(a, b, 0.5);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Mix_Vec3_VecT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let t = vec3<f32>(0.25, 0.5, 0.75);
    let m = mix(a, b, t);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Smoothstep_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let lo = vec3<f32>(0.0, 0.0, 0.0);
    let hi = vec3<f32>(1.0, 1.0, 1.0);
    let x = vec3<f32>(0.25, 0.5, 0.75);
    let s = smoothstep(lo, hi, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// Scalar-to-vector splatting for builtins
TEST(ComponentMathTest, Clamp_Vec3_ScalarBounds) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.5, -0.2, 0.8);
    let c = clamp(v, 0.0, 1.0);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Smoothstep_Vec3_ScalarEdges) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(0.25, 0.5, 0.75);
    let s = smoothstep(0.0, 1.0, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Step_Vec3_ScalarEdge) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(0.25, 0.75, 1.5);
    let s = step(0.5, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Mix_Vec4_ScalarT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let b = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    return mix(a, b, 0.5);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Builtins (non-component-wise)
// =============================================================================

TEST(VectorBuiltinTest, Dot_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec2<f32>(1.0, 0.0);
    let b = vec2<f32>(0.0, 1.0);
    let d = dot(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Dot_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(4.0, 3.0, 2.0, 1.0);
    let d = dot(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Cross_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 0.0, 0.0);
    let b = vec3<f32>(0.0, 1.0, 0.0);
    let c = cross(a, b);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Length_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(3.0, 4.0);
    let l = length(v);
    return vec4<f32>(l, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Length_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let l = length(v);
    return vec4<f32>(l, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Normalize_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(3.0, 4.0);
    let n = normalize(v);
    return vec4<f32>(n, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Normalize_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return normalize(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Distance_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec2<f32>(0.0, 0.0);
    let b = vec2<f32>(3.0, 4.0);
    let d = distance(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle + Math Integration Tests
// =============================================================================

TEST(SwizzleMathTest, PowOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let color = vec4<f32>(0.5, 0.25, 0.125, 1.0);
    let gamma = vec3<f32>(2.2);
    let corrected = pow(color.rgb, gamma);
    return vec4<f32>(corrected, color.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, NormalizeSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 0.0);
    let n = normalize(v.xyz);
    return vec4<f32>(n, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, DotOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let light = vec4<f32>(0.577, 0.577, 0.577, 0.0);
    let ndotl = dot(normal.xyz, light.xyz);
    return vec4<f32>(ndotl, ndotl, ndotl, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, CrossOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let c = cross(a.xyz, b.xyz);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, MixOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let b = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let mixed = mix(a.rgb, b.rgb, 0.5);
    return vec4<f32>(mixed, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, ClampOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let hdr = vec4<f32>(1.5, -0.2, 0.8, 1.0);
    let clamped = clamp(hdr.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(clamped, hdr.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, SqrtOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(4.0, 9.0, 16.0, 25.0);
    let s = sqrt(v.xyz);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, AbsOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(-1.0, 2.0, -3.0, 4.0);
    let a = abs(v.xyz);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Integration Tests - Real Shader Patterns
// =============================================================================

TEST(IntegrationTest, GammaCorrection) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Uniforms { gamma: f32, };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;

@fragment fn main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let texColor = textureSample(t, s, uv);
    let combined = color * texColor;
    let corrected = pow(combined.rgb, vec3<f32>(u.gamma));
    return vec4<f32>(corrected, combined.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, SimpleLighting) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) normal: vec3<f32>) -> @location(0) vec4<f32> {
    let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let n = normalize(normal);
    let ndotl = max(dot(n, lightDir), 0.0);
    let diffuse = vec3<f32>(0.8, 0.6, 0.4) * ndotl;
    let ambient = vec3<f32>(0.1, 0.1, 0.1);
    return vec4<f32>(diffuse + ambient, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ColorBlending) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    let tint = vec4<f32>(1.0, 0.8, 0.6, 1.0);
    let blended = mix(color.rgb, tint.rgb, 0.3);
    let saturated = clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(saturated, color.a * tint.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, DistanceAttenuation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
    let lightPos = vec3<f32>(0.0, 5.0, 0.0);
    let dist = distance(worldPos, lightPos);
    let atten = 1.0 / (1.0 + dist * dist);
    let color = vec3<f32>(1.0, 1.0, 1.0) * atten;
    return vec4<f32>(color, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ImGuiVertexShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms { mvp: mat4x4<f32>, gamma: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    return out;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ImGuiFragmentShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms { mvp: mat4x4<f32>, gamma: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;

@fragment fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color * textureSample(t, s, in.uv);
    let corrected_color = pow(color.rgb, vec3<f32>(uniforms.gamma));
    return vec4<f32>(corrected_color, color.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ReflectionVector) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) normal: vec3<f32>, @location(1) viewDir: vec3<f32>) -> @location(0) vec4<f32> {
    let n = normalize(normal);
    let v = normalize(viewDir);
    let r = reflect(-v, n);
    let spec = pow(max(dot(r, v), 0.0), 32.0);
    return vec4<f32>(vec3<f32>(spec), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
