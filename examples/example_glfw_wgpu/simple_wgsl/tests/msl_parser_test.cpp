#include <gtest/gtest.h>
#include <string>

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

struct ParseResult {
    bool success;
    std::string error;
    SsirModule* mod; // Caller owns via SsirModuleGuard
};

ParseResult ParseMsl(const std::string& msl) {
    ParseResult res = {false, "", nullptr};

    char* err = nullptr;
    MslToSsirOptions opts = {};
    opts.preserve_names = 1;

    MslToSsirResult result = msl_to_ssir(msl.c_str(), &opts, &res.mod, &err);

    if (result != MSL_TO_SSIR_OK) {
        res.error = "MSL parse failed: " + std::string(err ? err : "unknown");
        msl_to_ssir_free(err);
        return res;
    }

    msl_to_ssir_free(err);
    res.success = true;
    return res;
}

struct RoundtripResult {
    bool success;
    std::string error;
    std::string original_msl;
    std::string roundtrip_msl;
};

// WGSL -> AST -> Lower -> SSIR -> MSL -> parse MSL -> SSIR -> MSL
// Uses the direct SSIR path (no SPIR-V round-trip) for best fidelity.
RoundtripResult WgslToMslRoundtrip(const std::string& wgsl) {
    RoundtripResult res = {false, "", "", ""};

    // 1. WGSL -> AST -> Lower -> SSIR
    WgslAstNode* ast = wgsl_parse(wgsl.c_str());
    if (!ast) { res.error = "WGSL parse failed"; return res; }

    WgslResolver* resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); res.error = "Resolve failed"; return res; }

    WgslLowerOptions lower_opts = {};
    lower_opts.enable_debug_names = 1;
    WgslLower* lower = wgsl_lower_create(ast, resolver, &lower_opts);
    if (!lower) {
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        res.error = "Lower failed";
        return res;
    }

    const SsirModule* ssir = wgsl_lower_get_ssir(lower);

    // 2. SSIR -> MSL (first pass)
    char* msl1 = nullptr;
    char* err = nullptr;
    SsirToMslOptions msl_opts = {};
    msl_opts.preserve_names = 1;

    if (ssir_to_msl(ssir, &msl_opts, &msl1, &err) != SSIR_TO_MSL_OK) {
        res.error = "SSIR -> MSL (pass 1) failed: " + std::string(err ? err : "unknown");
        ssir_to_msl_free(err);
        wgsl_lower_destroy(lower);
        wgsl_resolver_free(resolver);
        wgsl_free_ast(ast);
        return res;
    }
    res.original_msl = msl1;
    ssir_to_msl_free(msl1);

    wgsl_lower_destroy(lower);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    // 3. Parse MSL -> SSIR
    SsirModule* mod2 = nullptr;
    MslToSsirOptions parse_opts = {};
    parse_opts.preserve_names = 1;

    if (msl_to_ssir(res.original_msl.c_str(), &parse_opts, &mod2, &err) != MSL_TO_SSIR_OK) {
        res.error = "MSL parse failed: " + std::string(err ? err : "unknown") +
                    "\nMSL was:\n" + res.original_msl;
        msl_to_ssir_free(err);
        return res;
    }
    SsirModuleGuard guard2(mod2);

    // 4. SSIR -> MSL (second pass)
    char* msl2 = nullptr;
    if (ssir_to_msl(mod2, &msl_opts, &msl2, &err) != SSIR_TO_MSL_OK) {
        res.error = "SSIR -> MSL (pass 2) failed: " + std::string(err ? err : "unknown");
        ssir_to_msl_free(err);
        return res;
    }
    res.roundtrip_msl = msl2;
    ssir_to_msl_free(msl2);

    res.success = true;
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// Direct MSL parsing tests
// ---------------------------------------------------------------------------

TEST(MslParser, ParseComputeKernel) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void main0(
            device float* data [[buffer(0)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            data[gid.x] = (data[gid.x] * 2.0);
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_COMPUTE);
}

TEST(MslParser, ParseVertexShader) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct vs_out {
            float4 pos [[position]];
        };

        vertex vs_out vs(
            uint vid [[vertex_id]]
        ) {
            vs_out _out = {};
            _out.pos = float4(0.0, 0.0, 0.0, 1.0);
            return _out;
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_VERTEX);
}

TEST(MslParser, ParseFragmentShader) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct fs_out {
            float4 color [[color(0)]];
        };

        fragment fs_out fs() {
            fs_out _out = {};
            _out.color = float4(1.0, 0.0, 0.0, 1.0);
            return _out;
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_FRAGMENT);
}

TEST(MslParser, ParseComputeWithUniformBuffer) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct Params {
            float scale;
        };

        kernel void cs(
            constant Params& params [[buffer(0)]],
            device float* data [[buffer(1)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            data[gid.x] = (data[gid.x] * params.scale);
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_COMPUTE);
}

TEST(MslParser, ParseVertexWithAttributes) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct vs_out {
            float4 pos [[position]];
            float3 color [[user(loc_0)]];
        };

        vertex vs_out vs(
            uint vid [[vertex_id]]
        ) {
            vs_out _out = {};
            _out.pos = float4(0.0, 0.0, 0.0, 1.0);
            _out.color = float3(1.0, 0.0, 0.0);
            return _out;
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_VERTEX);
}

TEST(MslParser, ParseFragmentWithStageIn) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct fs_in {
            float3 color [[user(loc_0)]];
        };

        struct fs_out {
            float4 color [[color(0)]];
        };

        fragment fs_out fs(
            fs_in _in [[stage_in]]
        ) {
            fs_out _out = {};
            _out.color = float4(_in.color, 1.0);
            return _out;
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    ASSERT_GT(res.mod->entry_point_count, 0u);
    EXPECT_EQ(res.mod->entry_points[0].stage, SSIR_STAGE_FRAGMENT);
}

// ---------------------------------------------------------------------------
// Re-emission from parsed MSL
// ---------------------------------------------------------------------------

TEST(MslParser, ParseAndReemit) {
    const char* msl = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void main0(
            device float* data [[buffer(0)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            data[gid.x] = (data[gid.x] * 2.0);
        }
    )";

    auto res = ParseMsl(msl);
    ASSERT_TRUE(res.success) << res.error;
    SsirModuleGuard guard(res.mod);

    // Re-emit to MSL
    char* msl2 = nullptr;
    char* err = nullptr;
    SsirToMslOptions opts = {};
    opts.preserve_names = 1;

    SsirToMslResult result = ssir_to_msl(res.mod, &opts, &msl2, &err);
    ASSERT_EQ(result, SSIR_TO_MSL_OK) << (err ? err : "unknown error");

    std::string output = msl2;
    ssir_to_msl_free(msl2);
    ssir_to_msl_free(err);

    EXPECT_NE(output.find("#include <metal_stdlib>"), std::string::npos);
    EXPECT_NE(output.find("kernel "), std::string::npos);
    EXPECT_NE(output.find("[[thread_position_in_grid]]"), std::string::npos);
    EXPECT_NE(output.find("device "), std::string::npos);
}

// ---------------------------------------------------------------------------
// Round-trip tests: WGSL -> SSIR -> MSL -> parse -> SSIR -> MSL
// ---------------------------------------------------------------------------

TEST(MslRoundtrip, ComputeShader) {
    const char* wgsl = R"(
        struct Buf { data: array<f32, 64> };
        @group(0) @binding(0) var<storage, read_write> buf: Buf;
        @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) gid: vec3u) {
            buf.data[gid.x] = f32(gid.x);
        }
    )";

    auto res = WgslToMslRoundtrip(wgsl);
    ASSERT_TRUE(res.success) << res.error;

    EXPECT_NE(res.original_msl.find("kernel "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("kernel "), std::string::npos);

    EXPECT_NE(res.original_msl.find("[[thread_position_in_grid]]"), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("[[thread_position_in_grid]]"), std::string::npos);

    EXPECT_NE(res.original_msl.find("device "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("device "), std::string::npos);
}

TEST(MslRoundtrip, VertexShader) {
    const char* wgsl = R"(
        @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
            let x = f32(vid);
            return vec4f(x, 0.0, 0.0, 1.0);
        }
    )";

    auto res = WgslToMslRoundtrip(wgsl);
    ASSERT_TRUE(res.success) << res.error;

    EXPECT_NE(res.original_msl.find("vertex "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("vertex "), std::string::npos);

    EXPECT_NE(res.original_msl.find("[[position]]"), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("[[position]]"), std::string::npos);

    EXPECT_NE(res.original_msl.find("[[vertex_id]]"), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("[[vertex_id]]"), std::string::npos);
}

TEST(MslRoundtrip, FragmentShader) {
    const char* wgsl = R"(
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";

    auto res = WgslToMslRoundtrip(wgsl);
    ASSERT_TRUE(res.success) << res.error;

    EXPECT_NE(res.original_msl.find("fragment "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("fragment "), std::string::npos);

    EXPECT_NE(res.original_msl.find("[[color(0)]]"), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("[[color(0)]]"), std::string::npos);
}

TEST(MslRoundtrip, UniformBuffer) {
    const char* wgsl = R"(
        struct UBO { val: f32 };
        @group(0) @binding(0) var<uniform> u: UBO;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(u.val, 0.0, 0.0, 1.0);
        }
    )";

    auto res = WgslToMslRoundtrip(wgsl);
    ASSERT_TRUE(res.success) << res.error;

    EXPECT_NE(res.original_msl.find("constant "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("constant "), std::string::npos);

    EXPECT_NE(res.original_msl.find("[[buffer("), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("[[buffer("), std::string::npos);
}

TEST(MslRoundtrip, ComputeWithArithmetic) {
    // Use arithmetic operations (not builtin functions) since the MSL parser
    // doesn't yet recognize all metal:: builtin function calls.
    const char* wgsl = R"(
        struct Buf { data: array<f32, 64> };
        @group(0) @binding(0) var<storage, read_write> buf: Buf;
        @compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) gid: vec3u) {
            buf.data[gid.x] = f32(gid.x) * 2.0 + 1.0;
        }
    )";

    auto res = WgslToMslRoundtrip(wgsl);
    ASSERT_TRUE(res.success) << res.error;

    EXPECT_NE(res.original_msl.find("kernel "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("kernel "), std::string::npos);

    EXPECT_NE(res.original_msl.find("device "), std::string::npos);
    EXPECT_NE(res.roundtrip_msl.find("device "), std::string::npos);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST(MslParser, InvalidMslDoesNotCrash) {
    // The MSL parser is lenient with unrecognized tokens/constructs.
    // This test ensures it doesn't crash on garbage input.
    const char* bad_msl = "this is not valid MSL at all {{{";

    SsirModule* mod = nullptr;
    char* err = nullptr;
    MslToSsirOptions opts = {};

    MslToSsirResult result = msl_to_ssir(bad_msl, &opts, &mod, &err);
    // Parser may or may not succeed, but it must not crash
    (void)result;

    if (mod) ssir_module_destroy(mod);
    msl_to_ssir_free(err);
}

TEST(MslParser, EmptyInputDoesNotCrash) {
    SsirModule* mod = nullptr;
    char* err = nullptr;
    MslToSsirOptions opts = {};

    MslToSsirResult result = msl_to_ssir("", &opts, &mod, &err);
    (void)result;

    if (mod) ssir_module_destroy(mod);
    msl_to_ssir_free(err);
}

TEST(MslParser, NullInputReturnsError) {
    SsirModule* mod = nullptr;
    char* err = nullptr;
    MslToSsirOptions opts = {};

    MslToSsirResult result = msl_to_ssir(nullptr, &opts, &mod, &err);
    EXPECT_NE(result, MSL_TO_SSIR_OK)
        << "Null input should fail";

    if (mod) ssir_module_destroy(mod);
    msl_to_ssir_free(err);
}

TEST(MslParser, ResultStringConversion) {
    EXPECT_STREQ(msl_to_ssir_result_string(MSL_TO_SSIR_OK), "Success");
    EXPECT_STREQ(msl_to_ssir_result_string(MSL_TO_SSIR_PARSE_ERROR), "Parse error");
    EXPECT_STREQ(msl_to_ssir_result_string(MSL_TO_SSIR_TYPE_ERROR), "Type error");
    EXPECT_STREQ(msl_to_ssir_result_string(MSL_TO_SSIR_UNSUPPORTED), "Unsupported feature");
}
