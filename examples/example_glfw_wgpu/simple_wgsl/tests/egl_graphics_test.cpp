#ifdef WGSL_HAS_EGL

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "egl_compute_harness.h"

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

GlslResult WgslToOpenGlsl(const char* wgsl_source, SsirStage stage) {
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

    SsirToGlslResult gres = ssir_to_glsl(ssir, stage, &gopts, &glsl, &glsl_err);
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

class EGLGraphicsTest : public ::testing::Test {
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

std::unique_ptr<egl_compute::Context> EGLGraphicsTest::ctx_;
std::string EGLGraphicsTest::skip_reason_ = "EGL context not initialized";

/* ==== vertex shader tests ==== */

TEST_F(EGLGraphicsTest, SimpleVertex) {
    auto g = WgslToOpenGlsl(R"(
        @vertex fn main() -> @builtin(position) vec4f {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_VERTEX_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, VertexWithInput) {
    auto g = WgslToOpenGlsl(R"(
        @vertex fn main(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_VERTEX_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, VertexTransform) {
    auto g = WgslToOpenGlsl(R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };
        @vertex fn main(vin: VertexInput) -> @builtin(position) vec4f {
            let scaled = vin.position * 0.5;
            return vec4f(scaled, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_VERTEX_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, VertexWithUniform) {
    auto g = WgslToOpenGlsl(R"(
        struct Transforms { mvp: mat4x4f };
        @group(0) @binding(0) var<uniform> xform: Transforms;
        @vertex fn main(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return xform.mvp * vec4f(pos, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_VERTEX_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

/* ==== fragment shader tests ==== */

TEST_F(EGLGraphicsTest, SimpleFragment) {
    auto g = WgslToOpenGlsl(R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_FRAGMENT_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, FragmentWithInput) {
    auto g = WgslToOpenGlsl(R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_FRAGMENT_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, FragmentMathOps) {
    auto g = WgslToOpenGlsl(R"(
        @fragment fn main() -> @location(0) vec4f {
            let a = 0.5;
            let b = abs(-0.3);
            let c = clamp(1.5, 0.0, 1.0);
            let d = max(0.2, 0.1);
            return vec4f(a, b, c, d);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_FRAGMENT_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, FragmentConditional) {
    auto g = WgslToOpenGlsl(R"(
        @fragment fn main(@location(0) value: f32) -> @location(0) vec4f {
            if (value > 0.5) {
                return vec4f(1.0, 0.0, 0.0, 1.0);
            } else {
                return vec4f(0.0, 0.0, 1.0, 1.0);
            }
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(g.success) << g.error;
    auto r = ctx_->compileShader(GL_FRAGMENT_SHADER, g.glsl);
    EXPECT_TRUE(r.success) << "GLSL:\n" << g.glsl << "\nGL error:\n" << r.info_log;
}

/* ==== linked program tests ==== */

TEST_F(EGLGraphicsTest, LinkedPassthrough) {
    auto vs = WgslToOpenGlsl(R"(
        @vertex fn main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
            return vec4f(pos, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(vs.success) << vs.error;

    auto fs = WgslToOpenGlsl(R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto r = ctx_->linkProgram(vs.glsl, fs.glsl);
    EXPECT_TRUE(r.success)
        << "VS:\n" << vs.glsl << "\nFS:\n" << fs.glsl << "\nLink error:\n" << r.info_log;
}

TEST_F(EGLGraphicsTest, LinkedVertexColor) {
    auto vs = WgslToOpenGlsl(R"(
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) color: vec3f,
        };
        @vertex fn main(@location(0) position: vec2f,
                        @location(1) color: vec3f) -> VOut {
            var out: VOut;
            out.pos = vec4f(position, 0.0, 1.0);
            out.color = color;
            return out;
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(vs.success) << vs.error;

    auto fs = WgslToOpenGlsl(R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto r = ctx_->linkProgram(vs.glsl, fs.glsl);
    EXPECT_TRUE(r.success)
        << "VS:\n" << vs.glsl << "\nFS:\n" << fs.glsl << "\nLink error:\n" << r.info_log;
}

/* ==== render-to-PNG tests ==== */

#include "stb_image_write.h"

TEST_F(EGLGraphicsTest, RenderSolidRed) {
    auto vs = WgslToOpenGlsl(R"(
        @vertex fn main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
            return vec4f(pos, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(vs.success) << vs.error;

    auto fs = WgslToOpenGlsl(R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto r = ctx_->renderToPixels(vs.glsl, fs.glsl, 256, 256);
    ASSERT_TRUE(r.success) << r.error;

    stbi_write_png("egl_solid_red.png", r.width, r.height, 4, r.pixels.data(), r.width * 4);

    /* Verify center pixel is red */
    size_t cx = (128 * 256 + 128) * 4;
    EXPECT_GE(r.pixels[cx + 0], 250);  /* R */
    EXPECT_LE(r.pixels[cx + 1], 5);    /* G */
    EXPECT_LE(r.pixels[cx + 2], 5);    /* B */
}

TEST_F(EGLGraphicsTest, RenderGradient) {
    auto vs = WgslToOpenGlsl(R"(
        @vertex fn main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
            return vec4f(pos, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(vs.success) << vs.error;

    /* Use gl_FragCoord to make a gradient: red increases left-to-right,
       green increases bottom-to-top (256x256) */
    auto fs = WgslToOpenGlsl(R"(
        @fragment fn main(@builtin(position) coord: vec4f) -> @location(0) vec4f {
            let u = coord.x / 256.0;
            let v = coord.y / 256.0;
            return vec4f(u, v, 0.5, 1.0);
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto r = ctx_->renderToPixels(vs.glsl, fs.glsl, 256, 256);
    ASSERT_TRUE(r.success) << r.error;

    stbi_write_png("egl_gradient.png", r.width, r.height, 4, r.pixels.data(), r.width * 4);

    /* Top-right corner should be bright red+green, bottom-left should be dark */
    size_t tr = (0 * 256 + 255) * 4;          /* top-right (row 0 after flip) */
    size_t bl = (255 * 256 + 0) * 4;          /* bottom-left */
    EXPECT_GE(r.pixels[tr + 0], 200);  /* R high at right */
    EXPECT_GE(r.pixels[tr + 1], 200);  /* G high at top */
    EXPECT_LE(r.pixels[bl + 0], 10);   /* R low at left */
    EXPECT_LE(r.pixels[bl + 1], 10);   /* G low at bottom */
}

TEST_F(EGLGraphicsTest, RenderCheckerboard) {
    auto vs = WgslToOpenGlsl(R"(
        @vertex fn main(@location(0) pos: vec2f) -> @builtin(position) vec4f {
            return vec4f(pos, 0.0, 1.0);
        }
    )", SSIR_STAGE_VERTEX);
    ASSERT_TRUE(vs.success) << vs.error;

    /* 8x8 checkerboard using floor + modulo */
    auto fs = WgslToOpenGlsl(R"(
        @fragment fn main(@builtin(position) coord: vec4f) -> @location(0) vec4f {
            let cx = u32(coord.x) / 32u;
            let cy = u32(coord.y) / 32u;
            let check = (cx + cy) % 2u;
            if (check == 0u) {
                return vec4f(1.0, 1.0, 1.0, 1.0);
            } else {
                return vec4f(0.2, 0.2, 0.2, 1.0);
            }
        }
    )", SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto r = ctx_->renderToPixels(vs.glsl, fs.glsl, 256, 256);
    ASSERT_TRUE(r.success) << r.error;

    stbi_write_png("egl_checkerboard.png", r.width, r.height, 4, r.pixels.data(), r.width * 4);

    /* Sample two adjacent cells: (16,16) should be white, (48,16) should be dark */
    size_t white_px = ((256 - 16) * 256 + 16) * 4;  /* GL y-flip: row 16 from bottom -> row 240 */
    size_t dark_px  = ((256 - 16) * 256 + 48) * 4;
    EXPECT_GE(r.pixels[white_px + 0], 250);
    EXPECT_LE(r.pixels[dark_px + 0], 60);
}

#endif // WGSL_HAS_EGL
