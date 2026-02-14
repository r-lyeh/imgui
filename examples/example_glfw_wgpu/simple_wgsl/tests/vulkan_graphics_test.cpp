#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef WGSL_HAS_VULKAN
#include "vulkan_graphics_harness.h"

class VulkanGraphicsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            ctx_ = std::make_unique<vk_graphics::GraphicsContext>();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Vulkan graphics not available: " << e.what();
        }
    }

    static void TearDownTestSuite() {
        ctx_.reset();
    }

    void SetUp() override {
        if (!ctx_) {
            GTEST_SKIP() << "Vulkan graphics context not initialized";
        }
    }

    static std::unique_ptr<vk_graphics::GraphicsContext> ctx_;
};

std::unique_ptr<vk_graphics::GraphicsContext> VulkanGraphicsTest::ctx_;

// Helper to extract RGBA from packed uint32_t (assuming RGBA8_Unorm layout)
inline void unpackRGBA(uint32_t pixel, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    r = (pixel >> 0) & 0xFF;
    g = (pixel >> 8) & 0xFF;
    b = (pixel >> 16) & 0xFF;
    a = (pixel >> 24) & 0xFF;
}

// Full-screen triangle vertices (covers entire viewport)
struct SimpleVertex {
    float x, y;
};

static const std::vector<SimpleVertex> kFullScreenTriangle = {
    {-1.0f, -1.0f},
    { 3.0f, -1.0f},
    {-1.0f,  3.0f},
};

// Test: Solid color fill - fragment shader outputs constant color
TEST_F(VulkanGraphicsTest, SolidColorFill) {
    // Compile vertex and fragment shaders separately to avoid duplicate type errors
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), width * height);

    // Check center pixel is red
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red channel should be ~255";
    EXPECT_LE(g, 5) << "Green channel should be ~0";
    EXPECT_LE(b, 5) << "Blue channel should be ~0";
    EXPECT_GE(a, 250) << "Alpha channel should be ~255";
}

// Test: Clear color verification
TEST_F(VulkanGraphicsTest, ClearColor) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Small triangle that doesn't cover the corners
    std::vector<SimpleVertex> small_tri = {
        {0.0f, 0.0f},
        {0.1f, 0.0f},
        {0.0f, 0.1f},
    };
    auto vb = ctx_->createVertexBuffer(small_tri);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);

    // Clear to green
    vk_graphics::ClearColor green = {0.0f, 1.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3}, {}, green);

    auto pixels = target.downloadAs<uint32_t>();

    // Check corner pixel (should be clear color = green)
    uint32_t corner = pixels[0];
    uint8_t r, g, b, a;
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0 (clear color)";
    EXPECT_GE(g, 250) << "Green should be ~255 (clear color)";
    EXPECT_LE(b, 5) << "Blue should be ~0 (clear color)";
}

// Test: Vertex attribute passing with color
TEST_F(VulkanGraphicsTest, VertexAttributes) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
            @location(1) color: vec3f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec3f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Vertex data: position (2 floats) + color (3 floats)
    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    // Full-screen triangle with blue color
    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        { 3.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f,  3.0f, 0.0f, 0.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    // Check center pixel is blue
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_LE(g, 5) << "Green should be ~0";
    EXPECT_GE(b, 250) << "Blue should be ~255";
}

// Test: Color interpolation across triangle
TEST_F(VulkanGraphicsTest, ColorInterpolation) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
            @location(1) color: vec3f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec3f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    // Triangle with RGB corners - full screen coverage
    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 1.0f, 0.0f, 0.0f}, // Red at bottom-left
        { 3.0f, -1.0f, 0.0f, 1.0f, 0.0f}, // Green at right
        {-1.0f,  3.0f, 0.0f, 0.0f, 1.0f}, // Blue at top
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    // Check bottom-left corner is reddish
    uint32_t bl = pixels[(height - 2) * width + 1];
    uint8_t r, g, b, a;
    unpackRGBA(bl, r, g, b, a);
    EXPECT_GT(r, 100) << "Bottom-left should have significant red";

    // Check that center has mixed colors (none should be 0 or 255)
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GT(r, 20) << "Center should have some red";
    EXPECT_GT(g, 20) << "Center should have some green";
}

// Test: Indexed drawing
TEST_F(VulkanGraphicsTest, IndexedDrawing) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 1.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Quad as 4 vertices
    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f,  1.0f},
    };

    // Two triangles forming the quad
    std::vector<uint16_t> indices = {0, 1, 2, 0, 2, 3};

    auto vb = ctx_->createVertexBuffer(vertices);
    auto ib = ctx_->createIndexBuffer(indices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->drawIndexed(pipeline, target, &vb, &ib, VK_INDEX_TYPE_UINT16,
                      {.index_count = 6});

    auto pixels = target.downloadAs<uint32_t>();

    // Check center pixel is yellow
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red should be ~255";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_LE(b, 5) << "Blue should be ~0";
}

// Test: Fragment shader math operations
TEST_F(VulkanGraphicsTest, FragmentMathOps) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            let a = 0.5;
            let b = abs(-0.3);
            let c = clamp(1.5, 0.0, 1.0);
            let d = max(0.2, 0.1);
            return vec4f(a, b, c, d);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    // r = 0.5 * 255 = 127.5
    EXPECT_NEAR(r, 128, 2) << "Red should be ~0.5";
    // g = 0.3 * 255 = 76.5
    EXPECT_NEAR(g, 77, 2) << "Green should be ~0.3";
    // b = 1.0 * 255 = 255
    EXPECT_GE(b, 250) << "Blue should be ~1.0";
    // a = 0.2 * 255 = 51
    EXPECT_NEAR(a, 51, 2) << "Alpha should be ~0.2";
}

// Test: Vertex position transformation
TEST_F(VulkanGraphicsTest, VertexTransform) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            let scaled = in.position * 0.5;
            return vec4f(scaled, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(0.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    // Full quad vertices, will be scaled to half
    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        {-1.0f,  1.0f},
        { 1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f,  1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);

    // Clear to black
    vk_graphics::ClearColor black = {0.0f, 0.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 6}, {}, black);

    auto pixels = target.downloadAs<uint32_t>();

    // Center should be cyan (inside scaled quad)
    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_GE(b, 250) << "Blue should be ~255";

    // Corner should be black (outside the scaled quad)
    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Corner red should be ~0 (clear)";
    EXPECT_LE(g, 5) << "Corner green should be ~0 (clear)";
    EXPECT_LE(b, 5) << "Corner blue should be ~0 (clear)";
}

// Test: Vector operations in shaders
TEST_F(VulkanGraphicsTest, VectorOperations) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        @vertex fn main(in: VertexInput) -> @builtin(position) vec4f {
            return vec4f(in.position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            let a = vec3f(0.5, 0.5, 0.5);
            let b = vec3f(0.5, 0.0, 0.5);
            let sum = a + b;
            let clamped = clamp(sum, vec3f(0.0), vec3f(1.0));
            return vec4f(clamped, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    EXPECT_GE(r, 250) << "Red should be 1.0";
    EXPECT_NEAR(g, 128, 2) << "Green should be 0.5";
    EXPECT_GE(b, 250) << "Blue should be 1.0";
}

// Test: Simple conditional in fragment shader
TEST_F(VulkanGraphicsTest, FragmentConditional) {
    const char* vs_source = R"(
        struct VertexInput {
            @location(0) position: vec2f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) ndc: vec2f,
        };

        @vertex fn main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.position = vec4f(in.position, 0.0, 1.0);
            out.ndc = in.position;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            if (ndc.x > 0.0) {
                return vec4f(1.0, 0.0, 0.0, 1.0);
            } else {
                return vec4f(0.0, 0.0, 1.0, 1.0);
            }
        }
    )";

    auto vs_result = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint8_t r, g, b, a;

    // Left side (x < 0) should be blue
    uint32_t left = pixels[(height / 2) * width + (width / 4)];
    unpackRGBA(left, r, g, b, a);
    EXPECT_LE(r, 5) << "Left side red should be ~0";
    EXPECT_GE(b, 250) << "Left side blue should be ~255";

    // Right side (x > 0) should be red
    uint32_t right = pixels[(height / 2) * width + (width * 3 / 4)];
    unpackRGBA(right, r, g, b, a);
    EXPECT_GE(r, 250) << "Right side red should be ~255";
    EXPECT_LE(b, 5) << "Right side blue should be ~0";
}

// ============================================================================
// GLSL Graphics Tests
// ============================================================================

TEST_F(VulkanGraphicsTest, GlslSolidColorFill) {
    const char* vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), width * height);

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 250) << "Red channel should be ~255";
    EXPECT_LE(g, 5) << "Green channel should be ~0";
    EXPECT_LE(b, 5) << "Blue channel should be ~0";
    EXPECT_GE(a, 250) << "Alpha channel should be ~255";
}

TEST_F(VulkanGraphicsTest, GlslVertexAttributes) {
    const char* vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 color;

        layout(location = 0) out vec3 fragColor;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragColor = color;
        }
    )";

    const char* fs_source = R"(
        #version 450
        layout(location = 0) in vec3 fragColor;
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(fragColor, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    struct ColorVertex {
        float x, y;
        float r, g, b;
    };

    std::vector<ColorVertex> vertices = {
        {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        { 3.0f, -1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f,  3.0f, 0.0f, 0.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(ColorVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColorVertex, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColorVertex, r)},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_LE(g, 5) << "Green should be ~0";
    EXPECT_GE(b, 250) << "Blue should be ~255";
}

TEST_F(VulkanGraphicsTest, GlslFragmentMathOps) {
    const char* vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            float a = 0.5;
            float b = abs(-0.3);
            float c = clamp(1.5, 0.0, 1.0);
            float d = max(0.2, 0.1);
            outColor = vec4(a, b, c, d);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTriangle);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);

    EXPECT_NEAR(r, 128, 2) << "Red should be ~0.5";
    EXPECT_NEAR(g, 77, 2) << "Green should be ~0.3";
    EXPECT_GE(b, 250) << "Blue should be ~1.0";
    EXPECT_NEAR(a, 51, 2) << "Alpha should be ~0.2";
}

TEST_F(VulkanGraphicsTest, GlslVertexTransform) {
    const char* vs_source = R"(
        #version 450
        layout(location = 0) in vec2 position;

        void main() {
            vec2 scaled = position * 0.5;
            gl_Position = vec4(scaled, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        #version 450
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4(0.0, 1.0, 1.0, 1.0);
        }
    )";

    auto vs_result = wgsl_test::CompileGlsl(vs_source, WGSL_STAGE_VERTEX);
    ASSERT_TRUE(vs_result.success) << vs_result.error;

    auto fs_result = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs_result.success) << fs_result.error;

    std::vector<SimpleVertex> vertices = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        {-1.0f,  1.0f},
        { 1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f,  1.0f},
    };

    auto vb = ctx_->createVertexBuffer(vertices);

    const uint32_t width = 64, height = 64;
    auto target = ctx_->createColorTarget(width, height);

    vk_graphics::GraphicsPipelineConfig config;
    config.vertex_spirv = vs_result.spirv.data();
    config.vertex_spirv_words = vs_result.spirv.size();
    config.vertex_entry = "main";
    config.fragment_spirv = fs_result.spirv.data();
    config.fragment_spirv_words = fs_result.spirv.size();
    config.fragment_entry = "main";
    config.vertex_stride = sizeof(SimpleVertex);
    config.vertex_attributes = {
        {0, VK_FORMAT_R32G32_SFLOAT, 0},
    };

    auto pipeline = ctx_->createPipeline(config);
    vk_graphics::ClearColor black = {0.0f, 0.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 6}, {}, black);

    auto pixels = target.downloadAs<uint32_t>();

    uint32_t center = pixels[(height / 2) * width + (width / 2)];
    uint8_t r, g, b, a;
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Red should be ~0";
    EXPECT_GE(g, 250) << "Green should be ~255";
    EXPECT_GE(b, 250) << "Blue should be ~255";

    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Corner red should be ~0 (clear)";
    EXPECT_LE(g, 5) << "Corner green should be ~0 (clear)";
    EXPECT_LE(b, 5) << "Corner blue should be ~0 (clear)";
}

#endif // WGSL_HAS_VULKAN
