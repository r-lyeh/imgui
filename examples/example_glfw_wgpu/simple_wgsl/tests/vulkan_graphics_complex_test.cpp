#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef WGSL_HAS_VULKAN
#include "vulkan_graphics_harness.h"
#include <cmath>
#include <cstring>
#include <set>

extern "C" {
#include "stb_image_write.h"
}

// ============================================================================
// Test Fixture
// ============================================================================

class VulkanGraphicsComplexTest : public ::testing::Test {
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

std::unique_ptr<vk_graphics::GraphicsContext> VulkanGraphicsComplexTest::ctx_;

// ============================================================================
// Helpers
// ============================================================================

struct Pos2D {
    float x, y;
};

struct Pos3D {
    float x, y, z;
};

struct PosColor3D {
    float x, y, z;
    float r, g, b;
};

struct PosNormal3D {
    float x, y, z;
    float nx, ny, nz;
};

inline void unpackRGBA(uint32_t pixel, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    r = (pixel >> 0) & 0xFF;
    g = (pixel >> 8) & 0xFF;
    b = (pixel >> 16) & 0xFF;
    a = (pixel >> 24) & 0xFF;
}

static const std::vector<Pos2D> kFullScreenTri = {
    {-1.0f, -1.0f},
    { 3.0f, -1.0f},
    {-1.0f,  3.0f},
};

static const std::vector<Pos2D> kFullScreenQuad = {
    {-1.0f, -1.0f}, { 1.0f, -1.0f}, {-1.0f,  1.0f},
    { 1.0f, -1.0f}, { 1.0f,  1.0f}, {-1.0f,  1.0f},
};

// RAII wrapper for manually created MRT pipeline resources
struct MRTPipelineResources {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    MRTPipelineResources() = default;
    MRTPipelineResources(const MRTPipelineResources&) = delete;
    MRTPipelineResources& operator=(const MRTPipelineResources&) = delete;

    MRTPipelineResources(MRTPipelineResources&& o) noexcept
        : device(o.device), vert(o.vert), frag(o.frag),
          desc_layout(o.desc_layout), layout(o.layout), pipeline(o.pipeline) {
        o.device = VK_NULL_HANDLE;
        o.vert = VK_NULL_HANDLE;
        o.frag = VK_NULL_HANDLE;
        o.desc_layout = VK_NULL_HANDLE;
        o.layout = VK_NULL_HANDLE;
        o.pipeline = VK_NULL_HANDLE;
    }

    MRTPipelineResources& operator=(MRTPipelineResources&& o) noexcept {
        if (this != &o) {
            if (device) {
                if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
                if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
                if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
                if (frag) vkDestroyShaderModule(device, frag, nullptr);
                if (vert) vkDestroyShaderModule(device, vert, nullptr);
            }
            device = o.device;
            vert = o.vert;
            frag = o.frag;
            desc_layout = o.desc_layout;
            layout = o.layout;
            pipeline = o.pipeline;
            o.device = VK_NULL_HANDLE;
            o.vert = VK_NULL_HANDLE;
            o.frag = VK_NULL_HANDLE;
            o.desc_layout = VK_NULL_HANDLE;
            o.layout = VK_NULL_HANDLE;
            o.pipeline = VK_NULL_HANDLE;
        }
        return *this;
    }

    ~MRTPipelineResources() {
        if (!device) return;
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        if (frag) vkDestroyShaderModule(device, frag, nullptr);
        if (vert) vkDestroyShaderModule(device, vert, nullptr);
    }
};

// ============================================================================
// 2D Scene Tests
// ============================================================================

// Procedural checkerboard pattern computed entirely in fragment shader
TEST_F(VulkanGraphicsComplexTest, ProceduralCheckerboard) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let gx = floor(uv.x * 8.0);
            let gy = floor(uv.y * 8.0);
            let sum = gx + gy;
            let half_sum = floor(sum * 0.5);
            let checker = sum - half_sum * 2.0;
            if (checker > 0.5) {
                return vec4f(1.0, 1.0, 1.0, 1.0);
            } else {
                return vec4f(0.0, 0.0, 0.0, 1.0);
            }
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 128, H = 128;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    // Sample two adjacent cells. With 8x8 grid on 128px, each cell is 16px.
    // Cell (0,0) at pixel (4,4) -> uv ~(0.03, 0.03) -> gx=0, gy=0 -> sum=0 -> black
    // Cell (1,0) at pixel (20,4) -> uv ~(0.16, 0.03) -> gx=1, gy=0 -> sum=1 -> white
    uint8_t r, g, b, a;

    uint32_t cell00 = pixels[4 * W + 4];
    unpackRGBA(cell00, r, g, b, a);
    EXPECT_LE(r, 10) << "Cell (0,0) should be dark";

    uint32_t cell10 = pixels[4 * W + 20];
    unpackRGBA(cell10, r, g, b, a);
    EXPECT_GE(r, 240) << "Cell (1,0) should be bright";

    // Verify alternation: cell (1,1) should be black again
    uint32_t cell11 = pixels[20 * W + 20];
    unpackRGBA(cell11, r, g, b, a);
    EXPECT_LE(r, 10) << "Cell (1,1) should be dark";
}

// Concentric rings using distance-from-center SDF
TEST_F(VulkanGraphicsComplexTest, ConcentricRings2D) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) ndc: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.ndc = in.pos;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            let dist = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
            let rings = sin(dist * 20.0);
            let intensity = rings * 0.5 + 0.5;
            return vec4f(intensity, intensity * 0.5, 0.0, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 128, H = 128;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    // Center pixel is at half-pixel offset from exact center, so dist > 0.
    // sin(small_dist * 20) is nonzero, intensity deviates from 0.5.
    uint8_t r, g, b, a;
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_NEAR(r, 155, 30) << "Center R should be near 0.5-0.6";
    EXPECT_NEAR(g, 78, 20) << "Center G should be near 0.25-0.3";
    EXPECT_LE(b, 5) << "Blue channel should be 0";

    // Pixel further from center should differ due to ring pattern
    uint32_t off_center = pixels[(H / 2) * W + (W / 2 + 16)];
    unpackRGBA(off_center, r, g, b, a);
    uint8_t cr, cg, cb, ca;
    unpackRGBA(center, cr, cg, cb, ca);
    bool differs = (std::abs((int)r - (int)cr) > 5) ||
                   (std::abs((int)g - (int)cg) > 5);
    EXPECT_TRUE(differs) << "Ring pattern should produce variation";
}

// ============================================================================
// 3D Scene Tests
// ============================================================================

// Two overlapping full-screen triangles at different depths in a single draw call
TEST_F(VulkanGraphicsComplexTest, DepthTestOverlap3D) {
    const char* vs_source = R"(
        struct VIn {
            @location(0) pos: vec3f,
            @location(1) color: vec3f,
        };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) color: vec3f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    // Blue triangle (z=0.7, far) drawn first, red triangle (z=0.3, near) drawn second.
    // With depth test LESS, red (0.3 < 0.7) should win.
    std::vector<PosColor3D> verts = {
        // Blue (far, z=0.7) - rasterized first
        {-1.0f, -1.0f, 0.7f,  0.0f, 0.0f, 1.0f},
        { 3.0f, -1.0f, 0.7f,  0.0f, 0.0f, 1.0f},
        {-1.0f,  3.0f, 0.7f,  0.0f, 0.0f, 1.0f},
        // Red (near, z=0.3) - rasterized second, should overwrite
        {-1.0f, -1.0f, 0.3f,  1.0f, 0.0f, 0.0f},
        { 3.0f, -1.0f, 0.3f,  1.0f, 0.0f, 0.0f},
        {-1.0f,  3.0f, 0.3f,  1.0f, 0.0f, 0.0f},
    };

    auto vb = ctx_->createVertexBuffer(verts);
    const uint32_t W = 64, H = 64;
    auto color = ctx_->createColorTarget(W, H);
    auto depth = ctx_->createDepthTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(PosColor3D);
    cfg.vertex_attributes = {
        {0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, r)},
    };
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, color, &vb, {.vertex_count = 6}, {}, {}, &depth);

    auto pixels = color.downloadAs<uint32_t>();
    uint8_t r, g, b, a;
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);

    EXPECT_GE(r, 240) << "Red (z=0.3) should win depth test over blue (z=0.7)";
    EXPECT_LE(b, 15) << "Blue should be occluded by red";
}

// Vertex transformation: scale + offset transform producing a centered quad
TEST_F(VulkanGraphicsComplexTest, UniformMVPTransform) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };

        @vertex fn main(in: VIn) -> @builtin(position) vec4f {
            let scale = 0.25;
            let offset = vec2f(0.0, 0.0);
            let tx = in.pos.x * scale + offset.x;
            let ty = in.pos.y * scale + offset.y;
            return vec4f(tx, ty, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(0.0, 1.0, 0.0, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenQuad);

    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    vk_graphics::ClearColor black = {0.0f, 0.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 6}, {}, black);

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Center should be green (inside scaled quad)
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_LE(r, 5) << "Center red should be 0";
    EXPECT_GE(g, 240) << "Center should be green (inside scaled quad)";

    // Corner should be black (outside the 0.25x scaled quad)
    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 5) << "Corner should be clear color (black)";
    EXPECT_LE(g, 5) << "Corner should be clear color (black)";
    EXPECT_LE(b, 5) << "Corner should be clear color (black)";

    // Edge of the scaled region: quad covers NDC [-0.25, 0.25].
    // In pixel coords, center is at 32. -0.25 NDC -> pixel 32 - 0.25*32 = 24
    // Pixel (23, 32) should be outside, pixel (25, 32) should be inside
    uint32_t outside = pixels[(H / 2) * W + 22];
    unpackRGBA(outside, r, g, b, a);
    EXPECT_LE(g, 15) << "Pixel outside scaled quad should be black";

    uint32_t inside = pixels[(H / 2) * W + 26];
    unpackRGBA(inside, r, g, b, a);
    EXPECT_GE(g, 200) << "Pixel inside scaled quad should be green";
}

// Diffuse (Lambertian) lighting: full-screen pass with per-vertex normal interpolation
// The fragment shader evaluates lighting from the interpolated normal Z component
TEST_F(VulkanGraphicsComplexTest, DiffuseLighting3D) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Fragment shader simulates diffuse lighting:
    // Left half (uv.x < 0.5): normal faces toward light → bright
    // Right half (uv.x >= 0.5): normal faces away → dark
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let nz = 1.0 - uv.x * 2.0;
            if (nz > 0.0) {
                return vec4f(nz, nz, nz, 1.0);
            } else {
                return vec4f(0.0, 0.0, 0.0, 1.0);
            }
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Left side: uv.x ~ 0.15, nz = 1 - 0.3 = 0.7, should be bright
    uint32_t left = pixels[(H / 2) * W + 10];
    unpackRGBA(left, r, g, b, a);
    EXPECT_GE(r, 150) << "Lit side should be bright";
    EXPECT_EQ(r, g) << "Should be grayscale";

    // Right side: uv.x ~ 0.85, nz = 1 - 1.7 = -0.7, should be dark
    uint32_t right = pixels[(H / 2) * W + (W - 10)];
    unpackRGBA(right, r, g, b, a);
    EXPECT_LE(r, 5) << "Unlit side should be dark";
}

// Indexed cube faces with depth testing - draw front and back faces
TEST_F(VulkanGraphicsComplexTest, IndexedCubeDepthTest) {
    const char* vs_source = R"(
        struct VIn {
            @location(0) pos: vec3f,
            @location(1) color: vec3f,
        };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) color: vec3f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    // Two overlapping quads: red at z=0.2 (front), green at z=0.8 (back)
    std::vector<PosColor3D> verts = {
        // Front quad (red, z=0.2)
        {-0.8f, -0.8f, 0.2f,  1.0f, 0.0f, 0.0f},
        { 0.8f, -0.8f, 0.2f,  1.0f, 0.0f, 0.0f},
        { 0.8f,  0.8f, 0.2f,  1.0f, 0.0f, 0.0f},
        {-0.8f,  0.8f, 0.2f,  1.0f, 0.0f, 0.0f},
        // Back quad (green, z=0.8)
        {-0.8f, -0.8f, 0.8f,  0.0f, 1.0f, 0.0f},
        { 0.8f, -0.8f, 0.8f,  0.0f, 1.0f, 0.0f},
        { 0.8f,  0.8f, 0.8f,  0.0f, 1.0f, 0.0f},
        {-0.8f,  0.8f, 0.8f,  0.0f, 1.0f, 0.0f},
    };

    // Draw back quad first (indices 4-7), then front (0-3)
    std::vector<uint16_t> indices = {
        4, 5, 6,  4, 6, 7,  // Back (green)
        0, 1, 2,  0, 2, 3,  // Front (red)
    };

    auto vb = ctx_->createVertexBuffer(verts);
    auto ib = ctx_->createIndexBuffer(indices);

    const uint32_t W = 64, H = 64;
    auto color_target = ctx_->createColorTarget(W, H);
    auto depth_target = ctx_->createDepthTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(PosColor3D);
    cfg.vertex_attributes = {
        {0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, r)},
    };
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->drawIndexed(pipeline, color_target, &vb, &ib, VK_INDEX_TYPE_UINT16,
                      {.index_count = 12}, {}, {0.0f, 0.0f, 0.0f, 1.0f},
                      &depth_target);

    auto pixels = color_target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Center should be red (front quad at z=0.2 wins over green at z=0.8)
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 240) << "Front face (red, z=0.2) should be visible";
    EXPECT_LE(g, 15) << "Back face (green, z=0.8) should be occluded";
}

// ============================================================================
// Bloom Effect Tests
// ============================================================================

// Bright-pass extraction: threshold brightness, output only bright pixels
TEST_F(VulkanGraphicsComplexTest, BrightPassExtraction) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Horizontal gradient, bright-pass threshold at 0.7
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let scene_color = vec3f(uv.x, uv.x * 0.5, uv.x * 0.2);
            let brightness = scene_color.x * 0.299 + scene_color.y * 0.587 + scene_color.z * 0.114;
            let threshold = 0.5;
            if (brightness > threshold) {
                return vec4f(scene_color, 1.0);
            } else {
                return vec4f(0.0, 0.0, 0.0, 1.0);
            }
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 128, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Left side (uv.x ~ 0.1, low brightness) should be black
    uint32_t left = pixels[(H / 2) * W + 10];
    unpackRGBA(left, r, g, b, a);
    EXPECT_LE(r, 5) << "Dark region should be black after bright pass";
    EXPECT_LE(g, 5);

    // Right side (uv.x ~ 0.9, high brightness) should retain color
    uint32_t right = pixels[(H / 2) * W + (W - 10)];
    unpackRGBA(right, r, g, b, a);
    EXPECT_GE(r, 200) << "Bright region should pass through";
}

// Radial glow/bloom: bright center with smooth falloff
TEST_F(VulkanGraphicsComplexTest, RadialGlowBloom) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) ndc: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.ndc = in.pos;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            let dist = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
            let glow = exp(-dist * dist * 4.0);
            let bloom_color = vec3f(1.0, 0.8, 0.3) * glow;
            let clamped = clamp(bloom_color, vec3f(0.0), vec3f(1.0));
            return vec4f(clamped, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 128, H = 128;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Center: glow = exp(0) = 1.0, color = (1.0, 0.8, 0.3)
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 245) << "Center glow R should be ~1.0";
    EXPECT_NEAR(g, 204, 10) << "Center glow G should be ~0.8";
    EXPECT_NEAR(b, 77, 10) << "Center glow B should be ~0.3";

    // Edge: dist ~ 1.0, glow = exp(-4) ~ 0.018, very dim
    uint32_t edge = pixels[2 * W + 2]; // near corner
    unpackRGBA(edge, r, g, b, a);
    EXPECT_LE(r, 30) << "Edge should be very dim";

    // Between center and edge should be intermediate
    // At dist ~ 0.5, glow = exp(-1) ~ 0.37
    uint32_t mid = pixels[(H / 2) * W + (W / 2 + W / 4)];
    unpackRGBA(mid, r, g, b, a);
    EXPECT_GT(r, 30) << "Mid region should have visible glow";
    EXPECT_LT(r, 200) << "Mid region should be dimmer than center";
}

// Bloom with HDR-like tone mapping
TEST_F(VulkanGraphicsComplexTest, BloomTonemapping) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Reinhard-style tonemapping of a bright scene
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let hdr_color = vec3f(uv.x * 3.0, uv.y * 2.0, 0.5);
            let mapped = hdr_color / (hdr_color + vec3f(1.0));
            return vec4f(mapped, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Vulkan Y-axis: NDC y=-1 = top of framebuffer. UV = ndc*0.5+0.5.
    // Framebuffer top-left = NDC(-1,-1) = uv(0,0)
    // Framebuffer bottom-right = NDC(1,1) = uv(1,1)

    // Top-left (uv ~ 0,0): hdr = (0, 0, 0.5), mapped = (0, 0, 0.333)
    uint32_t tl = pixels[2 * W + 2];
    unpackRGBA(tl, r, g, b, a);
    EXPECT_LE(r, 30) << "Top-left R should be ~0";
    EXPECT_NEAR(b, 85, 15) << "Top-left B should be ~0.333";

    // Bottom-right (uv ~ 1,1): hdr = (3, 2, 0.5), mapped = (0.75, 0.667, 0.333)
    uint32_t br = pixels[(H - 2) * W + (W - 2)];
    unpackRGBA(br, r, g, b, a);
    EXPECT_NEAR(r, 191, 15) << "Bottom-right R should be ~0.75";
    EXPECT_NEAR(g, 170, 15) << "Bottom-right G should be ~0.667";

    // All values should be < 255 (tonemapping clamps HDR)
    for (size_t i = 0; i < pixels.size(); i += 100) {
        unpackRGBA(pixels[i], r, g, b, a);
        EXPECT_LE(r, 255);
        EXPECT_LE(g, 255);
    }
}

// ============================================================================
// Blending Tests
// ============================================================================

// Alpha blending: semi-transparent red over green clear color
TEST_F(VulkanGraphicsComplexTest, AlphaBlendOverClear) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        @vertex fn main(in: VIn) -> @builtin(position) vec4f {
            return vec4f(in.pos, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 0.5);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};
    cfg.blend_enable = true;
    cfg.src_blend = VK_BLEND_FACTOR_SRC_ALPHA;
    cfg.dst_blend = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    auto pipeline = ctx_->createPipeline(cfg);
    // Clear to green, draw semi-transparent red on top
    vk_graphics::ClearColor green = {0.0f, 1.0f, 0.0f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3}, {}, green);

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Result: src * srcAlpha + dst * (1-srcAlpha)
    // = (1,0,0)*0.5 + (0,1,0)*0.5 = (0.5, 0.5, 0.0)
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_NEAR(r, 128, 5) << "Blended red should be ~0.5";
    EXPECT_NEAR(g, 128, 5) << "Blended green should be ~0.5";
    EXPECT_LE(b, 5) << "Blue should remain 0";
}

// Additive blending for glow accumulation
TEST_F(VulkanGraphicsComplexTest, AdditiveBlendGlow) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        @vertex fn main(in: VIn) -> @builtin(position) vec4f {
            return vec4f(in.pos, 0.0, 1.0);
        }
    )";

    const char* fs_source = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(0.3, 0.0, 0.0, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 64, H = 64;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};
    cfg.blend_enable = true;
    cfg.src_blend = VK_BLEND_FACTOR_ONE;
    cfg.dst_blend = VK_BLEND_FACTOR_ONE;

    auto pipeline = ctx_->createPipeline(cfg);

    // First draw: clear to dark blue, add red glow
    vk_graphics::ClearColor dark_blue = {0.0f, 0.0f, 0.3f, 1.0f};
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3}, {}, dark_blue);

    auto pixels = target.downloadAs<uint32_t>();
    uint8_t r, g, b, a;

    // Result: src*1 + dst*1 = (0.3,0,0) + (0,0,0.3) = (0.3, 0, 0.3)
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_NEAR(r, 77, 5) << "Additive red should be ~0.3";
    EXPECT_LE(g, 5) << "Green should be 0";
    EXPECT_NEAR(b, 77, 5) << "Additive blue (from clear) should be ~0.3";
}

// ============================================================================
// Multi-Attachment (MRT) Tests
// ============================================================================

// Fragment shader writes to two separate color attachments simultaneously
TEST_F(VulkanGraphicsComplexTest, DualColorAttachmentMRT) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Use GLSL for the MRT fragment shader (outputs to two locations)
    const char* fs_source = R"(
        #version 450
        layout(location = 0) in vec2 uv;
        layout(location = 0) out vec4 outColor0;
        layout(location = 1) out vec4 outColor1;

        void main() {
            outColor0 = vec4(1.0, 0.0, 0.0, 1.0);
            outColor1 = vec4(0.0, uv.x, uv.y, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileGlsl(fs_source, WGSL_STAGE_FRAGMENT);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 64, H = 64;

    // Create two color targets
    auto target0 = ctx_->createColorTarget(W, H);
    auto target1 = ctx_->createColorTarget(W, H);

    // Build pipeline manually for 2 color attachments
    MRTPipelineResources res;
    res.device = ctx_->device();

    // Shader modules
    VkShaderModuleCreateInfo vs_info = {};
    vs_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vs_info.codeSize = vs.spirv.size() * sizeof(uint32_t);
    vs_info.pCode = vs.spirv.data();
    ASSERT_EQ(vkCreateShaderModule(ctx_->device(), &vs_info, nullptr, &res.vert), VK_SUCCESS);

    VkShaderModuleCreateInfo fs_info = {};
    fs_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fs_info.codeSize = fs.spirv.size() * sizeof(uint32_t);
    fs_info.pCode = fs.spirv.data();
    ASSERT_EQ(vkCreateShaderModule(ctx_->device(), &fs_info, nullptr, &res.frag), VK_SUCCESS);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = res.vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = res.frag;
    stages[1].pName = "main";

    // Vertex input
    VkVertexInputBindingDescription bind_desc = {};
    bind_desc.stride = sizeof(Pos2D);
    bind_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr_desc = {};
    attr_desc.format = VK_FORMAT_R32G32_SFLOAT;

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind_desc;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr_desc;

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast = {};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    // Two color blend attachments (one per MRT target)
    VkPipelineColorBlendAttachmentState blend_atts[2] = {};
    blend_atts[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_atts[1] = blend_atts[0];

    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 2;
    cb.pAttachments = blend_atts;

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyn_states;

    // Empty descriptor set layout
    VkDescriptorSetLayoutCreateInfo dl_info = {};
    dl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ASSERT_EQ(vkCreateDescriptorSetLayout(ctx_->device(), &dl_info, nullptr, &res.desc_layout),
              VK_SUCCESS);

    VkPipelineLayoutCreateInfo pl_info = {};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &res.desc_layout;
    ASSERT_EQ(vkCreatePipelineLayout(ctx_->device(), &pl_info, nullptr, &res.layout), VK_SUCCESS);

    // Dynamic rendering with 2 color attachments
    VkFormat color_formats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    VkPipelineRenderingCreateInfo rendering = {};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 2;
    rendering.pColorAttachmentFormats = color_formats;

    VkGraphicsPipelineCreateInfo pi = {};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &rendering;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vps;
    pi.pRasterizationState = &rast;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = res.layout;

    ASSERT_EQ(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &pi, nullptr,
                                         &res.pipeline), VK_SUCCESS);

    // Record draw with 2 color attachments
    ctx_->executeCommands([&](VkCommandBuffer cmd) {
        ctx_->transitionImageLayout(cmd, target0.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        ctx_->transitionImageLayout(cmd, target1.handle(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color_atts[2] = {};
        color_atts[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_atts[0].imageView = target0.view();
        color_atts[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_atts[0].clearValue.color = {{0, 0, 0, 1}};

        color_atts[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_atts[1].imageView = target1.view();
        color_atts[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_atts[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_atts[1].clearValue.color = {{0, 0, 0, 1}};

        VkRenderingInfo ri = {};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = {{0, 0}, {W, H}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 2;
        ri.pColorAttachments = color_atts;

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline);

        VkViewport vp = {0, 0, (float)W, (float)H, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc = {{0, 0}, {W, H}};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        VkBuffer vbuf = vb.handle();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);

        ctx_->transitionImageLayout(cmd, target0.handle(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        ctx_->transitionImageLayout(cmd, target1.handle(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });

    // Verify target 0: solid red everywhere
    auto pixels0 = target0.downloadAs<uint32_t>();
    uint8_t r, g, b, a;
    uint32_t c0 = pixels0[(H / 2) * W + (W / 2)];
    unpackRGBA(c0, r, g, b, a);
    EXPECT_GE(r, 250) << "MRT target 0 should be red";
    EXPECT_LE(g, 5);
    EXPECT_LE(b, 5);

    // Verify target 1: UV-based gradient (green = uv.x, blue = uv.y)
    auto pixels1 = target1.downloadAs<uint32_t>();

    // Center: uv = (0.5, 0.5)
    uint32_t c1 = pixels1[(H / 2) * W + (W / 2)];
    unpackRGBA(c1, r, g, b, a);
    EXPECT_LE(r, 5) << "MRT target 1 R should be 0";
    EXPECT_NEAR(g, 128, 10) << "MRT target 1 G should be ~0.5 (uv.x)";
    EXPECT_NEAR(b, 128, 10) << "MRT target 1 B should be ~0.5 (uv.y)";

    // Vulkan Y: top-left = uv(0,0), bottom-right = uv(1,1)
    // Top-right: uv ~ (1, 0) -> G ~1.0, B ~0
    uint32_t tr = pixels1[2 * W + (W - 2)];
    unpackRGBA(tr, r, g, b, a);
    EXPECT_GE(g, 230) << "MRT target 1 top-right G should be ~1.0";
    EXPECT_LE(b, 20) << "MRT target 1 top-right B should be ~0";

    // Bottom-left: uv ~ (0, 1) -> G ~0, B ~1.0
    uint32_t bl = pixels1[(H - 2) * W + 2];
    unpackRGBA(bl, r, g, b, a);
    EXPECT_LE(g, 20) << "MRT target 1 bottom-left G should be ~0";
    EXPECT_GE(b, 230) << "MRT target 1 bottom-left B should be ~1.0";
}

// ============================================================================
// Image Export Tests
// ============================================================================

// Julia set fractal with smooth iteration coloring, exported as PNG
TEST_F(VulkanGraphicsComplexTest, JuliaSetFractal) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Julia set z = z^2 + c, c = (-0.7, 0.27015)
    // Smooth coloring via iteration count + log escape
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            var zr = (uv.x - 0.5) * 3.0;
            var zi = (uv.y - 0.5) * 3.0;
            let cr = -0.7;
            let ci = 0.27015;
            var iters = 0;
            var i = 0;
            for (i = 0; i < 200; i = i + 1) {
                let zr2 = zr * zr - zi * zi + cr;
                let zi2 = 2.0 * zr * zi + ci;
                zr = zr2;
                zi = zi2;
                let mag2 = zr * zr + zi * zi;
                if (mag2 <= 4.0) {
                    iters = iters + 1;
                }
            }
            let t = f32(iters) / 200.0;
            let r = 0.5 + 0.5 * sin(t * 6.28 * 3.0 + 0.0);
            let g = 0.5 + 0.5 * sin(t * 6.28 * 3.0 + 2.09);
            let b = 0.5 + 0.5 * sin(t * 6.28 * 3.0 + 4.19);
            if (iters >= 200) {
                return vec4f(0.0, 0.0, 0.0, 1.0);
            }
            return vec4f(r, g, b, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("julia_set.png", W, H, 4, pixels.data(), W * 4);

    // Center of the Julia set should be inside (black) or near boundary (colored)
    // Verify we have both interior and exterior pixels
    int dark = 0, colored = 0;
    for (auto px : pixels) {
        uint8_t r, g, b, a;
        unpackRGBA(px, r, g, b, a);
        if (r < 10 && g < 10 && b < 10) dark++;
        else colored++;
    }
    EXPECT_GT(dark, 1000) << "Julia set should have interior (dark) region";
    EXPECT_GT(colored, 1000) << "Julia set should have colored exterior";
}

// Voronoi cell pattern from hardcoded seed grid
TEST_F(VulkanGraphicsComplexTest, VoronoiCellPattern) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Voronoi: distance-to-nearest-seed computed per pixel
    // Uses unique let bindings per cell to avoid variable reassignment issues
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let su = uv.x * 5.0;
            let sv = uv.y * 5.0;
            let bx = floor(su);
            let by = floor(sv);

            var md = 10.0;
            var nc = 0.0;

            // Cell 0: (bx-1, by-1)
            let c0x = bx - 1.0;
            let c0y = by - 1.0;
            let s0x = c0x + fract(sin(c0x * 127.1 + c0y * 311.7) * 43758.5);
            let s0y = c0y + fract(sin(c0x * 269.5 + c0y * 183.3) * 43758.5);
            let d0x = su - s0x;
            let d0y = sv - s0y;
            let d0 = sqrt(d0x * d0x + d0y * d0y);
            if (d0 < md) { md = d0; nc = c0x * 7.0 + c0y * 13.0; }

            // Cell 1: (bx, by-1)
            let c1x = bx;
            let c1y = by - 1.0;
            let s1x = c1x + fract(sin(c1x * 127.1 + c1y * 311.7) * 43758.5);
            let s1y = c1y + fract(sin(c1x * 269.5 + c1y * 183.3) * 43758.5);
            let d1x = su - s1x;
            let d1y = sv - s1y;
            let d1 = sqrt(d1x * d1x + d1y * d1y);
            if (d1 < md) { md = d1; nc = c1x * 7.0 + c1y * 13.0; }

            // Cell 2: (bx+1, by-1)
            let c2x = bx + 1.0;
            let c2y = by - 1.0;
            let s2x = c2x + fract(sin(c2x * 127.1 + c2y * 311.7) * 43758.5);
            let s2y = c2y + fract(sin(c2x * 269.5 + c2y * 183.3) * 43758.5);
            let d2x = su - s2x;
            let d2y = sv - s2y;
            let d2 = sqrt(d2x * d2x + d2y * d2y);
            if (d2 < md) { md = d2; nc = c2x * 7.0 + c2y * 13.0; }

            // Cell 3: (bx-1, by)
            let c3x = bx - 1.0;
            let c3y = by;
            let s3x = c3x + fract(sin(c3x * 127.1 + c3y * 311.7) * 43758.5);
            let s3y = c3y + fract(sin(c3x * 269.5 + c3y * 183.3) * 43758.5);
            let d3x = su - s3x;
            let d3y = sv - s3y;
            let d3 = sqrt(d3x * d3x + d3y * d3y);
            if (d3 < md) { md = d3; nc = c3x * 7.0 + c3y * 13.0; }

            // Cell 4: (bx, by)
            let c4x = bx;
            let c4y = by;
            let s4x = c4x + fract(sin(c4x * 127.1 + c4y * 311.7) * 43758.5);
            let s4y = c4y + fract(sin(c4x * 269.5 + c4y * 183.3) * 43758.5);
            let d4x = su - s4x;
            let d4y = sv - s4y;
            let d4 = sqrt(d4x * d4x + d4y * d4y);
            if (d4 < md) { md = d4; nc = c4x * 7.0 + c4y * 13.0; }

            // Cell 5: (bx+1, by)
            let c5x = bx + 1.0;
            let c5y = by;
            let s5x = c5x + fract(sin(c5x * 127.1 + c5y * 311.7) * 43758.5);
            let s5y = c5y + fract(sin(c5x * 269.5 + c5y * 183.3) * 43758.5);
            let d5x = su - s5x;
            let d5y = sv - s5y;
            let d5 = sqrt(d5x * d5x + d5y * d5y);
            if (d5 < md) { md = d5; nc = c5x * 7.0 + c5y * 13.0; }

            // Cell 6: (bx-1, by+1)
            let c6x = bx - 1.0;
            let c6y = by + 1.0;
            let s6x = c6x + fract(sin(c6x * 127.1 + c6y * 311.7) * 43758.5);
            let s6y = c6y + fract(sin(c6x * 269.5 + c6y * 183.3) * 43758.5);
            let d6x = su - s6x;
            let d6y = sv - s6y;
            let d6 = sqrt(d6x * d6x + d6y * d6y);
            if (d6 < md) { md = d6; nc = c6x * 7.0 + c6y * 13.0; }

            // Cell 7: (bx, by+1)
            let c7x = bx;
            let c7y = by + 1.0;
            let s7x = c7x + fract(sin(c7x * 127.1 + c7y * 311.7) * 43758.5);
            let s7y = c7y + fract(sin(c7x * 269.5 + c7y * 183.3) * 43758.5);
            let d7x = su - s7x;
            let d7y = sv - s7y;
            let d7 = sqrt(d7x * d7x + d7y * d7y);
            if (d7 < md) { md = d7; nc = c7x * 7.0 + c7y * 13.0; }

            // Cell 8: (bx+1, by+1)
            let c8x = bx + 1.0;
            let c8y = by + 1.0;
            let s8x = c8x + fract(sin(c8x * 127.1 + c8y * 311.7) * 43758.5);
            let s8y = c8y + fract(sin(c8x * 269.5 + c8y * 183.3) * 43758.5);
            let d8x = su - s8x;
            let d8y = sv - s8y;
            let d8 = sqrt(d8x * d8x + d8y * d8y);
            if (d8 < md) { md = d8; nc = c8x * 7.0 + c8y * 13.0; }

            // Color from cell identity hash
            let cr = fract(sin(nc * 43.7) * 2747.1);
            let cg = fract(sin(nc * 31.1) * 3191.7);
            let cb = fract(sin(nc * 67.3) * 1571.3);
            // Darken near seed points
            let edge = clamp(md * 4.0, 0.0, 1.0);
            return vec4f(cr * edge, cg * edge, cb * edge, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("voronoi_cells.png", W, H, 4, pixels.data(), W * 4);

    // Verify cells exist: different cells should have different colors
    // Count distinct color clusters (quantize to 64-level bins)
    std::set<uint32_t> color_keys;
    for (auto px : pixels) {
        uint8_t r, g, b, a;
        unpackRGBA(px, r, g, b, a);
        uint32_t key = ((r / 64) << 8) | ((g / 64) << 4) | (b / 64);
        color_keys.insert(key);
    }
    EXPECT_GE(color_keys.size(), 5u) << "Should have multiple distinct cell colors";
}

// Sphere lit by directional light, computed from SDF normal
TEST_F(VulkanGraphicsComplexTest, NormalMappedSphere) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Sphere SDF: project UV onto sphere surface, compute normal, shade with Lambertian
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let cx = uv.x * 2.0 - 1.0;
            let cy = uv.y * 2.0 - 1.0;
            let r2 = cx * cx + cy * cy;
            if (r2 > 1.0) {
                return vec4f(0.05, 0.05, 0.1, 1.0);
            }
            // Normal on sphere surface
            let nz = sqrt(1.0 - r2);
            // Light direction: upper-right-forward (normalized ~0.577 each)
            let lx = 0.577;
            let ly = -0.577;
            let lz = 0.577;
            // Manual dot(normal, light)
            let ndotl = cx * lx + cy * ly + nz * lz;
            let diff = clamp(ndotl, 0.0, 1.0);
            // Specular: half-vector approximation
            let hx = lx;
            let hy = ly;
            let hz = lz + 1.0;
            let h_len = sqrt(hx * hx + hy * hy + hz * hz);
            let nhx = hx / h_len;
            let nhy = hy / h_len;
            let nhz = hz / h_len;
            let ndoth = cx * nhx + cy * nhy + nz * nhz;
            let spec_base = clamp(ndoth, 0.0, 1.0);
            let spec = spec_base * spec_base * spec_base * spec_base *
                       spec_base * spec_base * spec_base * spec_base;
            // Material: blueish sphere with white specular
            let color_r = 0.2 * diff + spec * 0.8;
            let color_g = 0.3 * diff + spec * 0.8;
            let color_b = 0.8 * diff + spec * 0.8;
            return vec4f(
                clamp(color_r, 0.0, 1.0),
                clamp(color_g, 0.0, 1.0),
                clamp(color_b, 0.0, 1.0),
                1.0
            );
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("normal_sphere.png", W, H, 4, pixels.data(), W * 4);

    // Center of sphere should be lit (blue-ish)
    uint8_t r, g, b, a;
    uint32_t center = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(b, 100) << "Center of sphere should have blue component";

    // Corner should be background color (dark)
    uint32_t corner = pixels[0];
    unpackRGBA(corner, r, g, b, a);
    EXPECT_LE(r, 20) << "Background should be dark";
}

// Classic demoscene plasma effect
TEST_F(VulkanGraphicsComplexTest, PlasmaEffect) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Overlapping sine waves creating a plasma effect
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let time = 2.5;
            let x = uv.x * 10.0;
            let y = uv.y * 10.0;
            let v1 = sin(x + time);
            let v2 = sin(y + time * 0.5);
            let v3 = sin(x + y + time * 0.7);
            let cx = x + 5.0 * sin(time * 0.3);
            let cy = y + 5.0 * sin(time * 0.5);
            let v4 = sin(sqrt(cx * cx + cy * cy) + time);
            let v = (v1 + v2 + v3 + v4) * 0.25;
            let r = sin(v * 3.14159) * 0.5 + 0.5;
            let g = sin(v * 3.14159 + 2.094) * 0.5 + 0.5;
            let b = sin(v * 3.14159 + 4.189) * 0.5 + 0.5;
            return vec4f(r, g, b, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("plasma.png", W, H, 4, pixels.data(), W * 4);

    // Plasma should have variety of colors - check we have a good distribution
    int r_high = 0, g_high = 0, b_high = 0;
    for (auto px : pixels) {
        uint8_t r, g, b, a;
        unpackRGBA(px, r, g, b, a);
        if (r > 180) r_high++;
        if (g > 180) g_high++;
        if (b > 180) b_high++;
    }
    EXPECT_GT(r_high, 500) << "Should have significant red regions";
    EXPECT_GT(g_high, 500) << "Should have significant green regions";
    EXPECT_GT(b_high, 500) << "Should have significant blue regions";
}

// Depth visualization: render overlapping colored geometry, export color + depth
TEST_F(VulkanGraphicsComplexTest, DepthVisualization) {
    const char* vs_source = R"(
        struct VIn {
            @location(0) pos: vec3f,
            @location(1) color: vec3f,
        };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) color: vec3f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 1.0);
            out.color = in.color;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    // Three overlapping quads at different depths with different colors
    std::vector<PosColor3D> verts = {
        // Red quad (z=0.2, front) - centered
        {-0.6f, -0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        { 0.6f, -0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        { 0.6f,  0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        {-0.6f, -0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        { 0.6f,  0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        {-0.6f,  0.6f, 0.2f,  1.0f, 0.0f, 0.0f},
        // Green quad (z=0.5, middle) - offset right+down
        {-0.3f, -0.3f, 0.5f,  0.0f, 1.0f, 0.0f},
        { 0.9f, -0.3f, 0.5f,  0.0f, 1.0f, 0.0f},
        { 0.9f,  0.9f, 0.5f,  0.0f, 1.0f, 0.0f},
        {-0.3f, -0.3f, 0.5f,  0.0f, 1.0f, 0.0f},
        { 0.9f,  0.9f, 0.5f,  0.0f, 1.0f, 0.0f},
        {-0.3f,  0.9f, 0.5f,  0.0f, 1.0f, 0.0f},
        // Blue quad (z=0.8, back) - offset left+up
        {-0.9f, -0.9f, 0.8f,  0.0f, 0.0f, 1.0f},
        { 0.3f, -0.9f, 0.8f,  0.0f, 0.0f, 1.0f},
        { 0.3f,  0.3f, 0.8f,  0.0f, 0.0f, 1.0f},
        {-0.9f, -0.9f, 0.8f,  0.0f, 0.0f, 1.0f},
        { 0.3f,  0.3f, 0.8f,  0.0f, 0.0f, 1.0f},
        {-0.9f,  0.3f, 0.8f,  0.0f, 0.0f, 1.0f},
    };

    auto vb = ctx_->createVertexBuffer(verts);
    const uint32_t W = 128, H = 128;
    auto color_target = ctx_->createColorTarget(W, H);
    auto depth_target = ctx_->createDepthTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(PosColor3D);
    cfg.vertex_attributes = {
        {0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, x)},
        {1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PosColor3D, r)},
    };
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;

    auto pipeline = ctx_->createPipeline(cfg);
    vk_graphics::ClearColor bg = {0.1f, 0.1f, 0.1f, 1.0f};
    ctx_->draw(pipeline, color_target, &vb, {.vertex_count = 18}, {}, bg, &depth_target);

    // Export color image
    auto color_pixels = color_target.downloadAs<uint32_t>();
    ASSERT_EQ(color_pixels.size(), W * H);
    stbi_write_png("depth_color.png", W, H, 4, color_pixels.data(), W * 4);

    // Export depth visualization as grayscale
    auto depth_bytes = depth_target.download();
    ASSERT_EQ(depth_bytes.size(), W * H * sizeof(float));
    const float* depth_data = reinterpret_cast<const float*>(depth_bytes.data());

    std::vector<uint8_t> depth_vis(W * H * 4);
    for (uint32_t i = 0; i < W * H; i++) {
        // Depth range [0, 1] -> grayscale (near=bright, far=dark)
        float d = depth_data[i];
        float val = (1.0f - d) * 255.0f;
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        uint8_t v = static_cast<uint8_t>(val);
        depth_vis[i * 4 + 0] = v;
        depth_vis[i * 4 + 1] = v;
        depth_vis[i * 4 + 2] = v;
        depth_vis[i * 4 + 3] = 255;
    }
    stbi_write_png("depth_vis.png", W, H, 4, depth_vis.data(), W * 4);

    // Verify: center should be red (front quad at z=0.2)
    uint8_t r, g, b, a;
    uint32_t center = color_pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center, r, g, b, a);
    EXPECT_GE(r, 200) << "Center should be red (front quad)";
    EXPECT_LE(g, 15) << "Center should not be green";

    // Verify depth at center should be near 0.2
    float center_depth = depth_data[(H / 2) * W + (W / 2)];
    EXPECT_NEAR(center_depth, 0.2f, 0.05f) << "Center depth should be ~0.2";
}

// ============================================================================
// Bloom & Blur Visual Tests (PNG Export)
// ============================================================================

// 1) 1D horizontal Gaussian blur on a hard step edge
TEST_F(VulkanGraphicsComplexTest, GaussianBlurStep) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            var blurred = 0.0;
            var wsum = 0.0;
            var scene_val = 0.0;
            var i = 0;
            for (i = 0; i < 9; i = i + 1) {
                let fi = f32(i) - 4.0;
                let offset = fi * 0.025;
                let sx = uv.x + offset;
                let w = exp(-fi * fi / 4.5);
                scene_val = 0.0;
                if (sx > 0.5) {
                    scene_val = 1.0;
                }
                blurred = blurred + scene_val * w;
                wsum = wsum + w;
            }
            blurred = blurred / wsum;
            return vec4f(blurred, 0.1, 1.0 - blurred, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("bloom_gaussian_step.png", W, H, 4, pixels.data(), W * 4);

    uint8_t r, g, b, a;

    // Far left: all samples below step -> blurred ~ 0
    unpackRGBA(pixels[(H / 2) * W + 32], r, g, b, a);
    EXPECT_LE(r, 10) << "Far left should be dark (below step)";
    EXPECT_GE(b, 240) << "Far left blue should be high (1 - blurred)";

    // Far right: all samples above step -> blurred ~ 1
    unpackRGBA(pixels[(H / 2) * W + 224], r, g, b, a);
    EXPECT_GE(r, 245) << "Far right should be bright (above step)";
    EXPECT_LE(b, 10) << "Far right blue should be low";

    // At step edge: Gaussian-weighted mix -> intermediate
    unpackRGBA(pixels[(H / 2) * W + 128], r, g, b, a);
    EXPECT_GT(r, 80) << "Step edge should have intermediate red";
    EXPECT_LT(r, 180) << "Step edge should not be fully bright";
}

// 2) 2D box blur applied to a hard circle, producing soft edges
TEST_F(VulkanGraphicsComplexTest, BoxBlurSoftCircle) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            var total = 0.0;
            var samples = 0;
            var idx = 0;
            for (idx = 0; idx < 25; idx = idx + 1) {
                let row = idx / 5;
                let col = idx - row * 5;
                let fdx = f32(col) - 2.0;
                let fdy = f32(row) - 2.0;
                let sx = uv.x + fdx * 0.02;
                let sy = uv.y + fdy * 0.02;
                let cx = sx * 2.0 - 1.0;
                let cy = sy * 2.0 - 1.0;
                let dist = sqrt(cx * cx + cy * cy);
                if (dist < 0.6) {
                    total = total + 1.0;
                }
                samples = samples + 1;
            }
            let v = total / f32(samples);
            return vec4f(v * 0.15, v * 0.5, v * 1.0, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("bloom_box_circle.png", W, H, 4, pixels.data(), W * 4);

    uint8_t r, g, b, a;

    // Center: all samples inside circle -> v ~ 1.0, blue ~ 255
    unpackRGBA(pixels[(H / 2) * W + (W / 2)], r, g, b, a);
    EXPECT_GE(b, 240) << "Center of circle should be bright blue";

    // Corner: all samples outside -> dark
    unpackRGBA(pixels[5 * W + 5], r, g, b, a);
    EXPECT_LE(b, 10) << "Corner should be dark (outside circle)";

    // Near circle edge (UV 0.8 -> cx=0.6, right at boundary): partial coverage
    unpackRGBA(pixels[(H / 2) * W + 205], r, g, b, a);
    EXPECT_GT(b, 20) << "Circle edge should have partial blur";
    EXPECT_LT(b, 240) << "Circle edge should not be fully bright";
}

// 3) Horizontal directional blur creating light streaks from two point lights
TEST_F(VulkanGraphicsComplexTest, DirectionalBlurStreaks) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) ndc: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.ndc = in.pos;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            var cr = 0.0;
            var cg = 0.0;
            var cb = 0.0;
            var wsum = 0.0;
            var i = 0;
            for (i = 0; i < 17; i = i + 1) {
                let fi = f32(i) - 8.0;
                let ox = fi * 0.03;
                let sx = ndc.x + ox;
                let sy = ndc.y;
                let d1x = sx - 0.3;
                let d1y = sy - 0.2;
                let l1 = exp(-(d1x * d1x + d1y * d1y) * 30.0);
                let d2x = sx + 0.4;
                let d2y = sy + 0.3;
                let l2 = exp(-(d2x * d2x + d2y * d2y) * 30.0);
                let w = exp(-fi * fi / 12.0);
                cr = cr + (l1 * 1.0 + l2 * 0.3) * w;
                cg = cg + (l1 * 0.7 + l2 * 0.6) * w;
                cb = cb + (l1 * 0.2 + l2 * 1.0) * w;
                wsum = wsum + w;
            }
            cr = cr / wsum;
            cg = cg / wsum;
            cb = cb / wsum;
            return vec4f(
                clamp(cr, 0.0, 1.0),
                clamp(cg, 0.0, 1.0),
                clamp(cb, 0.0, 1.0),
                1.0
            );
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("bloom_directional_streaks.png", W, H, 4, pixels.data(), W * 4);

    uint8_t r, g, b, a;

    // Near warm spot at NDC(0.3, 0.2) -> pixel (166, 154)
    unpackRGBA(pixels[154 * W + 166], r, g, b, a);
    EXPECT_GE(r, 150) << "Spot 1 should have strong red (warm)";
    EXPECT_GT(r, b) << "Spot 1 should be warmer than cool";

    // Near cool spot at NDC(-0.4, -0.3) -> pixel (77, 90)
    unpackRGBA(pixels[90 * W + 77], r, g, b, a);
    EXPECT_GE(b, 100) << "Spot 2 should have strong blue (cool)";

    // Far corner should be dark
    unpackRGBA(pixels[5 * W + 5], r, g, b, a);
    EXPECT_LE(r, 15) << "Corner should be dark";
    EXPECT_LE(b, 15) << "Corner should be dark";

    // Horizontal streak extends from spot 1: NDC(0.5, 0.2) -> pixel (192, 154)
    unpackRGBA(pixels[154 * W + 192], r, g, b, a);
    EXPECT_GT(r, 30) << "Horizontal streak should extend from spot 1";
}

// 4) Full bloom pipeline: scene + bright-pass + 2D Gaussian blur + composite + Reinhard tonemap
TEST_F(VulkanGraphicsComplexTest, BloomCompositeHDR) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) ndc: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.ndc = in.pos;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            let px = ndc.x;
            let py = ndc.y;
            let bg_r = 0.05 + (px * 0.5 + 0.5) * 0.1;
            let bg_g = 0.03 + (py * 0.5 + 0.5) * 0.08;
            let bg_b = 0.08;
            let e1x = px - 0.3;
            let e1y = py + 0.2;
            let emit1 = exp(-(e1x * e1x + e1y * e1y) * 25.0) * 3.0;
            let e2x = px + 0.4;
            let e2y = py - 0.3;
            let emit2 = exp(-(e2x * e2x + e2y * e2y) * 30.0) * 2.5;
            let scene_r = bg_r + emit1 * 1.0 + emit2 * 0.3;
            let scene_g = bg_g + emit1 * 0.8 + emit2 * 0.7;
            let scene_b = bg_b + emit1 * 0.2 + emit2 * 1.0;
            var bloom_r = 0.0;
            var bloom_g = 0.0;
            var bloom_b = 0.0;
            var bloom_w = 0.0;
            var idx = 0;
            for (idx = 0; idx < 49; idx = idx + 1) {
                let row = idx / 7;
                let col = idx - row * 7;
                let fdx = f32(col) - 3.0;
                let fdy = f32(row) - 3.0;
                let ox = fdx * 0.04;
                let oy = fdy * 0.04;
                let spx = px + ox;
                let spy = py + oy;
                let se1x = spx - 0.3;
                let se1y = spy + 0.2;
                let se1 = exp(-(se1x * se1x + se1y * se1y) * 25.0) * 3.0;
                let se2x = spx + 0.4;
                let se2y = spy - 0.3;
                let se2 = exp(-(se2x * se2x + se2y * se2y) * 30.0) * 2.5;
                let sr = se1 * 1.0 + se2 * 0.3;
                let sg = se1 * 0.8 + se2 * 0.7;
                let sb = se1 * 0.2 + se2 * 1.0;
                let luma = sr * 0.299 + sg * 0.587 + sb * 0.114;
                let gw = exp(-(fdx * fdx + fdy * fdy) / 4.5);
                if (luma > 0.4) {
                    bloom_r = bloom_r + sr * gw;
                    bloom_g = bloom_g + sg * gw;
                    bloom_b = bloom_b + sb * gw;
                }
                bloom_w = bloom_w + gw;
            }
            bloom_r = bloom_r / bloom_w;
            bloom_g = bloom_g / bloom_w;
            bloom_b = bloom_b / bloom_w;
            let final_r = scene_r + bloom_r * 0.5;
            let final_g = scene_g + bloom_g * 0.5;
            let final_b = scene_b + bloom_b * 0.5;
            let tr = final_r / (final_r + 1.0);
            let tg = final_g / (final_g + 1.0);
            let tb = final_b / (final_b + 1.0);
            return vec4f(tr, tg, tb, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("bloom_composite_hdr.png", W, H, 4, pixels.data(), W * 4);

    uint8_t r, g, b, a;

    // Near emissive spot 1 at NDC(0.3, -0.2) -> pixel (166, 102)
    unpackRGBA(pixels[102 * W + 166], r, g, b, a);
    EXPECT_GE(r, 100) << "Near emit 1 should be bright (tonemapped)";

    // Near emissive spot 2 at NDC(-0.4, 0.3) -> pixel (77, 166)
    unpackRGBA(pixels[166 * W + 77], r, g, b, a);
    EXPECT_GE(b, 80) << "Near emit 2 should have strong blue";

    // Background corner: dim but not black
    unpackRGBA(pixels[10 * W + 10], r, g, b, a);
    EXPECT_GE(r + g + b, 10) << "Background should not be completely black";
    EXPECT_LE(r, 100) << "Background should be dim";

    // Tonemapping prevents pure white
    int clipped = 0;
    for (size_t i = 0; i < pixels.size(); i += 64) {
        unpackRGBA(pixels[i], r, g, b, a);
        if (r == 255 && g == 255 && b == 255) clipped++;
    }
    EXPECT_EQ(clipped, 0) << "Tonemapping should prevent clipping to pure white";
}

// 5) Multi-scale bloom: 4 lights with different blur radii, anamorphic stretch, fog, ACES tonemap
TEST_F(VulkanGraphicsComplexTest, MultiScaleBloomFog) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) ndc: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.ndc = in.pos;
            return out;
        }
    )";

    const char* fs_source = R"(
        @fragment fn main(@location(0) ndc: vec2f) -> @location(0) vec4f {
            let px = ndc.x;
            let py = ndc.y;
            var hdr_r = 0.0;
            var hdr_g = 0.0;
            var hdr_b = 0.0;

            // Light 0: near, sharp, warm white (no blur)
            let l0x = px + 0.5;
            let l0y = py + 0.3;
            let l0 = exp(-(l0x * l0x + l0y * l0y) * 25.0) * 2.0;
            hdr_r = hdr_r + l0 * 1.0;
            hdr_g = hdr_g + l0 * 0.9;
            hdr_b = hdr_b + l0 * 0.7;

            // Light 1: mid, 9-tap diagonal blur, cyan
            var b1r = 0.0;
            var b1g = 0.0;
            var b1b = 0.0;
            var b1w = 0.0;
            var j = 0;
            for (j = 0; j < 9; j = j + 1) {
                let fj = f32(j) - 4.0;
                let ox = fj * 0.04;
                let oy = fj * 0.04;
                let sx = px - 0.3 + ox;
                let sy = py - 0.4 + oy;
                let s = exp(-(sx * sx + sy * sy) * 15.0) * 1.5;
                let w = exp(-fj * fj / 4.5);
                b1r = b1r + s * 0.1 * w;
                b1g = b1g + s * 0.8 * w;
                b1b = b1b + s * 1.0 * w;
                b1w = b1w + w;
            }
            hdr_r = hdr_r + b1r / b1w;
            hdr_g = hdr_g + b1g / b1w;
            hdr_b = hdr_b + b1b / b1w;

            // Light 2: far, 13-tap diagonal blur, magenta
            var b2r = 0.0;
            var b2g = 0.0;
            var b2b = 0.0;
            var b2w = 0.0;
            var k = 0;
            for (k = 0; k < 13; k = k + 1) {
                let fk = f32(k) - 6.0;
                let ox = fk * 0.05;
                let oy = fk * 0.05;
                let sx = px - 0.5 + ox;
                let sy = py + 0.5 + oy;
                let s = exp(-(sx * sx + sy * sy) * 5.0) * 2.0;
                let w = exp(-fk * fk / 8.0);
                b2r = b2r + s * 0.9 * w;
                b2g = b2g + s * 0.15 * w;
                b2b = b2b + s * 0.7 * w;
                b2w = b2w + w;
            }
            hdr_r = hdr_r + b2r / b2w;
            hdr_g = hdr_g + b2g / b2w;
            hdr_b = hdr_b + b2b / b2w;

            // Light 3: 17-tap horizontal-only blur, anamorphic gold streak
            var b3r = 0.0;
            var b3g = 0.0;
            var b3b = 0.0;
            var b3w = 0.0;
            var m = 0;
            for (m = 0; m < 17; m = m + 1) {
                let fm = f32(m) - 8.0;
                let ox = fm * 0.06;
                let sx = px + ox;
                let sy = py - 0.1;
                let s = exp(-(sx * sx * 20.0 + sy * sy * 40.0)) * 1.8;
                let w = exp(-fm * fm / 18.0);
                b3r = b3r + s * 1.0 * w;
                b3g = b3g + s * 0.75 * w;
                b3b = b3b + s * 0.15 * w;
                b3w = b3w + w;
            }
            hdr_r = hdr_r + b3r / b3w;
            hdr_g = hdr_g + b3g / b3w;
            hdr_b = hdr_b + b3b / b3w;

            // Atmospheric fog toward edges
            var apx = px;
            if (px < 0.0) { apx = -px; }
            var apy = py;
            if (py < 0.0) { apy = -py; }
            var edge = apx;
            if (apy > edge) { edge = apy; }
            let fog = clamp(edge * 0.3, 0.0, 0.35);
            hdr_r = hdr_r * (1.0 - fog) + 0.05 * fog;
            hdr_g = hdr_g * (1.0 - fog) + 0.07 * fog;
            hdr_b = hdr_b * (1.0 - fog) + 0.15 * fog;

            // ACES-inspired filmic tonemap
            let ar = (hdr_r * (2.51 * hdr_r + 0.03)) / (hdr_r * (2.43 * hdr_r + 0.59) + 0.14);
            let ag = (hdr_g * (2.51 * hdr_g + 0.03)) / (hdr_g * (2.43 * hdr_g + 0.59) + 0.14);
            let ab = (hdr_b * (2.51 * hdr_b + 0.03)) / (hdr_b * (2.43 * hdr_b + 0.59) + 0.14);

            return vec4f(
                clamp(ar, 0.0, 1.0),
                clamp(ag, 0.0, 1.0),
                clamp(ab, 0.0, 1.0),
                1.0
            );
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 256, H = 256;
    auto target = ctx_->createColorTarget(W, H);

    vk_graphics::GraphicsPipelineConfig cfg;
    cfg.vertex_spirv = vs.spirv.data();
    cfg.vertex_spirv_words = vs.spirv.size();
    cfg.fragment_spirv = fs.spirv.data();
    cfg.fragment_spirv_words = fs.spirv.size();
    cfg.vertex_stride = sizeof(Pos2D);
    cfg.vertex_attributes = {{0, VK_FORMAT_R32G32_SFLOAT, 0}};

    auto pipeline = ctx_->createPipeline(cfg);
    ctx_->draw(pipeline, target, &vb, {.vertex_count = 3});

    auto pixels = target.downloadAs<uint32_t>();
    ASSERT_EQ(pixels.size(), W * H);

    stbi_write_png("bloom_multilayer_fog.png", W, H, 4, pixels.data(), W * 4);

    uint8_t r, g, b, a;

    // Light 0 (warm) at NDC(-0.5, -0.3) -> pixel (64, 90)
    unpackRGBA(pixels[90 * W + 64], r, g, b, a);
    EXPECT_GE(r, 80) << "Near warm light should be bright";
    EXPECT_GT(r, b) << "Warm light should be warmer (red > blue)";

    // Light 1 (cyan) at NDC(0.3, 0.4) -> pixel (166, 179)
    unpackRGBA(pixels[179 * W + 166], r, g, b, a);
    EXPECT_GE(b, 50) << "Near cyan light should have blue";
    EXPECT_GE(g, 40) << "Near cyan light should have green";

    // Anamorphic streak: horizontal extent at NDC(0.5, -0.1) -> pixel (192, 115)
    unpackRGBA(pixels[115 * W + 192], r, g, b, a);
    EXPECT_GE(r, 20) << "Anamorphic streak should extend horizontally";

    // Edge fog: corners not pitch black
    unpackRGBA(pixels[2 * W + 2], r, g, b, a);
    EXPECT_GE(r + g + b, 3) << "Edge should have some fog color";

    // ACES tonemap prevents pure white
    int white_count = 0;
    for (size_t i = 0; i < pixels.size(); i += 32) {
        unpackRGBA(pixels[i], r, g, b, a);
        if (r >= 254 && g >= 254 && b >= 254) white_count++;
    }
    EXPECT_EQ(white_count, 0) << "ACES tonemap should prevent white clipping";
}

#endif // WGSL_HAS_VULKAN
