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

class VulkanGraphicsSceneTest : public ::testing::Test {
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

std::unique_ptr<vk_graphics::GraphicsContext> VulkanGraphicsSceneTest::ctx_;

// ============================================================================
// Helpers
// ============================================================================

struct Pos2D {
    float x, y;
};

static const std::vector<Pos2D> kFullScreenTri = {
    {-1.0f, -1.0f},
    { 3.0f, -1.0f},
    {-1.0f,  3.0f},
};

inline void unpackRGBA(uint32_t pixel, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    r = (pixel >> 0) & 0xFF;
    g = (pixel >> 8) & 0xFF;
    b = (pixel >> 16) & 0xFF;
    a = (pixel >> 24) & 0xFF;
}

// ============================================================================
// 3D Outdoor Scene: terrain, bloomy sun, trees, glossy sea
// ============================================================================

TEST_F(VulkanGraphicsSceneTest, Outdoor3DSceneWithBloom) {
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

    // Procedural 3D scene via ray-plane and ray-geometry intersection.
    // Camera at (0,3,0) looking along +Z.
    //  - Sky gradient with multi-layer bloomy sun
    //  - Green terrain (y=0 plane, z in [0,50])
    //  - Blue sea (y=0 plane, z>50) with glossy sun reflection
    //  - 3 trees (cylinder trunk + sphere canopy), inlined intersections
    const char* fs_source = R"(
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            // Camera ray
            let ndx = (uv.x - 0.5) * 2.0;
            let ndy = (0.5 - uv.y) * 2.0;
            let raw_x = ndx;
            let raw_y = ndy - 0.15;
            let raw_z = 1.2;
            let rlen = sqrt(raw_x * raw_x + raw_y * raw_y + raw_z * raw_z);
            let rx = raw_x / rlen;
            let ry = raw_y / rlen;
            let rz = raw_z / rlen;
            let cam_y = 3.0;

            // Sun direction (normalized)
            let slen = sqrt(0.3 * 0.3 + 0.45 * 0.45 + 0.8 * 0.8);
            let sx = 0.3 / slen;
            let sy = 0.45 / slen;
            let sz = 0.8 / slen;

            // --- Sky color ---
            // Note: var initializers with expressions are broken in this
            // WGSL compiler; use separate declaration + assignment.
            let sky_t = clamp(ry * 2.0 + 0.5, 0.0, 1.0);
            var cr = 0.0;
            var cg = 0.0;
            var cb = 0.0;
            cr = sky_t * 0.12 + (1.0 - sky_t) * 0.55;
            cg = sky_t * 0.25 + (1.0 - sky_t) * 0.65;
            cb = sky_t * 0.55 + (1.0 - sky_t) * 0.92;

            // --- Sun disk + bloom ---
            let sun_dot = rx * sx + ry * sy + rz * sz;
            let sun_ang = max(1.0 - sun_dot, 0.001);
            // Sun core disk (polynomial - avoids compiler if-conditional bug)
            let core_t = clamp(1.0 - sun_ang * 200.0, 0.0, 1.0);
            let sun_disk = core_t * core_t * core_t * core_t;
            cr = cr + sun_disk * 15.0;
            cg = cg + sun_disk * 13.0;
            cb = cb + sun_disk * 8.0;
            // Tight inner bloom
            let b1 = exp(-sun_ang * 50.0);
            cr = cr + b1 * 4.0;
            cg = cg + b1 * 3.0;
            cb = cb + b1 * 1.5;
            // Medium bloom halo
            let b2 = exp(-sun_ang * 12.0);
            cr = cr + b2 * 1.5;
            cg = cg + b2 * 1.0;
            cb = cb + b2 * 0.4;
            // Wide atmospheric glow
            let b3 = exp(-sun_ang * 4.0);
            cr = cr + b3 * 0.4;
            cg = cg + b3 * 0.2;
            cb = cb + b3 * 0.06;

            var best_t = 0.0;
            best_t = 100000.0;

            // --- Ground plane (y=0) ---
            if (ry < -0.001) {
                let gt = -cam_y / ry;
                if (gt > 0.0) {
                    let gz = gt * rz;
                    if (gz > 0.0) {
                        if (gz > 50.0) {
                            // Sea
                            best_t = gt;
                            cr = 0.02; cg = 0.06; cb = 0.3;
                            let refy = -ry;
                            let ref_dot = rx * sx + refy * sy + rz * sz;
                            let ref_ang = max(1.0 - ref_dot, 0.001);
                            let fbase = 1.0 - abs(ry);
                            let fresnel = 0.2 + 0.8 * fbase * fbase * fbase;
                            let refl1 = exp(-ref_ang * 40.0) * fresnel;
                            cr = cr + refl1 * 6.0;
                            cg = cg + refl1 * 5.0;
                            cb = cb + refl1 * 3.0;
                            let refl2 = exp(-ref_ang * 8.0) * fresnel * 0.3;
                            cr = cr + refl2 * 1.2;
                            cg = cg + refl2 * 0.8;
                            cb = cb + refl2 * 0.5;
                            let fog = clamp(gt / 400.0, 0.0, 0.65);
                            cr = cr * (1.0 - fog) + 0.45 * fog;
                            cg = cg * (1.0 - fog) + 0.55 * fog;
                            cb = cb * (1.0 - fog) + 0.75 * fog;
                        } else {
                            // Green terrain
                            best_t = gt;
                            let ndl = max(sy, 0.15);
                            cr = 0.08 * ndl;
                            cg = 0.42 * ndl;
                            cb = 0.04 * ndl;
                            let fog = clamp(gt / 120.0, 0.0, 0.7);
                            cr = cr * (1.0 - fog) + 0.45 * fog;
                            cg = cg * (1.0 - fog) + 0.55 * fog;
                            cb = cb * (1.0 - fog) + 0.75 * fog;
                        }
                    }
                }
            }

            // --- Tree A: pos (5, z=15), trunk h=3 r=0.3, canopy cy=3.8 r=1.5 ---
            let at_ox = 0.0 - 5.0;
            let at_oz = 0.0 - 15.0;
            let at_a = rx * rx + rz * rz;
            let at_hb = at_ox * rx + at_oz * rz;
            let at_c = at_ox * at_ox + at_oz * at_oz - 0.3 * 0.3;
            let at_disc = at_hb * at_hb - at_a * at_c;
            if (at_disc >= 0.0) {
                let t = (-at_hb - sqrt(at_disc)) / at_a;
                let y = cam_y + t * ry;
                if (t > 0.0 && t < best_t && y >= 0.0 && y <= 3.0) {
                    best_t = t;
                    let hx = t * rx;
                    let hz = t * rz;
                    let nx = (hx - 5.0) / 0.3;
                    let nz = (hz - 15.0) / 0.3;
                    let ndl = max(nx * sx + nz * sz, 0.15);
                    cr = 0.40 * ndl; cg = 0.22 * ndl; cb = 0.06 * ndl;
                }
            }
            let ac_ox = 0.0 - 5.0;
            let ac_oy = cam_y - 3.8;
            let ac_oz = 0.0 - 15.0;
            let ac_hb = ac_ox * rx + ac_oy * ry + ac_oz * rz;
            let ac_c = ac_ox * ac_ox + ac_oy * ac_oy + ac_oz * ac_oz - 1.5 * 1.5;
            let ac_disc = ac_hb * ac_hb - ac_c;
            if (ac_disc >= 0.0) {
                let t = -ac_hb - sqrt(ac_disc);
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    let hx = t * rx;
                    let hy = cam_y + t * ry;
                    let hz = t * rz;
                    let nx = (hx - 5.0) / 1.5;
                    let ny = (hy - 3.8) / 1.5;
                    let nz = (hz - 15.0) / 1.5;
                    let ndl = max(nx * sx + ny * sy + nz * sz, 0.15);
                    cr = 0.06 * ndl; cg = 0.48 * ndl; cb = 0.03 * ndl;
                }
            }

            // --- Tree B: pos (-3, z=10), trunk h=2.5 r=0.25, canopy cy=3.1 r=1.2 ---
            let bt_ox = 0.0 - -3.0;
            let bt_oz = 0.0 - 10.0;
            let bt_a = rx * rx + rz * rz;
            let bt_hb = bt_ox * rx + bt_oz * rz;
            let bt_c = bt_ox * bt_ox + bt_oz * bt_oz - 0.25 * 0.25;
            let bt_disc = bt_hb * bt_hb - bt_a * bt_c;
            if (bt_disc >= 0.0) {
                let t = (-bt_hb - sqrt(bt_disc)) / bt_a;
                let y = cam_y + t * ry;
                if (t > 0.0 && t < best_t && y >= 0.0 && y <= 2.5) {
                    best_t = t;
                    let hx = t * rx;
                    let hz = t * rz;
                    let nx = (hx + 3.0) / 0.25;
                    let nz = (hz - 10.0) / 0.25;
                    let ndl = max(nx * sx + nz * sz, 0.15);
                    cr = 0.38 * ndl; cg = 0.20 * ndl; cb = 0.05 * ndl;
                }
            }
            let bc_ox = 0.0 - -3.0;
            let bc_oy = cam_y - 3.1;
            let bc_oz = 0.0 - 10.0;
            let bc_hb = bc_ox * rx + bc_oy * ry + bc_oz * rz;
            let bc_c = bc_ox * bc_ox + bc_oy * bc_oy + bc_oz * bc_oz - 1.2 * 1.2;
            let bc_disc = bc_hb * bc_hb - bc_c;
            if (bc_disc >= 0.0) {
                let t = -bc_hb - sqrt(bc_disc);
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    let hx = t * rx;
                    let hy = cam_y + t * ry;
                    let hz = t * rz;
                    let nx = (hx + 3.0) / 1.2;
                    let ny = (hy - 3.1) / 1.2;
                    let nz = (hz - 10.0) / 1.2;
                    let ndl = max(nx * sx + ny * sy + nz * sz, 0.15);
                    cr = 0.05 * ndl; cg = 0.42 * ndl; cb = 0.02 * ndl;
                }
            }

            // --- Tree C: pos (1.5, z=25), trunk h=3.5 r=0.35, canopy cy=4.4 r=1.8 ---
            let ct_ox = 0.0 - 1.5;
            let ct_oz = 0.0 - 25.0;
            let ct_a = rx * rx + rz * rz;
            let ct_hb = ct_ox * rx + ct_oz * rz;
            let ct_c = ct_ox * ct_ox + ct_oz * ct_oz - 0.35 * 0.35;
            let ct_disc = ct_hb * ct_hb - ct_a * ct_c;
            if (ct_disc >= 0.0) {
                let t = (-ct_hb - sqrt(ct_disc)) / ct_a;
                let y = cam_y + t * ry;
                if (t > 0.0 && t < best_t && y >= 0.0 && y <= 3.5) {
                    best_t = t;
                    let hx = t * rx;
                    let hz = t * rz;
                    let nx = (hx - 1.5) / 0.35;
                    let nz = (hz - 25.0) / 0.35;
                    let ndl = max(nx * sx + nz * sz, 0.15);
                    cr = 0.42 * ndl; cg = 0.24 * ndl; cb = 0.07 * ndl;
                }
            }
            let cc_ox = 0.0 - 1.5;
            let cc_oy = cam_y - 4.4;
            let cc_oz = 0.0 - 25.0;
            let cc_hb = cc_ox * rx + cc_oy * ry + cc_oz * rz;
            let cc_c = cc_ox * cc_ox + cc_oy * cc_oy + cc_oz * cc_oz - 1.8 * 1.8;
            let cc_disc = cc_hb * cc_hb - cc_c;
            if (cc_disc >= 0.0) {
                let t = -cc_hb - sqrt(cc_disc);
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    let hx = t * rx;
                    let hy = cam_y + t * ry;
                    let hz = t * rz;
                    let nx = (hx - 1.5) / 1.8;
                    let ny = (hy - 4.4) / 1.8;
                    let nz = (hz - 25.0) / 1.8;
                    let ndl = max(nx * sx + ny * sy + nz * sz, 0.15);
                    cr = 0.04 * ndl; cg = 0.50 * ndl; cb = 0.03 * ndl;
                }
            }

            // --- Reinhard tonemapping ---
            cr = cr / (1.0 + cr);
            cg = cg / (1.0 + cg);
            cb = cb / (1.0 + cb);

            return vec4f(cr, cg, cb, 1.0);
        }
    )";

    auto vs = wgsl_test::CompileWgsl(vs_source);
    ASSERT_TRUE(vs.success) << vs.error;
    auto fs = wgsl_test::CompileWgsl(fs_source);
    ASSERT_TRUE(fs.success) << fs.error;

    auto vb = ctx_->createVertexBuffer(kFullScreenTri);
    const uint32_t W = 512, H = 512;
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

    stbi_write_png("outdoor_scene_3d.png", W, H, 4, pixels.data(), W * 4);

    // ---- Verification ----
    uint8_t r, g, b, a;

    // 1. Sky region - sample top-left, far from sun bloom (sun is upper-right)
    uint32_t sky_px = pixels[30 * W + 30];
    unpackRGBA(sky_px, r, g, b, a);
    EXPECT_GT(b, r) << "Sky should be blue-dominant (far from sun)";
    EXPECT_GT(b, 60) << "Sky should have noticeable blue";

    // 2. Terrain region (lower portion, y=380, center column) should be green-dominant
    uint32_t terrain_px = pixels[380 * W + 256];
    unpackRGBA(terrain_px, r, g, b, a);
    EXPECT_GT(g, r) << "Terrain should be green-dominant";
    EXPECT_GT(g, b) << "Terrain green should exceed blue";
    EXPECT_GT(g, 20) << "Terrain should have visible green";

    // 3. Sun bloom: the upper-right area should contain very bright pixels.
    //    Sun direction projects to approximately pixel (370, 45).
    int bright_count = 0;
    for (int sy = 20; sy < 80; sy++) {
        for (int sx = 340; sx < 420; sx++) {
            unpackRGBA(pixels[sy * W + sx], r, g, b, a);
            if (r > 200 && g > 150) {
                bright_count++;
            }
        }
    }
    EXPECT_GT(bright_count, 50)
        << "Sun bloom region should contain many bright warm pixels";

    // 4. Sea region: sample near the horizon (y ~ 230, center column).
    uint32_t sea_px = pixels[232 * W + 256];
    unpackRGBA(sea_px, r, g, b, a);
    EXPECT_GT(b, 40) << "Sea should have blue component";

    // 5. Tree presence: look for brown (trunk) and dark green (canopy) pixels.
    int brown_count = 0;
    int canopy_green_count = 0;
    for (uint32_t py = 160; py < 300; py++) {
        for (uint32_t px = 100; px < 450; px++) {
            unpackRGBA(pixels[py * W + px], r, g, b, a);
            if (r > 20 && g > 10 && r > g && g > b && b < r / 2) {
                brown_count++;
            }
            if (g > 15 && g > r * 2 && g > b * 2) {
                canopy_green_count++;
            }
        }
    }
    EXPECT_GT(brown_count, 10) << "Should have brown (trunk) pixels";
    EXPECT_GT(canopy_green_count, 100) << "Should have green (canopy) pixels";

    // 6. Overall variation: the scene should not be a single flat color.
    std::set<uint32_t> color_buckets;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        unpackRGBA(pixels[i], r, g, b, a);
        uint32_t key = ((r / 8) << 10) | ((g / 8) << 5) | (b / 8);
        color_buckets.insert(key);
    }
    EXPECT_GT(color_buckets.size(), 30u)
        << "Scene should have rich color variation";
}

// ============================================================================
// Fragment shader struct parameter: vertex passes color via struct,
// fragment receives it via a struct parameter and returns it.
// ============================================================================

TEST_F(VulkanGraphicsSceneTest, FragmentStructParam_PassthroughColor) {
    // Vertex shader outputs color via a struct (VsOut) with @location(0)
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            // Red for full-screen triangle
            out.col = vec4f(1.0, 0.0, 0.0, 1.0);
            return out;
        }
    )";

    // Fragment shader receives color via a struct parameter (FsIn) with @location(0)
    const char* fs_source = R"(
        struct FsIn {
            @location(0) col: vec4f,
        };
        @fragment fn main(in: FsIn) -> @location(0) vec4f {
            return in.col;
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
    ASSERT_EQ(pixels.size(), W * H);

    // The entire full-screen triangle should be red (vertex outputs red)
    uint8_t r, g, b, a;
    uint32_t center_px = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center_px, r, g, b, a);
    EXPECT_GT(r, 200) << "Center pixel should be red";
    EXPECT_LT(g, 20) << "Center pixel should have no green";
    EXPECT_LT(b, 20) << "Center pixel should have no blue";
}

// Fragment struct with multiple location fields
TEST_F(VulkanGraphicsSceneTest, FragmentStructParam_MultipleLocations) {
    const char* vs_source = R"(
        struct VIn { @location(0) pos: vec2f };
        struct VOut {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @vertex fn main(in: VIn) -> VOut {
            var out: VOut;
            out.pos = vec4f(in.pos, 0.0, 1.0);
            out.col = vec4f(0.0, 1.0, 0.0, 1.0);
            out.uv = in.pos * 0.5 + 0.5;
            return out;
        }
    )";

    // Fragment uses the uv field to modulate color
    const char* fs_source = R"(
        struct FsIn {
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @fragment fn main(in: FsIn) -> @location(0) vec4f {
            return vec4f(in.uv.x, in.col.g, in.uv.y, 1.0);
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
    ASSERT_EQ(pixels.size(), W * H);

    // Center pixel: uv.x ~0.5, col.g=1.0, uv.y ~0.5
    uint8_t r, g, b, a;
    uint32_t center_px = pixels[(H / 2) * W + (W / 2)];
    unpackRGBA(center_px, r, g, b, a);
    EXPECT_GT(g, 200) << "Green channel should be bright (from col.g=1.0)";
    // Red and blue from uv should be near 0.5 → ~128
    EXPECT_GT(r, 80) << "Red channel should be moderate (uv.x ~0.5)";
    EXPECT_LT(r, 200) << "Red channel should not be saturated";
}

#endif // WGSL_HAS_VULKAN
