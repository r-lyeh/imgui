#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <functional>

namespace vk_graphics {

class VulkanError : public std::runtime_error {
public:
    VulkanError(VkResult result, const char* msg)
        : std::runtime_error(std::string(msg) + " (VkResult: " + std::to_string(result) + ")")
        , result_(result) {}
    VkResult result() const { return result_; }
private:
    VkResult result_;
};

#define VK_CHECK(expr) do { \
    VkResult _res = (expr); \
    if (_res != VK_SUCCESS) throw VulkanError(_res, #expr); \
} while(0)

// Forward declarations
class GraphicsContext;
class Buffer;
class Image;
class GraphicsPipeline;

// Buffer usage flags
enum class BufferUsage {
    Vertex,       // Vertex buffer
    Index,        // Index buffer
    Uniform,      // Uniform buffer
    Storage,      // Storage buffer
    Staging       // CPU-visible staging buffer
};

// Image format presets
enum class ImageFormat {
    RGBA8_Unorm,      // VK_FORMAT_R8G8B8A8_UNORM
    RGBA8_Srgb,       // VK_FORMAT_R8G8B8A8_SRGB
    RGBA16_Float,     // VK_FORMAT_R16G16B16A16_SFLOAT
    RGBA32_Float,     // VK_FORMAT_R32G32B32A32_SFLOAT
    R32_Float,        // VK_FORMAT_R32_SFLOAT
    Depth32_Float,    // VK_FORMAT_D32_SFLOAT
};

// RAII buffer wrapper
class Buffer {
public:
    Buffer(GraphicsContext& ctx, size_t size, BufferUsage usage);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void upload(const void* data, size_t size, size_t offset = 0);
    void download(void* data, size_t size, size_t offset = 0);

    template<typename T>
    void upload(const std::vector<T>& data, size_t offset = 0) {
        upload(data.data(), data.size() * sizeof(T), offset);
    }

    template<typename T>
    std::vector<T> download(size_t count, size_t offset = 0) {
        std::vector<T> data(count);
        download(data.data(), count * sizeof(T), offset);
        return data;
    }

    VkBuffer handle() const { return buffer_; }
    size_t size() const { return size_; }
    BufferUsage usage() const { return usage_; }

private:
    GraphicsContext* ctx_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    BufferUsage usage_;
    void* mapped_ = nullptr;

    void cleanup();
};

// RAII image wrapper for render targets
class Image {
public:
    Image(GraphicsContext& ctx, uint32_t width, uint32_t height, ImageFormat format);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    // Download image data to CPU
    std::vector<uint8_t> download();

    // Download as typed data
    template<typename T>
    std::vector<T> downloadAs() {
        auto bytes = download();
        std::vector<T> result(bytes.size() / sizeof(T));
        std::memcpy(result.data(), bytes.data(), bytes.size());
        return result;
    }

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    VkFormat format() const { return format_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    size_t pixelSize() const;

    bool isDepthFormat() const {
        return format_ == VK_FORMAT_D32_SFLOAT;
    }

private:
    GraphicsContext* ctx_;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkFormat format_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    void cleanup();
};

// Vertex input attribute description
struct VertexAttribute {
    uint32_t location;
    VkFormat format;
    uint32_t offset;
};

// Graphics pipeline configuration
struct GraphicsPipelineConfig {
    // Shaders (required)
    const uint32_t* vertex_spirv = nullptr;
    size_t vertex_spirv_words = 0;
    const char* vertex_entry = "main";

    const uint32_t* fragment_spirv = nullptr;
    size_t fragment_spirv_words = 0;
    const char* fragment_entry = "main";

    // Vertex input
    uint32_t vertex_stride = 0;
    std::vector<VertexAttribute> vertex_attributes;

    // Render state
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_test = false;
    bool depth_write = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_LESS;

    // Color attachment format (for dynamic rendering)
    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;

    // Blending
    bool blend_enable = false;
    VkBlendFactor src_blend = VK_BLEND_FACTOR_SRC_ALPHA;
    VkBlendFactor dst_blend = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
};

// Graphics pipeline wrapper
class GraphicsPipeline {
public:
    GraphicsPipeline(GraphicsContext& ctx, const GraphicsPipelineConfig& config);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&& other) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return desc_set_layout_; }

private:
    GraphicsContext* ctx_;
    VkShaderModule vertex_module_ = VK_NULL_HANDLE;
    VkShaderModule fragment_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    void cleanup();
};

// Descriptor binding for draw calls
struct DescriptorBinding {
    uint32_t binding;
    Buffer* buffer;
    VkDescriptorType type;
};

// Draw command parameters
struct DrawParams {
    uint32_t vertex_count = 0;
    uint32_t instance_count = 1;
    uint32_t first_vertex = 0;
    uint32_t first_instance = 0;
};

struct DrawIndexedParams {
    uint32_t index_count = 0;
    uint32_t instance_count = 1;
    uint32_t first_index = 0;
    int32_t vertex_offset = 0;
    uint32_t first_instance = 0;
};

// Clear values
struct ClearColor {
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
};

// Main Vulkan graphics context
class GraphicsContext {
public:
    GraphicsContext();
    ~GraphicsContext();

    GraphicsContext(const GraphicsContext&) = delete;
    GraphicsContext& operator=(const GraphicsContext&) = delete;

    // Device access
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_device_; }
    VkQueue graphicsQueue() const { return graphics_queue_; }
    uint32_t graphicsQueueFamily() const { return graphics_queue_family_; }
    VkCommandPool commandPool() const { return command_pool_; }

    // Buffer creation
    Buffer createBuffer(size_t size, BufferUsage usage);

    template<typename T>
    Buffer createVertexBuffer(const std::vector<T>& data) {
        Buffer buf = createBuffer(data.size() * sizeof(T), BufferUsage::Vertex);
        buf.upload(data);
        return buf;
    }

    template<typename T>
    Buffer createIndexBuffer(const std::vector<T>& data) {
        Buffer buf = createBuffer(data.size() * sizeof(T), BufferUsage::Index);
        buf.upload(data);
        return buf;
    }

    template<typename T>
    Buffer createUniformBuffer(const T& data) {
        Buffer buf = createBuffer(sizeof(T), BufferUsage::Uniform);
        buf.upload(&data, sizeof(T));
        return buf;
    }

    Buffer createStorageBuffer(size_t size) {
        return createBuffer(size, BufferUsage::Storage);
    }

    // Image creation
    Image createImage(uint32_t width, uint32_t height, ImageFormat format);

    Image createColorTarget(uint32_t width, uint32_t height,
                            ImageFormat format = ImageFormat::RGBA8_Unorm) {
        return createImage(width, height, format);
    }

    Image createDepthTarget(uint32_t width, uint32_t height) {
        return createImage(width, height, ImageFormat::Depth32_Float);
    }

    // Pipeline creation
    GraphicsPipeline createPipeline(const GraphicsPipelineConfig& config);

    // Draw to image using dynamic rendering
    void draw(GraphicsPipeline& pipeline,
              Image& color_target,
              Buffer* vertex_buffer,
              const DrawParams& params,
              const std::vector<DescriptorBinding>& bindings = {},
              const ClearColor& clear = {},
              Image* depth_target = nullptr);

    void drawIndexed(GraphicsPipeline& pipeline,
                     Image& color_target,
                     Buffer* vertex_buffer,
                     Buffer* index_buffer,
                     VkIndexType index_type,
                     const DrawIndexedParams& params,
                     const std::vector<DescriptorBinding>& bindings = {},
                     const ClearColor& clear = {},
                     Image* depth_target = nullptr);

    // Execute a one-shot command buffer
    void executeCommands(std::function<void(VkCommandBuffer)> recorder);

    // Image layout transitions
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkImageLayout old_layout, VkImageLayout new_layout,
                               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Memory utilities
    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

    const VkPhysicalDeviceProperties& deviceProperties() const { return device_props_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_props_;
    VkPhysicalDeviceProperties device_props_;

    void createInstance();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createDescriptorPool();
    void cleanup();

    friend class Buffer;
    friend class Image;
    friend class GraphicsPipeline;
};

// Helper to convert ImageFormat enum to VkFormat
inline VkFormat toVkFormat(ImageFormat format) {
    switch (format) {
        case ImageFormat::RGBA8_Unorm:   return VK_FORMAT_R8G8B8A8_UNORM;
        case ImageFormat::RGBA8_Srgb:    return VK_FORMAT_R8G8B8A8_SRGB;
        case ImageFormat::RGBA16_Float:  return VK_FORMAT_R16G16B16A16_SFLOAT;
        case ImageFormat::RGBA32_Float:  return VK_FORMAT_R32G32B32A32_SFLOAT;
        case ImageFormat::R32_Float:     return VK_FORMAT_R32_SFLOAT;
        case ImageFormat::Depth32_Float: return VK_FORMAT_D32_SFLOAT;
        default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

} // namespace vk_graphics
