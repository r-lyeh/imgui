#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <functional>

namespace vk_compute {

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
class VulkanContext;
class Buffer;
class ComputePipeline;

// Buffer usage flags
enum class BufferUsage {
    Storage,      // Storage buffer (read/write)
    Uniform,      // Uniform buffer (read-only)
    Staging       // CPU-visible staging buffer for transfers
};

// RAII buffer wrapper
class Buffer {
public:
    Buffer(VulkanContext& ctx, size_t size, BufferUsage usage);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    // Upload data to buffer (for staging buffers, direct; otherwise uses staging)
    void upload(const void* data, size_t size, size_t offset = 0);

    // Download data from buffer
    void download(void* data, size_t size, size_t offset = 0);

    // Template helpers for typed data
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
    VulkanContext* ctx_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    BufferUsage usage_;
    void* mapped_ = nullptr;

    void cleanup();
};

// Compute pipeline wrapper
class ComputePipeline {
public:
    ComputePipeline(VulkanContext& ctx, const uint32_t* spirv, size_t word_count,
                    const char* entry_point = "main");
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;

    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return desc_set_layout_; }

private:
    VulkanContext* ctx_;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    void cleanup();
};

// Descriptor binding for dispatch
struct DescriptorBinding {
    uint32_t binding;
    Buffer* buffer;
    VkDescriptorType type;  // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
};

// Main Vulkan compute context
class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // Device access
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_device_; }
    VkQueue computeQueue() const { return compute_queue_; }
    uint32_t computeQueueFamily() const { return compute_queue_family_; }
    VkCommandPool commandPool() const { return command_pool_; }

    // Buffer creation helpers
    Buffer createBuffer(size_t size, BufferUsage usage);

    template<typename T>
    Buffer createStorageBuffer(const std::vector<T>& data) {
        Buffer buf = createBuffer(data.size() * sizeof(T), BufferUsage::Storage);
        buf.upload(data);
        return buf;
    }

    Buffer createStorageBuffer(size_t size) {
        return createBuffer(size, BufferUsage::Storage);
    }

    // Pipeline creation
    ComputePipeline createPipeline(const uint32_t* spirv, size_t word_count,
                                   const char* entry_point = "main");

    ComputePipeline createPipeline(const std::vector<uint32_t>& spirv,
                                   const char* entry_point = "main") {
        return createPipeline(spirv.data(), spirv.size(), entry_point);
    }

    // Dispatch compute shader
    void dispatch(ComputePipeline& pipeline,
                  const std::vector<DescriptorBinding>& bindings,
                  uint32_t group_count_x,
                  uint32_t group_count_y = 1,
                  uint32_t group_count_z = 1);

    // Execute a one-shot command buffer
    void executeCommands(std::function<void(VkCommandBuffer)> recorder);

    // Memory utilities
    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

    // Get device properties
    const VkPhysicalDeviceProperties& deviceProperties() const { return device_props_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = 0;
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
    friend class ComputePipeline;
};

} // namespace vk_compute
