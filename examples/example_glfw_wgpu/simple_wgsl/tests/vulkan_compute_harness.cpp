#include "vulkan_compute_harness.h"
#include <algorithm>
#include <cstring>

namespace vk_compute {

// ============================================================================
// VulkanContext
// ============================================================================

VulkanContext::VulkanContext() {
    createInstance();
    selectPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createDescriptorPool();
}

VulkanContext::~VulkanContext() {
    cleanup();
}

void VulkanContext::cleanup() {
    if (device_) {
        vkDeviceWaitIdle(device_);

        if (descriptor_pool_) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
        }
        if (command_pool_) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::createInstance() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "WGSL Compute Tests";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));
}

void VulkanContext::selectPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // Find a device with compute queue support
    for (auto device : devices) {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_family_count; i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical_device_ = device;
                compute_queue_family_ = i;
                vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_props_);
                vkGetPhysicalDeviceProperties(physical_device_, &device_props_);
                return;
            }
        }
    }

    throw std::runtime_error("No GPU with compute support found");
}

void VulkanContext::createLogicalDevice() {
    float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features = {};

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = 0;

    VK_CHECK(vkCreateDevice(physical_device_, &create_info, nullptr, &device_));
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

void VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));
}

void VulkanContext::createDescriptorPool() {
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    pool_info.maxSets = 100;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_));
}

uint32_t VulkanContext::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_props_.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memory_props_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

Buffer VulkanContext::createBuffer(size_t size, BufferUsage usage) {
    return Buffer(*this, size, usage);
}

ComputePipeline VulkanContext::createPipeline(const uint32_t* spirv, size_t word_count,
                                               const char* entry_point) {
    return ComputePipeline(*this, spirv, word_count, entry_point);
}

void VulkanContext::executeCommands(std::function<void(VkCommandBuffer)> recorder) {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device_, &alloc_info, &cmd));

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));
    recorder(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(compute_queue_));

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

void VulkanContext::dispatch(ComputePipeline& pipeline,
                              const std::vector<DescriptorBinding>& bindings,
                              uint32_t group_count_x,
                              uint32_t group_count_y,
                              uint32_t group_count_z) {
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    VkDescriptorSetLayout layout = pipeline.descriptorSetLayout();
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(device_, &alloc_info, &desc_set));

    // Update descriptor set
    std::vector<VkDescriptorBufferInfo> buffer_infos(bindings.size());
    std::vector<VkWriteDescriptorSet> writes(bindings.size());

    for (size_t i = 0; i < bindings.size(); i++) {
        buffer_infos[i].buffer = bindings[i].buffer->handle();
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = bindings[i].buffer->size();

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = bindings[i].binding;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].type;
        writes[i].pBufferInfo = &buffer_infos[i];
        writes[i].pImageInfo = nullptr;
        writes[i].pTexelBufferView = nullptr;
    }

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    // Record and submit
    executeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline.layout(), 0, 1, &desc_set, 0, nullptr);
        vkCmdDispatch(cmd, group_count_x, group_count_y, group_count_z);
    });

    // Free descriptor set
    vkFreeDescriptorSets(device_, descriptor_pool_, 1, &desc_set);
}

// ============================================================================
// Buffer
// ============================================================================

Buffer::Buffer(VulkanContext& ctx, size_t size, BufferUsage usage)
    : ctx_(&ctx), size_(size), usage_(usage) {

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkMemoryPropertyFlags mem_props;

    switch (usage) {
        case BufferUsage::Storage:
            buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsage::Uniform:
            buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsage::Staging:
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
    }

    VK_CHECK(vkCreateBuffer(ctx_->device(), &buffer_info, nullptr, &buffer_));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx_->device(), buffer_, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;

    // Try preferred properties first, fall back to just host visible for storage
    try {
        alloc_info.memoryTypeIndex = ctx_->findMemoryType(mem_reqs.memoryTypeBits, mem_props);
    } catch (const std::runtime_error&) {
        if (usage == BufferUsage::Storage) {
            // Fall back to host visible only
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            alloc_info.memoryTypeIndex = ctx_->findMemoryType(mem_reqs.memoryTypeBits, mem_props);
        } else {
            throw;
        }
    }

    VK_CHECK(vkAllocateMemory(ctx_->device(), &alloc_info, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(ctx_->device(), buffer_, memory_, 0));

    // Map memory for host-visible buffers
    VK_CHECK(vkMapMemory(ctx_->device(), memory_, 0, size, 0, &mapped_));
}

Buffer::~Buffer() {
    cleanup();
}

void Buffer::cleanup() {
    if (ctx_ && ctx_->device()) {
        if (mapped_) {
            vkUnmapMemory(ctx_->device(), memory_);
            mapped_ = nullptr;
        }
        if (buffer_) {
            vkDestroyBuffer(ctx_->device(), buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        if (memory_) {
            vkFreeMemory(ctx_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : ctx_(other.ctx_)
    , buffer_(other.buffer_)
    , memory_(other.memory_)
    , size_(other.size_)
    , usage_(other.usage_)
    , mapped_(other.mapped_) {
    other.ctx_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        usage_ = other.usage_;
        mapped_ = other.mapped_;
        other.ctx_ = nullptr;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.mapped_ = nullptr;
    }
    return *this;
}

void Buffer::upload(const void* data, size_t size, size_t offset) {
    if (offset + size > size_) {
        throw std::runtime_error("Buffer upload exceeds buffer size");
    }
    std::memcpy(static_cast<char*>(mapped_) + offset, data, size);
}

void Buffer::download(void* data, size_t size, size_t offset) {
    if (offset + size > size_) {
        throw std::runtime_error("Buffer download exceeds buffer size");
    }
    std::memcpy(data, static_cast<char*>(mapped_) + offset, size);
}

// ============================================================================
// ComputePipeline
// ============================================================================

ComputePipeline::ComputePipeline(VulkanContext& ctx, const uint32_t* spirv, size_t word_count,
                                  const char* entry_point)
    : ctx_(&ctx) {

    // Create shader module
    VkShaderModuleCreateInfo shader_info = {};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = word_count * sizeof(uint32_t);
    shader_info.pCode = spirv;

    VK_CHECK(vkCreateShaderModule(ctx_->device(), &shader_info, nullptr, &shader_module_));

    // Create descriptor set layout with common bindings
    // We create a flexible layout that supports up to 16 storage buffers
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    for (uint32_t i = 0; i < 16; i++) {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layout_bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
    layout_info.pBindings = layout_bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &desc_set_layout_));

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &desc_set_layout_;

    VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &pipeline_layout_info, nullptr, &layout_));

    // Create compute pipeline
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module_;
    pipeline_info.stage.pName = entry_point;
    pipeline_info.layout = layout_;

    VK_CHECK(vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &pipeline_info,
                                      nullptr, &pipeline_));
}

ComputePipeline::~ComputePipeline() {
    cleanup();
}

void ComputePipeline::cleanup() {
    if (ctx_ && ctx_->device()) {
        if (pipeline_) {
            vkDestroyPipeline(ctx_->device(), pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (layout_) {
            vkDestroyPipelineLayout(ctx_->device(), layout_, nullptr);
            layout_ = VK_NULL_HANDLE;
        }
        if (desc_set_layout_) {
            vkDestroyDescriptorSetLayout(ctx_->device(), desc_set_layout_, nullptr);
            desc_set_layout_ = VK_NULL_HANDLE;
        }
        if (shader_module_) {
            vkDestroyShaderModule(ctx_->device(), shader_module_, nullptr);
            shader_module_ = VK_NULL_HANDLE;
        }
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : ctx_(other.ctx_)
    , shader_module_(other.shader_module_)
    , desc_set_layout_(other.desc_set_layout_)
    , layout_(other.layout_)
    , pipeline_(other.pipeline_) {
    other.ctx_ = nullptr;
    other.shader_module_ = VK_NULL_HANDLE;
    other.desc_set_layout_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        shader_module_ = other.shader_module_;
        desc_set_layout_ = other.desc_set_layout_;
        layout_ = other.layout_;
        pipeline_ = other.pipeline_;
        other.ctx_ = nullptr;
        other.shader_module_ = VK_NULL_HANDLE;
        other.desc_set_layout_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vk_compute
