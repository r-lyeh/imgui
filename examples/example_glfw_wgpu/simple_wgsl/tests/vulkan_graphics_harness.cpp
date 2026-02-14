#include "vulkan_graphics_harness.h"
#include <algorithm>
#include <cstring>

namespace vk_graphics {

// ============================================================================
// GraphicsContext
// ============================================================================

GraphicsContext::GraphicsContext() {
    createInstance();
    selectPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createDescriptorPool();
}

GraphicsContext::~GraphicsContext() {
    cleanup();
}

void GraphicsContext::cleanup() {
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

void GraphicsContext::createInstance() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "WGSL Graphics Tests";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));
}

void GraphicsContext::selectPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // Find a device with graphics queue support
    for (auto device : devices) {
        // Check for Vulkan 1.3 support
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_family_count; i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                physical_device_ = device;
                graphics_queue_family_ = i;
                vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_props_);
                vkGetPhysicalDeviceProperties(physical_device_, &device_props_);
                return;
            }
        }
    }

    throw std::runtime_error("No GPU with Vulkan 1.3 graphics support found");
}

void GraphicsContext::createLogicalDevice() {
    float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = graphics_queue_family_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    // Enable Vulkan 1.3 features including dynamic rendering
    VkPhysicalDeviceVulkan13Features vulkan13_features = {};
    vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13_features.dynamicRendering = VK_TRUE;
    vulkan13_features.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 device_features2 = {};
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext = &vulkan13_features;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &device_features2;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.enabledExtensionCount = 0;

    VK_CHECK(vkCreateDevice(physical_device_, &create_info, nullptr, &device_));
    vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
}

void GraphicsContext::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = graphics_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));
}

void GraphicsContext::createDescriptorPool() {
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    pool_info.maxSets = 100;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_));
}

uint32_t GraphicsContext::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_props_.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memory_props_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

Buffer GraphicsContext::createBuffer(size_t size, BufferUsage usage) {
    return Buffer(*this, size, usage);
}

Image GraphicsContext::createImage(uint32_t width, uint32_t height, ImageFormat format) {
    return Image(*this, width, height, format);
}

GraphicsPipeline GraphicsContext::createPipeline(const GraphicsPipelineConfig& config) {
    return GraphicsPipeline(*this, config);
}

void GraphicsContext::executeCommands(std::function<void(VkCommandBuffer)> recorder) {
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

    VK_CHECK(vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(graphics_queue_));

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

void GraphicsContext::transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                             VkImageLayout old_layout, VkImageLayout new_layout,
                                             VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // Set access masks and pipeline stages based on layouts
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
    } else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    }

    if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkDependencyInfo dep_info = {};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep_info);
}

void GraphicsContext::draw(GraphicsPipeline& pipeline,
                            Image& color_target,
                            Buffer* vertex_buffer,
                            const DrawParams& params,
                            const std::vector<DescriptorBinding>& bindings,
                            const ClearColor& clear,
                            Image* depth_target) {
    VkDescriptorSet desc_set = VK_NULL_HANDLE;

    // Allocate and update descriptor set if needed
    if (!bindings.empty()) {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = pipeline.descriptorSetLayout();
        alloc_info.pSetLayouts = &layout;

        VK_CHECK(vkAllocateDescriptorSets(device_, &alloc_info, &desc_set));

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
        }

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    executeCommands([&](VkCommandBuffer cmd) {
        // Transition color target to attachment layout
        transitionImageLayout(cmd, color_target.handle(),
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Transition depth target if present
        if (depth_target) {
            transitionImageLayout(cmd, depth_target->handle(),
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        // Setup color attachment for dynamic rendering
        VkRenderingAttachmentInfo color_attachment = {};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = color_target.view();
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

        VkRenderingAttachmentInfo depth_attachment = {};
        if (depth_target) {
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = depth_target->view();
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil = {1.0f, 0};
        }

        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {{0, 0}, {color_target.width(), color_target.height()}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        if (depth_target) {
            rendering_info.pDepthAttachment = &depth_attachment;
        }

        vkCmdBeginRendering(cmd, &rendering_info);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());

        // Set viewport and scissor
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(color_target.width());
        viewport.height = static_cast<float>(color_target.height());
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor = {{0, 0}, {color_target.width(), color_target.height()}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Bind descriptor set if present
        if (desc_set != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.layout(), 0, 1, &desc_set, 0, nullptr);
        }

        // Bind vertex buffer if present
        if (vertex_buffer) {
            VkBuffer vb = vertex_buffer->handle();
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
        }

        vkCmdDraw(cmd, params.vertex_count, params.instance_count,
                  params.first_vertex, params.first_instance);

        vkCmdEndRendering(cmd);

        // Transition for readback
        transitionImageLayout(cmd, color_target.handle(),
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });

    // Free descriptor set
    if (desc_set != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device_, descriptor_pool_, 1, &desc_set);
    }
}

void GraphicsContext::drawIndexed(GraphicsPipeline& pipeline,
                                   Image& color_target,
                                   Buffer* vertex_buffer,
                                   Buffer* index_buffer,
                                   VkIndexType index_type,
                                   const DrawIndexedParams& params,
                                   const std::vector<DescriptorBinding>& bindings,
                                   const ClearColor& clear,
                                   Image* depth_target) {
    VkDescriptorSet desc_set = VK_NULL_HANDLE;

    if (!bindings.empty()) {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = pipeline.descriptorSetLayout();
        alloc_info.pSetLayouts = &layout;

        VK_CHECK(vkAllocateDescriptorSets(device_, &alloc_info, &desc_set));

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
        }

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    executeCommands([&](VkCommandBuffer cmd) {
        transitionImageLayout(cmd, color_target.handle(),
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        if (depth_target) {
            transitionImageLayout(cmd, depth_target->handle(),
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        VkRenderingAttachmentInfo color_attachment = {};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = color_target.view();
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

        VkRenderingAttachmentInfo depth_attachment = {};
        if (depth_target) {
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = depth_target->view();
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil = {1.0f, 0};
        }

        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {{0, 0}, {color_target.width(), color_target.height()}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        if (depth_target) {
            rendering_info.pDepthAttachment = &depth_attachment;
        }

        vkCmdBeginRendering(cmd, &rendering_info);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());

        VkViewport viewport = {};
        viewport.width = static_cast<float>(color_target.width());
        viewport.height = static_cast<float>(color_target.height());
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor = {{0, 0}, {color_target.width(), color_target.height()}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (desc_set != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.layout(), 0, 1, &desc_set, 0, nullptr);
        }

        if (vertex_buffer) {
            VkBuffer vb = vertex_buffer->handle();
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
        }

        if (index_buffer) {
            vkCmdBindIndexBuffer(cmd, index_buffer->handle(), 0, index_type);
        }

        vkCmdDrawIndexed(cmd, params.index_count, params.instance_count,
                         params.first_index, params.vertex_offset, params.first_instance);

        vkCmdEndRendering(cmd);

        transitionImageLayout(cmd, color_target.handle(),
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });

    if (desc_set != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device_, descriptor_pool_, 1, &desc_set);
    }
}

// ============================================================================
// Buffer
// ============================================================================

Buffer::Buffer(GraphicsContext& ctx, size_t size, BufferUsage usage)
    : ctx_(&ctx), size_(size), usage_(usage) {

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkMemoryPropertyFlags mem_props;

    switch (usage) {
        case BufferUsage::Vertex:
            buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsage::Index:
            buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsage::Uniform:
            buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsage::Storage:
            buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
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
    alloc_info.memoryTypeIndex = ctx_->findMemoryType(mem_reqs.memoryTypeBits, mem_props);

    VK_CHECK(vkAllocateMemory(ctx_->device(), &alloc_info, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(ctx_->device(), buffer_, memory_, 0));
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
// Image
// ============================================================================

Image::Image(GraphicsContext& ctx, uint32_t width, uint32_t height, ImageFormat format)
    : ctx_(&ctx), width_(width), height_(height), format_(toVkFormat(format)) {

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format_;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (isDepthFormat()) {
        image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    } else {
        image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    VK_CHECK(vkCreateImage(ctx_->device(), &image_info, nullptr, &image_));

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(ctx_->device(), image_, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = ctx_->findMemoryType(mem_reqs.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(ctx_->device(), &alloc_info, nullptr, &memory_));
    VK_CHECK(vkBindImageMemory(ctx_->device(), image_, memory_, 0));

    // Create image view
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (isDepthFormat()) {
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VK_CHECK(vkCreateImageView(ctx_->device(), &view_info, nullptr, &view_));
}

Image::~Image() {
    cleanup();
}

void Image::cleanup() {
    if (ctx_ && ctx_->device()) {
        if (view_) {
            vkDestroyImageView(ctx_->device(), view_, nullptr);
            view_ = VK_NULL_HANDLE;
        }
        if (image_) {
            vkDestroyImage(ctx_->device(), image_, nullptr);
            image_ = VK_NULL_HANDLE;
        }
        if (memory_) {
            vkFreeMemory(ctx_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
}

Image::Image(Image&& other) noexcept
    : ctx_(other.ctx_)
    , image_(other.image_)
    , memory_(other.memory_)
    , view_(other.view_)
    , format_(other.format_)
    , width_(other.width_)
    , height_(other.height_) {
    other.ctx_ = nullptr;
    other.image_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.view_ = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        image_ = other.image_;
        memory_ = other.memory_;
        view_ = other.view_;
        format_ = other.format_;
        width_ = other.width_;
        height_ = other.height_;
        other.ctx_ = nullptr;
        other.image_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.view_ = VK_NULL_HANDLE;
    }
    return *this;
}

size_t Image::pixelSize() const {
    switch (format_) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_R32_SFLOAT:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16;
        default:
            return 4;
    }
}

std::vector<uint8_t> Image::download() {
    size_t image_size = width_ * height_ * pixelSize();

    // Create staging buffer
    Buffer staging(*ctx_, image_size, BufferUsage::Staging);

    ctx_->executeCommands([&](VkCommandBuffer cmd) {
        // Copy image to staging buffer
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = isDepthFormat() ?
            VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width_, height_, 1};

        vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle(), 1, &region);
    });

    std::vector<uint8_t> data(image_size);
    staging.download(data.data(), image_size);
    return data;
}

// ============================================================================
// GraphicsPipeline
// ============================================================================

GraphicsPipeline::GraphicsPipeline(GraphicsContext& ctx, const GraphicsPipelineConfig& config)
    : ctx_(&ctx) {

    // Create shader modules
    VkShaderModuleCreateInfo vert_info = {};
    vert_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vert_info.codeSize = config.vertex_spirv_words * sizeof(uint32_t);
    vert_info.pCode = config.vertex_spirv;
    VK_CHECK(vkCreateShaderModule(ctx_->device(), &vert_info, nullptr, &vertex_module_));

    VkShaderModuleCreateInfo frag_info = {};
    frag_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    frag_info.codeSize = config.fragment_spirv_words * sizeof(uint32_t);
    frag_info.pCode = config.fragment_spirv;
    VK_CHECK(vkCreateShaderModule(ctx_->device(), &frag_info, nullptr, &fragment_module_));

    // Shader stages
    VkPipelineShaderStageCreateInfo shader_stages[2] = {};
    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = vertex_module_;
    shader_stages[0].pName = config.vertex_entry;

    shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = fragment_module_;
    shader_stages[1].pName = config.fragment_entry;

    // Vertex input
    VkVertexInputBindingDescription binding_desc = {};
    binding_desc.binding = 0;
    binding_desc.stride = config.vertex_stride;
    binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> attr_descs;
    for (const auto& attr : config.vertex_attributes) {
        VkVertexInputAttributeDescription desc = {};
        desc.location = attr.location;
        desc.binding = 0;
        desc.format = attr.format;
        desc.offset = attr.offset;
        attr_descs.push_back(desc);
    }

    VkPipelineVertexInputStateCreateInfo vertex_input = {};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (config.vertex_stride > 0 && !attr_descs.empty()) {
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding_desc;
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_descs.size());
        vertex_input.pVertexAttributeDescriptions = attr_descs.data();
    }

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = config.topology;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    // Viewport state (dynamic)
    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = config.polygon_mode;
    rasterizer.cullMode = config.cull_mode;
    rasterizer.frontFace = config.front_face;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.lineWidth = 1.0f;

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.sampleShadingEnable = VK_FALSE;

    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = config.depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = config.depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = config.depth_compare;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    // Color blending
    VkPipelineColorBlendAttachmentState color_blend_attachment = {};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = config.blend_enable ? VK_TRUE : VK_FALSE;
    if (config.blend_enable) {
        color_blend_attachment.srcColorBlendFactor = config.src_blend;
        color_blend_attachment.dstColorBlendFactor = config.dst_blend;
        color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;

    // Dynamic states
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = {};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    for (uint32_t i = 0; i < 16; i++) {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
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

    // Dynamic rendering format info (Vulkan 1.3)
    VkPipelineRenderingCreateInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &config.color_format;
    rendering_info.depthAttachmentFormat = config.depth_format;

    // Create graphics pipeline
    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &rendering_info;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = layout_;
    pipeline_info.renderPass = VK_NULL_HANDLE;  // Dynamic rendering - no render pass

    VK_CHECK(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &pipeline_info,
                                        nullptr, &pipeline_));
}

GraphicsPipeline::~GraphicsPipeline() {
    cleanup();
}

void GraphicsPipeline::cleanup() {
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
        if (fragment_module_) {
            vkDestroyShaderModule(ctx_->device(), fragment_module_, nullptr);
            fragment_module_ = VK_NULL_HANDLE;
        }
        if (vertex_module_) {
            vkDestroyShaderModule(ctx_->device(), vertex_module_, nullptr);
            vertex_module_ = VK_NULL_HANDLE;
        }
    }
}

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
    : ctx_(other.ctx_)
    , vertex_module_(other.vertex_module_)
    , fragment_module_(other.fragment_module_)
    , desc_set_layout_(other.desc_set_layout_)
    , layout_(other.layout_)
    , pipeline_(other.pipeline_) {
    other.ctx_ = nullptr;
    other.vertex_module_ = VK_NULL_HANDLE;
    other.fragment_module_ = VK_NULL_HANDLE;
    other.desc_set_layout_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
}

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        vertex_module_ = other.vertex_module_;
        fragment_module_ = other.fragment_module_;
        desc_set_layout_ = other.desc_set_layout_;
        layout_ = other.layout_;
        pipeline_ = other.pipeline_;
        other.ctx_ = nullptr;
        other.vertex_module_ = VK_NULL_HANDLE;
        other.fragment_module_ = VK_NULL_HANDLE;
        other.desc_set_layout_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vk_graphics
