// C99
#define _GNU_SOURCE // for asprintf
#define VK_USE_PLATFORM_DISPLAY_KHR
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// For DRM
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// For stat() and major/minor macros
#include <sys/stat.h>
#include <sys/sysmacros.h>

// --- Configuration ---
// On most single-GPU systems, this is "/dev/dri/card0".
// Change this to the DRM device you want to control.
#define DRM_DEVICE_PATH "/dev/dri/card1"


// --- Helper Macro and Function ---
// A simple utility for error handling
static void die(const char* msg) {
    fprintf(stderr, "Fatal error: %s\n", msg);
    exit(EXIT_FAILURE);
}

// Macro to check Vulkan results
#define VK_CHECK(call) \
    do { \
        VkResult result = call; \
        if (result != VK_SUCCESS) { \
            char* msg; \
            asprintf(&msg, "%s failed with error code %d", #call, result); \
            die(msg); \
        } \
    } while (0)

// --- Main Application State ---
struct App {
    // DRM state
    int drm_fd;
    uint32_t drm_connector_id;
    uint32_t drm_crtc_id;
    drmModeModeInfo drm_mode;

    // Vulkan core
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;

    // Swapchain
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    uint32_t image_count;
    VkImage* images;
    VkImageView* image_views;

    // Rendering
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence in_flight_fence;
    VkSemaphore image_available_semaphore;
    VkSemaphore render_finished_semaphore;

    // Dynamic rendering function pointers
    PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR;
    PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR;
};

// --- Function Prototypes ---
void init_drm(struct App* app);
void init_vulkan(struct App* app);
void render_frame(struct App* app);
void cleanup(struct App* app);

// --- Main Entry Point ---
int main() {
    struct App app = {0};

    // 1. Initialize DRM to find a connected display
    init_drm(&app);
    printf("DRM initialized for display %ux%u on %s\n", app.drm_mode.hdisplay, app.drm_mode.vdisplay, DRM_DEVICE_PATH);

    // 2. Initialize Vulkan, create a surface, device, and swapchain
    init_vulkan(&app);
    printf("Vulkan initialized successfully.\n");

    // 3. Record a command buffer to clear the screen to blue and present it
    render_frame(&app);
    printf("Frame rendered and presented. Check your display!\n");

    // Let the image stay on screen for a few seconds
    sleep(5);

    // 4. Clean up all resources
    cleanup(&app);
    printf("Cleanup complete.\n");

    return 0;
}

// --- Implementation ---

void init_drm(struct App* app) {
    // Open the primary DRM device
    app->drm_fd = open(DRM_DEVICE_PATH, O_RDWR);
    if (app->drm_fd < 0) {
        die("Could not open DRM device " DRM_DEVICE_PATH);
    }

    // Get DRM resources
    drmModeRes* resources = drmModeGetResources(app->drm_fd);
    if (!resources) {
        die("Could not get DRM resources");
    }

    // Find a connected connector
    drmModeConnector* connector = NULL;
    for (int i = 0; i < resources->count_connectors; i++) {
        drmModeConnector* current_connector = drmModeGetConnector(app->drm_fd, resources->connectors[i]);
        if (current_connector && current_connector->connection == DRM_MODE_CONNECTED) {
            connector = current_connector;
            break;
        }
        drmModeFreeConnector(current_connector);
    }

    if (!connector) {
        die("Could not find a connected connector");
    }

    // Find a suitable CRTC
    drmModeEncoder* encoder = drmModeGetEncoder(app->drm_fd, connector->encoder_id);
    if (!encoder) {
        die("Could not get encoder");
    }

    app->drm_connector_id = connector->connector_id;
    app->drm_crtc_id = encoder->crtc_id;
    memcpy(&app->drm_mode, &connector->modes[0], sizeof(drmModeModeInfo));

    // Free DRM resources
    drmModeFreeEncoder(encoder);
    drmModeFreeConnector(connector);
    drmModeFreeResources(resources);
}

void init_vulkan(struct App* app) {
    // --- 1. Create Instance ---
    const char* instance_extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_DISPLAY_EXTENSION_NAME,
        VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME // For matching DRM device to VkPhysicalDevice
    };
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "DRM Clear",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3
    };
    VkInstanceCreateInfo instance_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 3,
        .ppEnabledExtensionNames = instance_extensions
    };
    VK_CHECK(vkCreateInstance(&instance_ci, NULL, &app->instance));

    // --- 2. Select Physical Device (Robustly) ---
    struct stat drm_stat;
    if (fstat(app->drm_fd, &drm_stat) != 0) {
        die("Failed to stat DRM device");
    }
    
    PFN_vkGetPhysicalDeviceDrmPropertiesEXT vkGetPhysicalDeviceDrmPropertiesEXT =
        (PFN_vkGetPhysicalDeviceDrmPropertiesEXT)vkGetInstanceProcAddr(app->instance, "vkGetPhysicalDeviceDrmPropertiesEXT");
    if (!vkGetPhysicalDeviceDrmPropertiesEXT) {
        die("Failed to load vkGetPhysicalDeviceDrmPropertiesEXT");
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(app->instance, &device_count, NULL);
    if (device_count == 0) die("Failed to find GPUs with Vulkan support");
    VkPhysicalDevice devices[device_count];
    vkEnumeratePhysicalDevices(app->instance, &device_count, devices);

    app->physical_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceDrmPropertiesEXT drm_properties = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 device_properties = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drm_properties };
        vkGetPhysicalDeviceProperties2(devices[i], &device_properties);

        if (drm_properties.hasPrimary &&
            drm_properties.primaryMajor == major(drm_stat.st_rdev) &&
            drm_properties.primaryMinor == minor(drm_stat.st_rdev)) {
            app->physical_device = devices[i];
            printf("Found matching physical device for DRM node: %s\n", device_properties.properties.deviceName);
            break;
        }
    }

    if (app->physical_device == VK_NULL_HANDLE) {
        die("Could not find a Vulkan physical device for the given DRM fd");
    }

    // --- 3. Create DRM Surface ---
    uint32_t display_count;
    vkGetPhysicalDeviceDisplayPropertiesKHR(app->physical_device, &display_count, NULL);
    if (display_count == 0) die("No displays found on physical device. Are you running in a TTY?");
    VkDisplayPropertiesKHR display_properties[display_count];
    vkGetPhysicalDeviceDisplayPropertiesKHR(app->physical_device, &display_count, display_properties);

    VkDisplayKHR display = display_properties[0].display;

    uint32_t mode_count;
    vkGetDisplayModePropertiesKHR(app->physical_device, display, &mode_count, NULL);
    VkDisplayModePropertiesKHR mode_properties[mode_count];
    vkGetDisplayModePropertiesKHR(app->physical_device, display, &mode_count, mode_properties);

    VkDisplayModeKHR display_mode = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < mode_count; i++) {
        VkDisplayModeParametersKHR params = mode_properties[i].parameters;
        if (params.visibleRegion.width == app->drm_mode.hdisplay &&
            params.visibleRegion.height == app->drm_mode.vdisplay) {
            display_mode = mode_properties[i].displayMode;
            break;
        }
    }
    if (display_mode == VK_NULL_HANDLE) die("Could not find matching display mode");

    uint32_t plane_count;
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR(app->physical_device, &plane_count, NULL);
    VkDisplayPlanePropertiesKHR plane_properties[plane_count];
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR(app->physical_device, &plane_count, plane_properties);
    uint32_t plane_index = 0; // In a real app, find a plane compatible with the display

    VkDisplaySurfaceCreateInfoKHR surface_ci = {
        .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
        .displayMode = display_mode,
        .planeIndex = plane_index,
        .planeStackIndex = 0,
        .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .globalAlpha = 1.0f,
        .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
        .imageExtent = { app->drm_mode.hdisplay, app->drm_mode.vdisplay }
    };
    VK_CHECK(vkCreateDisplayPlaneSurfaceKHR(app->instance, &surface_ci, NULL, &app->surface));

    // --- 4. Create Logical Device and Queue ---
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(app->physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties queue_families[queue_family_count];
    vkGetPhysicalDeviceQueueFamilyProperties(app->physical_device, &queue_family_count, queue_families);

    app->queue_family_index = -1;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(app->physical_device, i, app->surface, &present_support);
        if (present_support && (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            app->queue_family_index = i;
            break;
        }
    }
    if (app->queue_family_index == -1) die("Could not find a suitable queue family");

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_feature = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        .dynamicRendering = VK_TRUE,
    };
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = app->queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    const char* device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo device_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &dynamic_rendering_feature,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = device_extensions,
    };

    VK_CHECK(vkCreateDevice(app->physical_device, &device_ci, NULL, &app->device));
    vkGetDeviceQueue(app->device, app->queue_family_index, 0, &app->queue);

    // --- 5. Create Swapchain ---
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app->physical_device, app->surface, &capabilities);

    app->swapchain_extent = capabilities.currentExtent;
    app->swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;

    VkSwapchainCreateInfoKHR swapchain_ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = app->surface,
        .minImageCount = capabilities.minImageCount > 0 ? capabilities.minImageCount : 1,
        .imageFormat = app->swapchain_format,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = app->swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE
    };
    VK_CHECK(vkCreateSwapchainKHR(app->device, &swapchain_ci, NULL, &app->swapchain));

    vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->image_count, NULL);
    app->images = malloc(sizeof(VkImage) * app->image_count);
    app->image_views = malloc(sizeof(VkImageView) * app->image_count);
    vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->image_count, app->images);

    for (uint32_t i = 0; i < app->image_count; i++) {
        VkImageViewCreateInfo iv_ci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = app->images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = app->swapchain_format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };
        VK_CHECK(vkCreateImageView(app->device, &iv_ci, NULL, &app->image_views[i]));
    }

    // --- 6. Create Command Pool and Command Buffer ---
    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = app->queue_family_index
    };
    VK_CHECK(vkCreateCommandPool(app->device, &pool_ci, NULL, &app->command_pool));

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = app->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VK_CHECK(vkAllocateCommandBuffers(app->device, &alloc_info, &app->command_buffer));

    // --- 7. Create Synchronization Objects ---
    VkSemaphoreCreateInfo semaphore_ci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
    VK_CHECK(vkCreateSemaphore(app->device, &semaphore_ci, NULL, &app->image_available_semaphore));
    VK_CHECK(vkCreateSemaphore(app->device, &semaphore_ci, NULL, &app->render_finished_semaphore));
    VK_CHECK(vkCreateFence(app->device, &fence_ci, NULL, &app->in_flight_fence));

    // --- 8. Load Dynamic Rendering Functions ---
    app->vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(app->device, "vkCmdBeginRenderingKHR");
    app->vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(app->device, "vkCmdEndRenderingKHR");
    if (!app->vkCmdBeginRenderingKHR || !app->vkCmdEndRenderingKHR) {
        die("Could not get dynamic rendering function pointers");
    }
}

void render_frame(struct App* app) {
    uint32_t image_index;

    vkWaitForFences(app->device, 1, &app->in_flight_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(app->device, 1, &app->in_flight_fence);
    VK_CHECK(vkAcquireNextImageKHR(app->device, app->swapchain, UINT64_MAX, app->image_available_semaphore, VK_NULL_HANDLE, &image_index));

    vkResetCommandBuffer(app->command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(app->command_buffer, &begin_info));

    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = app->images[image_index],
        .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
    };
    vkCmdPipelineBarrier(app->command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

    VkClearValue clear_color = {{{0.0f, 0.0f, 1.0f, 1.0f}}};
    VkRenderingAttachmentInfo color_attachment_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = app->image_views[image_index],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear_color
    };
    VkRenderingInfo rendering_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, app->swapchain_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_info,
    };
    app->vkCmdBeginRenderingKHR(app->command_buffer, &rendering_info);
    app->vkCmdEndRenderingKHR(app->command_buffer);

    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(app->command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

    VK_CHECK(vkEndCommandBuffer(app->command_buffer));

    VkSemaphore wait_semaphores[] = {app->image_available_semaphore};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {app->render_finished_semaphore};
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = wait_semaphores,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &app->command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signal_semaphores
    };
    VK_CHECK(vkQueueSubmit(app->queue, 1, &submit_info, app->in_flight_fence));

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signal_semaphores,
        .swapchainCount = 1,
        .pSwapchains = &app->swapchain,
        .pImageIndices = &image_index
    };
    VK_CHECK(vkQueuePresentKHR(app->queue, &present_info));
    vkQueueWaitIdle(app->queue);
}

void cleanup(struct App* app) {
    vkDeviceWaitIdle(app->device);

    vkDestroySemaphore(app->device, app->render_finished_semaphore, NULL);
    vkDestroySemaphore(app->device, app->image_available_semaphore, NULL);
    vkDestroyFence(app->device, app->in_flight_fence, NULL);
    vkDestroyCommandPool(app->device, app->command_pool, NULL);

    for (uint32_t i = 0; i < app->image_count; i++) {
        vkDestroyImageView(app->device, app->image_views[i], NULL);
    }
    free(app->image_views);
    free(app->images);

    vkDestroySwapchainKHR(app->device, app->swapchain, NULL);
    vkDestroyDevice(app->device, NULL);
    vkDestroySurfaceKHR(app->instance, app->surface, NULL);
    vkDestroyInstance(app->instance, NULL);

    close(app->drm_fd);
}