#ifndef WGVK_CONFIG_H_INCLUDED
#define WGVK_CONFIG_H_INCLUDED


// Detect and define DEFAULT_BACKEND based on the target platform
#if defined(_WIN32) || defined(_WIN64)
    // Windows platform detected
    // If msvc, default to DirectX, otherwise (e.g. w64devkit) use vulkan
    #ifdef _MSC_VER
        #define DEFAULT_BACKEND WGPUBackendType_D3D12
    #else
        #define DEFAULT_BACKEND WGPUBackendType_Vulkan
    #endif
#elif defined(__APPLE__) && defined(__MACH__)
    // Apple platform detected (macOS, iOS, etc.)
    #define DEFAULT_BACKEND WGPUBackendType_Metal

#elif defined(__linux__) || defined(__unix__) || defined(__FreeBSD__)
    // Linux or Unix-like platform detected
    #define DEFAULT_BACKEND WGPUBackendType_Vulkan

#else
    // Fallback to Vulkan for any other platforms
    #define DEFAULT_BACKEND WGPUBackendType_Vulkan
    #pragma message("Unknown platform. Defaulting to Vulkan as the backend.")
#endif

#ifndef MAX_COLOR_ATTACHMENTS
    #define MAX_COLOR_ATTACHMENTS 4
#endif
#ifndef USE_VMA_ALLOCATOR
    #define USE_VMA_ALLOCATOR 0
#endif
#ifndef VULKAN_USE_DYNAMIC_RENDERING
    #define VULKAN_USE_DYNAMIC_RENDERING 1
#endif
#ifndef VULKAN_ENABLE_RAYTRACING
    #define VULKAN_ENABLE_RAYTRACING 1
#endif
#if !defined(RL_MALLOC) && !defined(RL_CALLOC) && !defined(RL_REALLOC) && !defined(RL_FREE)
#define RL_MALLOC  malloc
#define RL_CALLOC  calloc
#define RL_REALLOC realloc
#define RL_FREE    free
#elif !defined(RL_MALLOC) || !defined(RL_CALLOC) || !defined(RL_REALLOC) || !defined(RL_FREE)
#error Must define all of RL_MALLOC, RL_CALLOC, RL_REALLOC and RL_FREE or none
#endif


#endif // CONFIG_H_INCLUDED


