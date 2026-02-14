cmake_minimum_required(VERSION 3.16)
project(simple_wgsl C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(WGSL_BUILD_TESTS "Build tests" ON)
option(WGSL_BUILD_EXPRESSION_TESTS "Build expression golden-file tests (requires expressions/ data)" OFF)
option(WGSL_USE_ASAN "Enable Address Sanitizer" OFF)
option(WGSL_BUILD_EGL_TESTS "Build EGL/OpenGL compute shader validation tests" OFF)
option(WGSL_BUILD_METAL_EXAMPLES "Build Metal/MSL example programs (macOS only)" OFF)
option(WGSL_BUILD_FUZZ "Build libFuzzer fuzz targets" OFF)

if(WGSL_USE_ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

add_library(wgsl_compiler STATIC
    wgsl_parser.c
    wgsl_resolve.c
    wgsl_lower.c
    wgsl_raise.c
    glsl_parser.c
    ssir.c
    ssir_to_spirv.c
    ssir_to_wgsl.c
    ssir_to_glsl.c
    ssir_to_msl.c
    ssir_to_hlsl.c
    spirv_to_ssir.c
    msl_parser.c
)

target_include_directories(wgsl_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_path(SPIRV_INCLUDE_DIR spirv/unified1/spirv.h
    HINTS
        $ENV{VULKAN_SDK}/include
        /usr/include
)
if(SPIRV_INCLUDE_DIR)
    target_include_directories(wgsl_compiler PUBLIC ${SPIRV_INCLUDE_DIR})
else()
    FetchContent_Declare(
        spirv_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG        vulkan-sdk-1.4.341
        GIT_SHALLOW    True
    )
    FetchContent_MakeAvailable(spirv_headers)
    target_include_directories(wgsl_compiler PUBLIC ${spirv_headers_SOURCE_DIR}/include)
endif()

add_executable(example example/example.c)
target_link_libraries(example wgsl_compiler)

if(WGSL_BUILD_METAL_EXAMPLES AND APPLE)
    enable_language(OBJC)

    add_executable(metal_compute example/metal_compute.m)
    target_link_libraries(metal_compute wgsl_compiler "-framework Metal" "-framework Foundation")

    add_executable(metal_triangle example/metal_triangle.m)
    target_link_libraries(metal_triangle wgsl_compiler "-framework Metal" "-framework Foundation")

    add_executable(msl_roundtrip example/msl_roundtrip.m)
    target_link_libraries(msl_roundtrip wgsl_compiler "-framework Metal" "-framework Foundation")
endif()

if(WGSL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WGSL_BUILD_FUZZ)
    add_subdirectory(fuzz)
endif()
