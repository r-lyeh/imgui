include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

find_package(Vulkan)

set(TEST_SOURCES
    parser_test.cpp
    resolver_test.cpp
    lower_test.cpp
    raise_test.cpp
    ssir_raise_test.cpp
    spirv_to_ssir_test.cpp
    integration_test.cpp
    glsl_parser_test.cpp
    glsl_raise_test.cpp
    ssir_to_hlsl_test.cpp
    ssir_to_msl_test.cpp
    msl_parser_test.cpp
)

if(WGSL_BUILD_EXPRESSION_TESTS)
    list(APPEND TEST_SOURCES expression_test.cpp)
endif()

if(Vulkan_FOUND)
    list(APPEND TEST_SOURCES
        vulkan_compute_harness.cpp
        vulkan_compute_test.cpp
        vulkan_graphics_harness.cpp
        vulkan_graphics_test.cpp
        vulkan_graphics_complex_test.cpp
        vulkan_graphics_scene_test.cpp
    )
endif()

if(WGSL_BUILD_EGL_TESTS)
    find_package(OpenGL COMPONENTS EGL OpenGL)
    if(OpenGL_EGL_FOUND)
        list(APPEND TEST_SOURCES
            egl_compute_harness.cpp
            egl_compute_test.cpp
            egl_graphics_test.cpp
        )
    else()
        message(WARNING "WGSL_BUILD_EGL_TESTS enabled but EGL not found")
    endif()
endif()

add_executable(wgsl_tests ${TEST_SOURCES})

target_link_libraries(wgsl_tests
    wgsl_compiler
    GTest::gtest
    GTest::gtest_main
)

if(Vulkan_FOUND)
    target_link_libraries(wgsl_tests Vulkan::Vulkan)
    target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_VULKAN=1)
endif()

if(WGSL_BUILD_EGL_TESTS AND OpenGL_EGL_FOUND)
    target_link_libraries(wgsl_tests OpenGL::EGL OpenGL::OpenGL)
    target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_EGL=1)
endif()

target_include_directories(wgsl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WGSL_BUILD_EXPRESSION_TESTS)
    target_compile_definitions(wgsl_tests PRIVATE
        WGSL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
endif()

include(GoogleTest)
gtest_discover_tests(wgsl_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES LABELS "unit"
)

add_test(NAME all_tests COMMAND wgsl_tests)
