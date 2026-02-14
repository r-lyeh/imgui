#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include <EGL/egl.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

namespace egl_compute {

struct ShaderCompileResult {
    bool success;
    std::string info_log;
};

struct RenderResult {
    bool success;
    std::string error;
    std::vector<uint8_t> pixels; /* RGBA8, top-down row order, width*height*4 bytes */
    int width;
    int height;
};

class Context {
public:
    Context();
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ShaderCompileResult compileShader(GLenum type, const std::string& glsl_source);
    ShaderCompileResult compileComputeShader(const std::string& glsl_source);
    ShaderCompileResult linkProgram(const std::string& vert_source,
                                    const std::string& frag_source);

    /* Render a fullscreen triangle with the given shaders and read back pixels */
    RenderResult renderToPixels(const std::string& vert_glsl,
                                const std::string& frag_glsl,
                                int width, int height);

    std::string glVersionString() const;

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
    ::EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;

    void cleanup();
};

} // namespace egl_compute
