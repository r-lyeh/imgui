#include "egl_compute_harness.h"

#include <cstring>
#include <stdexcept>

namespace egl_compute {

Context::Context() {
    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display_ == EGL_NO_DISPLAY)
        throw std::runtime_error("eglGetDisplay failed");

    EGLint major, minor;
    if (!eglInitialize(display_, &major, &minor))
        throw std::runtime_error("eglInitialize failed");

    if (!eglBindAPI(EGL_OPENGL_API))
        throw std::runtime_error("eglBindAPI(EGL_OPENGL_API) failed");

    EGLint config_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_NONE
    };
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(display_, config_attribs, &config, 1, &num_configs) || num_configs == 0)
        throw std::runtime_error("eglChooseConfig failed");

    EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, ctx_attribs);
    if (context_ == EGL_NO_CONTEXT)
        throw std::runtime_error("eglCreateContext failed - OpenGL 4.5 may not be supported");

    /* Try surfaceless first, fall back to 1x1 pbuffer */
    if (!eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_)) {
        EGLint pbuf_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        surface_ = eglCreatePbufferSurface(display_, config, pbuf_attribs);
        if (surface_ == EGL_NO_SURFACE) {
            cleanup();
            throw std::runtime_error("Neither surfaceless nor pbuffer available");
        }
        if (!eglMakeCurrent(display_, surface_, surface_, context_)) {
            cleanup();
            throw std::runtime_error("eglMakeCurrent failed");
        }
    }
}

Context::~Context() {
    cleanup();
}

void Context::cleanup() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (surface_ != EGL_NO_SURFACE)
            eglDestroySurface(display_, surface_);
        if (context_ != EGL_NO_CONTEXT)
            eglDestroyContext(display_, context_);
        eglTerminate(display_);
    }
    display_ = EGL_NO_DISPLAY;
    context_ = EGL_NO_CONTEXT;
    surface_ = EGL_NO_SURFACE;
}

ShaderCompileResult Context::compileShader(GLenum type, const std::string& glsl_source) {
    ShaderCompileResult result;
    result.success = false;

    GLuint shader = glCreateShader(type);
    const char* src = glsl_source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    GLint log_len;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
    if (log_len > 1) {
        result.info_log.resize(log_len);
        glGetShaderInfoLog(shader, log_len, nullptr, &result.info_log[0]);
        while (!result.info_log.empty() && result.info_log.back() == '\0')
            result.info_log.pop_back();
    }

    glDeleteShader(shader);
    result.success = (status == GL_TRUE);
    return result;
}

ShaderCompileResult Context::compileComputeShader(const std::string& glsl_source) {
    return compileShader(GL_COMPUTE_SHADER, glsl_source);
}

ShaderCompileResult Context::linkProgram(const std::string& vert_source,
                                          const std::string& frag_source) {
    ShaderCompileResult result;
    result.success = false;

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* vs_src = vert_source.c_str();
    glShaderSource(vs, 1, &vs_src, nullptr);
    glCompileShader(vs);

    GLint vs_status;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &vs_status);
    if (vs_status != GL_TRUE) {
        GLint log_len;
        glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &log_len);
        if (log_len > 1) {
            result.info_log.resize(log_len);
            glGetShaderInfoLog(vs, log_len, nullptr, &result.info_log[0]);
        }
        result.info_log = "vertex compile: " + result.info_log;
        glDeleteShader(vs);
        return result;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fs_src = frag_source.c_str();
    glShaderSource(fs, 1, &fs_src, nullptr);
    glCompileShader(fs);

    GLint fs_status;
    glGetShaderiv(fs, GL_COMPILE_STATUS, &fs_status);
    if (fs_status != GL_TRUE) {
        GLint log_len;
        glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &log_len);
        if (log_len > 1) {
            result.info_log.resize(log_len);
            glGetShaderInfoLog(fs, log_len, nullptr, &result.info_log[0]);
        }
        result.info_log = "fragment compile: " + result.info_log;
        glDeleteShader(vs);
        glDeleteShader(fs);
        return result;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint link_status;
    glGetProgramiv(prog, GL_LINK_STATUS, &link_status);

    GLint log_len;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_len);
    if (log_len > 1) {
        result.info_log.resize(log_len);
        glGetProgramInfoLog(prog, log_len, nullptr, &result.info_log[0]);
        while (!result.info_log.empty() && result.info_log.back() == '\0')
            result.info_log.pop_back();
    }

    glDetachShader(prog, vs);
    glDetachShader(prog, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glDeleteProgram(prog);

    result.success = (link_status == GL_TRUE);
    return result;
}

RenderResult Context::renderToPixels(const std::string& vert_glsl,
                                      const std::string& frag_glsl,
                                      int width, int height) {
    RenderResult result;
    result.success = false;
    result.width = width;
    result.height = height;

    /* Compile and link program */
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* vs_src = vert_glsl.c_str();
    glShaderSource(vs, 1, &vs_src, nullptr);
    glCompileShader(vs);
    GLint vs_status;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &vs_status);
    if (vs_status != GL_TRUE) {
        GLint len; glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &len);
        result.error.resize(len);
        glGetShaderInfoLog(vs, len, nullptr, &result.error[0]);
        result.error = "vertex compile: " + result.error;
        glDeleteShader(vs);
        return result;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fs_src = frag_glsl.c_str();
    glShaderSource(fs, 1, &fs_src, nullptr);
    glCompileShader(fs);
    GLint fs_status;
    glGetShaderiv(fs, GL_COMPILE_STATUS, &fs_status);
    if (fs_status != GL_TRUE) {
        GLint len; glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &len);
        result.error.resize(len);
        glGetShaderInfoLog(fs, len, nullptr, &result.error[0]);
        result.error = "fragment compile: " + result.error;
        glDeleteShader(vs);
        glDeleteShader(fs);
        return result;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint link_status;
    glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
        GLint len; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        result.error.resize(len);
        glGetProgramInfoLog(prog, len, nullptr, &result.error[0]);
        result.error = "link: " + result.error;
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(prog);
        return result;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    /* Create FBO with RGBA8 renderbuffer */
    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        result.error = "Framebuffer incomplete";
        glDeleteFramebuffers(1, &fbo);
        glDeleteRenderbuffers(1, &rbo);
        glDeleteProgram(prog);
        return result;
    }

    /* Fullscreen triangle: (-1,-1), (3,-1), (-1,3) covers all of clip space */
    float tri[] = { -1.f, -1.f,  3.f, -1.f,  -1.f, 3.f };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    /* Draw */
    glViewport(0, 0, width, height);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    /* Read pixels (GL gives bottom-up rows) */
    size_t row_bytes = (size_t)width * 4;
    result.pixels.resize((size_t)width * (size_t)height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, result.pixels.data());

    /* Flip to top-down for PNG */
    std::vector<uint8_t> row(row_bytes);
    for (int y = 0; y < height / 2; y++) {
        uint8_t* top = result.pixels.data() + y * row_bytes;
        uint8_t* bot = result.pixels.data() + (height - 1 - y) * row_bytes;
        memcpy(row.data(), top, row_bytes);
        memcpy(top, bot, row_bytes);
        memcpy(bot, row.data(), row_bytes);
    }

    /* Cleanup */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);

    result.success = true;
    return result;
}

std::string Context::glVersionString() const {
    const char* v = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    return v ? v : "(null)";
}

} // namespace egl_compute
