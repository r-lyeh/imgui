// FILE: glsl/compute_basic.glsl
// GLSL equivalent of wgsl/compute_basic.wgsl (adapted: storage buffer instead of texture)
#version 450

layout(local_size_x = 1) in;

layout(std430, set = 0, binding = 0) buffer OutputBuf {
    float data[];
} output_buf;

vec2 mandelIter(vec2 p, vec2 c) {
    float newr = p.x * p.x - p.y * p.y;
    float newi = 2.0 * p.x * p.y;
    return vec2(newr + c.x, newi + c.y);
}

const int maxiters = 150;

int mandelBailout(vec2 z) {
    vec2 zn = z;
    int i;
    for (i = 0; i < maxiters; i = i + 1) {
        zn = mandelIter(zn, z);
        if (sqrt(zn.x * zn.x + zn.y * zn.y) > 2.0) {
            break;
        }
    }
    return i;
}

void main() {
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;
    float scale = 3.5 / 256.0;
    float cx = float(px) * scale - 2.5;
    float cy = float(py) * scale - 1.75;
    vec2 c = vec2(cx, cy);
    int iters = mandelBailout(c);
    output_buf.data[py * 256u + px] = float(iters) / float(maxiters);
}
