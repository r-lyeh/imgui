// FILE: wgsl/transitive_vertex.wgsl
// EXPECT: entrypoints=main_vs:vertex
// EXPECT: bindings.main_vs=U@1:0

@group(1) @binding(0) var<uniform> U: mat4x4f;

fn useU(v: vec4f) -> vec4f {
    return U * v;
}

fn middle(v: vec4f) -> vec4f {
    return useU(v);
}

@vertex
fn main_vs(@location(0) p: vec3f) -> @builtin(position) vec4f {
    let v = vec4f(p, 1.0f);
    return middle(v);
}
