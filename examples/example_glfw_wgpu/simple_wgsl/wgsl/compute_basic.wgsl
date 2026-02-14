// FILE: wgsl/compute_basic.wgsl
// EXPECT: entrypoints=compute_main:compute
// EXPECT: bindings.compute_main=tex@0:0,scale@0:1,center@0:2
// EXPECT: globals_contains=maxiters

@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> scale: f32;
@group(0) @binding(2) var<uniform> center: vec2<f32>;

fn mandelIter(p: vec2f, c: vec2f) -> vec2f{
    let newr: f32 = p.x * p.x - p.y * p.y;
    let newi: f32 = 2.0f * p.x * p.y;
    return vec2f(newr + c.x, newi + c.y);
}

const maxiters: i32 = 150;

fn mandelBailout(z: vec2f) -> i32{
    var zn = z;
    var i: i32;
    for(i = 0;i < maxiters;i = i + 1){
        zn = mandelIter(zn, z);
        if(sqrt(zn.x * zn.x + zn.y * zn.y) > 2.0f){
            break;
        }
    }
    return i;
}

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: vec2f = (vec2f(id.xy) - 640.0f) * scale + center;
    let iters = mandelBailout(p);
    if(iters == maxiters){
        textureStore(tex, id.xy, vec4<f32>(0, 0, 0, 1.0f));
    }
}
