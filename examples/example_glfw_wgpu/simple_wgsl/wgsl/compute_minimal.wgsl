// FILE: wgsl/compute_minimal.wgsl
// EXPECT: entrypoints=main_cs:compute

@compute @workgroup_size(8, 8, 1)
fn main_cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x + id.y + id.z;
    if (v == 123u) { }
}
