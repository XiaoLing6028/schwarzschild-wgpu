// blackhole.wgsl — per-pixel ray tracer for a Schwarzschild black hole.
//
// GPU pipeline (what runs where):
//   1. VERTEX STAGE runs 3 times per frame. It emits one oversized triangle
//      that covers the whole viewport. There are no vertex buffers; the
//      positions come from @builtin(vertex_index).
//   2. The RASTERIZER turns that triangle into fragments, one per pixel, and
//      interpolates `ndc` across them.
//   3. FRAGMENT STAGE runs once per pixel, in parallel across the GPU. Each
//      invocation shoots one light ray from the camera and bends it step by step
//      around the black hole. It stops when the ray falls into the horizon,
//      escapes to the starfield, or passes through the accretion disk enough
//      times to become opaque.
//
// Units: geometric (G = c = 1). The Schwarzschild radius is r_s = 2M.
// The disk lies in the y = 0 plane.

struct Uniforms {
    cam_pos:   vec4<f32>, // xyz = camera position,  w = tan(fov/2)
    cam_right: vec4<f32>, // xyz = camera right axis, w = aspect ratio (w/h)
    cam_up:    vec4<f32>, // xyz = camera up axis,    w = time in seconds
    cam_fwd:   vec4<f32>, // xyz = camera forward,    w = r_s (Schwarzschild radius)
    disk:      vec4<f32>, // x = inner radius, y = outer radius, z = lensing strength, w = Doppler strength
    params:    vec4<f32>, // x = max steps, y = disk brightness, z = star brightness, w = step scale
    misc:      vec4<f32>, // x = 1 if the target is sRGB (GPU encodes gamma), y = disk rotation speed
};

// One uniform buffer, bound at group 0 / binding 0. The CPU rewrites it every frame.
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>, // -1..1 across the viewport, interpolated per pixel
};

// ---------------------------------------------------------------------------
// Vertex stage: full-screen triangle.
// vertex 0 -> (-1,-1), vertex 1 -> (3,-1), vertex 2 -> (-1,3).
// The part outside the viewport is clipped away for free.
// ---------------------------------------------------------------------------
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    let p = uv * 2.0 - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// ---------------------------------------------------------------------------
// Small procedural helpers. Generating stars and disk texture in the shader
// means we need no texture files.
// ---------------------------------------------------------------------------
fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// Smooth 3D value noise.
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, w.x);
    let x10 = mix(n010, n110, w.x);
    let x01 = mix(n001, n101, w.x);
    let x11 = mix(n011, n111, w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Background sky for a ray that escaped in direction `d` (unit vector).
// Two layers of hashed star cells, plus a faint galactic band.
fn starfield(d: vec3<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    for (var layer = 0; layer < 2; layer++) {
        let scale = select(90.0, 220.0, layer == 1);
        let p = d * scale;
        let cell = floor(p);
        let f = fract(p);
        let rnd = hash33(cell);
        let star_pos = 0.2 + 0.6 * rnd;               // where the star sits in its cell
        let dist = length(f - star_pos);
        let bright = pow(hash31(cell + 17.0), 18.0);  // most cells are nearly empty
        let tint = mix(vec3<f32>(1.0, 0.75, 0.55), vec3<f32>(0.65, 0.8, 1.0), rnd.z);
        col += tint * bright * smoothstep(0.18, 0.0, dist) * 4.0;
    }
    // Milky-way-like glow around a tilted great circle.
    let band_n = normalize(vec3<f32>(0.25, 1.0, 0.35));
    let b = dot(d, band_n);
    let band = exp(-b * b * 18.0) * (0.4 + 0.6 * noise3(d * 6.0));
    col += vec3<f32>(0.35, 0.3, 0.45) * band * 0.12;
    return col * u.params.z;
}

// Rough blackbody-like colour ramp. t ≈ 1 is white-hot, below is red/orange, above is blue.
fn blackbody(t: f32) -> vec3<f32> {
    var c = mix(vec3<f32>(1.0, 0.22, 0.02), vec3<f32>(1.0, 0.78, 0.5), smoothstep(0.15, 0.75, t));
    c = mix(c, vec3<f32>(0.95, 0.95, 1.0), smoothstep(0.75, 1.2, t));
    c = mix(c, vec3<f32>(0.6, 0.75, 1.0), smoothstep(1.2, 2.2, t));
    return c;
}

// Emitted colour (rgb) and opacity (a) where the ray `dir` crosses the disk at `hit`.
fn disk_sample(hit: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let rs = u.cam_fwd.w;
    let rin = u.disk.x;
    let rout = u.disk.y;
    let r = length(hit.xz);
    if (r < rin || r > rout) {
        return vec4<f32>(0.0);
    }

    // Thin-disk temperature profile T(r) ∝ r^-3/4 (1 - sqrt(rin/r))^1/4,
    // normalised so the hottest ring is about 1.
    let x = rin / r;
    let temp = 2.05 * pow(x, 0.75) * pow(max(1.0 - sqrt(x), 0.0), 0.25) + 0.05;

    // Keplerian orbital speed v = sqrt(M/r) (as a fraction of c), moving counter-clockwise seen from +y.
    let m = 0.5 * rs;
    let beta = min(sqrt(m / r), 0.95) * u.disk.w;
    let v_dir = normalize(vec3<f32>(-hit.z, 0.0, hit.x));
    let gamma = 1.0 / sqrt(1.0 - beta * beta);
    // Light reaching the camera travels opposite to the traced ray, so n = -dir.
    // Doppler factor g > 1 on the side moving toward us (blueshift), < 1 on the receding side.
    var g = 1.0 / (gamma * (1.0 - beta * dot(v_dir, -dir)));
    // Gravitational redshift from climbing out of the potential well.
    g *= sqrt(max(1.0 - rs / r, 0.02));

    // Swirling texture. The angle advances at the Keplerian angular velocity,
    // so inner rings orbit faster than outer ones.
    let omega = sqrt(m / (r * r * r)) * u.misc.y;
    let phi = atan2(hit.z, hit.x) - omega * u.cam_up.w * 4.0;
    let q = vec3<f32>(cos(phi) * 3.0, sin(phi) * 3.0, r * 0.9);
    let streaks = 0.55 * noise3(q) + 0.45 * noise3(q * 2.7 + 5.0);
    let density = 0.35 + 0.9 * streaks;

    // Fade both edges so the disk has no hard rim.
    let edge = smoothstep(rin, rin + 0.4, r) * (1.0 - smoothstep(rout * 0.75, rout, r));

    // Relativistic beaming: observed intensity scales roughly as g^3–g^4.
    let intensity = temp * temp * temp * temp * pow(g, 3.5) * density * u.params.y;
    let color = blackbody(temp * g * 0.8) * intensity * 0.9;
    let alpha = clamp(edge * density * 0.95, 0.0, 1.0);
    return vec4<f32>(color * edge, alpha);
}

// Photon acceleration. For a Schwarzschild null geodesic, this "Newtonian"
// force gives exactly the right orbit shapes (it reproduces the Binet equation
// u'' + u = 3Mu²):   a = -(3/2) r_s h² r̂ / r⁴, with h = |r × v|, which is conserved.
// `lensing` scales the force: 0 = straight rays, 1 = physical, >1 = exaggerated.
fn accel(p: vec3<f32>, h2: f32) -> vec3<f32> {
    let r2 = dot(p, p);
    let r5 = r2 * r2 * sqrt(r2);
    return -1.5 * u.cam_fwd.w * h2 * p / r5 * u.disk.z;
}

// ---------------------------------------------------------------------------
// Fragment stage: one ray per pixel.
// ---------------------------------------------------------------------------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tan_half_fov = u.cam_pos.w;
    let aspect = u.cam_right.w;
    let rs = u.cam_fwd.w;

    // Build the primary ray from the pinhole camera basis.
    var pos = u.cam_pos.xyz;
    var vel = normalize(u.cam_fwd.xyz
        + in.ndc.x * aspect * tan_half_fov * u.cam_right.xyz
        + in.ndc.y * tan_half_fov * u.cam_up.xyz);

    // Angular momentum squared (conserved along the ray).
    let hv = cross(pos, vel);
    let h2 = dot(hv, hv);

    let escape_r = max(length(pos) * 1.2, u.disk.y * 1.5) + 10.0;
    let max_steps = i32(u.params.x);
    let step_scale = u.params.w;

    var color = vec3<f32>(0.0);
    var transmittance = 1.0; // how much light from behind still gets through
    var captured = false;

    for (var i = 0; i < max_steps; i++) {
        let r = length(pos);
        if (r < rs) {           // inside the event horizon: no light escapes
            captured = true;
            break;
        }
        if (r > escape_r && dot(pos, vel) > 0.0) {
            break;              // heading off to infinity
        }

        // Adaptive step: long strides far away, short strides near the hole.
        let speed = length(vel);
        let dt = clamp(step_scale * (r - 0.5 * rs), 0.01, 4.0) / speed;

        // Leapfrog (kick-drift-kick) integration of the geodesic.
        let prev = pos;
        vel += 0.5 * dt * accel(pos, h2);
        pos += dt * vel;
        vel += 0.5 * dt * accel(pos, h2);

        // Did this step cross the disk plane (y = 0)? Find where, then composite front-to-back.
        if (prev.y * pos.y < 0.0) {
            let t = prev.y / (prev.y - pos.y);
            let hit = mix(prev, pos, t);
            let d = disk_sample(hit, normalize(vel));
            color += transmittance * d.rgb * d.a;
            transmittance *= 1.0 - d.a;
            if (transmittance < 0.01) {
                break;
            }
        }
    }

    // Whatever light is still transmitted comes from the (lensed) sky,
    // unless the ray ended in the black hole.
    if (!captured) {
        color += transmittance * starfield(normalize(vel));
    }

    // Tone mapping: compress HDR values into 0..1 without hard clipping.
    color = vec3<f32>(1.0) - exp(-color);

    // egui on wgpu usually uses a non-sRGB swapchain format, so we gamma-encode
    // ourselves. If the target is sRGB, the hardware does it on write.
    if (u.misc.x < 0.5) {
        color = pow(color, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(color, 1.0);
}
