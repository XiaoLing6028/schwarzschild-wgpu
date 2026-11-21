//! Real-time Schwarzschild black hole, ray traced on the GPU.
//!
//! How a frame is drawn:
//!   CPU (this file)                          GPU (blackhole.wgsl)
//!   ───────────────                          ────────────────────
//!   egui builds the UI and reads the mouse
//!   → we compute the camera basis
//!   → pack everything into `Uniforms`
//!   prepare(): queue.write_buffer(uniforms)  → uniform buffer in VRAM
//!   paint():   set pipeline + bind group,
//!              draw(0..3)                    → vertex shader: 1 full-screen triangle
//!                                            → fragment shader: 1 ray per pixel
//!   egui then draws its own widgets on top in the same render pass.
//!
//! The heavy work is all in the fragment shader. The CPU only uploads 112
//! bytes per frame.

use eframe::{egui, egui_wgpu, wgpu};
use std::time::Instant;

/// Data sent to the shader each frame. Its layout must match `struct Uniforms`
/// in the WGSL file. We use only `vec4`s, so there are no alignment surprises:
/// WGSL aligns `vec3` to 16 bytes, which is a classic source of bugs.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cam_pos: [f32; 4],   // xyz, w = tan(fov/2)
    cam_right: [f32; 4], // xyz, w = aspect
    cam_up: [f32; 4],    // xyz, w = time
    cam_fwd: [f32; 4],   // xyz, w = Schwarzschild radius
    disk: [f32; 4],      // inner, outer, lensing, doppler
    params: [f32; 4],    // max steps, disk brightness, star brightness, step scale
    misc: [f32; 4],      // srgb flag, disk rotation speed, unused, unused
}

/// GPU objects created once at startup. They live inside egui-wgpu's
/// `CallbackResources` type map, so the paint callback can reach them.
struct GpuResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

/// User-tweakable simulation parameters (the sliders).
struct Params {
    mass: f32,
    disk_inner: f32,
    disk_outer: f32,
    lensing: f32,
    doppler: f32,
    disk_brightness: f32,
    star_brightness: f32,
    disk_speed: f32,
    fov_deg: f32,
    max_steps: u32,
    step_scale: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mass: 1.0,
            disk_inner: 6.0, // ISCO = 3 r_s = 6M
            disk_outer: 20.0,
            lensing: 1.0,
            doppler: 1.0,
            disk_brightness: 1.0,
            star_brightness: 1.0,
            disk_speed: 1.0,
            fov_deg: 60.0,
            max_steps: 300,
            step_scale: 0.06,
        }
    }
}

struct BlackHoleApp {
    params: Params,
    // Orbit camera in spherical coordinates around the origin.
    yaw: f32,
    pitch: f32,
    distance: f32,
    start: Instant,
    fps: f32,
    target_is_srgb: bool,
}

impl BlackHoleApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        // eframe made the wgpu device, queue, and swapchain. We borrow them.
        let rs = cc.wgpu_render_state.as_ref()?;
        let device = &rs.device;

        // 1. Shader module: WGSL compiled at runtime by wgpu (naga).
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blackhole shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blackhole.wgsl").into()),
        });

        // 2. Uniform buffer: a small block of GPU memory that the CPU rewrites each frame.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blackhole uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Bind group layout: tells the pipeline "binding 0 is a uniform
        //    buffer visible to both shader stages".
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blackhole bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // 4. Bind group: the actual buffer plugged into that layout slot.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blackhole bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blackhole pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // 5. Render pipeline: shaders + fixed-function state (no vertex buffers,
        //    no depth, no blending). It must output to the same texture format
        //    egui renders to, because we draw inside egui's render pass.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blackhole pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: rs.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(), // triangle list
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        rs.renderer.write().callback_resources.insert(GpuResources {
            pipeline,
            bind_group,
            uniform_buffer,
        });

        Some(Self {
            params: Params::default(),
            yaw: 0.0,
            pitch: 8f32.to_radians(), // slightly above the disk: the classic lensed-disk view
            distance: 32.0,
            start: Instant::now(),
            fps: 0.0,
            target_is_srgb: rs.target_format.is_srgb(),
        })
    }

    /// Camera position and orthonormal basis (right, up, forward) looking at the origin.
    fn camera_basis(&self) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let pos = [self.distance * cp * sy, self.distance * sp, self.distance * cp * cy];
        let fwd = normalize([-pos[0], -pos[1], -pos[2]]);
        let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        (pos, right, up, fwd)
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.params;
        let rs = 2.0 * p.mass;

        ui.heading("Black hole");
        ui.label(format!("{:.0} FPS", self.fps));
        ui.separator();

        ui.add(egui::Slider::new(&mut p.mass, 0.2..=3.0).text("Mass M"));
        ui.label(format!("Event horizon r_s = 2M = {rs:.2}"));
        ui.add(egui::Slider::new(&mut p.lensing, 0.0..=3.0).text("Lensing strength"));

        ui.separator();
        ui.label("Accretion disk");
        ui.add(egui::Slider::new(&mut p.disk_inner, 1.0..=30.0).text("Inner radius"));
        ui.add(egui::Slider::new(&mut p.disk_outer, 2.0..=60.0).text("Outer radius"));
        if ui.button(format!("Snap inner to ISCO (6M = {:.1})", 6.0 * p.mass)).clicked() {
            p.disk_inner = 6.0 * p.mass;
        }
        ui.add(egui::Slider::new(&mut p.disk_brightness, 0.0..=4.0).text("Brightness"));
        ui.add(egui::Slider::new(&mut p.doppler, 0.0..=1.5).text("Doppler beaming"));
        ui.add(egui::Slider::new(&mut p.disk_speed, 0.0..=5.0).text("Rotation speed"));

        ui.separator();
        ui.label("Rendering");
        ui.add(egui::Slider::new(&mut p.star_brightness, 0.0..=3.0).text("Stars"));
        ui.add(egui::Slider::new(&mut p.fov_deg, 20.0..=110.0).text("FOV (deg)"));
        ui.add(egui::Slider::new(&mut p.max_steps, 50..=1000).text("Max ray steps"));
        ui.add(egui::Slider::new(&mut p.step_scale, 0.01..=0.2).text("Step size").logarithmic(true));

        ui.separator();
        if ui.button("Reset").clicked() {
            *p = Params::default();
            self.yaw = 0.0;
            self.pitch = 8f32.to_radians();
            self.distance = 32.0;
        }
        ui.small("Left-drag: orbit\nRight-drag / scroll: zoom");

        // Keep the parameters consistent: the disk can't start inside the horizon
        // and must have positive width.
        p.disk_inner = p.disk_inner.max(rs * 1.01);
        p.disk_outer = p.disk_outer.max(p.disk_inner + 0.5);
    }
}

impl eframe::App for BlackHoleApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Smoothed FPS readout.
        let dt = ui.input(|i| i.unstable_dt).max(1e-4);
        self.fps = if self.fps == 0.0 { 1.0 / dt } else { self.fps * 0.95 + 0.05 / dt };

        egui::Panel::left("controls")
            .resizable(false)
            .default_size(280.0)
            .show(ui, |ui| self.controls_ui(ui));

        egui::CentralPanel::no_frame().show(ui, |ui| {
            // Reserve the whole central area as a draggable canvas.
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

            // --- Mouse controls ---
            if response.dragged_by(egui::PointerButton::Primary) {
                let d = response.drag_delta();
                self.yaw -= d.x * 0.006;
                self.pitch = (self.pitch + d.y * 0.006).clamp(-1.55, 1.55);
            }
            if response.dragged_by(egui::PointerButton::Secondary) {
                self.distance *= (response.drag_delta().y * 0.01).exp();
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.distance *= (-scroll * 0.002).exp();
            }
            let rs = 2.0 * self.params.mass;
            self.distance = self.distance.clamp(rs * 1.5, 300.0);

            // --- Pack uniforms ---
            let (pos, right, up, fwd) = self.camera_basis();
            let p = &self.params;
            let uniforms = Uniforms {
                cam_pos: [pos[0], pos[1], pos[2], (p.fov_deg.to_radians() * 0.5).tan()],
                cam_right: [right[0], right[1], right[2], rect.width() / rect.height().max(1.0)],
                cam_up: [up[0], up[1], up[2], self.start.elapsed().as_secs_f32()],
                cam_fwd: [fwd[0], fwd[1], fwd[2], rs],
                disk: [p.disk_inner, p.disk_outer, p.lensing, p.doppler],
                params: [p.max_steps as f32, p.disk_brightness, p.star_brightness, p.step_scale],
                misc: [if self.target_is_srgb { 1.0 } else { 0.0 }, p.disk_speed, 0.0, 0.0],
            };

            // Queue our custom wgpu draw. egui runs it when it renders this rect.
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                BlackHoleCallback { uniforms },
            ));
        });

        // Animate continuously. With vsync on, this runs at the display's refresh rate.
        ui.ctx().request_repaint();
    }
}

/// One frame's worth of draw work, handed to egui-wgpu.
struct BlackHoleCallback {
    uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for BlackHoleCallback {
    /// Runs before the render pass starts. This is the place to upload data.
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res: &GpuResources = resources.get().expect("GPU resources missing");
        queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        Vec::new()
    }

    /// Runs inside egui's render pass. egui has already set the viewport and
    /// scissor to our rect, so the full-screen triangle fills only the 3D view.
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let res: &GpuResources = resources.get().expect("GPU resources missing");
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // 3 vertices, 1 instance, no vertex buffer
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Schwarzschild Black Hole"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "black-hole-sim",
        options,
        Box::new(|cc| {
            let app = BlackHoleApp::new(cc).ok_or("wgpu render state unavailable")?;
            Ok(Box::new(app))
        }),
    )
}
