//#![feature(const_slice_len)]

#[macro_use]
extern crate vulkano;

#[macro_use]
extern crate vulkano_shader_derive;

#[macro_use]
extern crate lazy_static;

extern crate cgmath;
extern crate image;

use cgmath::prelude::*;
use cgmath::{Vector2, Vector3};

use std::collections::HashSet;
use std::sync::Arc;

use std::io::prelude::*;
use std::ops::Deref;

use vulkano::buffer::*;
use vulkano::command_buffer::*;
use vulkano::descriptor::*;
use vulkano::device::*;
use vulkano::format::*;
use vulkano::framebuffer::*;
use vulkano::image::*;
use vulkano::instance::*;
use vulkano::pipeline::*;
use vulkano::sync::*;
use vulkano::sampler::*;

lazy_static! {
    pub static ref LAYERS: [&'static str; 4] = [
        "VK_LAYER_LUNARG_standard_validation",
        "VK_LAYER_LUNARG_parameter_validation",
        "VK_LAYER_LUNARG_monitor",
        "VK_LAYER_LUNARG_api_dump"
    ];
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
//layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 outColor;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    outColor = inColor;
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragTexCoord;
layout (location = 1) in vec3 inColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor * texture(texSampler, fragTexCoord).rgb, 1.0);
}
"]
    struct Dummy;
}

struct QueueFamilyHolder<'a> {
    pub graphics_q: QueueFamily<'a>,
    pub compute_q: QueueFamily<'a>,
    pub transfer_q: QueueFamily<'a>,
}

struct PriorityHolder {
    pub g: f32,
    pub c: f32,
    pub t: f32,
}

impl<'a> QueueFamilyHolder<'a> {
    pub fn to_vec_with_priorities(
        &self,
        priorities: PriorityHolder,
    ) -> Vec<(QueueFamily<'a>, f32)> {
        let mut set: HashSet<u32> = HashSet::new();
        set.insert(self.graphics_q.id());

        if !set.insert(self.compute_q.id()) {
            if set.insert(self.transfer_q.id()) {
                [self.graphics_q, self.transfer_q]
                    .iter()
                    .cloned()
                    .zip([priorities.g, priorities.t].iter().cloned())
                    .collect()
            } else {
                [self.graphics_q]
                    .iter()
                    .cloned()
                    .zip([priorities.g].iter().cloned())
                    .collect()
            }
        } else {
            if set.insert(self.transfer_q.id()) {
                [self.graphics_q, self.compute_q, self.transfer_q]
                    .iter()
                    .cloned()
                    .zip([priorities.g, priorities.c, priorities.t].iter().cloned())
                    .collect()
            } else {
                [self.graphics_q, self.compute_q]
                    .iter()
                    .cloned()
                    .zip([priorities.g, priorities.c].iter().cloned())
                    .collect()
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    pos: Vector3<f32>,
    colour: Vector3<f32>,
    //normal: Vector3<f32>,
    uv: Vector2<f32>,
}

impl Vertex {
    pub fn new() -> Vertex {
        Vertex {
            pos: Vector3::new(0f32, 0f32, 0f32),
            colour: Vector3::new(0f32, 0f32, 0f32),
            //normal: Vector3::new(0f32, 0f32, 0f32),
            uv: Vector2::new(0f32, 0f32),
        }
    }

    pub fn pos(mut self, p: Vector3<f32>) -> Self {
        self.pos = p;
        self
    }

    pub fn colour(mut self, c: Vector3<f32>) -> Self {
        self.colour = c;
        self
    }

    // pub fn normal(mut self, n: Vector3<f32>) -> Self {
    //     self.normal = n;
    //     self
    // }

    pub fn uv(mut self, u: Vector2<f32>) -> Self {
        self.uv = u;
        self
    }
}

fn init_instance() -> Arc<Instance> {
    Instance::new(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions::none(),
        LAYERS.iter().cloned(),
    )
    .expect("failed to create instance")
}

fn get_valid_queue_families<'a>(physdev: &'a PhysicalDevice) -> QueueFamilyHolder<'a> {
    let graphics_q = physdev
        .queue_families()
        .find(|x| x.supports_graphics())
        .expect("No queue families with graphics support found!");
    let compute_q = physdev
        .queue_families()
        .find(|x| x.supports_compute())
        .expect("No queue families with compute support found!");
    let transfer_q = physdev
        .queue_families()
        .find(|x| x.supports_transfers() && !(x.supports_graphics() || x.supports_compute()))
        .expect("No queue families with exclusive transfer support found!");

    println!(
        "Graphics q: {:#?}
Compute q: {:#?}
Transfer q: {:#?}",
        graphics_q, compute_q, transfer_q
    );

    QueueFamilyHolder {
        graphics_q: graphics_q,
        compute_q: compute_q,
        transfer_q: transfer_q,
    }
}

fn init_pipeline<'a>(device: Arc<Device>) -> () {}

fn main() {
    let instance = init_instance();

    let pdev = PhysicalDevice::enumerate(&instance)
        .find(|x| x.name().contains("GeForce"))
        .unwrap();
    println!("Selected physical device: {}", pdev.name());

    let q_families = get_valid_queue_families(&pdev);

    let (dev, submit_queues) = Device::new(
        pdev.clone(),
        &Features::none(),
        &DeviceExtensions::none(),
        q_families.to_vec_with_priorities(PriorityHolder {
            g: 1.0,
            c: 1.0,
            t: 1.0,
        }),
    )
    .expect("couldn't create device with requested features and exts.");

    let submit_queues: Vec<Arc<Queue>> = submit_queues.collect();

    let vert_shader = vs::Shader::load(dev.clone()).expect("Could not load vertex shader.");
    let frag_shader = fs::Shader::load(dev.clone()).expect("Could not load fragment shader.");

    let vert_size = std::mem::size_of::<Vertex>();
    let index_size = std::mem::size_of::<u32>();

    let vert_index_buf = {
        let verts: [Vertex; 4] = [
            Vertex::new().pos(Vector3::new(-1., -1., 0.)),
            Vertex::new().pos(Vector3::new(-1., 1., 0.)),
            Vertex::new().pos(Vector3::new(1., 1., 0.)),
            Vertex::new().pos(Vector3::new(1., -1., 0.)),
        ];

        let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];

        //let upload_size: usize = (verts.len() * std::mem::size_of::<Vertex>() + indices.len());

        let upload_data: &[u8] = unsafe {
            std::slice::from_raw_parts(
                verts.as_ptr() as *const u8,
                verts.len() * std::mem::size_of::<Vertex>(),
            )
            .iter()
            .cloned()
            .chain(
                std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len())
                    .iter()
                    .cloned(),
            )
            .collect::<Vec<u8>>()
            .as_slice()
        };

        let (a, mut b) = ImmutableBuffer::from_iter(
            upload_data.iter().cloned(),
            BufferUsage::index_buffer()
                | BufferUsage::vertex_buffer()
                | BufferUsage::transfer_source(),
            submit_queues[0].clone(),
        )
        .expect("Could not create vertex/index buffer.");

        b.flush()
            .expect("Could not upload vertex and index buffer data.");
        b.cleanup_finished();

        BufferSlice::from_typed_buffer_access(a)
    };

    let img = image::open("default.jpg")
        .expect("Could not open image.")
        .as_rgb8()
        .expect("Could not convert to R8G8B8 format.")
        .clone()
        .into_vec();

    let img2 = {
        let mut x: Vec<[u8; 4]> = Vec::new();
        for i in 0..img.len() - 3 {
            let r = img[i + 0];
            let g = img[i + 1];
            let b = img[i + 2];
            let a = 255;
            x.push([b, g, r, a]);
        }
        x
    };

    let img = {
        let (a, mut b) = ImmutableImage::from_iter(
            img2.iter().cloned(),
            Dimensions::Dim2d {
                width: 1366,
                height: 768,
            },
            Format::B8G8R8A8Unorm,
            submit_queues[0].clone(),
        )
        .expect("Could not create immutable image.");

        b.flush().expect("Could not upload image data.");
        b.cleanup_finished();

        a
    };

    let sampler = Sampler::simple_repeat_linear_no_mipmap(dev.clone());


    let storage_img = StorageImage::with_usage(
        dev.clone(),
        Dimensions::Dim2d {
            width: 800,
            height: 600,
        },
        Format::B8G8R8A8Unorm,
        ImageUsage {
            transfer_source: true,
            transfer_destination: true,
            //color_attachment: true,

            ..ImageUsage::none()
        },
        q_families
            .to_vec_with_priorities(PriorityHolder {
                g: 0.,
                c: 0.,
                t: 0.,
            })
            .iter()
            .map(|x| x.0),
    )
    .expect("Could not create render target image.");

    let render_pass = Arc::new(
        single_pass_renderpass!(
            dev.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::B8G8R8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).expect("Could not create render pass.")
    );

    let frame_buffers = Arc::new(Framebuffer::start(render_pass.clone())
        .add(storage_img.clone())
        .expect("Could not add image to framebuffer.")
        .build()
        .expect("Could not create framebuffer.")
    );

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_single_buffer::<[u8]>()
            .vertex_shader(vert_shader.main_entry_point(), ())
            .fragment_shader(frag_shader.main_entry_point(), ())
            .viewports_scissors(
                [(
                    viewport::Viewport {
                        depth_range: 0. ..1.,
                        dimensions: [800., 600.],
                        origin: [0., 0.],
                    },
                    viewport::Scissor {
                        origin: [0, 0],
                        dimensions: [800, 600],
                    },
                )]
                    .iter()
                    .cloned(),
            )
            .depth_clamp(true)
            .depth_write(false)
            .front_face_clockwise()
            .cull_mode_back()
            .build(dev.clone())
            .expect("Could not build pipeline.")
    );

    let descset = Arc::new(
        descriptor_set::PersistentDescriptorSet::start(
            graphics_pipeline.clone(), 
            0
        ).add_sampled_image(
            img.clone(), 
            sampler
        ).expect("Could not add sampled image to descriptor.")
        .build()
        .expect("Could not create descriptor set.")
    );

    println!(
        "Device: {:#?}
Submit queues[0]: {:#?}",
        dev, submit_queues[0]
    );

    let cmd_buf = AutoCommandBufferBuilder::primary_one_time_submit(
        dev.clone(), 
        submit_queues[0].family()
    ).expect("Could not create draw command buffer.")
    .begin_render_pass(frame_buffers, false, vec![[0.0, 0.0, 1.0, 1.0].into()])
    .expect("Could not record render pass begin command.")
    .draw_indexed(
        graphics_pipeline.clone(), 
        &DynamicState::none(), 
        vert_index_buf.slice(
            0..(4 * vert_size)
        ).expect("Could not index vertex range. "), 
        vert_index_buf.slice(
            (4 * vert_size)..(4 * vert_size + 6 * index_size)
        ).expect("Could not index index range. "), 
        descset.clone(), 
        ()
    );

    loop {
        let mut x = String::new();
        let y = std::io::stdin();
        if y.lock().read_line(&mut x).is_ok() {
            break;
        }
    }
}
