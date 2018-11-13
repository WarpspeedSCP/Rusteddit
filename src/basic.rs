use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder, Window};

// use cgmath::prelude::*;
// use cgmath::{Vector2, Vector3};

use std::collections::HashSet;
use std::sync::Arc;

use crate::shaders;

use std::io::prelude::*;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{Dimensions, ImageUsage, ImmutableImage, StorageImage, SwapchainImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::{Viewport, Scissor};
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{acquire_next_image, AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;



lazy_static! {
    pub static ref LAYERS: &'static [&'static str] = &[
        "VK_LAYER_LUNARG_standard_validation",
//        "VK_LAYER_LUNARG_monitor",
//        "VK_LAYER_LUNARG_api_dump"
    ];
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
    pub inPosition: [f32; 3],
    pub inColour: [f32; 3],
    //normal: Vector3<f32>,
    pub inTexCoord: [f32; 2],
}

//trace_macros!(true);
impl_vertex!(Vertex, inPosition, inColour, inTexCoord);
//trace_macros!(false);

impl Vertex {
    pub fn new() -> Vertex {
        Vertex {
            inPosition: [0.; 3],
            inColour: [1.; 3],
            //normal: Vector3::new(0f32, 0f32, 0f32),
            inTexCoord: [0.; 2],
        }
    }

    pub fn pos(mut self, p: [f32; 3]) -> Self {
        self.inPosition = p;
        self
    }

    pub fn colour(mut self, c: [f32; 3]) -> Self {
        self.inColour = c;
        self
    }

    // pub fn normal(mut self, n: [f32; 3]) -> Self {
    //     self.normal = n;
    //     self
    // }

    pub fn uv(mut self, u: [f32; 2]) -> Self {
        self.inTexCoord = u;
        self
    }
}

fn init_instance() -> Arc<Instance> {
    Instance::new(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions {
            khr_surface: true,
            khr_xcb_surface: true,
//            khr_xlib_surface: true,
            ..InstanceExtensions::none()
        },
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

//fn init_pipeline<'a>(device: Arc<Device>) -> () {}

pub fn main() {
    let instance = init_instance();

    let pdev = PhysicalDevice::enumerate(&instance)
        .find(|x| x.name().contains("GeForce"))
        .unwrap();
    println!("Selected physical device: {}", pdev.name());

    let q_families = get_valid_queue_families(&pdev);

    let (dev, submit_queues) = Device::new(
        pdev.clone(),
        &Features {
            depth_clamp: true,
            ..Features::none()
        },
        &DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        },
        q_families.to_vec_with_priorities(PriorityHolder {
            g: 1.0,
            c: 1.0,
            t: 1.0,
        }),
    )
    .expect("couldn't create device with requested features and exts.");

    let submit_queues: Vec<Arc<Queue>> = submit_queues.collect();
    println!("{}", submit_queues.len());

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_resizable(false)
        .with_dimensions((800u32, 600u32).into())
        .build_vk_surface(&events_loop, instance.clone())
        .expect("Could not create vulkan window surface.");
    if !surface.is_supported(submit_queues[0].family()).expect("Could not retrieve surface capabilities.") {
        return;
    }

    let surface_caps = surface
        .capabilities(pdev.clone())
        .expect("Could not get capabilities of surface.");

    let dimensions = surface_caps.current_extent.unwrap_or([800, 600]);
    let alpha = surface_caps
        .supported_composite_alpha
        .iter()
        .next()
        .unwrap();
    let format = Format::B8G8R8A8Unorm; //surface_caps.supported_formats[0].0;

    let (mut swapchain, mut swapchain_images) = Swapchain::new(
        dev.clone(),
        surface.clone(),
        if (3 >= surface_caps.min_image_count) && (3 <= surface_caps.max_image_count.unwrap_or(4)) {
            3
        } else {
            surface_caps.min_image_count
        },
        format,
        dimensions,
        1,
        surface_caps.supported_usage_flags,
        &submit_queues[0],
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        true,
        None,
    )
    .expect("failed to create swapchain");

    let vert_shader = shaders::basic_vs::Shader::load(dev.clone()).expect("Could not load vertex shader.");
    let frag_shader = shaders::basic_fs::Shader::load(dev.clone()).expect("Could not load fragment shader.");

    let vert_size = std::mem::size_of::<Vertex>();
    let index_size = std::mem::size_of::<u32>();

    let vert_buf = {
        let verts = &[
            Vertex::new().pos([-1., -1., 0.]).uv([0., 0.]), //.colour([1., 0., 0.]),
            Vertex::new().pos([-1., 1., 0.]).uv([0., 1.]),  //.colour([1., 1., 0.]),
            Vertex::new().pos([1., 1., 0.]).uv([1., 1.]),   //.colour([0., 1., 1.]),
            Vertex::new().pos([1., -1., 0.]).uv([1., 0.]),  //.colour([0., 0., 1.]),
        ];

        let (a, mut b) = ImmutableBuffer::from_iter(
            verts.iter().cloned(),
            BufferUsage::vertex_buffer(),
            submit_queues[1].clone(),
        )
        .expect("Could not create vertex/index buffer.");

        b.flush()
            .expect("Could not upload vertex and index buffer data.");
        b.cleanup_finished();

        a
    };

    let index_buf = {
        let indices = &[0u32, 1, 2, 2, 3, 0];

        let (a, mut b) = ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            BufferUsage::index_buffer(),
            submit_queues[1].clone(),
        )
        .expect("Could not create vertex/index buffer.");

        b.flush()
            .expect("Could not upload vertex and index buffer data.");
        b.cleanup_finished();

        a
    };

    let mut w = 0;
    let mut h = 0;

    let mut previous_frame_end: Box<GpuFuture>;

    let img = {
        let i = image::open("default.jpg").expect("Could not open image.");
        println!("{:#?}", i.color());

        let i = i
            .as_rgb8()
            .expect("Could not convert to R8G8B8 format.")
            .clone();
        w = i.dimensions().0;
        h = i.dimensions().1;
        i
    };
    let img = {
        let mut x: Vec<[u8; 4]> = Vec::new();
        for i in img.pixels() {
            let r = i[0];
            let g = i[1];
            let b = i[2];
            let a = 0xFF; //i[3];
                          //let res = (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | (a as u32) << 0;

            x.push([b, g, r, a]);
        }

        let (a, mut b) = ImmutableImage::from_iter(
            x.iter().cloned(),
            Dimensions::Dim2d {
                width: w,
                height: h,
            },
            Format::B8G8R8A8Unorm,
            submit_queues[0].clone(),
        )
        .expect("Could not create immutable image.");

        b.flush().expect("Could not upload image data.");

        previous_frame_end = Box::new(b) as Box<_>;

        a
    };

    let sampler = Sampler::simple_repeat_linear_no_mipmap(dev.clone());

    let storage_img = StorageImage::with_usage(
        dev.clone(),
        Dimensions::Dim2d {
            width: w,
            height: h,
        },
        Format::B8G8R8A8Unorm,
        ImageUsage {
            transfer_source: true,
            transfer_destination: true,
            color_attachment: true,
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

    let storage_buf = CpuAccessibleBuffer::from_iter(
        dev.clone(),
        BufferUsage::transfer_destination(),
        vec![[0u8; 4]; (w * h) as usize].iter().cloned(),
    )
    .expect("Could not create CPU vissible buffer.");

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
        )
        .expect("Could not create render pass."),
    );

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };
    let mut frame_buffers = [
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[0].clone())
            .expect("Could not add image to framebuffer.")
            .build()
            .expect("Could not create framebuffer.")
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[1].clone())
            .expect("Could not add image to framebuffer.")
            .build()
            .expect("Could not create framebuffer.")
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[2].clone())
            .expect("Could not add image to framebuffer.")
            .build()
            .expect("Could not create framebuffer.")
        )
    ];
    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader.main_entry_point(), ())
            .fragment_shader(frag_shader.main_entry_point(), ())
            .viewports_scissors(
                [(
                    Viewport {
                        depth_range: 0. ..1.,
                        dimensions: [800., 600.],
                        origin: [0., 0.],
                    },
                    Scissor {
                        origin: [0, 0],
                        dimensions: [800, 600],
                    },
                )].iter().cloned(),
            )
            .depth_clamp(false)
            .depth_write(false)
            .front_face_counter_clockwise()
            .cull_mode_back()
            .build(dev.clone())
            .expect("Could not build pipeline."),
    );
    let descset = Arc::new(
        PersistentDescriptorSet::start(graphics_pipeline.clone(), 0)
            .add_sampled_image(img.clone(), sampler)
            .expect("Could not add sampled image to descriptor.")
            .build()
            .expect("Could not create descriptor set."),
    );
    println!(
        "Device: {:#?}
Submit queues[0]: {:#?}",
        dev, submit_queues[0]
    );



    let cmd_bufs = vec![
        Arc::new(
            AutoCommandBufferBuilder::primary(dev.clone(), submit_queues[0].family())
                .expect("Could not create draw command buffer.")
                .begin_render_pass(
                    frame_buffers[0].clone(),
                    false,
                    vec![[0.5, 0.5, 0.5, 1.0].into()],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    index_buf.clone(),
                    descset.clone(),
                    (),
                )
                .expect("Could not record indexed draw command.")
                .end_render_pass()
                .expect("Could not record render pass end command.")
                // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                // .expect("Could not record image to buffer copy op.")
                .build()
                .expect("Could not build command buffer."),
        ),
        Arc::new(
            AutoCommandBufferBuilder::primary(dev.clone(), submit_queues[0].family())
                .expect("Could not create draw command buffer.")
                .begin_render_pass(
                    frame_buffers[1].clone(),
                    false,
                    vec![[0.5, 0.5, 0.5, 1.0].into()],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    index_buf.clone(),
                    descset.clone(),
                    (),
                )
                .expect("Could not record indexed draw command.")
                .end_render_pass()
                .expect("Could not record render pass end command.")
                .build()
                .expect("Could not build command buffer."),
        ),
        Arc::new(
            AutoCommandBufferBuilder::primary(dev.clone(), submit_queues[0].family())
                .expect("Could not create draw command buffer.")
                .begin_render_pass(
                    frame_buffers[2].clone(),
                    false,
                    vec![[0.5, 0.5, 0.5, 1.0].into()],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    index_buf.clone(),
                    descset.clone(),
                    (),
                )
                .expect("Could not record indexed draw command.")
                .end_render_pass()
                .expect("Could not record render pass end command.")
                .build()
                .expect("Could not build command buffer."),
        ),
        Arc::new(
            AutoCommandBufferBuilder::primary(dev.clone(), submit_queues[0].family())
                .expect("Could not create draw command buffer.")
                .begin_render_pass(
                    frame_buffers[2].clone(),
                    false,
                    vec![[0.5, 0.5, 0.5, 1.0].into()],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    index_buf.clone(),
                    descset.clone(),
                    (),
                )
                .expect("Could not record indexed draw command.")
                .end_render_pass()
                
                .expect("Could not record render pass end command.")
                .build()
                .expect("Could not build command buffer."),
        ),
    ];

    // let buffer_content = storage_buf.read().unwrap();
    // let buffer_content = {
    //     let mut x: Vec<u8> = Vec::new();
    //     for i in buffer_content[..].iter() {
    //         x.extend(i.iter().cloned());
    //     }
    //     x
    // };
    // let final_img =
    //     image::ImageBuffer::<image::Bgra<u8>, _>::from_raw(w, h, &buffer_content[..]).unwrap();
    // final_img.save("triangle.png").unwrap();
    
    let mut done = false;

    loop {
        previous_frame_end.cleanup_finished();

        let (image_num, acquire_future) = acquire_next_image(swapchain.clone(), None).unwrap();
        //println!("Rendering to image: {}", image_num);
        
        let future = previous_frame_end
           .join(acquire_future)
           .then_execute(submit_queues[0].clone(), cmd_bufs[image_num].clone())
           .expect("Could not execute command buffer.")
           .then_swapchain_present(submit_queues[0].clone(), swapchain.clone(), image_num)
           .then_signal_fence_and_flush()
           .expect("Could not signal fence after submit.");
        future.wait(None).expect("Timed out while waiting");

        previous_frame_end = Box::new(future);

        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent {
                    event: winit::WindowEvent::CloseRequested,
                    ..
                } => done = true,
                _ => (),
            }
        });
        if done {
            return;
        }
    }
}
