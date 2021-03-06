use crate::shaders::*;
use log;

use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, Window, WindowBuilder};

// use cgmath::prelude::*;
// use cgmath::{Vector2, Vector3};

use std::collections::HashSet;
use std::sync::Arc;

use std::sync::mpsc::{channel, Receiver, Sender};

use std::io::prelude::*;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::{ClearValue, Format, FormatDesc};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{
    AttachmentImage, Dimensions, ImageUsage, ImmutableImage, StorageImage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::depth_stencil::*;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract, GraphicsPipelineBuilder};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain;
use vulkano::swapchain::{
    acquire_next_image, AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use crate::{
    get_valid_queue_families, init_instance, model, shaders, InternalVertex, PriorityHolder,
    VertexPC, VertexPCNT, VertexPCT,
};

pub struct Renderer {
    pub surface: Arc<Surface<Window>>,
    pub events_loop: EventsLoop,
    pub device: Arc<Device>,
    pub current_op_future: Box<GpuFuture>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub queues: Vec<Arc<Queue>>,
    pub pipelines: Pipelines,
    pub render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pub framebuffers: Vec<
        std::sync::Arc<
            Framebuffer<
                Arc<dyn RenderPassAbstract + Sync + Send>,
                (
                    (
                        (((), Arc<AttachmentImage>), Arc<SwapchainImage<Window>>),
                        Arc<AttachmentImage>,
                    ),
                    Arc<AttachmentImage>,
                ),
            >,
        >,
    >, //Vec<Arc<FramebufferAbstract + Send + Sync>>,
    pub features: Features,
    pub camera: crate::Camera,
    pub width: u32,
    pub height: u32,
    pub prev_time: std::time::Instant,
    pub done: bool,
    //uniform_buffer_pool:         CpuBufferPool<vs::ty::Data>,
    //surface_uniform_buffer_pool: CpuBufferPool<surface_vs::ty::Data>,
    //draw_text:                   DrawText,
    //os_input_tx:                 Sender<Event>,
    //render_rx:                   Receiver<GraphicsMessage>,
    //frame_durations:             Vec<Duration>,
    //prev_fullscreen: Option<bool>,
}

pub struct Pipelines {
    pub image_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub bg_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
    //pub line_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub font_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub image_3d_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub colour_3d_pipe: Arc<GraphicsPipelineAbstract + Send + Sync>,
}

impl Renderer {
    //pub fn init(os_input_tx: Sender<Event>, device_name: Option<String>) -> Sender<GraphicsMessage> {
    //    let (render_tx, render_rx) = channel();
    //
    //    thread::spawn(move || {
    //        let mut graphics = VulkanGraphics::new(os_input_tx, render_rx, device_name);
    //        graphics.run();
    //    });
    //    render_tx
    //}
    //os_input_tx: Sender<Event>, render_rx: Receiver<GraphicsMessage>, device_name: Option<String>

    pub fn new(width: u32, height: u32) -> Self {
        let instance = init_instance();

        let pdev = PhysicalDevice::enumerate(&instance)
            .find(|x| x.name().contains("GeForce"))
            .unwrap_or_else(|| {
                eprintln!("Could not select preferred device, choosing first available device.");
                PhysicalDevice::enumerate(&instance)
                    .next()
                    .expect("Could not find any suitable devices.")
            });

        let supported_ftrs = pdev.supported_features();
        let enabled_ftrs = Features { ..Features::none() };

        let q_families = get_valid_queue_families(&pdev);

        println!("Selected physical device: {}", pdev.name());

        let (device, submit_queues) = Device::new(
            pdev.clone(),
            &enabled_ftrs,
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

        let mut events_loop = EventsLoop::new();

        let surface = WindowBuilder::new()
            .with_resizable(true)
            .with_dimensions((width, height).into())
            .build_vk_surface(&events_loop, instance.clone())
            .expect("Could not create vulkan window surface.");
        if !surface
            .is_supported(submit_queues[0].family())
            .expect("Could not retrieve surface capabilities.")
        {
            println!("Creation of vulkan surfaces is not supported on this device!");
            panic!()
        }

        let surface_caps = surface
            .capabilities(pdev.clone())
            .expect("Could not get capabilities of surface.");

        let dimensions = surface_caps.current_extent.unwrap_or([width, height]);
        let dimensions_float = [dimensions[0] as f32, dimensions[1] as f32];

        let alpha = surface_caps
            .supported_composite_alpha
            .iter()
            .next()
            .unwrap();
        let swapchain_format = surface_caps.supported_formats[0].0;

        let bg_vs = shaders::solid_colour_bg_vs::Shader::load(device.clone())
            .expect("Could not load bg vertex shader.");
        let bg_fs = shaders::solid_colour_bg_fs::Shader::load(device.clone())
            .expect("Could not load bg fragment shader.");

        let static_image_vs = shaders::basic_vs::Shader::load(device.clone())
            .expect("Could not load static image vertex shader.");
        let static_image_fs = shaders::basic_fs::Shader::load(device.clone())
            .expect("Could not load static image fragment shader.");

        let image_3d_vs = shaders::image_3d_render_vs::Shader::load(device.clone())
            .expect("Could not load 3d image vertex shader.");
        let image_3d_fs = shaders::image_3d_render_fs::Shader::load(device.clone())
            .expect("Could not load 3d image vertex shader.");

        let colour_3d_vs = shaders::colour_3d_render_vs::Shader::load(device.clone())
            .expect("Could not load 3d image vertex shader.");
        let colour_3d_fs = shaders::colour_3d_render_fs::Shader::load(device.clone())
            .expect("Could not load 3d image vertex shader.");

        let (mut swapchain, mut swapchain_images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            if (3 >= surface_caps.min_image_count)
                && (3 <= surface_caps.max_image_count.unwrap_or(4))
            {
                3
            } else {
                surface_caps.min_image_count
            },
            swapchain_format,
            dimensions,
            1,
            surface_caps.supported_usage_flags,
            &submit_queues[0],
            SurfaceTransform::Identity,
            alpha,
            if surface_caps.present_modes.supports(PresentMode::Immediate) {
                PresentMode::Immediate
            } else if surface_caps.present_modes.supports(PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else {
                PresentMode::Fifo // guaranteed to be supported
            },
            true,
            None,
        )
        .expect("failed to create swapchain");

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    multisampled_color: {
                        load:    Clear,
                        store:   DontCare,
                        format:  swapchain.format(),
                        samples: 4,
                    },
                    resolve_color: {
                        load:    DontCare,
                        store:   Store,
                        format:  swapchain.format(),
                        samples: 1,
                    },
                    multisampled_depth: {
                        load:    Clear,
                        store:   DontCare,
                        format:  Format::D16Unorm,
                        samples: 4,
                    },
                    resolve_depth: {
                        load:    DontCare,
                        store:   Store,
                        format:  Format::D16Unorm,
                        samples: 1,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                    }
                },
                pass: {
                    color: [multisampled_color],
                    depth_stencil: {multisampled_depth},
                    resolve: [resolve_color]
                }
            )
            .unwrap(),
        ) as Arc<RenderPassAbstract + Send + Sync>;

        let depth =
            AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();
        let multisampled_depth = AttachmentImage::transient_multisampled(
            device.clone(),
            dimensions,
            4,
            Format::D16Unorm,
        )
        .unwrap();
        let multisampled_image = AttachmentImage::transient_multisampled(
            device.clone(),
            dimensions,
            4,
            swapchain.format(),
        )
        .unwrap();
        let mut framebuffers: Vec<
            std::sync::Arc<
                Framebuffer<
                    Arc<dyn RenderPassAbstract + Sync + Send>,
                    (
                        (
                            (((), Arc<AttachmentImage>), Arc<SwapchainImage<Window>>),
                            Arc<AttachmentImage>,
                        ),
                        Arc<AttachmentImage>,
                    ),
                >,
            >,
        > = Vec::new();

        framebuffers.extend(swapchain_images.iter().map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(multisampled_image.clone())
                    .unwrap()
                    .add(image.clone())
                    .unwrap()
                    .add(multisampled_depth.clone())
                    .unwrap()
                    .add(depth.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        }));

        let wireframe = false;

        let builder = GraphicsPipeline::start()
            .vertex_input_single_buffer::<VertexPC>()
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: dimensions_float,
            }))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap());

        let bg_pipe = if wireframe {
            builder.polygon_mode_line()
        } else {
            builder
        }
        .vertex_shader(bg_vs.main_entry_point(), ())
        .fragment_shader(bg_fs.main_entry_point(), ())
        .depth_stencil(DepthStencil {
            depth_write: true,
            depth_compare: Compare::LessOrEqual,
            depth_bounds_test: DepthBounds::Disabled,
            stencil_front: Default::default(),
            stencil_back: Default::default(),
        })
        .front_face_counter_clockwise()
        .cull_mode_back()
        .build(device.clone())
        .expect("Could not build background pipeline.");

        let image_3d_pipe = GraphicsPipeline::start()
            .vertex_input_single_buffer::<VertexPCT>()
            //.viewports_dynamic_scissors_irrelevant(1)
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: dimensions_float,
            }))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_shader(image_3d_vs.main_entry_point(), ())
            .fragment_shader(image_3d_fs.main_entry_point(), ())
            .depth_stencil(DepthStencil {
                depth_write: true,
                depth_compare: Compare::LessOrEqual,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .front_face_counter_clockwise()
            //            .cull_mode_back()
            .build(device.clone())
            .expect("Could not build image pipeline.");

        let colour_3d_pipe = GraphicsPipeline::start()
            .vertex_input_single_buffer::<VertexPC>()
            //.viewports_dynamic_scissors_irrelevant(1)
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: dimensions_float,
            }))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_shader(colour_3d_vs.main_entry_point(), ())
            .fragment_shader(colour_3d_fs.main_entry_point(), ())
            .depth_stencil(DepthStencil {
                depth_write: true,
                depth_compare: Compare::LessOrEqual,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .front_face_counter_clockwise()
            //            .cull_mode_back()
            .build(device.clone())
            .expect("Could not build image pipeline.");

        let image_pipe = GraphicsPipeline::start()
            .vertex_input_single_buffer::<VertexPCT>()
            //.viewports_dynamic_scissors_irrelevant(1)
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: dimensions_float,
            }))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_shader(static_image_vs.main_entry_point(), ())
            .fragment_shader(static_image_fs.main_entry_point(), ())
            .depth_stencil(DepthStencil {
                depth_write: true,
                depth_compare: Compare::LessOrEqual,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .front_face_counter_clockwise()
            .cull_mode_back()
            .build(device.clone())
            .expect("Could not build image pipeline.");

        let font_pipe = GraphicsPipeline::start()
            .vertex_input_single_buffer::<VertexPCT>()
            .viewports(std::iter::once(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: dimensions_float,
            }))
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_shader(static_image_vs.main_entry_point(), ())
            .fragment_shader(static_image_fs.main_entry_point(), ())
            .depth_stencil(DepthStencil {
                depth_write: true,
                depth_compare: Compare::LessOrEqual,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .front_face_counter_clockwise()
            .cull_mode_back()
            .build(device.clone())
            .expect("Could not build font pipeline.");

        let camera = crate::Camera {
            pos: cgmath::Vector3::new(1., 0., 0.),
            dir: cgmath::Vector3::new(1., 0., 0.),
            up: cgmath::Vector3::unit_y(),
            pitch: 0.,
        };

        Renderer {
            surface: surface,
            events_loop: events_loop,
            device: device.clone(),
            current_op_future: Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>,
            swapchain: swapchain,
            queues: submit_queues,
            pipelines: Pipelines {
                image_pipe: Arc::new(image_pipe),
                bg_pipe: Arc::new(bg_pipe),
                font_pipe: Arc::new(font_pipe),
                image_3d_pipe: Arc::new(image_3d_pipe),
                colour_3d_pipe: Arc::new(colour_3d_pipe),
                //line_pipe:
            },
            camera: camera,
            render_pass: render_pass,
            framebuffers: framebuffers,
            features: enabled_ftrs,
            width: width,
            height: height,
            prev_time: std::time::Instant::now(),
            done: false,
        }
    }

    pub fn render(mut self, models: Vec<&model::ModelBase>) -> Self {
        self.current_op_future.cleanup_finished();
        // use std::time::Instant;
        // let current_time = Instant::now();
        // let _delta = current_time.duration_since(self.prev_time).as_millis();
        // //            let elapsed_time =  current_time.duration_since(start_time.clone()).as_millis();
        // self.prev_time = current_time;

        //if let Some(ubo) = model.ubo.clone() {
        //    let mut write_lock = ubo
        //        .write()
        //        .expect("Could not lock uniform buffer for write access.");
        //    use std::ops::DerefMut;
        //    let x = write_lock.deref_mut();
        //    x.model = (cgmath::Matrix4::from_angle_x(cgmath::Rad(_delta as f32))).into();
        //}

        //println!("Rendering to image: {}", image_num);

        let cmdbufs = {
            let mut cmdbufs = Vec::new();
            for (i, x) in self.framebuffers.iter().enumerate() {
                unsafe {
                    cmdbufs.push(Arc::new(
                        AutoCommandBufferBuilder::primary_one_time_submit(
                            self.queues[0].device().clone(),
                            self.queues[0].family(),
                        )
                        .expect("Could not create main cmd buffer.")
                        .begin_render_pass(
                            x.clone(),
                            true,
                            vec![
                                [0.0, 0.0, 0.0, 1.0].into(),
                                ClearValue::None,
                                1f32.into(),
                                ClearValue::None,
                            ],
                        )
                        .expect("Could not begin render pass.")
                        .execute_commands(
                            models
                                .iter()
                                .map(|y| y.get_cmd_bufs()[i].clone())
                                .collect::<Vec<Arc<_>>>(),
                        )
                        .expect("Could not add secondary command buffer.")
                        .end_render_pass()
                        .expect("Could not end render pass.")
                        .build()
                        .unwrap(),
                    ));
                }
            }
            cmdbufs
        };

        let (image_num, acquire_future) = acquire_next_image(self.swapchain.clone(), None).unwrap();
        let future = self
            .current_op_future
            .join(acquire_future)
            .then_execute(self.queues[0].clone(), cmdbufs[image_num].clone())
            .expect("Could not execute command buffer.")
            .then_swapchain_present(self.queues[0].clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush()
            .expect("Could not signal fence after submit.");
        future.wait(None).expect("Timed out while waiting");

        self.current_op_future = Box::new(future);

        self
    }

    pub fn handle_events(mut self) -> Self {
        let mut t = false;
        let sensitivity = 0.01;
        let mut del = (0f64, 0f64);
        self.events_loop.poll_events(|ev| match ev {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => t = true,
            winit::Event::DeviceEvent {
                event: winit::DeviceEvent::MouseMotion { delta: m_delta },
                ..
            } => {
                println!("{:#?}", &m_delta);
                del = m_delta;
            }
            winit::Event::DeviceEvent {
                event: winit::DeviceEvent::Key(key_info),
                ..
            } => {
                if key_info.virtual_keycode == Some(winit::VirtualKeyCode::Escape) {
                    t = true;
                }
            }
            _ => (),
        });

        self.camera.update(del, sensitivity);
        self.done = t;
        self
    }
}

pub fn main() {
    let mut r = Renderer::new(1024, 1024);
    let (mut m1, mut m2) = {
        let img = image::open(&std::path::Path::new(
            "/home/warpspeedscp/vkrust/chalet.jpg",
        ))
        .expect("Could not load model texture.")
        .as_rgb8()
        .expect("Could not get RGB representation of image.")
        .clone();

        let mut img_vec: Vec<[u8; 4]> = Vec::new();
        let mut w: u32 = 1024;
        let mut h: u32 = 1024;

        {
            let x = img.dimensions();
            w = x.0;
            h = x.1;
        }

        for i in img.pixels() {
            let r = i[0];
            let g = i[1];
            let b = i[2];
            let a = 0xFF;

            img_vec.push([r, g, b, a]);
        }

        (
            model::Model::new(
                //vertices,
                //indices,
                vec![
                    VertexPC::new()
                        .pos([-1., -1., 0.]) /*.uv([0., 0.])*/
                        .colour([1., 0., 0.]),
                    VertexPC::new()
                        .pos([-1., 1., 0.]) /*.uv([0., 1.])*/
                        .colour([1., 1., 0.]),
                    VertexPC::new()
                        .pos([1., 1., 0.]) /*.uv([1., 1.])*/
                        .colour([0., 1., 1.]),
                    VertexPC::new()
                        .pos([1., -1., 0.]) /*.uv([1., 0.])*/
                        .colour([0., 0., 1.]),
                ],
                vec![0, 1, 2, 2, 3, 0],
                vec![], //img_vec,
                Format::R8G8B8A8Unorm,
                Dimensions::Dim2d {
                    width: w,
                    height: h,
                },
                r.queues[0].clone(),
                r.pipelines.colour_3d_pipe.clone(),
                0,
                true,
            )
            .expect("Could not create model."),
            model::Model::new(
                vec![
                    VertexPCT::new()
                        .pos([-1., -1., 0.])
                        .uv([0., 0.])
                        .colour([1., 0., 0.]),
                    VertexPCT::new()
                        .pos([-1., 1., 0.])
                        .uv([0., 1.])
                        .colour([1., 1., 0.]),
                    VertexPCT::new()
                        .pos([1., 1., 0.])
                        .uv([1., 1.])
                        .colour([0., 1., 1.]),
                    VertexPCT::new()
                        .pos([1., -1., 0.])
                        .uv([1., 0.])
                        .colour([0., 0., 1.]),
                ],
                vec![0, 1, 2, 2, 3, 0],
                img_vec,
                Format::R8G8B8A8Unorm,
                Dimensions::Dim2d {
                    width: w,
                    height: h,
                },
                r.queues[0].clone(),
                r.pipelines.image_3d_pipe.clone(),
                0,
                true,
            )
            .expect("Could not create model."),
        )
    };

    {
        let mut_m1 = &mut m1;
        for _ in 0..r.framebuffers.len() {
            mut_m1.cmd_bufs.push(Arc::new({
                AutoCommandBufferBuilder::secondary_graphics_simultaneous_use(
                    r.queues[0].device().clone(),
                    r.queues[0].family(),
                    Subpass::from(r.render_pass.clone(), 0).unwrap(),
                )
                .expect("Could not create draw command buffer.")
                .draw_indexed(
                    mut_m1.pipeline.clone(),
                    &DynamicState::none(),
                    vec![mut_m1.vertices.clone()],
                    mut_m1.indices.clone(),
                    /*{
                        let a: Vec<
                            std::sync::Arc<
                                (dyn vulkano::descriptor::DescriptorSet
                                     + std::marker::Sync
                                     + std::marker::Send
                                     + 'static),
                            >,
                        > = Vec::new();
                        a
                    }*/
                    vec![mut_m1.descriptor_set.clone().unwrap()],
                    (),
                )
                .expect("Could not record indexed draw command.")
                // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                // .expect("Could not record image to buffer copy op.")
                .build()
                .expect("Could not build command buffer.")
            }));
        }

        let mut_m2 = &mut m2;
        for _ in 0..r.framebuffers.len() {
            mut_m2.cmd_bufs.push(Arc::new({
                AutoCommandBufferBuilder::secondary_graphics(
                    r.queues[0].device().clone(),
                    r.queues[0].family(),
                    Subpass::from(r.render_pass.clone(), 0).unwrap(),
                )
                .expect("Could not create draw command buffer.")
                .draw_indexed(
                    mut_m2.pipeline.clone(),
                    &DynamicState::none(),
                    vec![mut_m2.vertices.clone()],
                    mut_m2.indices.clone(),
                    vec![mut_m2.descriptor_set.clone().unwrap()],
                    shaders::image_3d_render_vs::ty::ColorBlock {
                        Color: cgmath::Vector4::unit_x().into(),
                    },
                )
                .expect("Could not record indexed draw command.")
                // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                // .expect("Could not record image to buffer copy op.")
                .build()
                .expect("Could not build command buffer.")
            }));
        }
    }
    {
        use cgmath::*;
        // let m1 = &mut m1;
        let m1_ubo = m1.ubo.clone().unwrap();
        let mut write_lock = m1_ubo
            .write()
            .expect("Could not lock uniform buffer for write access.");
        use std::ops::DerefMut;
        let x = write_lock.deref_mut();
        x.view = (Matrix4::look_at(
            Point3::from_homogeneous(r.camera.pos.extend(1.)),
            Point3::from_homogeneous((r.camera.pos + r.camera.dir).extend(1.)),
            r.camera.up,
        ))
        .into();

        x.proj = cgmath::perspective(Deg(60.), 4. / 3., 0.001, 1000.0).into();
        x.proj[1][1] *= -1.;

        // let m2 = Arc::get_mut(&mut m2).unwrap();
        let m1_ubo = m2.ubo.clone().unwrap();
        let mut write_lock = m1_ubo
            .write()
            .expect("Could not lock uniform buffer for write access.");
        let x = write_lock.deref_mut();
        x.view = (Matrix4::look_at(
            Point3::from_homogeneous(r.camera.pos.extend(1.)),
            Point3::from_homogeneous((r.camera.pos + r.camera.dir).extend(1.)),
            r.camera.up,
        ))
        .into();
        x.model = (Matrix4::from_scale(2f32)).into();
        x.proj = cgmath::perspective(Deg(45.), 4. / 3., 0.01, 100.0).into();
        x.proj[1][1] *= -1.;

        r.camera.pos = Vector3::unit_y();
        r.camera.dir = Vector3::unit_z();
    }

    let mut colourTable: Vec<[f32; 4]> = vec![
        cgmath::Vector4::unit_w().into(),
        cgmath::Vector4::unit_x().into(),
        cgmath::Vector4::unit_y().into(),
        cgmath::Vector4::unit_z().into(),
    ];
    let mut colourTable = colourTable.into_iter().cycle();

    let mut old_cmd_bufs: Vec<Arc<vulkano::command_buffer::AutoCommandBuffer>>;

    while !r.done {
        {
            use crate::model::ModelBase;
            use cgmath::*;
            use std::mem::replace;
            use std::ops::DerefMut;

            old_cmd_bufs = m2.cmd_bufs;
            m2.cmd_bufs = Vec::new();
            for _ in 0..r.framebuffers.len() {
                m2.cmd_bufs.push(Arc::new({
                    AutoCommandBufferBuilder::secondary_graphics(
                        r.queues[0].device().clone(),
                        r.queues[0].family(),
                        Subpass::from(r.render_pass.clone(), 0).unwrap(),
                    )
                    .expect("Could not create draw command buffer.")
                    .draw_indexed(
                        m2.pipeline.clone(),
                        &DynamicState::none(),
                        vec![m2.vertices.clone()],
                        m2.indices.clone(),
                        vec![m2.descriptor_set.clone().unwrap()],
                        shaders::image_3d_render_vs::ty::ColorBlock {
                            Color: colourTable.next().unwrap(),
                        },
                    )
                    .expect("Could not record indexed draw command.")
                    // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                    // .expect("Could not record image to buffer copy op.")
                    .build()
                    .expect("Could not build command buffer.")
                }));
            }

            {
                let m1_ubo = m1.ubo.clone().unwrap();
                let mut write_lock = m1_ubo
                    .write()
                    .expect("Could not lock uniform buffer for write access.");
                let x = write_lock.deref_mut();
                x.model = (Matrix4::from_translation(Vector3::unit_z() * 4.)
                    + Matrix4::from_translation(Vector3::unit_y() * 2.))
                .into();
                x.view = (Matrix4::look_at(
                    Point3::from_homogeneous(r.camera.pos.extend(1.)),
                    Point3::from_homogeneous((r.camera.pos + r.camera.dir).extend(1.)),
                    r.camera.up,
                ))
                .into();
            }
            {
                let m2_ubo = m2.ubo.clone().unwrap();
                let mut write_lock = m2_ubo
                    .write()
                    .expect("Could not lock uniform buffer for write access.");
                let x = write_lock.deref_mut();
                x.model = ((Matrix4::from_translation(Vector3::unit_z() * 3.)
                    + Matrix4::from_translation(Vector3::unit_y() * 2.5))
                    * Matrix4::from_scale(0.25))
                .into();
                x.view = (Matrix4::look_at(
                    Point3::from_homogeneous(r.camera.pos.extend(1.)),
                    Point3::from_homogeneous((r.camera.pos + r.camera.dir).extend(1.)),
                    r.camera.up,
                ))
                .into();
            }
        } //m2.clone(),
        r = r.render(vec![&m1, &m2]).handle_events();
    }
}
