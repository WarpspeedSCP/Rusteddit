use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder};

//use cgmath::prelude::*;
use cgmath::{Vector3, Point3, Matrix4, Rad, SquareMatrix};

use std::sync::Arc;

use std::io::prelude::*;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{AttachmentImage, Dimensions, ImageUsage, ImmutableImage, StorageImage, SwapchainImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::{Viewport, Scissor};
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{acquire_next_image, AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;



use crate::{get_valid_queue_families, init_instance, PriorityHolder, VertexPCNT, shaders, UBO};


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
            sampler_anisotropy: true,
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
    let format = Format::B8G8R8A8Unorm;

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

    let vert_shader = shaders::diffuse_lighting_vs::Shader::load(dev.clone()).expect("Could not load vertex shader.");
    let frag_shader = shaders::diffuse_lighting_fs::Shader::load(dev.clone()).expect("Could not load fragment shader.");

    let vertex_size = std::mem::size_of::<VertexPCNT>();
    let index_size = 4usize;

    let (vertices, indices) = {
        use tobj;
        use std::path::Path;
        let mesh = &tobj::load_obj(&Path::new("/home/warpspeedscp/vkrust/chalet.obj")).unwrap().0[0].mesh;
        
        let (positions, normals) = {
            let mut i = 0usize;
            let mut p: Vec<[f32; 3]> = Vec::new();
            let mut n: Vec<[f32; 3]> = Vec::new();
            while i < mesh.positions.len() - 3 {
                p.push(
                    [
                        mesh.positions[i], 
                        mesh.positions[i + 1], 
                        mesh.positions[i + 2]
                    ]
                );
                n.push(
                    [
                        mesh.normals[i], 
                        mesh.normals[i + 1], 
                        mesh.normals[i + 2]
                    ]
                );
                i += 3;
            }

            (p, n)
        };


        let indices = mesh.indices.clone();

        let uvs = {
            let mut i = 0usize;
            let mut u: Vec<[f32; 2]> = Vec::new();
            while i < mesh.texcoords.len() - 2 {
                u.push(
                    [
                        mesh.texcoords[i], 
                        mesh.texcoords[i + 1]
                    ]
                );
                i += 2;
            }

            u
        };

        let mut tmp: Vec<VertexPCNT> = Vec::new();
        for (vert, norm, uv) in izip!( positions, normals, uvs ) {
            tmp.push(VertexPCNT::new().pos(vert).normal(norm).uv(uv))
        }
        
        (tmp, indices)
    };

    let (vert_buf, ind_buf) = {
        
        let (a, mut b) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(), 
            BufferUsage::vertex_buffer(), 
            submit_queues[1].clone()
        ).expect("Could not create vertex buffer.");

        b.flush().expect("Could not upload vertex buffer data.");
        b.cleanup_finished();

        let (c, mut d) = ImmutableBuffer::from_iter(
            indices.iter().cloned(), 
            BufferUsage::index_buffer(), 
            submit_queues[1].clone()
        ).expect("Could not create index buffer.");

        d.flush().expect("Could not upload index buffer data.");
        d.cleanup_finished();

        (a, c)
    };

    let mut previous_frame_end: Box<GpuFuture>;

    let mut w: u32;
    let mut h: u32;

    let texture = {
        let img = image::open(&std::path::Path::new("/home/warpspeedscp/vkrust/chalet.jpg")).expect("Could not load model texture.").as_rgb8().expect("Could not get RGB representation of image.").clone();
        let mut img_vec: Vec<[u8; 4]> = Vec::new();
        
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

        let (a, mut b) = ImmutableImage::from_iter(
            img_vec.iter().cloned(),
            Dimensions::Dim2d {
                width: w,
                height: h,
            },
            Format::R8G8B8A8Unorm,
            submit_queues[0].clone(),
        )
        .expect("Could not create immutable image.");

        b.flush().expect("Could not upload image data.");

        previous_frame_end = Box::new(b) as Box<_>;
        a
    };

    let depth_buffer = AttachmentImage::transient(
        dev.clone(), 
        dimensions, 
        Format::D16Unorm
    ).unwrap();

    let sampler = Sampler::new(
        dev.clone(), 
        Filter::Nearest, 
        Filter::Nearest, 
        MipmapMode::Linear, 
        SamplerAddressMode::ClampToEdge, 
        SamplerAddressMode::ClampToEdge, 
        SamplerAddressMode::ClampToEdge, 
        0., 
        4., 
        1., 
        1.
    ).expect("Could not create sampler.");

    let storage_img = StorageImage::with_usage(
        dev.clone(),
        Dimensions::Dim2d {
            width: w,
            height: h,
        },
        Format::R8G8B8A8Unorm,
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

    use cgmath::Deg;
    let mut ubo = UBO {
        proj: cgmath::perspective(
            Deg(45.), 
            4./3., 
            0.01, 
            100.0
        ).into(),
        model: (Matrix4::from_scale(3.)).into(),
        view: (
            Matrix4::look_at(
                Point3::new(4., 4., 1.0), 
                Point3::new(0.0, 0.0, 0.0), 
                Vector3::new(0.0, 0.0, 1.0)
            )
        ).into()
    };

    ubo.proj[1][1] *= -1.;

    let ubo_buf = CpuAccessibleBuffer::from_data(dev.clone(),  BufferUsage::uniform_buffer(), ubo.clone()).expect("Could not upload uniform buffer data.");

    let render_pass = Arc::new(
        single_pass_renderpass!(
            dev.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::B8G8R8A8Unorm,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,  //D24Unorm_S8Uint,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).expect("Could not create render pass.")
    );

    let mut frame_buffers = [
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[0].clone())
            .expect("Could not add image to framebuffer.")
            .add(depth_buffer.clone())
            .expect("Could not add depth buffer to framebuffer")
            .build()
            .expect("Could not create framebuffer.")
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[1].clone())
            .expect("Could not add image to framebuffer.")
            .add(depth_buffer.clone())
            .expect("Could not add depth buffer to framebuffer")
            .build()
            .expect("Could not create framebuffer.")
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(swapchain_images[2].clone())
            .expect("Could not add image to framebuffer.")
            .add(depth_buffer.clone())
            .expect("Could not add depth buffer to framebuffer")
            .build()
            .expect("Could not create framebuffer.")
        )
    ];

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .vertex_input_single_buffer::<VertexPCNT>()
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
        //.cull_mode_back()
        .depth_stencil_simple_depth()
        .build(dev.clone())
        .expect("Could not build pipeline."),
    );

    let descset = Arc::new(
        PersistentDescriptorSet::start(graphics_pipeline.clone(), 0)
        .add_buffer(ubo_buf.clone())
        .expect("Could not add UBO to descriptor set.")
        .add_sampled_image(texture.clone(), sampler.clone())
        .expect("Could not add sampled image to descriptor set.")
        .build()
        .expect("Could bot create descriptor set.")
    );

    let cmd_bufs = vec![
        Arc::new(
            AutoCommandBufferBuilder::primary(dev.clone(), submit_queues[0].family())
                .expect("Could not create draw command buffer.")
                .begin_render_pass(
                    frame_buffers[0].clone(),
                    false,
                    vec![
                        [0.1, 0.1, 0.1, 1.0].into(),
                        1f32.into()
                    ],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    ind_buf.clone(),
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
                    vec![
                        [0.1, 0.1, 0.1, 1.0].into(),
                        1f32.into()
                    ],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    ind_buf.clone(),
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
                    vec![
                        [0.1, 0.1, 0.1, 1.0].into(),
                        1f32.into()
                    ],
                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    ind_buf.clone(),
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
                    vec![
                        [0.1, 0.1, 0.1, 1.0].into(),
                        1f32.into()
                    ],

                )
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vert_buf.clone(),
                    ind_buf.clone(),
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

    use std::time::{Duration, Instant};
    let start_time = Instant::now();
    //let mut prev_time = start_time.clone();

        //     currentTime = std::chrono::high_resolution_clock::now();
        // delta = std::chrono::duration<float, std::chrono::seconds::period>(
        //             currentTime - prevTime)
        //             .count();
        // time = std::chrono::duration<float, std::chrono::seconds::period>(
        //            currentTime - startTime)
        //            .count();

        // d->model = glm::rotate(glm::mat4(1), (float)time * glm::radians(45.0f), glm::vec3(0, 0, 1));

    let mut done = false;
    loop {
        previous_frame_end.cleanup_finished();


        {
            let current_time = Instant::now();
            //let delta = prev_time.duration_since(current_time.clone()).as_millis();
            let elapsed_time =  current_time.duration_since(start_time.clone()).as_millis();



            let mut write_lock = ubo_buf.write().expect("Could not lock uniform buffer for write access.");
            use std::ops::DerefMut;
            let x = write_lock.deref_mut();
            x.model = (Matrix4::from_angle_z(Deg(0.1) * elapsed_time as f32) * Matrix4::from_scale(2.)).into();
        }

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