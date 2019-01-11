use vulkano::format::FormatDesc;
use vulkano::framebuffer::RenderPass;
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, Window, WindowBuilder};

use cgmath::prelude::*;
use cgmath::{Matrix4, Point3, Rad, SquareMatrix, Vector3};

use std::sync::Arc;

use std::io::prelude::*;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{
    AttachmentImage, Dimensions, ImageUsage, ImmutableImage, StorageImage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain;

use vulkano::swapchain::{
    acquire_next_image, AcquireError, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use model::Model;

use crate::{
    get_valid_queue_families, init_instance, project_on_plane, revolve_left, revolve_up, shaders,
    PriorityHolder, VertexPCNT, UBO,
};

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
        .with_resizable(true)
        //        .with_dimensions((800u32, 600u32).into())
        .build_vk_surface(&events_loop, instance.clone())
        .expect("Could not create vulkan window surface.");
    if !surface
        .is_supported(submit_queues[0].family())
        .expect("Could not retrieve surface capabilities.")
    {
        return;
    }

    let mut window = surface.window();

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

    let (mut swapchain, swapchain_images) = Swapchain::new(
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

    let vert_shader = shaders::diffuse_lighting_vs::Shader::load(dev.clone())
        .expect("Could not load vertex shader.");
    let frag_shader = shaders::diffuse_lighting_fs::Shader::load(dev.clone())
        .expect("Could not load fragment shader.");

    let vertex_size = std::mem::size_of::<VertexPCNT>();
    let index_size = 4usize;

    let mut previous_frame_end: Box<GpuFuture>;

    let w: u32;
    let h: u32;

    let mut depth_buffer =
        AttachmentImage::transient(dev.clone(), dimensions, Format::D16Unorm).unwrap();

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
        )
        .expect("Could not create render pass."),
    );

    let mut frame_buffers = vec![
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(swapchain_images[0].clone())
                .expect("Could not add image to framebuffer.")
                .add(depth_buffer.clone())
                .expect("Could not add depth buffer to framebuffer")
                .build()
                .expect("Could not create framebuffer."),
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(swapchain_images[1].clone())
                .expect("Could not add image to framebuffer.")
                .add(depth_buffer.clone())
                .expect("Could not add depth buffer to framebuffer")
                .build()
                .expect("Could not create framebuffer."),
        ),
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(swapchain_images[2].clone())
                .expect("Could not add image to framebuffer.")
                .add(depth_buffer.clone())
                .expect("Could not add depth buffer to framebuffer")
                .build()
                .expect("Could not create framebuffer."),
        ),
    ];

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_single_buffer::<VertexPCNT>()
            .vertex_shader(vert_shader.main_entry_point(), ())
            .fragment_shader(frag_shader.main_entry_point(), ())
            // .viewports_scissors(
            //     [(
            //         Viewport {
            //             depth_range: 0. ..1.,
            //             dimensions: {
            //                 let fd: Vec<f32> = dimensions.iter().map(|x| *x as f32).collect();
            //                 [fd[0], fd[1]]
            //             },
            //             origin: [0., 0.],
            //         },
            //         Scissor {
            //             origin: [0, 0],
            //             dimensions: dimensions.clone(),
            //         },
            //     )]
            //         .iter()
            //         .cloned(),
            // )
            .viewports_dynamic_scissors_irrelevant(1)
            .depth_clamp(false)
            .front_face_counter_clockwise()
            //.cull_mode_back()
            .depth_stencil_simple_depth()
            .build(dev.clone())
            .expect("Could not build pipeline."),
    );

    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: Some(vec![Viewport {
            depth_range: 0. ..1.,
            dimensions: {
                let fd: Vec<f32> = dimensions.iter().map(|x| *x as f32).collect();
                [fd[0], fd[1]]
            },
            origin: [0., 0.],
        }]),
        scissors: None,
    };

    let mut model = {
        let (vertices, indices) = {
            use std::path::Path;
            use tobj;
            let mesh = &tobj::load_obj(&Path::new("/home/warpspeedscp/vkrust/chalet.obj"))
                .unwrap()
                .0[0]
                .mesh;

            let (positions, normals) = {
                let mut i = 0usize;
                let mut p: Vec<[f32; 3]> = Vec::new();
                let mut n: Vec<[f32; 3]> = Vec::new();
                while i < mesh.positions.len() - 3 {
                    p.push([
                        mesh.positions[i],
                        mesh.positions[i + 1],
                        mesh.positions[i + 2],
                    ]);
                    n.push([mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]]);
                    i += 3;
                }

                (p, n)
            };

            let indices = mesh.indices.clone();

            let uvs = {
                let mut i = 0usize;
                let mut u: Vec<[f32; 2]> = Vec::new();
                while i < mesh.texcoords.len() - 2 {
                    u.push([mesh.texcoords[i], mesh.texcoords[i + 1]]);
                    i += 2;
                }

                u
            };

            let mut tmp: Vec<VertexPCNT> = Vec::new();
            for (vert, norm, uv) in izip!(positions, normals, uvs) {
                tmp.push(VertexPCNT::new().pos(vert).normal(norm).uv(uv))
            }

            (tmp, indices)
        };

        let img = image::open(&std::path::Path::new(
            "/home/warpspeedscp/vkrust/chalet.jpg",
        ))
        .expect("Could not load model texture.")
        .as_rgb8()
        .expect("Could not get RGB representation of image.")
        .clone();

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

        Model::new(
            vertices,
            indices,
            img_vec,
            Format::R8G8B8A8Unorm,
            Dimensions::Dim2d {
                width: w,
                height: h,
            },
            submit_queues[1].clone(),
            graphics_pipeline.clone(),
            0,
            true,
        )
        .expect("Could not create model.")
    };

    {
        use cgmath::Deg;
        let mut ubo = UBO {
            proj: cgmath::perspective(Deg(45.), 4. / 3., 0.01, 100.0).into(),
            model: (Matrix4::from_scale(2.)).into(),
            view: (Matrix4::look_at(
                Point3::new(4., 4., 1.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ))
            .into(),
        };

        ubo.proj[1][1] *= -1.;

        let ubo_buf = model.ubo.clone().unwrap();

        let mut write_lock = ubo_buf
            .write()
            .expect("Could not lock uniform buffer for write access.");
        use std::ops::DerefMut;
        let x = write_lock.deref_mut();

        x.proj = ubo.proj;
        x.view = ubo.view;
        x.model = ubo.model;
    }

    {
        let vertices = model.vertices.clone();
        let indices = model.indices.clone();
        let desc_set = model.descriptor_set.clone();

        model.cmd_bufs = crate::model::gen_cmd_bufs(
            model.cmd_bufs,
            submit_queues[0].clone(),
            &frame_buffers[..],
            |queue: Arc<Queue>, fb: Arc<FramebufferAbstract + Sync + Send>| {
                AutoCommandBufferBuilder::primary_simultaneous_use(
                    queue.device().clone(),
                    queue.family(),
                )
                .expect("Could not create draw command buffer.")
                .begin_render_pass(fb, false, vec![[0.1, 0.1, 0.1, 1.0].into(), 1f32.into()])
                .expect("Could not record render pass begin command.")
                .draw_indexed(
                    graphics_pipeline.clone(),
                    &dynamic_state,
                    vertices.clone(),
                    indices.clone(),
                    desc_set.clone().unwrap(),
                    (),
                )
                .expect("Could not record indexed draw command.")
                .end_render_pass()
                .expect("Could not record render pass end command.")
                // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                // .expect("Could not record image to buffer copy op.")
                .build()
                .expect("Could not build command buffer.")
            },
        );
    }

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
    .expect("Could not create CPU visible buffer.");

    use std::time::Instant;
    let start_time = Instant::now();
    let mut prev_time = start_time.clone();

    let mut camera = crate::Camera {
        pos: Vector3::new(4., 0., 0.),
        dir: Vector3::new(-1., 0., 0.),
        up: Vector3::unit_y(),
        pitch: 0.,
    };

    let mut recreate_swapchain = false;
    let mut done = false;
    let mut left = false;
    let mut right = false;
    let mut up = false;
    let mut down = false;

    let ubo = model.ubo.clone().unwrap();

    previous_frame_end = Box::new(sync::now(dev.clone()));

    loop {
        previous_frame_end.cleanup_finished();

        {
            if recreate_swapchain {
                // Get the new dimensions of the window.
                let dimensions = if let Some(dimensions) = window.get_inner_size() {
                    let dimensions: (u32, u32) =
                        dimensions.to_physical(window.get_hidpi_factor()).into();
                    [dimensions.0, dimensions.1]
                } else {
                    return;
                };

                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimension(dimensions) {
                        Ok(r) => r,
                        // This error tends to happen when the user is manually resizing the window.
                        // Simply restarting the loop is the easiest way to fix this issue.
                        Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                        Err(err) => panic!("{:?}", err),
                    };

                depth_buffer =
                    AttachmentImage::transient(dev.clone(), dimensions, Format::D16Unorm).unwrap();;

                swapchain = new_swapchain;
                // Because framebuffers contains an Arc on the old swapchain, we need to
                // recreate framebuffers as well.
                let nfb = window_size_dependent_setup(
                    &new_images,
                    &depth_buffer,
                    render_pass.clone(),
                    &mut dynamic_state,
                );

                unsafe {
                    use std::mem;
                    frame_buffers = mem::transmute::<
                        Vec<Arc<FramebufferAbstract + Send + Sync>>,
                        Vec<
                            Arc<
                                Framebuffer<
                                    Arc<RenderPass<_>>,
                                    (
                                        (
                                            (),
                                            std::sync::Arc<
                                                vulkano::image::SwapchainImage<winit::Window>,
                                            >,
                                        ),
                                        std::sync::Arc<vulkano::image::AttachmentImage>,
                                    ),
                                >,
                            >,
                        >,
                    >(nfb);
                }

                {
                    let vertices = model.vertices.clone();
                    let indices = model.indices.clone();
                    let desc_set = model.descriptor_set.clone();

                    while model.cmd_bufs.len() > 0 {
                        model.cmd_bufs.pop();
                    }
                    model.cmd_bufs = crate::model::gen_cmd_bufs(
                        model.cmd_bufs,
                        submit_queues[0].clone(),
                        &frame_buffers[..],
                        |queue: Arc<Queue>, fb: Arc<FramebufferAbstract + Sync + Send>| {
                            AutoCommandBufferBuilder::primary_simultaneous_use(
                                queue.device().clone(),
                                queue.family(),
                            )
                            .expect("Could not create draw command buffer.")
                            .begin_render_pass(
                                fb,
                                false,
                                vec![[0.1, 0.1, 0.1, 1.0].into(), 1f32.into()],
                            )
                            .expect("Could not record render pass begin command.")
                            .draw_indexed(
                                graphics_pipeline.clone(),
                                &dynamic_state,
                                vertices.clone(),
                                indices.clone(),
                                desc_set.clone().unwrap(),
                                (),
                            )
                            .expect("Could not record indexed draw command.")
                            .end_render_pass()
                            .expect("Could not record render pass end command.")
                            // .copy_image_to_buffer(storage_img.clone(), storage_buf.clone())
                            // .expect("Could not record image to buffer copy op.")
                            .build()
                            .expect("Could not build command buffer.")
                        },
                    );
                }

                recreate_swapchain = false;
            }

            let current_time = Instant::now();
            let _delta = current_time.duration_since(prev_time).as_millis();
            //let elapsed_time =  current_time.duration_since(start_time.clone()).as_millis();
            prev_time = current_time;

            events_loop.poll_events(|ev| match ev {
                winit::Event::WindowEvent {
                    event: winit::WindowEvent::CloseRequested,
                    ..
                } => done = true,
                winit::Event::DeviceEvent {
                    event: winit::DeviceEvent::MouseMotion { delta: m_delta },
                    ..
                } => {
                    let sensitivity = 0.01;
                    camera.update(m_delta, sensitivity);
                }
                winit::Event::DeviceEvent {
                    event: winit::DeviceEvent::Key(key_info),
                    ..
                } => {
                    if key_info.virtual_keycode == Some(winit::VirtualKeyCode::A) {
                        if key_info.state == winit::ElementState::Pressed {
                            left = true;
                        } else {
                            left = false;
                        }
                    } else if key_info.virtual_keycode == Some(winit::VirtualKeyCode::D) {
                        if key_info.state == winit::ElementState::Pressed {
                            right = true;
                        } else {
                            right = false;
                        }
                    } else if key_info.virtual_keycode == Some(winit::VirtualKeyCode::W) {
                        if key_info.state == winit::ElementState::Pressed {
                            up = true;
                        } else {
                            up = false;
                        }
                    } else if key_info.virtual_keycode == Some(winit::VirtualKeyCode::S) {
                        if key_info.state == winit::ElementState::Pressed {
                            down = true;
                        } else {
                            down = false;
                        }
                    } else if key_info.virtual_keycode == Some(winit::VirtualKeyCode::Escape) {
                        done = true;
                    }
                }
                _ => (),
            });

            if left {
                //println!("camera pos: {:#?}\ncamera dir {:#?}", camera.pos, camera.dir);
                camera.pos -= 0.01 * camera.dir.cross(Vector3::unit_y()).normalize();
            }
            if right {
                camera.pos += 0.01 * camera.dir.cross(Vector3::unit_y()).normalize();
            }
            if up {
                camera.pos += 0.01 * camera.dir; //project_on_plane(&camera.dir, &Vector3::unit_y());
            }
            if down {
                camera.pos -= 0.01 * camera.dir; //project_on_plane(&camera.dir, &Vector3::unit_y());
            }

            match ubo.write() {
                Ok(mut write_lock) => {
                    use std::ops::DerefMut;
                    let x = write_lock.deref_mut();
                    x.view = (Matrix4::look_at(
                        Point3::from_homogeneous(camera.pos.extend(1.)),
                        Point3::from_homogeneous((camera.pos + camera.dir).extend(1.)),
                        camera.up,
                    ))
                    .into();
                }
                Err(x) => {
                    //eprintln!("{:#?}", x);
                }
            }
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        //        let (image_num, acquire_future) = acquire_next_image(swapchain.clone(), None).unwrap();
        //println!("Rendering to image: {}", image_num);

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(submit_queues[0].clone(), model.cmd_bufs[image_num].clone())
            .expect("Could not execute command buffer.")
            .then_swapchain_present(submit_queues[0].clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        //future.wait(None).expect("Timed out while waiting");

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(dev.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(dev.clone())) as Box<_>;
            }
        }

        //previous_frame_end = Box::new(future);

        if done {
            return;
        }
    }
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    depth_buffer: &Arc<AttachmentImage>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .expect("Could not add image to framebuffer.")
                    .add(depth_buffer.clone())
                    .expect("Could not add depth buffer to framebuffer")
                    .build()
                    .expect("Could not create framebuffer."),
            ) as Arc<FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
