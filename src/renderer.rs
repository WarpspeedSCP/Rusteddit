
use log;
use crate::shaders::*;

use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder, Window};

// use cgmath::prelude::*;
// use cgmath::{Vector2, Vector3};

use std::collections::HashSet;
use std::sync::Arc;

use std::sync::mpsc::{Sender, Receiver, channel};

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
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{acquire_next_image, AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, Surface};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;


pub struct Renderer {
    surface:                     Arc<Surface<Window>>,
    events_loop:                 EventsLoop,
    device:                      Arc<Device>,
    future:                      Box<GpuFuture>,
    swapchain:                   Arc<Swapchain<Window>>,
    queue:                       Arc<Queue>,
    pipelines:                   Pipelines,
    shaders:                     Shaders,
    render_pass:                 Arc<RenderPassAbstract + Send + Sync>,
    framebuffers:                Vec<Arc<FramebufferAbstract + Send + Sync>>,
    //uniform_buffer_pool:         CpuBufferPool<vs::ty::Data>,
    //surface_uniform_buffer_pool: CpuBufferPool<surface_vs::ty::Data>,
    //draw_text:                   DrawText,
    //os_input_tx:                 Sender<Event>,
    //render_rx:                   Receiver<GraphicsMessage>,
    //frame_durations:             Vec<Duration>,
    fps:                         String,
    width:                       u32,
    height:                      u32,
prev_fullscreen: Option<bool>,
}

struct Pipelines {

}

struct Shaders {

}

