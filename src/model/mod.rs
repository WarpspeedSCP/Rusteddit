use cgmath::prelude::*;
use cgmath::{Matrix4, Point3, Rad, SquareMatrix, Vector3};
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::GraphicsPipelineAbstract;

use std::sync::Arc;

use std::io::prelude::*;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::*;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::{Format, FormatDesc};
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

use crate::{InternalVertex, UBO};

pub mod font;

trait InternalCommandBuffer {}

impl InternalCommandBuffer for AutoCommandBuffer {}

//#[derive(Debug)]
pub struct Model<T: InternalVertex, F: FormatDesc> {
    pub vertices: Arc<ImmutableBuffer<[T]>>,
    pub indices: Arc<ImmutableBuffer<[u32]>>,
    pub texture: Option<Arc<ImmutableImage<F>>>,
    pub descriptor_set: Option<Arc<DescriptorSet + Sync + Send>>,
    pub tex_sampler: Option<Arc<Sampler>>,
    pub ubo: Option<Arc<CpuAccessibleBuffer<UBO>>>,
    pub cmd_bufs: Vec<Arc<AutoCommandBuffer>>,
    pub pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub cpu_verts: Vec<T>,
    pub cpu_inds: Vec<u32>,
}

impl<'a, T, F> Model<T, F>
where
    T: InternalVertex + 'static,
    F: 'static + Sync + Send + FormatDesc + vulkano::format::AcceptsPixels<[u8; 4]>,
{
    pub fn new(
        verts: Vec<T>,
        inds: Vec<u32>,
        texture: Vec<[u8; 4]>,
        format: F,
        dimensions: Dimensions,
        queue: Arc<Queue>,
        pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>,
        set_id: usize,
        ubo: bool,
    ) -> Result<Self, vulkano::sync::FlushError> {
        let (vert_buf, mut vert_future) = ImmutableBuffer::from_iter(
            verts.iter().cloned(),
            BufferUsage::vertex_buffer(),
            queue.clone(),
        )
        .expect("Could not create vertex buffer for model.");

        let (ind_buf, mut ind_future) = ImmutableBuffer::from_iter(
            inds.iter().cloned(),
            BufferUsage::index_buffer(),
            queue.clone(),
        )
        .expect("Could not create index buffer for model.");

        let (tex, tex_future) = if texture.len() > 0 {
            let x =
                ImmutableImage::from_iter(texture.into_iter(), dimensions, format, queue.clone())
                    .expect("Could not create image for model texture.");
            (Some(x.0), Box::new(x.1) as Box<GpuFuture>)
        } else {
            (
                None,
                Box::new(vulkano::sync::now(queue.device().clone())) as Box<GpuFuture>,
            )
        };

        let ubo_buf = if ubo {
            Some(
                CpuAccessibleBuffer::from_data(
                    queue.device().clone(),
                    BufferUsage::uniform_buffer(),
                    UBO {
                        model: cgmath::Matrix4::from_value(1.).into(),
                        view: cgmath::Matrix4::from_value(1.).into(),
                        proj: cgmath::Matrix4::from_value(1.).into(),
                    },
                )
                .expect("Could not create uniform buffer for model."),
            )
        } else {
            None
        };

        let desc_set = PersistentDescriptorSet::start(pipeline.clone(), set_id);

        match (ubo_buf, tex) {
            (Some(ubo_buf), Some(tex)) => {
                let tex_sampler = Sampler::simple_repeat_linear_no_mipmap(queue.device().clone());

                let desc_set = desc_set
                    .add_buffer(ubo_buf.clone())
                    .expect("Could not create UBO descriptor.")
                    .add_sampled_image(tex.clone(), tex_sampler.clone())
                    .expect("Could not add image to descriptor.")
                    .build()
                    .expect("Could not create descriptor set.");

                match vert_future.join(ind_future).join(tex_future).flush() {
                    Ok(_) => Ok(Model {
                        vertices: vert_buf,
                        indices: ind_buf,
                        texture: Some(tex),
                        tex_sampler: Some(tex_sampler),
                        descriptor_set: Some(Arc::new(desc_set)),
                        ubo: Some(ubo_buf),
                        cmd_bufs: vec![],
                        cpu_verts: verts,
                        cpu_inds: inds,
                        pipeline: pipeline,
                    }),
                    Err(x) => Err(x),
                }
            }
            (None, Some(tex)) => {
                let tex_sampler = Sampler::simple_repeat_linear_no_mipmap(queue.device().clone());

                let desc_set = desc_set
                    .add_sampled_image(tex.clone(), tex_sampler.clone())
                    .expect("Could not add image to descriptor.")
                    .build()
                    .expect("Could not create descriptor set.");

                match vert_future.join(ind_future).join(tex_future).flush() {
                    Ok(_) => Ok(Model {
                        vertices: vert_buf,
                        indices: ind_buf,
                        texture: Some(tex),
                        tex_sampler: Some(tex_sampler),
                        descriptor_set: Some(Arc::new(desc_set)),
                        ubo: None,
                        cmd_bufs: vec![],
                        cpu_verts: verts,
                        cpu_inds: inds,
                        pipeline: pipeline,
                    }),
                    Err(x) => Err(x),
                }
            }

            (Some(ubo_buf), None) => {
                let desc_set = desc_set
                    .add_buffer(ubo_buf.clone())
                    .expect("Could not create UBO descriptor.")
                    .build()
                    .expect("Could bot create descriptor set.");

                match vert_future.join(ind_future).join(tex_future).flush() {
                    Ok(_) => Ok(Model {
                        vertices: vert_buf,
                        indices: ind_buf,
                        texture: None,
                        tex_sampler: None,
                        descriptor_set: Some(Arc::new(desc_set)),
                        ubo: Some(ubo_buf),
                        cmd_bufs: vec![],
                        cpu_verts: verts,
                        cpu_inds: inds,
                        pipeline: pipeline,
                    }),
                    Err(x) => Err(x),
                }
            }

            (None, None) => match vert_future.join(ind_future).join(tex_future).flush() {
                Ok(_) => Ok(Model {
                    vertices: vert_buf,
                    indices: ind_buf,
                    texture: None,
                    tex_sampler: None,
                    descriptor_set: None,
                    ubo: None,
                    cmd_bufs: vec![],
                    cpu_verts: verts,
                    cpu_inds: inds,
                    pipeline: pipeline,
                }),
                Err(x) => Err(x),
            },
        }
    }
}

pub trait ModelBase {
    fn get_cmd_bufs(&self) -> &Vec<Arc<AutoCommandBuffer>>;
    fn get_cmd_bufs_mut(&mut self) -> &mut Vec<Arc<AutoCommandBuffer>>;
}

impl<T> ModelBase for Model<T, vulkano::format::Format>
where
    T: InternalVertex + Sized,
{
    fn get_cmd_bufs(&self) -> &Vec<Arc<AutoCommandBuffer>> {
        &self.cmd_bufs
    }

    fn get_cmd_bufs_mut(&mut self) -> &mut Vec<Arc<AutoCommandBuffer>> {
        &mut self.cmd_bufs
    }
}
