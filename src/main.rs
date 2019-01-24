//#![feature(const_slice_len)]
#![feature(range_contains)]
#![feature(duration_as_u128)]
#![feature(trace_macros)]

#[macro_use]
extern crate vulkano;

#[macro_use]
extern crate vulkano_shaders;

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;

#[macro_use]
extern crate itertools;

extern crate cgmath;
extern crate image;
extern crate ordered_float;
extern crate tobj;
extern crate vulkano_win;
extern crate winit;

mod basic;
mod model;
mod renderer;
mod shaders;
//mod three_d;
//mod resizable;

use std::collections::HashSet;
use std::sync::Arc;

use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};

lazy_static! {
    pub static ref LAYERS: &'static [&'static str] = &[
        "VK_LAYER_LUNARG_standard_validation",
//        "VK_LAYER_LUNARG_monitor",
 //       "VK_LAYER_LUNARG_api_dump"
    ];
}

pub struct QueueFamilyHolder<'a> {
    pub graphics_q: QueueFamily<'a>,
    pub compute_q: QueueFamily<'a>,
    pub transfer_q: QueueFamily<'a>,
}

pub struct PriorityHolder {
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
pub struct VertexPCT {
    pub inPosition: [f32; 3],
    pub inColour: [f32; 3],
    pub inTexCoord: [f32; 2],
}

impl_vertex!(VertexPCT, inPosition, inColour, inTexCoord);

impl VertexPCT {
    pub fn new() -> Self {
        Self {
            inPosition: [0.; 3],
            inColour: [1.; 3],
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

    pub fn uv(mut self, u: [f32; 2]) -> Self {
        self.inTexCoord = u;
        self
    }
}

pub trait InternalVertex: Clone + Sync + Send {}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexPCNT {
    pub inPosition: [f32; 3],
    pub inColour: [f32; 3],
    pub inNormal: [f32; 3],
    pub inTexCoord: [f32; 2],
}

impl_vertex!(VertexPCNT, inPosition, inColour, inNormal, inTexCoord);

impl VertexPCNT {
    pub fn new() -> Self {
        Self {
            inPosition: [0.; 3],
            inColour: [1.; 3],
            inNormal: [1.; 3],
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

    pub fn normal(mut self, n: [f32; 3]) -> Self {
        self.inNormal = n;
        self
    }

    pub fn uv(mut self, u: [f32; 2]) -> Self {
        self.inTexCoord = u;
        self
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexPNT {
    pub inPosition: [f32; 3],
    pub inNormal: [f32; 3],
    pub inTexCoord: [f32; 2],
}

impl_vertex!(VertexPNT, inPosition, inNormal, inTexCoord);

impl VertexPNT {
    pub fn new() -> Self {
        Self {
            inPosition: [0.; 3],
            inNormal: [1.; 3],
            inTexCoord: [0.; 2],
        }
    }

    pub fn pos(mut self, p: [f32; 3]) -> Self {
        self.inPosition = p;
        self
    }

    pub fn normal(mut self, n: [f32; 3]) -> Self {
        self.inNormal = n;
        self
    }

    pub fn uv(mut self, u: [f32; 2]) -> Self {
        self.inTexCoord = u;
        self
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexPC {
    pub inPosition: [f32; 3],
    pub inColour: [f32; 3],
}

impl VertexPC {
    pub fn new() -> Self {
        Self {
            inPosition: [0.; 3],
            inColour: [1.; 3],
        }
    }

    pub fn colour(mut self, c: [f32; 3]) -> Self {
        self.inColour = c;
        self
    }

    pub fn pos(mut self, p: [f32; 3]) -> Self {
        self.inPosition = p;
        self
    }

}

impl_vertex!(VertexPC, inPosition, inColour);

impl InternalVertex for VertexPCNT {}
impl InternalVertex for VertexPCT {}
impl InternalVertex for VertexPNT {}
impl InternalVertex for VertexPC {}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct UBO {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct InstanceUBO {}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Camera {
    pub pos: cgmath::Vector3<f32>,
    pub dir: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pitch: f64,
}

impl Camera {
    pub fn update(&mut self, m_delta: (f64, f64), sensitivity: f64) {
        let x_delta = m_delta.0 * sensitivity;
        let mut y_delta = m_delta.1 * sensitivity;
        self.pitch += y_delta;

        if self.pitch > 1.55334303 {
            self.pitch = 1.553;
            y_delta = 0.;
        }
        if self.pitch < -1.55334303 {
            self.pitch = -1.553;
            y_delta = 0.;
        }

        use ordered_float::OrderedFloat;
        if OrderedFloat::from(x_delta) != OrderedFloat::from(0.) {
            self.dir = revolve_left(x_delta as f32, &self.dir, &self.up);
        }

        if OrderedFloat::from(y_delta) != OrderedFloat::from(0.) {
            let x = revolve_up(y_delta as f32, &self.dir, &self.up);
            self.dir = x.0;
        }
        // For full 3d movement.
        //self.up = x.1;
    }
}

pub fn init_instance() -> Arc<Instance> {
    Instance::new(
        Some(&app_info_from_cargo_toml!()),
        &InstanceExtensions {
            khr_surface: true,
            //            khr_xcb_surface: true,
            khr_xlib_surface: true,
            ..InstanceExtensions::none()
        },
        LAYERS.iter().cloned(),
    )
    .expect("failed to create instance")
}

pub fn get_valid_queue_families<'a>(physdev: &'a PhysicalDevice) -> QueueFamilyHolder<'a> {
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
        .find(|x| x.supports_transfers() && !(x.supports_graphics() || x.supports_compute())).unwrap_or_else(
            || { 
                eprintln!("No queue families with exclusive transfer support found! Switching to using universal queue family for transfer ops."); 
                graphics_q.clone() 
            }
        );

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

pub fn revolve_left(
    rads: f32,
    eye: &cgmath::Vector3<f32>,
    up: &cgmath::Vector3<f32>,
) -> cgmath::Vector3<f32> {
    rotate(rads, up) * eye
}

pub fn revolve_up(
    rads: f32,
    eye: &cgmath::Vector3<f32>,
    up: &cgmath::Vector3<f32>,
) -> (cgmath::Vector3<f32>, cgmath::Vector3<f32>) {
    let c = eye.cross(up.clone());
    (rotate(rads, &c) * eye, rotate(rads, &c) * up)
}

pub fn project_on_vec(u: &cgmath::Vector3<f32>, v: &cgmath::Vector3<f32>) -> cgmath::Vector3<f32> {
    v * (cgmath::dot(u.clone(), v.clone()) / (v.x * v.x + v.y * v.y + v.z * v.z))
}

pub fn project_on_plane(
    u: &cgmath::Vector3<f32>,
    n: &cgmath::Vector3<f32>,
) -> cgmath::Vector3<f32> {
    u - project_on_vec(u, n)
}

pub fn rotate(rads: f32, axis: &cgmath::Vector3<f32>) -> cgmath::Matrix3<f32> {
    let aSin = rads.sin();
    let aCos = rads.cos();

    use cgmath::*;
    let c1 = Matrix3::from_value(aCos);

    let mut c2 = Matrix3::from_value(0f32);

    let n = axis.normalize();

    c2[0][0] = n.x * n.x;
    c2[0][1] = n.x * n.y;
    c2[0][2] = n.x * n.z;

    c2[1][0] = n.y * n.x;
    c2[1][1] = n.y * n.y;
    c2[1][2] = n.y * n.z;

    c2[2][0] = n.z * n.x;
    c2[2][1] = n.z * n.y;
    c2[2][2] = n.z * n.z;

    c2 *= 1f32 - aCos;

    let mut c3 = Matrix3::from_value(0f32);

    c3[0][1] = -n.z;
    c3[0][2] = n.y;

    c3[1][0] = n.z;
    c3[1][2] = -n.x;

    c3[2][0] = -n.y;
    c3[2][1] = n.x;

    c3 *= aSin;

    return c1 + c2 + c3;
}

fn main() {
    //three_d::main()
    renderer::main();
}

// fn screenshot(device: Arc<vulkano::device::Device>, image: Arc<vulkano::image::SwapchainImage> ) {
// }
