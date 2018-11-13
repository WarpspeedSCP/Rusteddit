//#![feature(const_slice_len)]
#![feature(trace_macros)]

#[macro_use]
extern crate vulkano;

#[macro_use]
extern crate vulkano_shaders;

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;

extern crate image;
extern crate vulkano_win;
extern crate winit;

mod basic;
mod shaders;

fn main() {
    basic::main();
}