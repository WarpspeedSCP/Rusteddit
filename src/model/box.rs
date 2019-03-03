use vulkano::image::Dimensions;
use ::model::Model;

#[derive(Debug, Clone, Default)]
struct BoxOffset {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32
}

pub type Margin = BoxOffset;
pub type Padding = BoxOffset;

pub trait UI {
    /// Set margin.
    fn set_margin(&mut Self, Margin) -> &mut Self;
    
    /// Set padding.
    fn set_padding(&mut Self, Padding) -> &mut Self;

    /// Get padding.
    fn get_padding(&Self) -> &Padding;

    /// Get margin.
    fn get_margin(&Self) -> &Margin;

}

#[derive(Debug, Clone)]
pub struct Box: UI {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    margin: Margin,
    padding: Padding,
}

impl Box {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x: x,
            y: y,
            width: w,
            height: h,
            margin: Margin::default(),
            padding: Padding::default()
        }
    }

    
}

impl UI for Box {

    fn get_margin(&Self) -> &Margin {
        &self.margin
    }

    fn get_padding(&Self) -> &Padding {
        &self.padding
    }

    fn set_padding(&mut Self, padding: Padding) -> &mut Self {
        self.padding = padding;
        &mut self
    }

    fn set_margin(&mut Self, margin: Margin) -> &mut Self {
        self.margin = margin;
        &mut self
    }

}


pub fn draw_box() {
    
}
