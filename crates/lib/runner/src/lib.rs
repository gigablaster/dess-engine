use dess_backend::{vulkan::FrameContext, BackendError, ResourcePool};
use dess_common::GameTime;

mod runner;

pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}

pub struct RenderContext<'a> {
    pub frame: &'a FrameContext<'a>,
    pub pool: &'a ResourcePool<'a>,
}

pub trait Client {
    fn tick(&mut self, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(&self, context: RenderContext) -> Result<(), BackendError>;
}
