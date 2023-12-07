use dess_backend::{vulkan::FrameContext, BackendError, BufferPool, ResourcePool};
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
    pub resource_pool: &'a ResourcePool<'a>,
    pub buffer_pool: &'a BufferPool<'a>,
}

pub trait Client {
    fn tick(&mut self, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(&self, context: RenderContext) -> Result<(), BackendError>;
}
