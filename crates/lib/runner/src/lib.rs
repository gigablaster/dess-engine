use std::sync::Arc;

use dess_backend::{vulkan::FrameContext, BackendError};
use dess_common::GameTime;

mod runner;

use dess_engine::{BufferPool, ResourceManager, ResourcePool};
pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}

pub struct RenderContext<'a> {
    pub frame: &'a FrameContext<'a>,
    pub resource_pool: &'a ResourcePool<'a>,
    pub buffer_pool: &'a BufferPool,
}

pub struct UpdateContext {
    pub resource_manager: Arc<ResourceManager>,
}

pub trait Client {
    fn init(&mut self, context: UpdateContext);
    fn tick(&mut self, context: UpdateContext, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(&self, context: RenderContext) -> Result<(), BackendError>;
}
