use std::sync::Arc;

use dess_backend::{Device, Error, FrameContext};
use dess_common::GameTime;

mod runner;

use dess_engine::{BufferPool, PipelineCache, ResourceManager};
pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}

pub struct UpdateContext {
    pub resource_manager: Arc<ResourceManager>,
}

pub struct InitContext<'a> {
    pub resource_manager: Arc<ResourceManager>,
    pub pipeline_cache: &'a PipelineCache<'a>,
    pub render_device: &'a Device,
    pub buffer_pool: &'a BufferPool,
}

pub trait Client {
    fn init(&mut self, context: InitContext);
    fn tick(&mut self, context: UpdateContext, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(&self, context: &FrameContext) -> Result<(), Error>;
}
