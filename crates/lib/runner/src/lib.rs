use std::sync::Arc;

use dess_backend::{BackendError, FrameContext};
use dess_common::GameTime;

mod runner;

use dess_engine::{
    BufferPool, PipelineCache, PipelineCacheBuilder, ResourceManager, TemporaryImagePool,
};
pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}

pub struct RenderContext<'a> {
    pub frame: &'a FrameContext<'a>,
    pub resource_pool: &'a TemporaryImagePool<'a>,
    pub buffer_pool: &'a BufferPool,
    pub pipeline_cache: &'a PipelineCache,
}

pub struct UpdateContext {
    pub resource_manager: Arc<ResourceManager>,
}

pub struct InitContext<'a> {
    pub resource_manager: Arc<ResourceManager>,
    pub pipeline_cache: &'a PipelineCacheBuilder<'a>,
}

pub trait Client {
    fn init(&mut self, context: InitContext);
    fn tick(&mut self, context: UpdateContext, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(&self, context: RenderContext) -> Result<(), BackendError>;
}
