use std::sync::Arc;

use dess_backend::{
    BackendError, BackendResult, Device, Format, FrameContext, ImageAspect, ImageUsage,
};
use dess_common::GameTime;

mod runner;

use dess_engine::{
    BufferPool, Error, PipelineCache, PoolImageDesc, RelativeImageSize, ResourceManager,
    TemporaryImage, TemporaryImagePool,
};
pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}

pub struct RenderContext<'a> {
    pub device: &'a Device,
    pub frame: &'a FrameContext<'a>,
    temporary_image_pool: &'a TemporaryImagePool<'a>,
}

impl<'a> RenderContext<'a> {
    pub fn get_temporary_render_target(
        &self,
        format: Format,
        usage: ImageUsage,
        aspect: ImageAspect,
        resolution: RelativeImageSize,
    ) -> BackendResult<TemporaryImage> {
        let backbuffer_dims = [self.frame.render_area.width, self.frame.render_area.height];
        self.temporary_image_pool.temp_image(
            backbuffer_dims,
            PoolImageDesc {
                format,
                aspect,
                usage,
                resolution,
            },
        )
    }
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
    fn render(&self, context: RenderContext) -> Result<(), BackendError>;
}
