use ash::vk;
use dess_backend::{vulkan::FrameContext, BackendError};
use dess_common::GameTime;

mod runner;

pub use runner::*;

#[derive(Debug, PartialEq, Eq)]
pub enum ClientState {
    Continue,
    Exit,
}
pub trait Client {
    fn tick(&mut self, dt: GameTime) -> ClientState;
    fn hidden(&mut self, value: bool);
    fn render(
        &self,
        context: FrameContext,
        render_area: vk::Rect2D,
        target_view: vk::ImageView,
        target_layout: vk::ImageLayout,
    ) -> Result<(), BackendError>;
}
