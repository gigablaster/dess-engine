use ash::vk::{self, Rect2D};
use dess_backend::{
    vulkan::{FrameContext, RenderAttachment},
    DrawStream,
};
use dess_common::GameTime;
use dess_runner::{Client, Runner};

#[derive(Default)]
struct ClearBackbuffer {}

impl Client for ClearBackbuffer {
    fn tick(&mut self, _dt: GameTime) -> dess_runner::ClientState {
        dess_runner::ClientState::Continue
    }

    fn render(
        &self,
        context: FrameContext,
        render_area: Rect2D,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> Result<(), dess_backend::BackendError> {
        let color_attachment = RenderAttachment::new(view, layout)
            .store_output()
            .clear_input(vk::ClearValue {
                color: vk::ClearColorValue {
                    uint32: [255, 0, 0, 255],
                },
            });
        context.execute(
            render_area,
            &[color_attachment],
            None,
            [DrawStream::default()].into_iter(),
        )
    }

    fn hidden(&mut self, _value: bool) {}
}

fn main() {
    let mut runner = Runner::new(
        ClearBackbuffer::default(),
        "Dess Engine - Clear backbuffer example",
    );
    runner.run();
}
