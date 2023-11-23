use ash::vk::{self};
use dess_backend::{
    barrier, vulkan::RenderAttachment, DrawStream, PoolImageDesc, RelativeImageSize,
};
use dess_common::GameTime;
use dess_runner::{Client, RenderContext, Runner};

#[derive(Default)]
struct ClearBackbuffer {}

impl Client for ClearBackbuffer {
    fn tick(&mut self, _dt: GameTime) -> dess_runner::ClientState {
        dess_runner::ClientState::Continue
    }

    fn render(&self, context: RenderContext) -> Result<(), dess_backend::BackendError> {
        {
            let temp = context
                .pool
                .temp_image(
                    [
                        context.frame.render_area.extent.width,
                        context.frame.render_area.extent.height,
                    ],
                    PoolImageDesc {
                        format: vk::Format::R16G16B16A16_SFLOAT,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                        resolution: RelativeImageSize::Backbuffer,
                    },
                )
                .unwrap();
            let color_attachment = RenderAttachment::new(
                temp.as_ref().view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )
            .store_output()
            .clear_input(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [100.0, 200.0, 400.0, 1.0],
                },
            });
            context.frame.barrier(
                &[barrier::undefined_to_color_attachment(
                    &temp,
                    context.frame.universal_queue,
                )],
                &[],
            );
            context
                .frame
                .execute(
                    context.frame.render_area,
                    &[color_attachment],
                    None,
                    [DrawStream::default()].into_iter(),
                )
                .unwrap();
            context.frame.barrier(
                &[barrier::color_attachment_to_sampled(
                    &temp,
                    context.frame.universal_queue,
                )],
                &[],
            );
        }

        let color_attachment =
            RenderAttachment::new(context.frame.target_view, context.frame.target_layout)
                .store_output()
                .clear_input(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.125, 0.25, 0.6, 1.0],
                    },
                });
        context.frame.execute(
            context.frame.render_area,
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
