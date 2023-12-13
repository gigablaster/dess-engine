use ash::vk::{self};
use dess_backend::{
    vulkan::{Barrier, DescriptorHandle, RenderAttachment},
    DrawStream,
};
use dess_common::GameTime;
use dess_engine::{AssetCacheFns, AssetHandle, PoolImageDesc, RelativeImageSize, RenderModel};
use dess_runner::{Client, RenderContext, Runner, UpdateContext};

#[derive(Default)]
struct ClearBackbuffer {
    model: AssetHandle<RenderModel>,
}

impl Client for ClearBackbuffer {
    fn tick(&mut self, _context: UpdateContext, _dt: GameTime) -> dess_runner::ClientState {
        dess_runner::ClientState::Continue
    }

    fn render(&self, context: RenderContext) -> Result<(), dess_backend::BackendError> {
        {
            let temp = context
                .resource_pool
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
            let color_attachment =
                RenderAttachment::new(temp.view(), vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store_output()
                    .clear_input(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [100.0, 200.0, 400.0, 1.0],
                        },
                    });
            context.frame.execute(
                context.frame.render_area,
                &[color_attachment],
                None,
                [DrawStream::new(DescriptorHandle::invalid())].into_iter(),
                &[Barrier::color_to_attachment(temp.image())],
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
            [DrawStream::new(DescriptorHandle::invalid())].into_iter(),
            &[],
        );

        Ok(())
    }

    fn hidden(&mut self, _value: bool) {}

    fn init(&mut self, context: UpdateContext) {
        self.model = context
            .asset_cache
            .request_model("models/Avocado/Avocado.gltf");
    }
}

fn main() {
    let mut runner = Runner::new(
        ClearBackbuffer::default(),
        "Dess Engine - Clear backbuffer example",
    );
    runner.run();
}
