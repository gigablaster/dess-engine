use dess_assets::{ContentSource, GltfSource};
use dess_backend::{
    ClearRenderTarget, DrawStream, Format, ImageAspect, ImageLayout, ImageUsage,
    {BindGroupHandle, ImageBarrier, RenderTarget},
};
use dess_common::GameTime;
use dess_engine::{
    ModelCollection, PoolImageDesc, RelativeImageSize, ResourceHandle, ResourceLoader,
};
use dess_runner::{Client, RenderContext, Runner, UpdateContext};

#[derive(Default)]
struct ClearBackbuffer {
    model: ResourceHandle<ModelCollection>,
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
                        context.frame.render_area.width,
                        context.frame.render_area.height,
                    ],
                    PoolImageDesc {
                        format: Format::RGBA16_SFLOAT,
                        aspect: ImageAspect::Color,
                        usage: ImageUsage::Sampled | ImageUsage::ColorTarget,
                        resolution: RelativeImageSize::Backbuffer,
                    },
                )
                .unwrap();
            let color_attachment = RenderTarget::new(temp.view(), ImageLayout::ColorTarget)
                .store_output()
                .clear_input(ClearRenderTarget::Color([0.5, 0.5, 0.5, 1.0]));
            context.frame.execute(
                context.frame.render_area,
                &[color_attachment],
                None,
                [DrawStream::new(BindGroupHandle::invalid())].into_iter(),
                &[ImageBarrier::color_to_attachment(temp.image())],
            );
        }

        let color_attachment =
            RenderTarget::new(context.frame.target_view, ImageLayout::ColorTarget)
                .store_output()
                .clear_input(ClearRenderTarget::Color([0.125, 0.25, 0.5, 1.0]));
        context.frame.execute(
            context.frame.render_area,
            &[color_attachment],
            None,
            [DrawStream::new(BindGroupHandle::invalid())].into_iter(),
            &[],
        );

        Ok(())
    }

    fn hidden(&mut self, _value: bool) {}

    fn init(&mut self, context: UpdateContext) {
        self.model = context
            .resource_manager
            .request_model(GltfSource::new("models/Avocado/Avocado.gltf").get_ref());
    }
}

fn main() {
    let mut runner = Runner::new(
        ClearBackbuffer::default(),
        "Dess Engine - Clear backbuffer example",
    );
    runner.run();
}
