// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{marker::PhantomData, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use parking_lot::Mutex;

use crate::{
    barrier, BackendError, BackendResult, DeferedPass, DrawStream, Format, ImageLayout, ImageView,
    RenderArea,
};

use super::{
    frame::Frame, BindGroupStorage, BufferHandle, BufferSlice, BufferStorage, Device, ImageHandle,
    ImageStorage, RasterPipelineStorage,
};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum RenderTargetLoadOp {
    Clear,
    Load,
    #[default]
    Discard,
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum RenderTargetStoreOp {
    Store,
    #[default]
    Discard,
}

impl From<RenderTargetLoadOp> for vk::AttachmentLoadOp {
    fn from(value: RenderTargetLoadOp) -> Self {
        match value {
            RenderTargetLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            RenderTargetLoadOp::Load => vk::AttachmentLoadOp::LOAD,
            RenderTargetLoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

impl From<RenderTargetStoreOp> for vk::AttachmentStoreOp {
    fn from(value: RenderTargetStoreOp) -> Self {
        match value {
            RenderTargetStoreOp::Store => vk::AttachmentStoreOp::STORE,
            RenderTargetStoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClearRenderTarget {
    Color([f32; 4]),
    DepthStencil(f32, u32),
}

impl From<ClearRenderTarget> for vk::ClearValue {
    fn from(value: ClearRenderTarget) -> Self {
        match value {
            ClearRenderTarget::Color(color) => vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
            ClearRenderTarget::DepthStencil(depth, stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub format: Format,
    pub target: ImageView,
    pub layout: ImageLayout,
    pub load: RenderTargetLoadOp,
    pub store: RenderTargetStoreOp,
    pub clear: Option<ClearRenderTarget>,
}

impl RenderTarget {
    pub fn new(format: Format, target: ImageView, layout: ImageLayout) -> Self {
        Self {
            format,
            target,
            layout,
            load: RenderTargetLoadOp::default(),
            store: RenderTargetStoreOp::default(),
            clear: None,
        }
    }

    pub fn clear_input(mut self, color: ClearRenderTarget) -> Self {
        self.load = RenderTargetLoadOp::Clear;
        self.clear = Some(color);

        self
    }

    pub fn load_input(mut self) -> Self {
        self.load = RenderTargetLoadOp::Load;
        self.clear = None;

        self
    }

    pub fn store_output(mut self) -> Self {
        self.store = RenderTargetStoreOp::Store;

        self
    }

    fn build(&self) -> vk::RenderingAttachmentInfo {
        let mut builder = vk::RenderingAttachmentInfo::builder()
            .load_op(self.load.into())
            .store_op(self.store.into())
            .image_view(self.target)
            .image_layout(self.layout.into());
        if let Some(clear) = self.clear {
            builder = builder.clear_value(clear.into());
        }

        builder.build()
    }
}

const MAX_BARRIERS: usize = 32;
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

pub(crate) struct Pass<'a> {
    color_attachments: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_attachment: Option<RenderTarget>,
    render_area: RenderArea,
    streams: Vec<DrawStream>,
    barriers: ArrayVec<ImageBarrier, MAX_BARRIERS>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Pass<'a> {
    fn submit_one_stream(
        &self,
        context: &'a ExecutionContext<'a>,
        color_attachments: &[vk::Format],
        depth_attachment: Option<vk::Format>,
        stream: &DrawStream,
    ) -> BackendResult<vk::CommandBuffer> {
        let cb = context.frame.get_or_create_secondary_buffer(
            &context.device.raw,
            context.device.queue_familt_index,
        )?;
        let mut inheretence = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(color_attachments)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        if let Some(format) = depth_attachment {
            inheretence = inheretence.depth_attachment_format(format);
        }
        unsafe {
            context.device.raw.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(
                        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                            | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                    )
                    .inheritance_info(
                        &vk::CommandBufferInheritanceInfo::builder()
                            .push_next(&mut inheretence)
                            .build(),
                    )
                    .build(),
            )
        }
        .unwrap();
        unsafe {
            context
                .device
                .raw
                .cmd_set_viewport(cb, 0, &[self.render_area.into()]);
            context
                .device
                .raw
                .cmd_set_scissor(cb, 0, &[self.render_area.into()]);
        }

        stream.execute(context, cb).unwrap();
        unsafe { context.device.raw.end_command_buffer(cb).unwrap() };
        Ok(cb)
    }
}
impl<'a> DeferedPass for Pass<'a> {
    fn execute(&self, context: &ExecutionContext) -> BackendResult<()> {
        puffin::profile_function!();
        let color_attachments = self
            .color_attachments
            .iter()
            .map(RenderTarget::build)
            .collect::<ArrayVec<_, 8>>();
        let depth_attachment = self.depth_attachment.map(|x| x.build());
        let mut image_barriers = ArrayVec::<_, MAX_BARRIERS>::new();
        for barrier in &self.barriers {
            image_barriers.push(barrier.build(context.images, context.universal_queue)?);
        }

        let dependency = vk::DependencyInfo::builder()
            .image_memory_barriers(&image_barriers)
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();

        unsafe {
            context
                .device
                .raw
                .cmd_pipeline_barrier2(context.frame.main_cb.raw, &dependency)
        };

        let info = vk::RenderingInfo::builder()
            .render_area(self.render_area.into())
            .color_attachments(&color_attachments)
            .flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .layer_count(1);
        let info = if let Some(depth_attachment) = depth_attachment.as_ref() {
            info.depth_attachment(depth_attachment)
        } else {
            info
        };
        unsafe {
            context
                .device
                .raw
                .cmd_begin_rendering(context.frame.main_cb.raw, &info)
        };
        let color_attachments = self
            .color_attachments
            .iter()
            .map(|x| x.format.into())
            .collect::<Vec<_>>();
        let depth_attachment = self.depth_attachment.map(|x| x.format.into());
        let cbs = Arc::new(Mutex::new(vec![
            vk::CommandBuffer::null();
            self.streams.len()
        ]));

        {
            puffin::profile_scope!("Generate comand buffers");
            rayon::scope(|s| {
                for (index, stream) in self.streams.iter().enumerate() {
                    let cbs = cbs.clone();
                    let color_attachments = color_attachments.clone();
                    s.spawn(move |_| {
                        let cb = self.submit_one_stream(
                            context,
                            &color_attachments,
                            depth_attachment,
                            stream,
                        );
                        cbs.lock()[index] = cb.unwrap();
                    })
                }
            });
        };
        {
            puffin::profile_scope!("Execute command buffers");
            unsafe {
                context
                    .device
                    .raw
                    .cmd_execute_commands(context.frame.main_cb.raw, &cbs.lock())
            };
        }

        unsafe {
            context
                .device
                .raw
                .cmd_end_rendering(context.frame.main_cb.raw)
        };

        Ok(())
    }
}

pub struct FrameContext<'a> {
    pub render_area: RenderArea,
    pub target_view: ImageView,
    pub(crate) frame: &'a Frame,
    pub(crate) temp_buffer_handle: BufferHandle,
    pub(crate) passes: Mutex<Vec<Box<dyn DeferedPass>>>,
}

#[derive(Debug, Clone, Copy)]
pub enum BarrierType {
    ColorToAttachment,
    DepthToAttachment,
    ColorFromAttachmentToSampled,
    DepthFromAttachmentToSampled,
    ColorAttachmentToAttachment,
    DepthAttachmentToAttachment,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageBarrier {
    pub image: ImageHandle,
    pub ty: BarrierType,
}

impl ImageBarrier {
    pub fn depth_to_attachment(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::DepthToAttachment,
        }
    }

    pub fn color_to_attachment(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::ColorToAttachment,
        }
    }

    pub fn depth_attachment_to_sampled(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::DepthFromAttachmentToSampled,
        }
    }

    pub fn color_attachment_to_sampled(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::ColorFromAttachmentToSampled,
        }
    }

    pub fn depth_attachment_to_attachment(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::DepthAttachmentToAttachment,
        }
    }

    pub fn color_attachment_to_attachment(image: ImageHandle) -> Self {
        Self {
            image,
            ty: BarrierType::ColorAttachmentToAttachment,
        }
    }

    fn build(
        &self,
        images: &ImageStorage,
        queue_family_index: u32,
    ) -> BackendResult<vk::ImageMemoryBarrier2> {
        let image = images.get(self.image).ok_or(BackendError::InvalidHandle)?;
        let barrier = match self.ty {
            BarrierType::ColorToAttachment => {
                barrier::undefined_to_color_attachment(image, queue_family_index)
            }
            BarrierType::DepthToAttachment => {
                barrier::undefined_to_depth_attachment(image, queue_family_index)
            }
            BarrierType::ColorFromAttachmentToSampled => {
                barrier::color_attachment_to_sampled(image, queue_family_index)
            }
            BarrierType::DepthFromAttachmentToSampled => {
                barrier::depth_attachment_to_sampled(image, queue_family_index)
            }
            BarrierType::ColorAttachmentToAttachment => {
                barrier::color_write_to_write(image, queue_family_index)
            }
            BarrierType::DepthAttachmentToAttachment => {
                barrier::depth_write_to_write(image, queue_family_index)
            }
        };

        Ok(barrier)
    }
}

impl<'a> FrameContext<'a> {
    /// Allocate temporary memory on GPU and copy data there.
    ///
    /// Buffer slive is only valid during current frame, no need to free it in any way
    pub fn temp_allocate<T: Sized>(&self, data: &[T]) -> BackendResult<BufferSlice> {
        let offset = self.frame.temp_allocate(data)?;
        Ok(BufferSlice::new(self.temp_buffer_handle, offset))
    }

    /// Record render pass
    ///
    /// Render pass is a set of draw streams and barriers that executed before
    /// drawing start.
    ///
    /// Actual recording is done later, might be multithreaded.
    pub fn execute(
        &self,
        area: RenderArea,
        color_attachments: &[RenderTarget],
        depth_attachment: Option<RenderTarget>,
        streams: impl Iterator<Item = DrawStream>,
        barriers: &[ImageBarrier],
    ) {
        let pass = Pass {
            color_attachments: color_attachments
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth_attachment,
            render_area: area,
            streams: streams.collect(),
            barriers: barriers
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_BARRIERS>>(),
            _marker: PhantomData,
        };
        self.passes.lock().push(Box::new(pass));
    }
}

pub(crate) struct ExecutionContext<'a> {
    pub universal_queue: u32,
    pub device: &'a Device,
    pub frame: &'a Frame,
    pub images: &'a ImageStorage,
    pub buffers: &'a BufferStorage,
    pub pipelines: &'a RasterPipelineStorage,
    pub descriptors: &'a BindGroupStorage,
}

impl<'a> ExecutionContext<'a> {
    pub fn execute<I>(&self, passes: I) -> BackendResult<()>
    where
        I: Iterator<Item = &'a Box<dyn DeferedPass>>,
    {
        puffin::profile_function!();
        for pass in passes {
            pass.execute(self)?;
        }

        Ok(())
    }
}
