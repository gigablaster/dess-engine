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

use std::mem;

use arrayvec::ArrayVec;
use ash::vk::{self};
use parking_lot::Mutex;

use crate::{barrier, BackendError, BackendResult, DeferedPass, DrawStream};

use super::{
    frame::Frame, BufferHandle, BufferSlice, BufferStorage, DescriptorStorage, Device, ImageHandle,
    ImageStorage, PipelineStorage,
};

#[derive(Clone, Copy)]
pub struct RenderAttachment {
    pub target: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: Option<vk::ClearValue>,
}

impl RenderAttachment {
    pub fn new(target: vk::ImageView, layout: vk::ImageLayout) -> Self {
        Self {
            target,
            layout,
            load: vk::AttachmentLoadOp::DONT_CARE,
            store: vk::AttachmentStoreOp::DONT_CARE,
            clear: None,
        }
    }

    pub fn clear_input(mut self, color: vk::ClearValue) -> Self {
        self.load = vk::AttachmentLoadOp::CLEAR;
        self.clear = Some(color);

        self
    }

    pub fn load_input(mut self) -> Self {
        self.load = vk::AttachmentLoadOp::LOAD;
        self.clear = None;

        self
    }

    pub fn store_output(mut self) -> Self {
        self.store = vk::AttachmentStoreOp::STORE;

        self
    }

    fn build(&self) -> vk::RenderingAttachmentInfo {
        vk::RenderingAttachmentInfo::builder()
            .clear_value(self.clear.unwrap_or_default())
            .load_op(self.load)
            .store_op(self.store)
            .image_view(self.target)
            .image_layout(self.layout)
            .build()
    }
}

const MAX_BARRIERS: usize = 32;
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

pub(crate) struct Pass {
    color_attachments: ArrayVec<RenderAttachment, MAX_COLOR_ATTACHMENTS>,
    depth_attachment: Option<RenderAttachment>,
    rende_area: vk::Rect2D,
    streams: Vec<DrawStream>,
    barriers: ArrayVec<Barrier, MAX_BARRIERS>,
}

impl DeferedPass for Pass {
    fn execute(&self, context: &ExecutionContext) -> BackendResult<()> {
        puffin::profile_function!();
        let color_attachments = self
            .color_attachments
            .iter()
            .map(RenderAttachment::build)
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
            .render_area(self.rende_area)
            .color_attachments(&color_attachments)
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

        for stream in &self.streams {
            stream.execute(context, context.frame.main_cb.raw)?;
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
    pub render_area: vk::Rect2D,
    pub target_view: vk::ImageView,
    pub target_layout: vk::ImageLayout,
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
pub struct Barrier {
    pub image: ImageHandle,
    pub ty: BarrierType,
}

impl Barrier {
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
        let image = images
            .get_hot(self.image)
            .ok_or(BackendError::InvalidHandle)?;
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
        Ok(BufferSlice::new(
            self.temp_buffer_handle,
            offset,
            mem::size_of_val(data) as u32,
        ))
    }

    /// Record render pass
    ///
    /// Render pass is a set of draw streams and barriers that executed before
    /// drawing start.
    ///
    /// Actual recording is done later, might be multithreaded.
    pub fn execute(
        &self,
        area: vk::Rect2D,
        color_attachments: &[RenderAttachment],
        depth_attachment: Option<RenderAttachment>,
        streams: impl Iterator<Item = DrawStream>,
        barriers: &[Barrier],
    ) {
        let pass = Pass {
            color_attachments: color_attachments
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth_attachment,
            rende_area: area,
            streams: streams.collect(),
            barriers: barriers
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_BARRIERS>>(),
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
    pub pipelines: &'a PipelineStorage,
    pub descriptors: &'a DescriptorStorage,
}

impl<'a> ExecutionContext<'a> {
    pub fn execute<I>(&self, passes: I) -> BackendResult<()>
    where
        I: Iterator<Item = &'a Box<dyn DeferedPass>>,
    {
        for pass in passes {
            pass.execute(self)?;
        }

        Ok(())
    }
}
