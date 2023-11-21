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

use arrayvec::ArrayVec;
use ash::vk::{self};

use crate::{BackendError, BackendResult, DrawStream};

use super::{
    frame::Frame, BufferHandle, BufferSlice, BufferStorage, DescriptorHandle, DescriptorStorage,
    Device, ImageHandle, ImageStorage, ImageViewDesc, PipelineHandle, PipelineStorage,
};

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

pub struct FrameContext<'a> {
    pub(crate) device: &'a Device,
    pub(crate) frame: &'a Frame,
    pub(crate) images: &'a ImageStorage,
    pub(crate) buffers: &'a BufferStorage,
    pub(crate) descriptors: &'a DescriptorStorage,
    pub(crate) pipelins: &'a PipelineStorage,
    pub(crate) temp_buffer_handle: BufferHandle,
}

pub struct RenderContext<'a> {
    frame: &'a FrameContext<'a>,
}

impl<'a> FrameContext<'a> {
    pub fn render(&'a self) -> BackendResult<RenderContext<'a>> {
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();

        unsafe {
            self.device
                .raw
                .begin_command_buffer(self.frame.main_cb.raw, &info)
        }?;

        Ok(RenderContext { frame: self })
    }

    pub fn temp_allocate<T: Sized>(&self, data: &[T]) -> BackendResult<BufferSlice> {
        let offset = self.frame.temp_allocate(data)?;
        Ok(BufferSlice::new(self.temp_buffer_handle, offset))
    }
}

impl<'a> RenderContext<'a> {
    pub fn resolve_buffer(&self, handle: BufferHandle) -> Option<vk::Buffer> {
        self.frame.buffers.get_hot(handle).copied()
    }

    pub fn resolve_descriptor_set(&self, handle: DescriptorHandle) -> Option<vk::DescriptorSet> {
        self.frame.descriptors.get_hot(handle).copied()
    }

    pub fn resolve_pipeline(
        &self,
        handle: PipelineHandle,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        self.frame.pipelins.get(handle.index()).copied()
    }

    pub fn resolve_image_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> BackendResult<vk::ImageView> {
        let image = self
            .frame
            .images
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?;
        image.get_or_create_view(&self.frame.device.raw, desc)
    }

    pub fn execute(
        &self,
        area: vk::Rect2D,
        color_attachments: &[RenderAttachment],
        depth_attachment: Option<RenderAttachment>,
        _stream: DrawStream,
    ) -> BackendResult<()> {
        let color_attachments = color_attachments
            .iter()
            .map(RenderAttachment::build)
            .collect::<ArrayVec<_, 8>>();
        let depth_attachment = depth_attachment.map(|x| x.build());

        let info = vk::RenderingInfo::builder()
            .render_area(area)
            .color_attachments(&color_attachments);
        let info = if let Some(depth_attachment) = depth_attachment.as_ref() {
            info.depth_attachment(depth_attachment)
        } else {
            info
        };
        unsafe {
            self.frame
                .device
                .raw
                .cmd_begin_rendering(self.frame.frame.main_cb.raw, &info)
        };

        // TODO:: execute draw stream

        unsafe {
            self.frame
                .device
                .raw
                .cmd_end_rendering(self.frame.frame.main_cb.raw)
        };

        Ok(())
    }
}

impl<'a> Drop for RenderContext<'a> {
    fn drop(&mut self) {
        unsafe {
            self.frame
                .device
                .raw
                .end_command_buffer(self.frame.frame.main_cb.raw)
        }
        .unwrap();
    }
}
