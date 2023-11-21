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

use std::cmp::{max, min};

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
    device: &'a ash::Device,
    cb: vk::CommandBuffer,
    buffers: &'a BufferStorage,
    images: &'a ImageStorage,
    descriptors: &'a DescriptorStorage,
    pipelins: &'a PipelineStorage,
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

        Ok(RenderContext {
            cb: self.frame.main_cb.raw,
            device: &self.device.raw,
            buffers: self.buffers,
            images: self.images,
            descriptors: self.descriptors,
            pipelins: self.pipelins,
        })
    }

    pub fn temp_allocate<T: Sized>(&self, data: &[T]) -> BackendResult<BufferSlice> {
        let offset = self.frame.temp_allocate(data)?;
        Ok(BufferSlice::new(self.temp_buffer_handle, offset))
    }
}

impl<'a> RenderContext<'a> {
    fn resolve_buffer(&self, handle: BufferHandle) -> Option<vk::Buffer> {
        self.buffers.get_hot(handle).copied()
    }

    fn resolve_descriptor_set(&self, handle: DescriptorHandle) -> Option<vk::DescriptorSet> {
        self.descriptors.get_hot(handle).copied()
    }

    fn resolve_pipeline(
        &self,
        handle: PipelineHandle,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        self.pipelins.get(handle.index()).copied()
    }

    pub fn get_or_create_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> BackendResult<vk::ImageView> {
        let image = self
            .images
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?;
        image.get_or_create_view(self.device, desc)
    }

    pub fn execute(
        &self,
        area: vk::Rect2D,
        color_attachments: &[RenderAttachment],
        depth_attachment: Option<RenderAttachment>,
        streams: impl Iterator<Item = DrawStream>,
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
        unsafe { self.device.cmd_begin_rendering(self.cb, &info) };

        for stream in streams {
            self.execute_stream(stream)?;
        }

        unsafe { self.device.cmd_end_rendering(self.cb) };

        Ok(())
    }

    fn execute_stream(&self, stream: DrawStream) -> BackendResult<()> {
        let mut pipeline_layout = vk::PipelineLayout::null();
        let mut dynamic_buffer_offsets = [u32::MAX; 2];
        let mut dynamic_buffer_offset_count = 0usize;
        for command in stream.iter() {
            match command {
                crate::DrawCommand::BindPipeline(handle) => {
                    let (pipeline, layout) = self
                        .resolve_pipeline(handle)
                        .ok_or(BackendError::InvalidHandle)?;
                    if pipeline == vk::Pipeline::null() || layout == vk::PipelineLayout::null() {
                        return Err(BackendError::Fail);
                    }
                    unsafe {
                        self.device.cmd_bind_pipeline(
                            self.cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline,
                        )
                    };
                    pipeline_layout = layout;
                }
                crate::DrawCommand::BindVertexBuffer(index, slice) => {
                    let buffer = self
                        .resolve_buffer(slice.buffer)
                        .ok_or(BackendError::InvalidHandle)?;
                    unsafe {
                        self.device.cmd_bind_vertex_buffers(
                            self.cb,
                            index,
                            &[buffer],
                            &[slice.offset as u64],
                        )
                    };
                }
                crate::DrawCommand::UnbindVertexBuffer(index) => unsafe {
                    self.device
                        .cmd_bind_vertex_buffers(self.cb, index, &[vk::Buffer::null()], &[0])
                },
                crate::DrawCommand::BindIndexBuffer(slice) => {
                    let buffer = self
                        .resolve_buffer(slice.buffer)
                        .ok_or(BackendError::InvalidHandle)?;
                    unsafe {
                        self.device.cmd_bind_index_buffer(
                            self.cb,
                            buffer,
                            slice.offset as _,
                            vk::IndexType::UINT16,
                        )
                    }
                }
                crate::DrawCommand::SetDynamicBufferOffset(index, offset) => {
                    dynamic_buffer_offsets[index as usize] = offset;
                    dynamic_buffer_offset_count = max(index as usize, dynamic_buffer_offset_count);
                }
                crate::DrawCommand::UnsetDynamicBufferOffset(index) => {
                    dynamic_buffer_offset_count = min(index as usize, dynamic_buffer_offset_count);
                }
                crate::DrawCommand::BindDescriptorSet(index, handle) => {
                    let descriptor_set = self
                        .resolve_descriptor_set(handle)
                        .ok_or(BackendError::InvalidHandle)?;
                    let mut dynamic_offsets = ArrayVec::<_, 2>::new();
                    for offset in dynamic_buffer_offsets
                        .iter()
                        .take(dynamic_buffer_offset_count)
                    {
                        dynamic_offsets.push(*offset);
                    }
                    unsafe {
                        self.device.cmd_bind_descriptor_sets(
                            self.cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            index,
                            &[descriptor_set],
                            &dynamic_offsets,
                        )
                    };
                }
                crate::DrawCommand::UnbindDescriptorSet(index) => unsafe {
                    self.device.cmd_bind_descriptor_sets(
                        self.cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        index,
                        &[vk::DescriptorSet::null()],
                        &[],
                    );
                },
                crate::DrawCommand::Draw(first_index, triangle_count) => unsafe {
                    self.device
                        .cmd_draw(self.cb, triangle_count * 3, 1, first_index, 0)
                },
                crate::DrawCommand::DrawInstanced(_, _, _) => unimplemented!(),
            }
        }
        Ok(())
    }
}

impl<'a> Drop for RenderContext<'a> {
    fn drop(&mut self) {
        unsafe { self.device.end_command_buffer(self.cb) }.unwrap();
    }
}
