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
use parking_lot::Mutex;

use crate::{barrier, BackendError, BackendResult, DrawStream};

use super::{
    frame::Frame, BufferHandle, BufferSlice, BufferStorage, DescriptorStorage, Device, ImageHandle,
    ImageStorage, PipelineStorage, PER_DRAW_BINDING_SLOT,
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
const MAX_COLOR_ATTACHMENTS: usize = 8;

pub(crate) struct Pass {
    color_attachments: ArrayVec<RenderAttachment, MAX_COLOR_ATTACHMENTS>,
    depth_attachment: Option<RenderAttachment>,
    rende_area: vk::Rect2D,
    streams: Vec<DrawStream>,
    barriers: ArrayVec<Barrier, MAX_BARRIERS>,
}

pub struct FrameContext<'a> {
    pub render_area: vk::Rect2D,
    pub target_view: vk::ImageView,
    pub target_layout: vk::ImageLayout,
    pub(crate) frame: &'a Frame,
    pub(crate) temp_buffer_handle: BufferHandle,
    pub(crate) passes: Mutex<Vec<Pass>>,
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
        self.passes.lock().push(pass);
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
        I: Iterator<Item = &'a Pass>,
    {
        for pass in passes {
            self.execute_pass(pass)?;
        }

        Ok(())
    }

    fn execute_pass(&self, pass: &Pass) -> BackendResult<()> {
        let color_attachments = pass
            .color_attachments
            .iter()
            .map(RenderAttachment::build)
            .collect::<ArrayVec<_, 8>>();
        let depth_attachment = pass.depth_attachment.map(|x| x.build());
        let mut image_barriers = ArrayVec::<_, MAX_BARRIERS>::new();
        for barrier in &pass.barriers {
            image_barriers.push(barrier.build(self.images, self.universal_queue)?);
        }

        let dependency = vk::DependencyInfo::builder()
            .image_memory_barriers(&image_barriers)
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();

        unsafe {
            self.device
                .raw
                .cmd_pipeline_barrier2(self.frame.main_cb.raw, &dependency)
        };

        let info = vk::RenderingInfo::builder()
            .render_area(pass.rende_area)
            .color_attachments(&color_attachments)
            .layer_count(1);
        let info = if let Some(depth_attachment) = depth_attachment.as_ref() {
            info.depth_attachment(depth_attachment)
        } else {
            info
        };
        unsafe {
            self.device
                .raw
                .cmd_begin_rendering(self.frame.main_cb.raw, &info)
        };

        for stream in &pass.streams {
            self.execute_stream(stream)?;
        }

        unsafe { self.device.raw.cmd_end_rendering(self.frame.main_cb.raw) };

        Ok(())
    }

    fn execute_stream(&self, stream: &DrawStream) -> BackendResult<()> {
        let mut pipeline_layout = vk::PipelineLayout::null();
        let mut dynamic_buffer_offsets = [u32::MAX; 2];
        let mut dynamic_buffer_offset_count = 0usize;
        for command in stream.iter() {
            match command {
                crate::DrawCommand::BindPipeline(handle) => {
                    let (pipeline, layout) = self
                        .pipelines
                        .get(handle.index())
                        .copied()
                        .ok_or(BackendError::InvalidHandle)?;
                    unsafe {
                        self.device.raw.cmd_bind_pipeline(
                            self.frame.main_cb.raw,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline,
                        )
                    };
                    pipeline_layout = layout;
                }
                crate::DrawCommand::BindVertexBuffer(index, slice) => {
                    let buffer = self
                        .buffers
                        .get_hot(slice.buffer)
                        .copied()
                        .ok_or(BackendError::InvalidHandle)?;
                    unsafe {
                        self.device.raw.cmd_bind_vertex_buffers(
                            self.frame.main_cb.raw,
                            index,
                            &[buffer],
                            &[slice.offset as u64],
                        )
                    };
                }
                crate::DrawCommand::UnbindVertexBuffer(index) => unsafe {
                    self.device.raw.cmd_bind_vertex_buffers(
                        self.frame.main_cb.raw,
                        index,
                        &[vk::Buffer::null()],
                        &[0],
                    )
                },
                crate::DrawCommand::BindIndexBuffer(slice) => {
                    let buffer = self
                        .buffers
                        .get_hot(slice.buffer)
                        .copied()
                        .ok_or(BackendError::InvalidHandle)?;
                    unsafe {
                        self.device.raw.cmd_bind_index_buffer(
                            self.frame.main_cb.raw,
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
                        .descriptors
                        .get_hot(handle)
                        .copied()
                        .ok_or(BackendError::InvalidHandle)?;
                    if descriptor_set == vk::DescriptorSet::null() {
                        return Err(BackendError::DescriptorIsntReady);
                    }
                    let mut dynamic_offsets = ArrayVec::<_, 2>::new();
                    if index == PER_DRAW_BINDING_SLOT {
                        for offset in dynamic_buffer_offsets
                            .iter()
                            .take(dynamic_buffer_offset_count)
                        {
                            dynamic_offsets.push(*offset);
                        }
                    }
                    unsafe {
                        self.device.raw.cmd_bind_descriptor_sets(
                            self.frame.main_cb.raw,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            index,
                            &[descriptor_set],
                            &dynamic_offsets,
                        )
                    };
                }
                crate::DrawCommand::UnbindDescriptorSet(index) => unsafe {
                    self.device.raw.cmd_bind_descriptor_sets(
                        self.frame.main_cb.raw,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        index,
                        &[vk::DescriptorSet::null()],
                        &[],
                    );
                },
                crate::DrawCommand::Draw(triangle_count) => unsafe {
                    self.device
                        .raw
                        .cmd_draw(self.frame.main_cb.raw, triangle_count * 3, 1, 0, 0)
                },
                crate::DrawCommand::DrawInstanced(_, _) => unimplemented!(),
            }
        }
        Ok(())
    }
}
