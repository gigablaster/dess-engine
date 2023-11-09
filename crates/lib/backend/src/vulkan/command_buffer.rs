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

use std::{slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self, CommandBufferUsageFlags, FenceCreateFlags};
use vk_sync::{cmd::pipeline_barrier, BufferBarrier, GlobalBarrier, ImageBarrier};

use crate::{GpuResource, RenderError};

use super::{
    FboCacheKey, Image, ImageViewDesc, RenderPass, MAX_ATTACHMENTS, MAX_COLOR_ATTACHMENTS,
};

pub(crate) struct CommandPool {
    pool: vk::CommandPool,
    free_cbs: Vec<CommandBuffer>,
    processing_cbs: Vec<CommandBuffer>,
    level: vk::CommandBufferLevel,
}

impl CommandPool {
    pub fn new(
        device: &ash::Device,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
        level: vk::CommandBufferLevel,
    ) -> Result<Self, RenderError> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family)
            .flags(flags)
            .build();
        let pool = unsafe { device.create_command_pool(&command_pool_info, None) }?;

        Ok(Self {
            pool,
            free_cbs: Vec::new(),
            processing_cbs: Vec::new(),
            level,
        })
    }
    pub fn get_or_create(&mut self, device: &ash::Device) -> Result<CommandBuffer, RenderError> {
        if let Some(cb) = self.free_cbs.pop() {
            Ok(cb)
        } else {
            Ok(CommandBuffer::new(device, self.pool, self.level)?)
        }
    }

    pub fn retire(&mut self, cb: CommandBuffer) {
        assert_eq!(
            self.pool, cb.pool,
            "Command buffer wasn't created from same pool"
        );
        self.processing_cbs.push(cb);
    }

    pub fn reset(&self, device: &ash::Device) -> Result<(), RenderError> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;

        Ok(())
    }

    pub fn recycle(&mut self) {
        self.free_cbs.append(&mut self.processing_cbs);
    }
}

impl GpuResource for CommandPool {
    fn free(&self, device: &ash::Device) {
        self.free_cbs.iter().for_each(|cb| cb.free(device));
        self.processing_cbs.iter().for_each(|cb| cb.free(device));
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CommandBuffer {
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CommandBuffer {
    fn new(
        device: &ash::Device,
        pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> Result<Self, RenderError> {
        let command_buffer_allocation_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(level)
            .build();

        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocation_info)? }[0];

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED)
            .build();

        let fence = unsafe { device.create_fence(&fence_create_info, None)? };

        Ok(Self {
            pool,
            raw: command_buffer,
            fence,
        })
    }

    pub fn wait(&self, device: &ash::Device) -> Result<(), RenderError> {
        unsafe { device.wait_for_fences(slice::from_ref(&self.fence), true, u64::MAX) }?;
        Ok(())
    }

    pub fn reset(&self, device: &ash::Device) -> Result<(), RenderError> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        Ok(())
    }

    pub fn record<F: FnOnce(CommandBufferRecorder)>(
        &self,
        device: &ash::Device,
        cb: F,
    ) -> Result<(), RenderError> {
        let recorder = CommandBufferRecorder::new(device, &self.raw)?;
        cb(recorder);

        Ok(())
    }

    pub fn raw(&self) -> vk::CommandBuffer {
        self.raw
    }

    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl GpuResource for CommandBuffer {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_command_buffers(self.pool, &[self.raw]);
            device.destroy_fence(self.fence, None);
        }
    }
}

pub struct RenderPassAttachment<'a> {
    pub image: &'a Arc<Image>,
    pub clear: vk::ClearValue,
}

impl<'a> RenderPassAttachment<'a> {
    pub fn color(image: &'a Arc<Image>, clear: [f32; 4]) -> Self {
        Self {
            image,
            clear: vk::ClearValue {
                color: vk::ClearColorValue { float32: clear },
            },
        }
    }

    pub fn depth(image: &'a Arc<Image>, depth: f32) -> Self {
        Self {
            image,
            clear: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
            },
        }
    }
}

pub struct CommandBufferRecorder<'a> {
    pub device: &'a ash::Device,
    pub cb: &'a vk::CommandBuffer,
}

impl<'a> CommandBufferRecorder<'a> {
    pub(self) fn new(
        device: &'a ash::Device,
        cb: &'a vk::CommandBuffer,
    ) -> Result<Self, RenderError> {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe { device.begin_command_buffer(*cb, &begin_info) }?;
        Ok(Self { device, cb })
    }

    pub fn render_pass<F: FnOnce(RenderPassRecorder)>(
        &self,
        render_pass: &RenderPass,
        color_attachments: &[RenderPassAttachment],
        depth_attachment: Option<&RenderPassAttachment>,
        cb: F,
    ) -> Result<(), RenderError> {
        let clear_values = color_attachments
            .iter()
            .chain(depth_attachment)
            .map(|attachment| attachment.clear)
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        let color_attachments = color_attachments
            .iter()
            .map(|attachment| attachment.image)
            .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
        let depth_attachment = depth_attachment
            .iter()
            .map(|attachment| attachment.image)
            .next();
        let dims = color_attachments
            .iter()
            .chain(depth_attachment.iter())
            .map(|image| image.desc().extent)
            .next()
            .expect("Render pass needs at least one attachment");

        let color_attachment_descs = color_attachments
            .iter()
            .map(|image| image.desc())
            .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();

        let depth_attachment_desc = depth_attachment.map(|image| image.desc());

        let key = FboCacheKey::new(dims, &color_attachment_descs, depth_attachment_desc);
        let framebuffer = render_pass.get_or_create_fbo(key)?;

        let views = color_attachments
            .into_iter()
            .map(|attachment| {
                attachment
                    .get_or_create_view(ImageViewDesc::default())
                    .unwrap()
            })
            .chain(depth_attachment.iter().map(|attachment| {
                attachment
                    .get_or_create_view(
                        ImageViewDesc::default().aspect_mask(vk::ImageAspectFlags::DEPTH),
                    )
                    .unwrap()
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let mut render_pass_attachment_desc = vk::RenderPassAttachmentBeginInfo::builder()
            .attachments(&views)
            .build();

        let begin_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw())
            .framebuffer(framebuffer)
            .clear_values(&clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: dims[0] as _,
                    height: dims[1] as _,
                },
            })
            .push_next(&mut render_pass_attachment_desc)
            .build();

        unsafe {
            self.device.cmd_begin_render_pass(
                *self.cb,
                &begin_pass_info,
                vk::SubpassContents::INLINE,
            )
        };

        cb(RenderPassRecorder {
            device: self.device,
            cb: self.cb,
        });

        Ok(())
    }

    pub fn barrier(
        &self,
        global: Option<GlobalBarrier>,
        buffers: &[BufferBarrier],
        images: &[ImageBarrier],
    ) {
        pipeline_barrier(self.device, *self.cb, global, buffers, images)
    }
}

impl<'a> Drop for CommandBufferRecorder<'a> {
    fn drop(&mut self) {
        unsafe { self.device.end_command_buffer(*self.cb) }.unwrap();
    }
}

pub struct RenderPassRecorder<'a> {
    pub device: &'a ash::Device,
    pub cb: &'a vk::CommandBuffer,
}

impl<'a> RenderPassRecorder<'a> {
    pub fn bind_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(*self.cb, vk::PipelineBindPoint::GRAPHICS, pipeline)
        };
    }

    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.device
                .cmd_set_viewport(*self.cb, 0, slice::from_ref(&viewport))
        };
    }

    pub fn set_scissor(&self, scissor: vk::Rect2D) {
        unsafe {
            self.device
                .cmd_set_scissor(*self.cb, 0, slice::from_ref(&scissor))
        };
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(*self.cb, buffer, offset, vk::IndexType::UINT16)
        };
    }

    pub fn bind_vertex_buffer(&self, buffer: vk::Buffer, offset: u64) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(*self.cb, 0, slice::from_ref(&buffer), &[offset])
        };
    }

    pub fn bind_descriptor_set(
        &self,
        slot: u32,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
    ) {
        unsafe {
            self.device
                .cmd_bind_descriptor_sets(*self.cb, bind_point, layout, slot, &[set], &[])
        }
    }

    pub fn draw(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                *self.cb,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                0,
            )
        };
    }
}

impl<'a> Drop for RenderPassRecorder<'a> {
    fn drop(&mut self) {
        unsafe { self.device.cmd_end_render_pass(*self.cb) };
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Semaphore {
    pub raw: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: &ash::Device) -> Result<Semaphore, RenderError> {
        let info = vk::SemaphoreCreateInfo::builder().build();
        Ok(Semaphore {
            raw: unsafe { device.create_semaphore(&info, None) }?,
        })
    }
}

impl GpuResource for Semaphore {
    fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.raw, None) };
    }
}
