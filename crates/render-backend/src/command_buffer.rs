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

use std::slice;

use arrayvec::ArrayVec;
use ash::vk::{self, CommandBufferUsageFlags, FenceCreateFlags};
use vk_sync::{cmd::pipeline_barrier, BufferBarrier, GlobalBarrier, ImageBarrier};

use crate::BufferView;

use super::{
    BackendResult, FboCacheKey, FreeGpuResource, Image, Pipeline, RenderPass, MAX_ATTACHMENTS,
    MAX_COLOR_ATTACHMENTS,
};

pub struct CommandBuffer {
    pub pool: vk::CommandPool,
    pub raw: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl CommandBuffer {
    pub fn new(device: &ash::Device, queue_family_index: u32) -> BackendResult<Self> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .build();
        let pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }?;

        let command_buffer_allocation_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();

        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocation_info)? }[0];

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED)
            .build();

        let fence = unsafe { device.create_fence(&fence_create_info, None)? };

        Ok(Self {
            raw: command_buffer,
            fence,
            pool,
        })
    }

    pub fn record<F: FnOnce(CommandBufferRecorder)>(
        &self,
        device: &ash::Device,
        cb: F,
    ) -> BackendResult<()> {
        let recorder = CommandBufferRecorder::new(device, &self.raw)?;
        cb(recorder);

        Ok(())
    }

    pub fn wait(&self, device: &ash::Device) -> BackendResult<()> {
        unsafe { device.wait_for_fences(slice::from_ref(&self.fence), true, u64::MAX) }?;
        Ok(())
    }

    pub fn reset(&self, device: &ash::Device) -> BackendResult<()> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        Ok(())
    }
}

impl FreeGpuResource for CommandBuffer {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_command_buffers(self.pool, slice::from_ref(&self.raw));
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

pub struct RenderPassAttachment<'a> {
    pub image: &'a Image,
    pub clear: vk::ClearValue,
}

impl<'a> RenderPassAttachment<'a> {
    pub fn new(image: &'a Image, clear: vk::ClearValue) -> Self {
        Self { image, clear }
    }
}

pub struct CommandBufferRecorder<'a> {
    pub device: &'a ash::Device,
    pub cb: &'a vk::CommandBuffer,
}

impl<'a> CommandBufferRecorder<'a> {
    pub(self) fn new(device: &'a ash::Device, cb: &'a vk::CommandBuffer) -> BackendResult<Self> {
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
        depth_attachment: Option<RenderPassAttachment>,
        cb: F,
    ) {
        let clear_values = color_attachments
            .iter()
            .chain(depth_attachment.iter())
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
        let key = FboCacheKey::new(&color_attachments, depth_attachment);
        let dims = color_attachments
            .iter()
            .chain(depth_attachment.iter())
            .map(|image| image.desc.extent)
            .next();
        if let Some(dims) = dims {
            let framebuffer = render_pass
                .fbo_cache
                .get_or_create(self.device, key)
                .unwrap();

            let begin_pass_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass.raw)
                .framebuffer(framebuffer)
                .clear_values(&clear_values)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: dims[0] as _,
                        height: dims[1] as _,
                    },
                })
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
        } else {
            panic!("Can't start render pass without attachments");
        }
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
    pub fn bind_pipeline(&self, pipeline: &Pipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                *self.cb,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            )
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
