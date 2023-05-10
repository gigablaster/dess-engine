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

use crate::BackendResult;

use super::{
    Device, FboCacheKey, FreeGpuResource, Image, RenderPass, MAX_ATTACHMENTS, MAX_COLOR_ATTACHMENTS,
};

pub struct CommandBuffer {
    pub raw: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub pool: vk::CommandPool,
}

impl CommandBuffer {
    pub(crate) fn new(device: &ash::Device, pool: vk::CommandPool) -> BackendResult<Self> {
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

    pub fn record<'a>(&'a self, device: &'a Device) -> BackendResult<CommandBufferRecorder<'a>> {
        CommandBufferRecorder::new(&device.raw, &self.raw)
    }
}

impl FreeGpuResource for CommandBuffer {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_command_buffers(self.pool, slice::from_ref(&self.raw));
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

    pub fn render_pass(
        &self,
        device: &ash::Device,
        render_pass: &RenderPass,
        color_attachments: &[RenderPassAttachment],
        depth_attachment: Option<RenderPassAttachment>,
    ) -> RenderPassRecorder {
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
            let framebuffer = render_pass.fbo_cache.get_or_create(device, key).unwrap();

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

            RenderPassRecorder {
                device: self.device,
                cb: self.cb,
            }
        } else {
            panic!("Can't start render pass without attachments");
        }
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

impl<'a> Drop for RenderPassRecorder<'a> {
    fn drop(&mut self) {
        unsafe { self.device.cmd_end_render_pass(*self.cb) };
    }
}
