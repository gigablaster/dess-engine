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

use std::marker::PhantomData;

use arrayvec::ArrayVec;
use ash::vk::{
    self, ClearColorValue, ClearDepthStencilValue, ClearDepthStencilValueBuilder, ClearValue,
    CommandBufferUsageFlags, FenceCreateFlags, SubpassContents,
};

use crate::BackendResult;

use super::{Framebuffer, GpuResource, QueueFamily, RenderPass, MAX_ATTACHMENTS};

pub struct CommandBuffer {
    device: ash::Device,
    pub raw: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub pool: vk::CommandPool,
}

impl CommandBuffer {
    pub fn new(device: &ash::Device, queue_family: &QueueFamily) -> BackendResult<Self> {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family.index)
            .build();

        let pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

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
            device: device.clone(),
            raw: command_buffer,
            fence,
            pool,
        })
    }

    pub fn begin(&self) -> BackendResult<CommandBufferGenerator> {
        unsafe {
            self.device
                .reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())
        }?;
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe { self.device.begin_command_buffer(self.raw, &begin_info) }?;

        Ok(CommandBufferGenerator {
            device: self.device.clone(),
            cb: self.raw,
            phantom: PhantomData,
        })
    }
}

pub struct CommandBufferGenerator<'a> {
    device: ash::Device,
    cb: vk::CommandBuffer,
    phantom: PhantomData<&'a CommandBuffer>,
}

impl<'a> Drop for CommandBufferGenerator<'a> {
    fn drop(&mut self) {
        unsafe { self.device.end_command_buffer(self.cb) }.unwrap();
    }
}

impl<'a> CommandBufferGenerator<'a> {
    pub fn begin_pass(&self, render_pass: &RenderPass, framebuffer: &Framebuffer) {
        let clear = render_pass
            .color_attachments
            .iter()
            .map(|_| ClearValue {
                color: ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            })
            .chain(render_pass.depth_attachment.as_ref().map(|_| ClearValue {
                depth_stencil: ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        let pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw)
            .framebuffer(framebuffer.raw)
            .clear_values(&clear)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: framebuffer.extent,
            })
            .build();

        unsafe {
            self.device
                .cmd_begin_render_pass(self.cb, &pass_begin_info, SubpassContents::INLINE)
        };
    }

    pub fn end_pass(&self) {
        unsafe { self.device.cmd_end_render_pass(self.cb) };
    }
}

impl GpuResource for CommandBuffer {
    fn free(&mut self, device: &ash::Device, _: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
        }
    }
}
