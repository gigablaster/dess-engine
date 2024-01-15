// Copyright (C) 2023-2024 gigablaster

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

use std::{
    collections::HashMap,
    sync::Arc,
    thread::{self},
};

use ash::vk::{self, CommandBuffer};
use parking_lot::Mutex;

use crate::{AsVulkanCommandBuffer, CommandBufferRecorder, Device, Result, RenderPass, AsVulkan};

use super::{DropList, GpuAllocator};

const PREALLOCATED_COMMAND_BUFFERS: usize = 8;

#[derive(Debug)]
struct SecondaryCommandBufferPool {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    free: Vec<vk::CommandBuffer>,
}

pub struct SecondaryCommandBuffer<'a> {
    device: &'a ash::Device,
    cb: vk::CommandBuffer
}

impl<'a> SecondaryCommandBuffer<'a> {
    pub fn record(self, render_pass: &'a RenderPass, subpass: usize, framebuffer: vk::Framebuffer) -> CommandBufferRecorder {
        CommandBufferRecorder::secondary(self.device, self.cb, render_pass.as_vk(), subpass, framebuffer)
    }
}

impl SecondaryCommandBufferPool {
    pub fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )
        }?;

        Ok(Self {
            pool,
            free: Vec::new(),
            buffers: Vec::new(),
        })
    }

    pub fn get_or_create(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer> {
        if let Some(cb) = self.free.pop() {
            Ok(cb)
        } else {
            let mut buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .command_buffer_count(PREALLOCATED_COMMAND_BUFFERS as _)
                        .level(vk::CommandBufferLevel::SECONDARY),
                )
            }?;
            self.buffers.append(&mut buffers.clone());
            self.free.append(&mut buffers);
            let cb = self.free.pop().unwrap();

            Ok(cb)
        }
    }

    pub fn reset(&mut self, device: &ash::Device) {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
            .unwrap();
        self.free.clear();
        for it in &self.buffers {
            self.free.push(*it);
        }
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

#[derive(Debug)]
pub struct Frame {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    finished: vk::Semaphore,
    drop_list: DropList,
    per_thread_buffers: Mutex<HashMap<thread::ThreadId, SecondaryCommandBufferPool>>,
}

#[derive(Debug)]
pub struct FrameContext<'a> {
    pub(crate) device: &'a Device,
    pub(crate) frame: Option<Arc<Frame>>,
}

impl<'a> Drop for FrameContext<'a> {
    fn drop(&mut self) {
        self.device.end_frame(self.frame.take().unwrap());
    }
}

impl<'a> FrameContext<'a> {
    pub fn record_command_buffer(&self) -> CommandBufferRecorder {
        CommandBufferRecorder::primary(self.device.get(), self.frame.clone().unwrap().cb)
    }
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub(crate) fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self> {
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )?;
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build(),
            )?[0];
            let fence = device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )?;
            let finished =
                device.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)?;
            let drop_list = DropList::default();
            Ok(Self {
                pool,
                cb,
                fence,
                finished,
                drop_list,
                per_thread_buffers: Mutex::default(),
            })
        }
    }

    pub(crate) fn reset(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
    ) -> Result<()> {
        self.drop_list.purge(device, memory_allocator);
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        self.per_thread_buffers
            .lock()
            .iter_mut()
            .for_each(|(_, x)| x.reset(device));
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
        }

        Ok(())
    }

    pub(crate) fn free(&mut self, device: &ash::Device, memory_allocator: &mut GpuAllocator) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.finished, None);
        }
        self.drop_list.purge(device, memory_allocator);
        self.per_thread_buffers
            .lock()
            .drain()
            .for_each(|(_, x)| x.free(device));
    }

    pub(crate) fn assign_drop_list(&mut self, drop_list: DropList) {
        self.drop_list = drop_list;
    }

    pub fn secondary_buffer(
        &self,
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandBuffer> {
        let thread_id = thread::current().id();
        let mut pools = self.per_thread_buffers.lock();
        if let Some(pool) = pools.get_mut(&thread_id) {
            pool.get_or_create(device)
        } else {
            let mut pool = SecondaryCommandBufferPool::new(device, queue_family_index).unwrap();
            let cb = pool.get_or_create(device);
            pools.insert(thread_id, pool);

            cb
        }
    }

    pub(crate) fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl<'a> AsVulkanCommandBuffer for FrameContext<'a> {
    fn command_buffer(&self) -> vk::CommandBuffer {
        self.frame.clone().unwrap().cb
    }

    fn fence(&self) -> vk::Fence {
        self.frame.clone().unwrap().fence
    }
}
