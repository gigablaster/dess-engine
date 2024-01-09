// Copyright (C) 2024 gigablaster

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

use ash::vk;

use crate::{Device, Result};

#[derive(Debug)]
pub struct CommandBuffer {
    device: Arc<Device>,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CommandBuffer {
    pub fn graphics<S: AsRef<str>>(device: &Arc<Device>, name: Option<S>) -> Result<Self> {
        Self::new(device, device.main_queue_index(), name)
    }

    pub fn transfer<S: AsRef<str>>(device: &Arc<Device>, name: Option<S>) -> Result<Self> {
        Self::new(device, device.transfer_queue_index(), name)
    }

    fn new<S: AsRef<str>>(device: &Arc<Device>, queue_index: u32, name: Option<S>) -> Result<Self> {
        unsafe {
            let pool = device.get().create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )?;
            let cb = device.get().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build(),
            )?[0];
            let fence = device.get().create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )?;
            if let Some(name) = name {
                device.set_object_name(pool, format!("{} - Pool", name.as_ref()));
                device.set_object_name(cb, format!("{} - CB", name.as_ref()));
                device.set_object_name(fence, format!("{} - Fence", name.as_ref()));
            }
            Ok(Self {
                device: device.clone(),
                pool,
                cb,
                fence,
            })
        }
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device
                .get()
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
        }?;

        Ok(())
    }

    pub fn record(&self) -> CommandBufferRecorder {
        CommandBufferRecorder {
            device: self.device.get(),
            cb: self.cb,
        }
    }

    pub fn get(&self) -> vk::CommandBuffer {
        self.cb
    }

    pub fn fence(&self) -> vk::Fence {
        self.fence
    }

    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.device
                .get()
                .wait_for_fences(slice::from_ref(&self.fence), true, u64::MAX)?;
            self.device
                .get()
                .reset_fences(slice::from_ref(&self.fence))?;
        }

        Ok(())
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.get().destroy_fence(self.fence, None);
            self.device.get().destroy_command_pool(self.pool, None);
        }
    }
}

pub struct CommandBufferRecorder<'a> {
    device: &'a ash::Device,
    cb: vk::CommandBuffer,
}

impl<'a> CommandBufferRecorder<'a> {
    pub(crate) fn new(device: &'a ash::Device, cb: vk::CommandBuffer) -> Self {
        unsafe {
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
        }
        .unwrap();
        Self { device, cb }
    }
}

impl<'a> Drop for CommandBufferRecorder<'a> {
    fn drop(&mut self) {
        unsafe { self.device.end_command_buffer(self.cb).unwrap() };
    }
}

pub trait AsVulkanCommandBuffer {
    fn fence(&self) -> vk::Fence;
    fn command_buffer(&self) -> vk::CommandBuffer;
}

impl AsVulkanCommandBuffer for CommandBuffer {
    fn fence(&self) -> vk::Fence {
        self.fence
    }

    fn command_buffer(&self) -> vk::CommandBuffer {
        self.cb
    }
}
