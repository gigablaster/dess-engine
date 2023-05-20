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

use ash::vk;

use super::{BackendResult, CommandBuffer, FreeGpuResource, QueueFamily};

pub struct FrameContext {
    pub presentation_cb: CommandBuffer,
    pub main_cb: CommandBuffer,
    pub pool: vk::CommandPool,
    pub render_finished: vk::Semaphore,
}

impl FrameContext {
    pub fn new(device: &ash::Device, queue_family: &QueueFamily) -> BackendResult<Self> {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family.index)
            .build();

        let pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

        let semaphore_info = vk::SemaphoreCreateInfo::builder()
            .flags(vk::SemaphoreCreateFlags::default())
            .build();

        let render_finished = unsafe { device.create_semaphore(&semaphore_info, None) }?;

        Ok(Self {
            presentation_cb: CommandBuffer::new(device, pool)?,
            main_cb: CommandBuffer::new(device, pool)?,
            pool,
            render_finished,
        })
    }

    pub(crate) fn reset(&self, device: &ash::Device) -> BackendResult<()> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;

        Ok(())
    }
}

impl FreeGpuResource for FrameContext {
    fn free(&self, device: &ash::Device) {
        self.presentation_cb.free(device);
        self.main_cb.free(device);
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_semaphore(self.render_finished, None);
        }
    }
}
