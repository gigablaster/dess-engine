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

use crate::{Device, Instance};

use super::{BackendResult, CommandBuffer, FreeGpuResource, QueueFamily};

pub struct FrameContext {
    pub presentation_cb: CommandBuffer,
    pub main_cb: CommandBuffer,
    pub render_finished: vk::Semaphore,
}

impl FrameContext {
    pub(crate) fn new(
        instance: &Instance,
        device: &ash::Device,
        queue_family: &QueueFamily,
        name: &str,
    ) -> BackendResult<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder()
            .flags(vk::SemaphoreCreateFlags::default())
            .build();

        let render_finished = unsafe { device.create_semaphore(&semaphore_info, None) }?;
        let presentation_cb = CommandBuffer::new(device, queue_family.index)?;
        let main_cb = CommandBuffer::new(device, queue_family.index)?;

        Device::set_object_name_impl(
            instance,
            device,
            presentation_cb.raw,
            &format!("{} - present", name),
        )?;

        Device::set_object_name_impl(instance, device, main_cb.raw, &format!("{} - main", name))?;

        Ok(Self {
            presentation_cb,
            main_cb,
            render_finished,
        })
    }

    pub(crate) fn reset(&self, device: &ash::Device) -> BackendResult<()> {
        self.main_cb.reset(device)?;
        self.presentation_cb.reset(device)?;

        Ok(())
    }
}

impl FreeGpuResource for FrameContext {
    fn free(&self, device: &ash::Device) {
        self.presentation_cb.free(device);
        self.main_cb.free(device);
        unsafe {
            device.destroy_semaphore(self.render_finished, None);
        }
    }
}
