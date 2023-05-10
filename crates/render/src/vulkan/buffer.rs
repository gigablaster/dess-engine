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

use std::sync::Arc;

use ash::vk;
use gpu_alloc::{Request, UsageFlags};

use crate::{Allocation, BackendResult};

use super::Device;

pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: UsageFlags,
    pub aligment: Option<u64>,
}

pub struct Buffer {
    device: Arc<Device>,
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub allocation: Option<Allocation>,
}

impl Buffer {
    pub fn new(device: &Arc<Device>, desc: BufferDesc, name: Option<&str>) -> BackendResult<Self> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(desc.size as _)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.raw.create_buffer(&buffer_create_info, None) }?;
        let mut requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        if let Some(aligment) = desc.aligment {
            requirements.alignment = requirements.alignment.max(aligment);
        }

        let allocation = device.allocate(|allocator, device| unsafe {
            allocator.alloc(
                device,
                Request {
                    size: requirements.size,
                    align_mask: requirements.alignment,
                    usage: desc.memory_usage,
                    memory_types: !0,
                },
            )
        })?;

        unsafe {
            device
                .raw
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
        }?;

        if let Some(name) = name {
            device.set_object_name(buffer, name)?;
        }

        Ok(Self {
            device: device.clone(),
            raw: buffer,
            desc,
            allocation: Some(allocation),
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.drop_resources(|drop_list| {
            if let Some(allocation) = self.allocation.take() {
                drop_list.drop_memory(allocation);
            }
            drop_list.drop_buffer(self.raw);
        })
    }
}
