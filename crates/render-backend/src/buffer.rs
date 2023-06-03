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

use core::slice;
use std::{ptr::NonNull, sync::Arc};

use ash::vk::{self, BufferCreateInfo};
use gpu_alloc::{Dedicated, MemoryBlock, Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;

use crate::{BackendError, Device};

use super::BackendResult;

#[derive(Debug)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub memory_location: UsageFlags,
    pub alignment: Option<u64>,
    pub dedicated: bool,
}

impl BufferDesc {
    pub fn gpu_only(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: false,
        }
    }

    pub fn host_only(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: UsageFlags::HOST_ACCESS,
            alignment: None,
            dedicated: false,
        }
    }

    pub fn upload(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: UsageFlags::UPLOAD,
            alignment: None,
            dedicated: false,
        }
    }

    pub fn aligment(mut self, aligment: u64) -> Self {
        self.alignment = Some(aligment);
        self
    }

    pub fn dedicated(mut self, value: bool) -> Self {
        self.dedicated = value;
        self
    }
}

#[derive(Debug)]
pub struct Buffer {
    device: Arc<Device>,
    pub raw: vk::Buffer,
    pub desc: BufferDesc,
    pub allocation: Option<MemoryBlock<vk::DeviceMemory>>,
}

impl Buffer {
    fn new(
        device: &Arc<Device>,
        desc: BufferDesc,
        queue_family_index: u32,
        name: Option<&str>,
    ) -> BackendResult<Self> {
        let buffer_create_info = BufferCreateInfo::builder()
            .size(desc.size as _)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(slice::from_ref(&queue_family_index))
            .build();
        let buffer = unsafe { device.raw.create_buffer(&buffer_create_info, None) }?;
        if let Some(name) = name {
            device.set_object_name(buffer, name)?;
        }
        let requirement = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        let aligment = if let Some(aligment) = desc.alignment {
            aligment.max(requirement.alignment)
        } else {
            requirement.alignment
        };
        let request = Request {
            size: requirement.size,
            align_mask: aligment,
            memory_types: requirement.memory_type_bits,
            usage: desc.memory_location,
        };
        let allocation = if desc.dedicated {
            unsafe {
                device.allocator().alloc_with_dedicated(
                    AshMemoryDevice::wrap(&device.raw),
                    request,
                    Dedicated::Required,
                )
            }?
        } else {
            unsafe {
                device
                    .allocator()
                    .alloc(AshMemoryDevice::wrap(&device.raw), request)
            }?
        };
        unsafe {
            device
                .raw
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
        }?;

        Ok(Self {
            device: device.clone(),
            raw: buffer,
            desc,
            allocation: Some(allocation),
        })
    }

    pub fn graphics(
        device: &Arc<Device>,
        desc: BufferDesc,
        name: Option<&str>,
    ) -> BackendResult<Self> {
        Self::new(device, desc, device.graphics_queue.family.index, name)
    }

    pub fn transfer(
        device: &Arc<Device>,
        desc: BufferDesc,
        name: Option<&str>,
    ) -> BackendResult<Self> {
        Self::new(device, desc, device.transfer_queue.family.index, name)
    }

    pub fn map(&mut self) -> BackendResult<NonNull<u8>> {
        if let Some(allocation) = &mut self.allocation {
            let ptr = unsafe {
                allocation.map(AshMemoryDevice::wrap(&self.device.raw), 0, self.desc.size)
            }?
            .as_ptr() as *mut u8;

            Ok(NonNull::new(ptr).unwrap())
        } else {
            Err(BackendError::MemoryNotAllocated)
        }
    }

    pub fn unmap(&mut self) {
        if let Some(allocation) = &mut self.allocation {
            unsafe { allocation.unmap(AshMemoryDevice::wrap(&self.device.raw)) };
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.with_drop_list(|droplist| {
            droplist.drop_buffer(self.raw);
            if let Some(allocation) = self.allocation.take() {
                droplist.free_memory(allocation);
            }
        })
    }
}
