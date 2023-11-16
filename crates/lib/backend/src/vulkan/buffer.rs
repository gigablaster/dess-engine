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

use std::{hash::Hash, ptr::NonNull, sync::Arc};

use ash::vk::{self, BufferCreateInfo, Handle};
use gpu_alloc::{Dedicated, Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;
use vk_sync::AccessType;

use crate::BackendError;

use super::{Device, GpuMemory, MemoryAllocationInfo};

#[derive(Debug, Clone, Copy)]
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

    pub fn shared(size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
            memory_location: UsageFlags::HOST_ACCESS | UsageFlags::FAST_DEVICE_ACCESS,
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
    raw: vk::Buffer,
    desc: BufferDesc,
    allocation: Option<GpuMemory>,
    access: Vec<AccessType>,
}

impl Hash for Buffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.raw.as_raw());
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for Buffer {}

unsafe impl Send for Buffer {}

impl Buffer {
    pub fn new(device: &Arc<Device>, desc: BufferDesc) -> Result<Self, BackendError> {
        let buffer_create_info = BufferCreateInfo::builder()
            .size(desc.size as _)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[device.queue_family_index()])
            .build();
        let buffer = unsafe { device.raw().create_buffer(&buffer_create_info, None) }?;
        let requirement = unsafe { device.raw().get_buffer_memory_requirements(buffer) };
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
                    AshMemoryDevice::wrap(device.raw()),
                    request,
                    Dedicated::Required,
                )
            }?
        } else {
            unsafe {
                device
                    .allocator()
                    .alloc(AshMemoryDevice::wrap(device.raw()), request)
            }?
        };
        unsafe {
            device
                .raw()
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
        }?;

        Ok(Self {
            access: Self::create_access_type(&desc),
            device: device.clone(),
            raw: buffer,
            desc,
            allocation: Some(allocation),
        })
    }

    fn create_access_type(desc: &BufferDesc) -> Vec<AccessType> {
        let mut flags = Vec::new();
        if desc.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
            flags.push(AccessType::VertexBuffer);
        }
        if desc.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
            flags.push(AccessType::IndexBuffer);
        }
        if desc.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
            flags.push(AccessType::AnyShaderReadUniformBuffer);
        }
        if desc.usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
            flags.push(AccessType::IndirectBuffer);
        }
        if desc.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
            flags.push(AccessType::TransferWrite);
        }
        if desc.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
            flags.push(AccessType::TransferRead);
        }

        flags
    }

    pub fn access_type(&self) -> &[AccessType] {
        &self.access
    }

    pub fn map(&mut self) -> Result<NonNull<u8>, BackendError> {
        if let Some(allocation) = &mut self.allocation {
            let ptr = unsafe {
                allocation.map(AshMemoryDevice::wrap(self.device.raw()), 0, self.desc.size)
            }?
            .as_ptr();

            Ok(NonNull::new(ptr).unwrap())
        } else {
            panic!("Buffer is already retired");
        }
    }

    pub fn unmap(&mut self) {
        if let Some(allocation) = &mut self.allocation {
            unsafe { allocation.unmap(AshMemoryDevice::wrap(self.device.raw())) };
        }
    }

    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    pub fn name(&self, name: &str) {
        self.device.set_object_name(self.raw, name);
    }

    pub fn allocation_info(&self) -> Option<MemoryAllocationInfo> {
        self.allocation.as_ref().map(|memory| MemoryAllocationInfo {
            memory: memory.memory(),
            offset: memory.offset(),
            size: memory.size(),
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut drop_list = self.device.drop_list();
        drop_list.drop_buffer(self.raw);
        if let Some(allocation) = self.allocation.take() {
            drop_list.free_memory(allocation);
        }
    }
}
