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

use std::ptr::NonNull;

use ash::vk::{self, BufferCreateInfo};
use gpu_alloc::{Dedicated, MemoryBlock, Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;

use super::{droplist::DropList, Device, MapError, ResourceCreateError, Retire};

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
    raw: vk::Buffer,
    desc: BufferDesc,
    allocation: Option<MemoryBlock<vk::DeviceMemory>>,
}

unsafe impl Send for Buffer {}

impl Buffer {
    /// Отображает буфер на память если тот выделен в памяти хоста
    ///
    /// # Паникует
    ///
    /// Если буфер уже отмечен на удаление
    pub fn map(&mut self, device: &ash::Device) -> Result<NonNull<u8>, MapError> {
        if let Some(allocation) = &mut self.allocation {
            let ptr = unsafe { allocation.map(AshMemoryDevice::wrap(device), 0, self.desc.size) }?
                .as_ptr() as *mut u8;

            Ok(NonNull::new(ptr).unwrap())
        } else {
            panic!("Buffer is already retired");
        }
    }

    /// Отменяет отображение буфера на память
    pub fn unmap(&mut self, device: &ash::Device) {
        if let Some(allocation) = &mut self.allocation {
            unsafe { allocation.unmap(AshMemoryDevice::wrap(device)) };
        }
    }

    /// Возвращает описание буфера
    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }
}

impl Retire for Buffer {
    fn retire(&mut self, drop_list: &mut DropList) {
        drop_list.drop_buffer(self.raw);
        if let Some(allocation) = self.allocation.take() {
            drop_list.free_memory(allocation);
        }
    }
}

impl Device {
    /// Создаёт GPU буффер
    ///
    /// Может быть вызвано из разных потоков, синхронизация только для доступа к аллокатуру
    pub fn create_buffer(
        &self,
        desc: BufferDesc,
        name: Option<&str>,
    ) -> Result<Buffer, ResourceCreateError> {
        let buffer_create_info = BufferCreateInfo::builder()
            .size(desc.size as _)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[self.queue_index()])
            .build();
        let buffer = unsafe { self.raw().create_buffer(&buffer_create_info, None) }?;
        if let Some(name) = name {
            self.set_object_name(buffer, name);
        }
        let requirement = unsafe { self.raw().get_buffer_memory_requirements(buffer) };
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
                self.allocator().alloc_with_dedicated(
                    AshMemoryDevice::wrap(self.raw()),
                    request,
                    Dedicated::Required,
                )
            }?
        } else {
            unsafe {
                self.allocator()
                    .alloc(AshMemoryDevice::wrap(self.raw()), request)
            }?
        };
        unsafe {
            self.raw()
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
        }?;

        Ok(Buffer {
            raw: buffer,
            desc,
            allocation: Some(allocation),
        })
    }
}
