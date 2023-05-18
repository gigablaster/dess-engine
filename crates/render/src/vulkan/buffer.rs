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
use std::{mem::size_of, ptr::copy_nonoverlapping};

use ash::vk;
use log::debug;

use super::{
    memory::{allocate_vram, DynamicAllocator, RingAllocator},
    BackendResult, FreeGpuResource, PhysicalDevice,
};

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl Buffer {
    pub fn new(buffer: vk::Buffer, offset: u64, size: u64) -> Self {
        Self {
            buffer,
            offset,
            size,
        }
    }
}

/// Giant buffer that used for all static geometry data.
/// Clients are supposed to suballocate buffers by calling alloc and
/// free them by calling free.
pub struct BufferAllocator {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocator: DynamicAllocator,
}

impl BufferAllocator {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        size: u64,
        flags: vk::BufferUsageFlags,
    ) -> BackendResult<Self> {
        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (_, memory) = allocate_vram(
            device,
            pdevice,
            requirement.size,
            requirement.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            None,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
        let allocator = DynamicAllocator::new(size, 64.max(requirement.alignment));

        Ok(Self {
            buffer,
            memory,
            allocator,
        })
    }

    pub fn allocate(&mut self, size: u64) -> BackendResult<Buffer> {
        let block = self.allocator.alloc(size)?;
        debug!("Allocate buffer size {}", size);
        Ok(Buffer {
            buffer: self.buffer,
            offset: block.offset as _,
            size: block.size as _,
        })
    }

    pub fn free(&mut self, buffer: Buffer) -> BackendResult<()> {
        self.allocator.free(buffer.offset as _)?;

        Ok(())
    }
}

impl FreeGpuResource for BufferAllocator {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct RingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ring: RingAllocator,
    mapping: *mut u8,
}

impl RingBuffer {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        size: u64,
        flags: vk::BufferUsageFlags,
    ) -> BackendResult<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(flags)
            .size(size)
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (_, memory) = allocate_vram(
            device,
            pdevice,
            requirement.size,
            requirement.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            None,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
        let ring = RingAllocator::new(
            size,
            pdevice
                .properties
                .limits
                .min_uniform_buffer_offset_alignment,
        );
        let mapping =
            unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8 };

        Ok(Self {
            buffer,
            ring,
            size,
            memory,
            mapping,
        })
    }

    pub fn push<T: Sized>(&self, data: &[T]) -> Buffer {
        let size = (size_of::<T>() * data.len()) as u64;
        let block = self.ring.allocate(size);
        unsafe {
            copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.mapping.add(block.offset as _),
                size as _,
            )
        };

        Buffer::new(self.buffer, block.offset, block.size)
    }

    pub fn commit(&self, device: &ash::Device) -> BackendResult<()> {
        let range = vk::MappedMemoryRange::builder()
            .size(self.size)
            .offset(0)
            .memory(self.memory)
            .build();
        unsafe { device.flush_mapped_memory_ranges(slice::from_ref(&range)) }?;

        Ok(())
    }
}

impl FreeGpuResource for RingBuffer {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}
