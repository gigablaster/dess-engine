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

use std::{
    cell::Cell,
    ptr::{copy_nonoverlapping, NonNull},
    sync::atomic::{AtomicU32, Ordering},
};

use ash::vk::{self};
use dess_common::Align;
use gpu_alloc_ash::AshMemoryDevice;

use crate::{vulkan::Device, BackendError, BackendResult};

use super::{
    CommandBuffer, DescriptorAllocator, DropList, GpuAllocator, GpuMemory, UniformStorage,
};

pub(crate) const MAX_TEMP_MEMORY: u32 = 16 * 1024 * 1024;
const ALIGMENT: u32 = 256;

pub struct Frame {
    pub(crate) pool: vk::CommandPool,
    pub(crate) main_cb: CommandBuffer,
    pub(crate) present_cb: CommandBuffer,
    pub(crate) finished: vk::Semaphore,
    pub(crate) temp_buffer: vk::Buffer,
    pub(crate) temp_mapping: NonNull<u8>,
    pub(crate) temp_memory: Option<GpuMemory>,
    pub(crate) temp_top: AtomicU32,
    pub(crate) drop_list: Cell<DropList>,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        queue_family_index: u32,
    ) -> BackendResult<Self> {
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )?;
            let finished =
                device.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)?;
            let temp_buffer = device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .size(MAX_TEMP_MEMORY as _)
                    .build(),
                None,
            )?;
            let requirements = device.get_buffer_memory_requirements(temp_buffer);
            let mut memory = Device::allocate_impl(
                device,
                allocator,
                requirements,
                gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | gpu_alloc::UsageFlags::HOST_ACCESS,
                true,
            )?;
            device.bind_buffer_memory(temp_buffer, *memory.memory(), memory.offset())?;
            let temp_mapping =
                memory.map(AshMemoryDevice::wrap(device), 0, MAX_TEMP_MEMORY as _)?;
            let drop_list = DropList::default();
            Ok(Self {
                pool,
                main_cb: CommandBuffer::primary(device, pool)?,
                present_cb: CommandBuffer::primary(device, pool)?,
                finished,
                temp_buffer,
                temp_mapping,
                temp_memory: Some(memory),
                temp_top: AtomicU32::new(0),
                drop_list: Cell::new(drop_list),
            })
        }
    }

    pub(crate) fn reset(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut DescriptorAllocator,
        uniforms: &mut UniformStorage,
    ) -> BackendResult<()> {
        self.drop_list
            .get_mut()
            .purge(device, memory_allocator, descriptor_allocator, uniforms);
        self.temp_top.store(0, Ordering::Release);
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;

        Ok(())
    }

    pub(crate) fn free(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut DescriptorAllocator,
        uniforms: &mut UniformStorage,
    ) {
        if let Some(memory) = self.temp_memory.take() {
            self.main_cb.free(device);
            self.present_cb.free(device);
            unsafe {
                memory_allocator.dealloc(AshMemoryDevice::wrap(device), memory);
                device.destroy_command_pool(self.pool, None);
                device.destroy_semaphore(self.finished, None);
            }
            self.drop_list.get_mut().purge(
                device,
                memory_allocator,
                descriptor_allocator,
                uniforms,
            );
        }
    }

    pub fn temp_allocate<T: Sized>(&self, data: &[T]) -> BackendResult<u32> {
        let bytes = std::mem::size_of_val(data);
        let offset = self
            .allocate(bytes as _)
            .ok_or(BackendError::OutOfTempMemory)?;
        unsafe {
            copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.temp_mapping.as_ptr().offset(offset as _),
                bytes,
            );
        }

        Ok(offset)
    }

    fn allocate(&self, size: u32) -> Option<u32> {
        self.temp_top
            .fetch_update(Ordering::Release, Ordering::SeqCst, |x| {
                let new_top = (x + size).align(ALIGMENT);
                if new_top <= MAX_TEMP_MEMORY {
                    Some(new_top)
                } else {
                    None
                }
            })
            .ok()
    }
}
