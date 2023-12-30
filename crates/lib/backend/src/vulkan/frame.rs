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
};

use ash::vk::{self};
use dess_common::BumpAllocator;

use crate::{BackendError, BackendResult};

use super::{CommandBuffer, DescriptorAllocator, DropList, GpuAllocator, UniformStorage};

pub(crate) const MAX_TEMP_MEMORY: usize = 16 * 1024 * 1024;
const ALIGMENT: usize = 256;

pub struct Frame {
    pub(crate) pool: vk::CommandPool,
    pub(crate) main_cb: CommandBuffer,
    pub(crate) present_cb: CommandBuffer,
    pub(crate) finished: vk::Semaphore,
    pub(crate) temp_mapping: NonNull<u8>,
    pub(crate) temp_allocator: BumpAllocator,
    pub(crate) temp_offset: usize,
    pub(crate) drop_list: Cell<DropList>,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub(crate) fn new(
        device: &ash::Device,
        temp_offset: usize,
        temp_mapping: NonNull<u8>,
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
            let drop_list = DropList::default();
            Ok(Self {
                pool,
                main_cb: CommandBuffer::primary(device, pool)?,
                present_cb: CommandBuffer::primary(device, pool)?,
                finished,
                temp_mapping,
                temp_offset,
                temp_allocator: BumpAllocator::new(MAX_TEMP_MEMORY, ALIGMENT),
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
        self.temp_allocator.reset();
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
        self.main_cb.free(device);
        self.present_cb.free(device);
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_semaphore(self.finished, None);
        }
        self.drop_list
            .get_mut()
            .purge(device, memory_allocator, descriptor_allocator, uniforms);
    }

    pub fn temp_allocate<T: Sized>(&self, data: &[T]) -> BackendResult<u32> {
        let bytes = std::mem::size_of_val(data);
        let offset = self
            .temp_allocator
            .allocate(bytes as _)
            .ok_or(BackendError::OutOfTempMemory)?;
        unsafe {
            copy_nonoverlapping(
                data.as_ptr() as *const u8,
                #[allow(clippy::ptr_offset_with_cast)]
                self.temp_mapping
                    .as_ptr()
                    .offset((self.temp_offset + offset) as _),
                bytes,
            );
        }

        Ok((offset + self.temp_offset) as u32)
    }
}
