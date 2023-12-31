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
    collections::HashMap,
    ptr::{copy_nonoverlapping, NonNull},
    thread::{self},
};

use ash::vk::{self};
use dess_common::BumpAllocator;
use parking_lot::Mutex;

use crate::{BackendError, BackendResult};

use super::{CommandBuffer, DescriptorAllocator, DropList, GpuAllocator, UniformStorage};

pub(crate) const MAX_TEMP_MEMORY: usize = 16 * 1024 * 1024;
const ALIGMENT: usize = 256;
const PREALLOCATED_COMMAND_BUFFERS: usize = 8;

struct SecondaryCommandBufferPool {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    free: Vec<vk::CommandBuffer>,
}

impl SecondaryCommandBufferPool {
    pub fn new(device: &ash::Device, queue_family_index: u32) -> BackendResult<Self> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )
        }?;

        Ok(Self {
            pool,
            free: Vec::new(),
            buffers: Vec::new(),
        })
    }

    pub fn get_or_create(&mut self, device: &ash::Device) -> BackendResult<vk::CommandBuffer> {
        if let Some(cb) = self.free.pop() {
            Ok(cb)
        } else {
            let mut buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .command_buffer_count(PREALLOCATED_COMMAND_BUFFERS as _)
                        .level(vk::CommandBufferLevel::SECONDARY),
                )
            }?;
            self.buffers.append(&mut buffers.clone());
            self.free.append(&mut buffers);
            let cb = self.free.pop().unwrap();

            Ok(cb)
        }
    }

    pub fn reset(&mut self, device: &ash::Device) {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
            .unwrap();
        self.free.clear();
        for it in &self.buffers {
            self.free.push(*it);
        }
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

pub struct Frame {
    pub(crate) pool: vk::CommandPool,
    pub(crate) main_cb: CommandBuffer,
    pub(crate) present_cb: CommandBuffer,
    pub(crate) finished: vk::Semaphore,
    pub(crate) temp_mapping: NonNull<u8>,
    pub(crate) temp_allocator: BumpAllocator,
    pub(crate) temp_offset: usize,
    pub(crate) drop_list: Cell<DropList>,
    per_thread_buffers: Mutex<HashMap<thread::ThreadId, SecondaryCommandBufferPool>>,
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
                per_thread_buffers: Mutex::default(),
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
        self.per_thread_buffers
            .lock()
            .iter_mut()
            .for_each(|(_, x)| x.reset(device));

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
        self.per_thread_buffers
            .lock()
            .drain()
            .for_each(|(_, x)| x.free(device));
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

    pub fn get_or_create_secondary_buffer(
        &self,
        device: &ash::Device,
        queue_family_index: u32,
    ) -> BackendResult<vk::CommandBuffer> {
        let thread_id = thread::current().id();
        let mut pools = self.per_thread_buffers.lock();
        if let Some(pool) = pools.get_mut(&thread_id) {
            pool.get_or_create(device)
        } else {
            let mut pool = SecondaryCommandBufferPool::new(device, queue_family_index).unwrap();
            let cb = pool.get_or_create(device);
            pools.insert(thread_id, pool);

            cb
        }
    }
}
