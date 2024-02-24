// Copyright (C) 2024 gigablaster

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
use std::{
    marker::PhantomData,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::vk;
use dess_backend::{Buffer, BufferCreateDesc, Device};
use dess_common::BumpAllocator;

use crate::{Error, GpuBufferWriterImpl};

const PAGES: usize = 2;

#[derive(Debug)]
pub struct TempBuffer {
    device: Arc<Device>,
    buffer: Arc<Buffer>,
    current_frame: AtomicUsize,
    page_size: usize,
    allocator: BumpAllocator,
    mapping: NonNull<u8>,
}

impl TempBuffer {
    pub fn new(device: &Arc<Device>, page_size: usize) -> dess_backend::Result<Self> {
        let mut buffer = Buffer::new(
            device,
            BufferCreateDesc::shared(page_size * PAGES)
                .dedicated(true)
                .usage(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
                .name("Temp buffer"),
        )?;
        let mapping = buffer.map()?;
        Ok(Self {
            device: device.clone(),
            buffer: Arc::new(buffer),
            current_frame: AtomicUsize::new(0),
            page_size,
            allocator: BumpAllocator::new(page_size),
            mapping,
        })
    }

    pub fn get(&self) -> Arc<Buffer> {
        self.buffer.clone()
    }

    pub fn next_frame(&self) {
        self.current_frame.fetch_add(1, Ordering::AcqRel);
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<usize, Error> {
        Ok(self
            .allocator
            .allocate(size, alignment as _)
            .ok_or(Error::OutOfSpace)?
            + (self.current_frame.load(Ordering::Acquire) % PAGES) * self.page_size)
    }

    fn push<T: Sized + Copy>(&self, alignment: usize, data: &[T]) -> Result<usize, Error> {
        let size = mem::size_of_val(data);
        let offset = self.allocate(size, alignment)?;
        let src = data.as_ptr() as *const u8;
        unsafe { copy_nonoverlapping(src, self.mapping.as_ptr().add(offset), size) };
        Ok(offset)
    }

    pub fn push_data<T: Sized + Copy>(&self, data: &T) -> Result<usize, Error> {
        let alignment = self
            .device
            .physical_device()
            .properties()
            .limits
            .min_uniform_buffer_offset_alignment;
        self.push(alignment as _, slice::from_ref(data))
    }

    pub fn push_bufer<T: Sized + Copy>(&self, data: &[T]) -> Result<usize, Error> {
        let alignment = self
            .device
            .physical_device()
            .properties()
            .limits
            .min_storage_buffer_offset_alignment;
        self.push(alignment as _, data)
    }

    fn write<T: Sized + Copy>(
        &self,
        alignment: usize,
        count: usize,
    ) -> Result<GpuBufferWriterImpl<T>, Error> {
        let size = mem::size_of::<T>();
        let offset = self.allocate(size, alignment)?;
        Ok(GpuBufferWriterImpl {
            ptr: self.mapping,
            offset,
            count,
            cursor: 0,
            _marker: PhantomData,
        })
    }

    pub fn write_buffer<T: Sized + Copy>(
        &self,
        count: usize,
    ) -> Result<GpuBufferWriterImpl<T>, Error> {
        let alignment = self
            .device
            .physical_device()
            .properties()
            .limits
            .min_storage_buffer_offset_alignment;
        self.write(alignment as _, count)
    }
}
