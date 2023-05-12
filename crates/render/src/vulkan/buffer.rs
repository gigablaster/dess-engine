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

use std::{sync::Arc, io::{Write, self, Seek}, ptr::copy_nonoverlapping, slice::from_raw_parts, mem::size_of};

use ash::vk;
use gpu_alloc::{Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;

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

    pub fn map(&mut self, offset: usize, size: usize) -> BackendResult<MappedBuffer> {
        if let Some(allocation) = &mut self.allocation {
            let memory_device = AshMemoryDevice::wrap(&self.device.raw);
            let ptr = unsafe { allocation.map(memory_device, offset as _, size) }?;

            Ok(MappedBuffer { data: ptr.as_ptr(), size, cursor: 0 })
        } else {
            Err(crate::BackendError::Other("Buffer isn't allocated".into()))
        }
    }

    pub fn unmap(&mut self) -> BackendResult<()> {
        if let Some(allocation) = &mut self.allocation {
            let memory_device = AshMemoryDevice::wrap(&self.device.raw);
            unsafe { allocation.unmap(memory_device) };

            Ok(())
        } else {
            Err(crate::BackendError::Other("Buffer isn't allocated".into()))
        }
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

pub struct MappedBuffer {
    data: *mut u8,
    size: usize,
    cursor: usize
}

impl Write for MappedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let to_write = (self.size - self.cursor).min(buf.len());
        if to_write == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "No space in buffer"));
        }

        unsafe { copy_nonoverlapping(buf.as_ptr(), self.data.add(self.cursor), to_write) };

        self.cursor += to_write;

        Ok(to_write)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for MappedBuffer {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let offset = match pos {
            io::SeekFrom::Start(offset) => offset,
            io::SeekFrom::End(offset) => self.validate_offset(self.size as i64 + offset)?,
            io::SeekFrom::Current(offset) => self.validate_offset(self.cursor as i64 + offset)?
        };
        self.cursor = offset as _;
        Ok(offset)
    }
}

impl MappedBuffer {
    fn validate_offset(&self, offset: i64) -> io::Result<u64> {
        let offset = self.cursor as i64 + offset;
        if offset < 0 || offset >= self.size as _ {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Can't seek outside of mapped buffer"))
        } else {
            Ok(offset as u64)
        }
    }

    pub fn push<T:Sized>(&mut self, data: &[T]) -> BackendResult<usize> {
        let ptr = data.as_ptr() as *const u8;
        let data = unsafe { from_raw_parts(ptr, data.len() * size_of::<T>()) };
        match self.write(data) {
            Ok(count) => Ok(count / size_of::<T>()),
            Err(err) => Err(crate::BackendError::Other(err.to_string()))
        }
    }
}
