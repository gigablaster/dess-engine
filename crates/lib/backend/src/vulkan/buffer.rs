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

use ash::vk::{self};
use bitflags::bitflags;
use parking_lot::Mutex;

use crate::{BackendError, BackendResult};

use super::{
    AsVulkan, BufferHandle, BufferSlice, Device, DropList, GpuAllocator, GpuMemory, ImageHandle,
    Instance, ToDrop,
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: usize,
    pub ty: BufferUsage,
}

#[derive(Debug)]
pub struct Buffer {
    pub(crate) raw: vk::Buffer,
    pub desc: BufferDesc,
    pub(crate) memory: Option<GpuMemory>,
}

impl ToDrop for Buffer {
    fn to_drop(&mut self, drop_list: &mut DropList) {
        if let Some(memory) = self.memory.take() {
            drop_list.free_memory(memory);
            drop_list.drop_buffer(self.raw);
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCreateDesc<'a> {
    pub size: usize,
    pub ty: BufferUsage,
    pub alignment: Option<u64>,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    memory_location: gpu_alloc::UsageFlags,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: usize, ty: BufferUsage) -> Self {
        Self {
            size,
            ty,
            memory_location: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn host(size: usize, ty: BufferUsage) -> Self {
        Self {
            size,
            ty,
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn upload(size: usize, ty: BufferUsage) -> Self {
        Self {
            size,
            ty,
            memory_location: gpu_alloc::UsageFlags::UPLOAD,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn shared(size: usize, ty: BufferUsage) -> Self {
        Self {
            size,
            ty,
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: true,
            name: None,
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

    pub fn name(mut self, value: &'a str) -> Self {
        self.name = Some(value);
        self
    }

    fn build(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::builder()
            .usage(self.ty.into())
            .size(self.size as _)
            .build()
    }
}

impl AsVulkan<vk::Buffer> for Buffer {
    fn as_vk(&self) -> vk::Buffer {
        self.raw
    }
}

impl Device {
    pub fn create_buffer(&self, desc: BufferCreateDesc) -> BackendResult<BufferHandle> {
        let buffer =
            Self::create_buffer_impl(&self.instance, &self.raw, &self.memory_allocator, desc)?;
        Ok(self.buffer_storage.write().push(buffer.raw, buffer))
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        self.destroy_resource(handle, &self.buffer_storage);
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.destroy_resource(handle, &self.image_storage);
    }

    pub fn upload_buffer<T: Sized>(&self, target: BufferSlice, data: &[T]) -> BackendResult<()> {
        let buffers = self.buffer_storage.read();
        let buffer = buffers
            .get_cold(target.handle)
            .ok_or(BackendError::InvalidHandle)?;
        self.staging
            .lock()
            .upload_buffer(self, buffer, target.offset as _, data)
    }

    pub fn get_buffer_desc(&self, handle: BufferHandle) -> BackendResult<BufferDesc> {
        Ok(self
            .buffer_storage
            .read()
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .desc)
    }

    fn create_buffer_impl(
        instance: &Instance,
        device: &ash::Device,
        allocator: &Mutex<GpuAllocator>,
        desc: BufferCreateDesc,
    ) -> BackendResult<Buffer> {
        let buffer = unsafe { device.create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = Self::allocate_impl(
            device,
            &mut allocator.lock(),
            requirements,
            desc.memory_location,
            desc.dedicated,
        )?;
        unsafe { device.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;
        if let Some(name) = desc.name {
            Self::set_object_name_impl(instance, device, buffer, name);
        }
        Ok(Buffer {
            raw: buffer,
            desc: BufferDesc {
                size: desc.size,
                ty: desc.ty,
            },
            memory: Some(memory),
        })
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const Vertex = 1;
        const Index = 2;
        const Uniform = 4;
        const Storage = 8;
        const Destination = 16;
        const Source = 32;
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(value: BufferUsage) -> Self {
        let mut result = vk::BufferUsageFlags::empty();
        if value.contains(BufferUsage::Vertex) {
            result |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if value.contains(BufferUsage::Index) {
            result |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if value.contains(BufferUsage::Storage) {
            result |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if value.contains(BufferUsage::Uniform) {
            result |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if value.contains(BufferUsage::Destination) {
            result |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if value.contains(BufferUsage::Source) {
            result |= vk::BufferUsageFlags::TRANSFER_SRC;
        }

        result
    }
}
