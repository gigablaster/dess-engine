// Copyright (C) 2023-2024 gigablaster

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

use std::{ptr::NonNull, sync::Arc};

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use smol_str::SmolStr;

use crate::Result;

use super::{AsVulkan, Device, GpuMemory};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub name: Option<SmolStr>,
}

#[derive(Debug)]
pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    desc: BufferDesc,
    memory: Option<GpuMemory>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCreateDesc<'a> {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
    pub alignment: Option<u64>,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    memory_location: gpu_alloc::UsageFlags,
}

impl<'a> BufferCreateDesc<'a> {
    pub fn gpu(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn host(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn upload(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::UPLOAD,
            alignment: None,
            dedicated: false,
            name: None,
        }
    }

    pub fn shared(size: usize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_location: gpu_alloc::UsageFlags::HOST_ACCESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            alignment: None,
            dedicated: true,
            name: None,
        }
    }

    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
        self
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
            .usage(self.usage)
            .size(self.size as _)
            .build()
    }
}

impl AsVulkan<vk::Buffer> for Buffer {
    fn as_vk(&self) -> vk::Buffer {
        self.raw
    }
}

impl Buffer {
    pub fn new(device: &Arc<Device>, desc: BufferCreateDesc) -> Result<Self> {
        let buffer = unsafe { device.get().create_buffer(&desc.build(), None) }?;
        let requirements = unsafe { device.get().get_buffer_memory_requirements(buffer) };
        let memory = device.allocate(requirements, desc.memory_location, desc.dedicated)?;
        unsafe {
            device
                .get()
                .bind_buffer_memory(buffer, *memory.memory(), memory.offset())
        }?;
        if let Some(name) = desc.name {
            device.set_object_name(buffer, name);
        }
        Ok(Self {
            device: device.clone(),
            raw: buffer,
            desc: BufferDesc {
                size: desc.size,
                usage: desc.usage,
                name: desc.name.map(|x| x.into()),
            },
            memory: Some(memory),
        })
    }

    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    pub fn map(&mut self) -> Result<NonNull<u8>> {
        let memory = self
            .memory
            .as_mut()
            .expect("Buffer must point to allocated data");
        Ok(unsafe { memory.map(AshMemoryDevice::wrap(self.device.get()), 0, self.desc.size) }?)
    }

    pub fn unmap(&mut self) {
        if let Some(memory) = &mut self.memory {
            unsafe { memory.unmap(AshMemoryDevice::wrap(self.device.get())) };
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            self.device.with_drop_list(|drop_list| {
                drop_list.drop_memory(memory);
                drop_list.drop_buffer(self.raw);
            })
        }
    }
}
