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

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor_ash::AshDescriptorDevice;

use crate::{GpuDescriptorAllocator, GpuDescriptorSet};

use super::{GpuAllocator, GpuMemory};

#[derive(Debug, Default)]
pub struct DropList {
    memory: Vec<GpuMemory>,
    views: Vec<vk::ImageView>,
    images: Vec<vk::Image>,
    buffers: Vec<vk::Buffer>,
    descriptor_sets: Vec<GpuDescriptorSet>,
}

impl DropList {
    pub fn drop_memory(&mut self, memory: GpuMemory) {
        self.memory.push(memory);
    }

    pub fn drop_view(&mut self, view: vk::ImageView) {
        self.views.push(view);
    }

    pub fn drop_image(&mut self, image: vk::Image) {
        self.images.push(image);
    }

    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers.push(buffer);
    }

    pub fn drop_descriptor_set(&mut self, ds: GpuDescriptorSet) {
        self.descriptor_sets.push(ds);
    }

    pub fn purge(
        &mut self,
        device: &ash::Device,
        memory_allocator: &mut GpuAllocator,
        descriptor_allocator: &mut GpuDescriptorAllocator,
    ) {
        self.memory
            .drain(..)
            .for_each(|x| unsafe { memory_allocator.dealloc(AshMemoryDevice::wrap(device), x) });
        self.views
            .drain(..)
            .for_each(|x| unsafe { device.destroy_image_view(x, None) });
        self.images
            .drain(..)
            .for_each(|x| unsafe { device.destroy_image(x, None) });
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(device),
                self.descriptor_sets.drain(..),
            )
        };
    }
}
