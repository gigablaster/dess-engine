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

use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use gpu_descriptor_ash::AshDescriptorDevice;

use super::{DescriptorAllocator, DescriptorSet, GpuAllocator, GpuMemory, UniformStorage};

const CAPACITY: usize = 65535;

#[derive(Debug)]
pub struct DropList {
    memory_to_free: Vec<GpuMemory>,
    images_to_free: Vec<vk::Image>,
    image_views_to_free: Vec<vk::ImageView>,
    buffers_to_free: Vec<vk::Buffer>,
    descriptors_to_free: Vec<DescriptorSet>,
    uniforms_to_free: Vec<usize>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            memory_to_free: Vec::with_capacity(CAPACITY),
            images_to_free: Vec::with_capacity(CAPACITY),
            image_views_to_free: Vec::with_capacity(CAPACITY),
            buffers_to_free: Vec::with_capacity(CAPACITY),
            descriptors_to_free: Vec::with_capacity(CAPACITY),
            uniforms_to_free: Vec::with_capacity(CAPACITY),
        }
    }
}

impl DropList {
    pub fn drop_image(&mut self, image: vk::Image) {
        self.images_to_free.push(image);
    }

    pub fn drop_image_view(&mut self, view: vk::ImageView) {
        self.image_views_to_free.push(view);
    }

    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers_to_free.push(buffer);
    }

    pub fn free_memory(&mut self, block: GpuMemory) {
        self.memory_to_free.push(block);
    }

    pub fn free_descriptor_set(&mut self, descriptor_set: DescriptorSet) {
        self.descriptors_to_free.push(descriptor_set);
    }

    pub fn free_uniform(&mut self, uniform: usize) {
        self.uniforms_to_free.push(uniform);
    }

    pub(crate) fn purge(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        descriptor_allocator: &mut DescriptorAllocator,
        uniforms: &mut UniformStorage,
    ) {
        self.image_views_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_image_view(view, None) };
        });
        self.images_to_free.drain(..).for_each(|image| {
            unsafe { device.destroy_image(image, None) };
        });
        self.buffers_to_free.drain(..).for_each(|buffer| {
            unsafe { device.destroy_buffer(buffer, None) };
        });
        self.memory_to_free.drain(..).for_each(|block| {
            unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), block) };
        });
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(device),
                self.descriptors_to_free.drain(..),
            )
        };
        self.uniforms_to_free
            .drain(..)
            .for_each(|x| uniforms.dealloc(x));

        self.memory_to_free.shrink_to(CAPACITY);
        self.image_views_to_free.shrink_to(CAPACITY);
        self.images_to_free.shrink_to(CAPACITY);
        self.buffers_to_free.shrink_to(CAPACITY);
        self.descriptors_to_free.shrink_to(CAPACITY);
        self.uniforms_to_free.shrink_to(CAPACITY);
    }
}
