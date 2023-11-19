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

mod buffer;
mod descriptors;
mod device;
mod drop_list;
mod frame;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod pipeline_cache;
mod program;
mod staging;
mod swapchain;
mod uniforms;

use std::marker::PhantomData;

use ash::vk::{self};
pub use buffer::*;
pub use descriptors::*;
pub use device::*;
pub use drop_list::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
pub use program::*;
pub use staging::*;
pub use swapchain::*;
pub use uniforms::*;

pub type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub type GpuMemory = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub type DescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub type DescriptorSet = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;

pub trait ToDrop {
    fn to_drop(&mut self, drop_list: &mut DropList);
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct Index<T>(u32, PhantomData<T>);

impl<T> Index<T> {
    pub(crate) fn new(value: usize) -> Self {
        Self(value as u32, PhantomData)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}
