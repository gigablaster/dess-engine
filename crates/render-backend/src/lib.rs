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
mod command_buffer;
mod device;
mod droplist;
mod error;
mod frame_context;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod render_pass;
mod swapchain;

pub use self::image::*;

use ash::vk;
pub use buffer::*;
pub use command_buffer::*;
pub use device::*;
pub use error::*;
pub use frame_context::*;
pub use instance::*;
pub use droplist::*;
pub use physical_device::*;
pub use pipeline::*;
pub use render_pass::*;
pub use swapchain::*;

pub trait FreeGpuResource {
    fn free(&self, device: &ash::Device);
}

pub type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub type GpuMemory = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub type DescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub type DescriptorSet = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;
