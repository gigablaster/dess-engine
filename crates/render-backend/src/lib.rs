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
mod staging;
mod swapchain;

pub use self::image::*;
use ash::vk;
pub use buffer::*;
pub use command_buffer::*;
pub use device::*;
pub use error::*;
pub use frame_context::*;
pub use instance::*;
use log::debug;
pub use physical_device::*;
pub use pipeline::*;
pub use render_pass::*;
pub use swapchain::*;

pub trait FreeGpuResource {
    fn free(&self, device: &ash::Device);
}

fn allocate_vram(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    size: u64,
    mask: u32,
    desired_flags: vk::MemoryPropertyFlags,
    required_flags: Option<vk::MemoryPropertyFlags>,
) -> BackendResult<vk::DeviceMemory> {
    let mut index = find_memory(pdevice, mask, desired_flags);
    if index.is_none() {
        if let Some(required_flags) = required_flags {
            index = find_memory(pdevice, mask, required_flags);
        }
    }

    if let Some(index) = index {
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .build();
        let mem = unsafe { device.allocate_memory(&alloc_info, None) }?;

        debug!(
            "Allocate {} bytes flags {:?}/{:?} type {}",
            size, desired_flags, required_flags, index
        );

        Ok(mem)
    } else {
        Err(BackendError::VramTypeNotFound)
    }
}

fn find_memory(pdevice: &PhysicalDevice, mask: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    let memory_prop = pdevice.memory_properties;
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & mask != 0 && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}
