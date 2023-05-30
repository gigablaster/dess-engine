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
mod descriptors;
mod error;
mod megabuffer;
mod render;
mod staging;

use ash::vk;
pub use descriptors::DescriptorHandle;
pub use error::*;
pub use megabuffer::AllocatedBuffer;
pub use render::*;
pub use staging::*;

pub type Index = u16;

pub type DescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub type DescriptorSet = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;
