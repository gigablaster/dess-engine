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

mod bind_group;
mod buffer;
mod device;
mod drop_list;
mod frame;
mod frame_context;
mod image;
mod instance;
mod physical_device;
mod pipeline;
mod program;
mod staging;
mod swapchain;
mod uniforms;

use std::marker::PhantomData;

use ash::vk::{self};
pub use bind_group::*;
use bitflags::bitflags;
pub use buffer::*;
pub use device::*;
use drop_list::*;
pub use frame_context::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
pub use program::*;
use speedy::{Readable, Writable};
use staging::*;
pub use swapchain::*;
use uniforms::*;

pub type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub type GpuMemory = gpu_alloc::MemoryBlock<vk::DeviceMemory>;
pub type DescriptorAllocator =
    gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>;
pub type DescriptorSet = gpu_descriptor::DescriptorSet<vk::DescriptorSet>;

pub trait ToDrop {
    fn to_drop(&mut self, drop_list: &mut DropList);
}

#[derive(Debug)]
pub struct Index<T>(u32, PhantomData<T>);

impl<T> Copy for Index<T> {}

impl<T> Clone for Index<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Index<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Index<T> {}

unsafe impl<T> Send for Index<T> {}
unsafe impl<T> Sync for Index<T> {}

impl<T> Index<T> {
    pub(crate) fn new(value: usize) -> Self {
        Self(value as u32, PhantomData)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }

    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl<T> Default for Index<T> {
    fn default() -> Self {
        Self(u32::MAX, PhantomData)
    }
}

impl<T> From<Index<T>> for u32 {
    fn from(value: Index<T>) -> Self {
        value.0
    }
}

impl<T> From<u32> for Index<T> {
    fn from(value: u32) -> Self {
        Index(value, PhantomData)
    }
}

impl<T> std::hash::Hash for Index<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

pub(crate) trait AsVulkan<T> {
    fn as_vk(&self) -> T;
}

impl AsVulkan<vk::Image> for vk::Image {
    fn as_vk(&self) -> vk::Image {
        self.to_owned()
    }
}

impl AsVulkan<vk::Buffer> for vk::Buffer {
    fn as_vk(&self) -> vk::Buffer {
        self.to_owned()
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
#[allow(non_camel_case_types)]
pub enum Format {
    #[default]
    INVALID,
    R8_UNORM,
    R8_SNORM,
    R8_USCALED,
    R8_SSCALED,
    R8_UINT,
    R8_SINT,
    R8_SRGB,
    RG8_UNORM,
    RG8_SNORM,
    RG8_USCALED,
    RG8_SSCALED,
    RG8_UINT,
    RG8_SINT,
    RG8_SRGB,
    RGB8_UNORM,
    RGB8_SNORM,
    RGB8_USCALED,
    RGB8_SSCALED,
    RGB8_UINT,
    RGB8_SINT,
    RGB8_SRGB,
    RGBA8_UNORM,
    RGBA8_SNORM,
    RGBA8_USCALED,
    RGBA8_SSCALED,
    RGBA8_UINT,
    RGBA8_SINT,
    RGBA8_SRGB,
    BGR8_UNORM,
    BGR8_SNORM,
    BGR8_USCALED,
    BGR8_SSCALED,
    BGR8_UINT,
    BGR8_SINT,
    BGR8_SRGB,
    BGRA8_UNORM,
    BGRA8_SNORM,
    BGRA8_USCALED,
    BGRA8_SSCALED,
    BGRA8_UINT,
    BGRA8_SINT,
    BGRA8_SRGB,
    R16_UNORM,
    R16_SNORM,
    R16_USCALED,
    R16_SSCALED,
    R16_UINT,
    R16_SINT,
    R16_SFLOAT,
    RG16_UNORM,
    RG16_SNORM,
    RG16_USCALED,
    RG16_SSCALED,
    RG16_UINT,
    RG16_SINT,
    RG16_SFLOAT,
    RGB16_UNORM,
    RGB16_SNORM,
    RGB16_USCALED,
    RGB16_SSCALED,
    RGB16_UINT,
    RGB16_SINT,
    RGB16_SFLOAT,
    RGBA16_UNORM,
    RGBA16_SNORM,
    RGBA16_USCALED,
    RGBA16_SSCALED,
    RGBA16_UINT,
    RGBA16_SINT,
    RGBA16_SFLOAT,
    R32_UINT,
    R32_SINT,
    R32_SFLOAT,
    RG32_UINT,
    RG32_SINT,
    RG32_SFLOAT,
    RGB32_UINT,
    RGB32_SINT,
    RGB32_SFLOAT,
    RGBA32_UINT,
    RGBA32_SINT,
    RGBA32_SFLOAT,
    D16,
    D24,
    D32,
    D16_S8,
    D24_S8,
    BC1_RGB_UNORM,
    BC1_RGB_SRGB,
    BC1_RGBA_UNORM,
    BC1_RGBA_SRGB,
    BC2_UNORM,
    BC2_SRGB,
    BC3_UNORM,
    BC3_SRGB,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC6_UFLOAT,
    BC6_SFLOAT,
    BC7_UNORM,
    BC7_SRGB,
}

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        match value {
            Format::INVALID => vk::Format::UNDEFINED,
            Format::R8_UNORM => vk::Format::R8_UNORM,
            Format::R8_SNORM => vk::Format::R8_SNORM,
            Format::R8_USCALED => vk::Format::R8_USCALED,
            Format::R8_SSCALED => vk::Format::R8_SSCALED,
            Format::R8_UINT => vk::Format::R8_UINT,
            Format::R8_SINT => vk::Format::R8_SINT,
            Format::R8_SRGB => vk::Format::R8_SRGB,
            Format::RG8_UNORM => vk::Format::R8G8_UNORM,
            Format::RG8_SNORM => vk::Format::R8G8_SNORM,
            Format::RG8_USCALED => vk::Format::R8G8_USCALED,
            Format::RG8_SSCALED => vk::Format::R8G8_SSCALED,
            Format::RG8_UINT => vk::Format::R8G8_UINT,
            Format::RG8_SINT => vk::Format::R8G8_SINT,
            Format::RG8_SRGB => vk::Format::R8G8_SRGB,
            Format::RGB8_UNORM => vk::Format::R8G8B8_UNORM,
            Format::RGB8_SNORM => vk::Format::R8G8B8_SNORM,
            Format::RGB8_USCALED => vk::Format::R8G8B8_USCALED,
            Format::RGB8_SSCALED => vk::Format::R8G8B8_SSCALED,
            Format::RGB8_UINT => vk::Format::R8G8B8_UINT,
            Format::RGB8_SINT => vk::Format::R8G8B8_SINT,
            Format::RGB8_SRGB => vk::Format::R8G8B8_SRGB,
            Format::RGBA8_UNORM => vk::Format::R8G8B8A8_UNORM,
            Format::RGBA8_SNORM => vk::Format::R8G8B8A8_SNORM,
            Format::RGBA8_USCALED => vk::Format::R8G8B8A8_USCALED,
            Format::RGBA8_SSCALED => vk::Format::R8G8B8A8_SSCALED,
            Format::RGBA8_UINT => vk::Format::R8G8B8A8_UINT,
            Format::RGBA8_SINT => vk::Format::R8G8B8A8_SINT,
            Format::RGBA8_SRGB => vk::Format::R8G8B8A8_SRGB,
            Format::BGR8_UNORM => vk::Format::B8G8R8_UNORM,
            Format::BGR8_SNORM => vk::Format::B8G8R8_SNORM,
            Format::BGR8_USCALED => vk::Format::B8G8R8_USCALED,
            Format::BGR8_SSCALED => vk::Format::B8G8R8_SSCALED,
            Format::BGR8_UINT => vk::Format::B8G8R8_UINT,
            Format::BGR8_SINT => vk::Format::B8G8R8_UINT,
            Format::BGR8_SRGB => vk::Format::B8G8R8_SRGB,
            Format::BGRA8_UNORM => vk::Format::B8G8R8A8_UNORM,
            Format::BGRA8_SNORM => vk::Format::B8G8R8A8_SNORM,
            Format::BGRA8_USCALED => vk::Format::B8G8R8A8_USCALED,
            Format::BGRA8_SSCALED => vk::Format::B8G8R8A8_SSCALED,
            Format::BGRA8_UINT => vk::Format::B8G8R8A8_UINT,
            Format::BGRA8_SINT => vk::Format::B8G8R8A8_SINT,
            Format::BGRA8_SRGB => vk::Format::B8G8R8A8_SRGB,
            Format::R16_UNORM => vk::Format::R16_UNORM,
            Format::R16_SNORM => vk::Format::R16_SNORM,
            Format::R16_USCALED => vk::Format::R16_USCALED,
            Format::R16_SSCALED => vk::Format::R16_SSCALED,
            Format::R16_UINT => vk::Format::R16_UINT,
            Format::R16_SINT => vk::Format::R16_SINT,
            Format::R16_SFLOAT => vk::Format::R16_SFLOAT,
            Format::RG16_UNORM => vk::Format::R16G16_UNORM,
            Format::RG16_SNORM => vk::Format::R16G16_SNORM,
            Format::RG16_USCALED => vk::Format::R16G16_USCALED,
            Format::RG16_SSCALED => vk::Format::R16G16_SSCALED,
            Format::RG16_UINT => vk::Format::R16G16_UINT,
            Format::RG16_SINT => vk::Format::R16G16_SINT,
            Format::RG16_SFLOAT => vk::Format::R16G16_SFLOAT,
            Format::RGB16_UNORM => vk::Format::R16G16B16_UNORM,
            Format::RGB16_SNORM => vk::Format::R16G16B16_SNORM,
            Format::RGB16_USCALED => vk::Format::R16G16B16_USCALED,
            Format::RGB16_SSCALED => vk::Format::R16G16B16_USCALED,
            Format::RGB16_UINT => vk::Format::R16G16B16_UINT,
            Format::RGB16_SINT => vk::Format::R16G16B16_SINT,
            Format::RGB16_SFLOAT => vk::Format::R16G16B16_SFLOAT,
            Format::RGBA16_UNORM => vk::Format::R16G16B16A16_UNORM,
            Format::RGBA16_SNORM => vk::Format::R16G16B16A16_SNORM,
            Format::RGBA16_USCALED => vk::Format::R16G16B16A16_USCALED,
            Format::RGBA16_SSCALED => vk::Format::R16G16B16A16_SSCALED,
            Format::RGBA16_UINT => vk::Format::R16G16B16A16_UINT,
            Format::RGBA16_SINT => vk::Format::R16G16B16A16_SINT,
            Format::RGBA16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
            Format::R32_UINT => vk::Format::R32_UINT,
            Format::R32_SINT => vk::Format::R32_SINT,
            Format::R32_SFLOAT => vk::Format::R32_SFLOAT,
            Format::RG32_UINT => vk::Format::R32G32_UINT,
            Format::RG32_SINT => vk::Format::R32G32_SINT,
            Format::RG32_SFLOAT => vk::Format::R32G32_SFLOAT,
            Format::RGB32_UINT => vk::Format::R32G32B32_UINT,
            Format::RGB32_SINT => vk::Format::R32G32B32_SINT,
            Format::RGB32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
            Format::RGBA32_UINT => vk::Format::R32G32B32A32_UINT,
            Format::RGBA32_SINT => vk::Format::R32G32B32A32_SINT,
            Format::RGBA32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
            Format::D16 => vk::Format::D16_UNORM,
            Format::D24 => vk::Format::X8_D24_UNORM_PACK32,
            Format::D32 => vk::Format::D32_SFLOAT,
            Format::D16_S8 => vk::Format::D16_UNORM_S8_UINT,
            Format::D24_S8 => vk::Format::D24_UNORM_S8_UINT,
            Format::BC1_RGB_UNORM => vk::Format::BC1_RGB_UNORM_BLOCK,
            Format::BC1_RGB_SRGB => vk::Format::BC1_RGB_SRGB_BLOCK,
            Format::BC1_RGBA_UNORM => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Format::BC1_RGBA_SRGB => vk::Format::BC1_RGBA_SRGB_BLOCK,
            Format::BC2_UNORM => vk::Format::BC2_UNORM_BLOCK,
            Format::BC2_SRGB => vk::Format::BC2_SRGB_BLOCK,
            Format::BC3_UNORM => vk::Format::BC3_UNORM_BLOCK,
            Format::BC3_SRGB => vk::Format::BC3_SRGB_BLOCK,
            Format::BC4_UNORM => vk::Format::BC4_UNORM_BLOCK,
            Format::BC4_SNORM => vk::Format::BC4_SNORM_BLOCK,
            Format::BC5_UNORM => vk::Format::BC5_UNORM_BLOCK,
            Format::BC5_SNORM => vk::Format::BC5_SNORM_BLOCK,
            Format::BC6_UFLOAT => vk::Format::BC6H_UFLOAT_BLOCK,
            Format::BC6_SFLOAT => vk::Format::BC6H_SFLOAT_BLOCK,
            Format::BC7_UNORM => vk::Format::BC7_UNORM_BLOCK,
            Format::BC7_SRGB => vk::Format::BC7_SRGB_BLOCK,
        }
    }
}

bitflags! {
#[derive(Debug, Default, Hash, Eq, PartialEq, Clone, Copy)]
pub struct ShaderStage: u32 {
    const None = 0;
    const Vertex = 1;
    const Fragment = 2;
    const Graphics = 3;
    const Compute = 4;
}

}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        let mut result = vk::ShaderStageFlags::empty();
        if value.contains(ShaderStage::Vertex) {
            result |= vk::ShaderStageFlags::VERTEX;
        }
        if value.contains(ShaderStage::Fragment) {
            result |= vk::ShaderStageFlags::FRAGMENT;
        }
        if value.contains(ShaderStage::Compute) {
            result |= vk::ShaderStageFlags::COMPUTE;
        }
        result
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct RenderArea {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl From<RenderArea> for vk::Viewport {
    fn from(value: RenderArea) -> Self {
        Self {
            x: value.x as _,
            y: value.y as _,
            width: value.width as _,
            height: value.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}
impl From<RenderArea> for vk::Rect2D {
    fn from(value: RenderArea) -> Self {
        Self {
            offset: vk::Offset2D {
                x: value.x as i32,
                y: value.y as i32,
            },
            extent: vk::Extent2D {
                width: value.width,
                height: value.height,
            },
        }
    }
}

impl RenderArea {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn aspect_ratio(&self) -> f32 {
        (self.width as f32) / (self.height as f32)
    }
}
