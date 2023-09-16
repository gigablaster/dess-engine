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

#[derive(Debug)]
pub enum RenderError {
    Vulkan(vk::Result),
    Loading(ash::LoadingError),
    MemoryAllocation(gpu_alloc::AllocationError),
    MemoryMap(gpu_alloc::MapError),
    DescriptorAllocation(gpu_descriptor::AllocationError),
    NoSuitableDevice,
    ExtensionNotFound(String),
    NoSuitableQueue,
    Reflection(rspirv_reflect::ReflectError),
    NoSuitableFormat,
    OutOfUnifromsSpace,
}

impl From<vk::Result> for RenderError {
    fn from(value: vk::Result) -> Self {
        RenderError::Vulkan(value)
    }
}

impl From<gpu_alloc::AllocationError> for RenderError {
    fn from(value: gpu_alloc::AllocationError) -> Self {
        RenderError::MemoryAllocation(value)
    }
}

impl From<gpu_alloc::MapError> for RenderError {
    fn from(value: gpu_alloc::MapError) -> Self {
        RenderError::MemoryMap(value)
    }
}

impl From<ash::LoadingError> for RenderError {
    fn from(value: ash::LoadingError) -> Self {
        RenderError::Loading(value)
    }
}

impl From<rspirv_reflect::ReflectError> for RenderError {
    fn from(value: rspirv_reflect::ReflectError) -> Self {
        RenderError::Reflection(value)
    }
}

impl From<gpu_descriptor::AllocationError> for RenderError {
    fn from(value: gpu_descriptor::AllocationError) -> Self {
        RenderError::DescriptorAllocation(value)
    }
}
