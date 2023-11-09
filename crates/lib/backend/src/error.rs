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
pub enum BackendError {
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
    PipelineCreatingFailed,
}

impl From<vk::Result> for BackendError {
    fn from(value: vk::Result) -> Self {
        BackendError::Vulkan(value)
    }
}

impl From<gpu_alloc::AllocationError> for BackendError {
    fn from(value: gpu_alloc::AllocationError) -> Self {
        BackendError::MemoryAllocation(value)
    }
}

impl From<gpu_alloc::MapError> for BackendError {
    fn from(value: gpu_alloc::MapError) -> Self {
        BackendError::MemoryMap(value)
    }
}

impl From<ash::LoadingError> for BackendError {
    fn from(value: ash::LoadingError) -> Self {
        BackendError::Loading(value)
    }
}

impl From<rspirv_reflect::ReflectError> for BackendError {
    fn from(value: rspirv_reflect::ReflectError) -> Self {
        BackendError::Reflection(value)
    }
}

impl From<gpu_descriptor::AllocationError> for BackendError {
    fn from(value: gpu_descriptor::AllocationError) -> Self {
        BackendError::DescriptorAllocation(value)
    }
}
