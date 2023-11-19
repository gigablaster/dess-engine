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
    OutOfMemory,
    TooManyObjects,
    NotSupported,
    NotFound,
    OutOfAllocatedSpace,
    NoSuitableDevice,
    ShaderReflectionFailed,
    ExtensionNotFound(String),
    NoSuitableQueue,
    Fail,
    InvalidHandle,
    OutOfTempMemory,
    TooBig,
}

impl From<vk::Result> for BackendError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_FORMAT_NOT_SUPPORTED
            | vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR => Self::NotSupported,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY
            | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            | vk::Result::ERROR_OUT_OF_POOL_MEMORY => Self::OutOfMemory,
            vk::Result::ERROR_TOO_MANY_OBJECTS => Self::TooManyObjects,
            _ => Self::Fail,
        }
    }
}

impl From<gpu_alloc::AllocationError> for BackendError {
    fn from(value: gpu_alloc::AllocationError) -> Self {
        match value {
            gpu_alloc::AllocationError::NoCompatibleMemoryTypes => Self::NotSupported,
            gpu_alloc::AllocationError::OutOfDeviceMemory
            | gpu_alloc::AllocationError::OutOfHostMemory => Self::OutOfMemory,
            gpu_alloc::AllocationError::TooManyObjects => Self::TooManyObjects,
        }
    }
}

impl From<gpu_alloc::MapError> for BackendError {
    fn from(value: gpu_alloc::MapError) -> Self {
        match value {
            gpu_alloc::MapError::NonHostVisible => Self::NotSupported,
            gpu_alloc::MapError::AlreadyMapped | gpu_alloc::MapError::MapFailed => Self::Fail,
            gpu_alloc::MapError::OutOfDeviceMemory | gpu_alloc::MapError::OutOfHostMemory => {
                Self::OutOfMemory
            }
        }
    }
}

impl From<ash::LoadingError> for BackendError {
    fn from(value: ash::LoadingError) -> Self {
        match value {
            ash::LoadingError::LibraryLoadFailure(..) => Self::Fail,
            ash::LoadingError::MissingEntryPoint(..) => Self::NotFound,
        }
    }
}

impl From<rspirv_reflect::ReflectError> for BackendError {
    fn from(_value: rspirv_reflect::ReflectError) -> Self {
        Self::ShaderReflectionFailed
    }
}

impl From<gpu_descriptor::AllocationError> for BackendError {
    fn from(_value: gpu_descriptor::AllocationError) -> Self {
        Self::OutOfMemory
    }
}
