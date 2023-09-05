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
use gpu_alloc::AllocationError;

#[derive(Debug)]
pub enum ResourceCreateError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    NoCompatibleMemory,
    TooManyObjects,
}

impl From<vk::Result> for ResourceCreateError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => ResourceCreateError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => ResourceCreateError::OutOfHostMemory,
            _ => panic!("Unknown error {}", value),
        }
    }
}

impl From<AllocationError> for ResourceCreateError {
    fn from(value: AllocationError) -> Self {
        match value {
            AllocationError::NoCompatibleMemoryTypes => ResourceCreateError::NoCompatibleMemory,
            AllocationError::OutOfDeviceMemory => ResourceCreateError::OutOfDeviceMemory,
            AllocationError::OutOfHostMemory => ResourceCreateError::OutOfHostMemory,
            AllocationError::TooManyObjects => ResourceCreateError::TooManyObjects,
        }
    }
}

#[derive(Debug)]
pub enum CreateError {
    OutOfDeviceMemory,
    OutOfHostMemory,
}

impl From<vk::Result> for CreateError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => CreateError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => CreateError::OutOfHostMemory,
            _ => panic!("Unknown error {}", value),
        }
    }
}

pub type MapError = gpu_alloc::MapError;

#[derive(Debug)]
pub enum AcquireError {
    Timeout,
    NotReady,
    Suboptimal,
    OutOfHostMemory,
    OutOfDeviceMemory,
    DeviceLost,
    OutOfDate,
    SurfaceLost,
    FullScreenExclusiveModeLost,
}

impl From<vk::Result> for AcquireError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::TIMEOUT => AcquireError::Timeout,
            vk::Result::NOT_READY => AcquireError::NotReady,
            vk::Result::SUBOPTIMAL_KHR => AcquireError::Suboptimal,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => AcquireError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => AcquireError::OutOfHostMemory,
            vk::Result::ERROR_DEVICE_LOST => AcquireError::DeviceLost,
            vk::Result::ERROR_OUT_OF_DATE_KHR => AcquireError::OutOfDate,
            vk::Result::ERROR_SURFACE_LOST_KHR => AcquireError::SurfaceLost,
            vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
                AcquireError::FullScreenExclusiveModeLost
            }
            _ => panic!("Unknown error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum InstanceCreateError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    LayerNotPresent,
    ExtensionNotPresent,
    IncompatibeDriver,
    LibraryLoadingFailed,
    EntryPointNotFound,
}

impl From<vk::Result> for InstanceCreateError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => InstanceCreateError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => InstanceCreateError::OutOfHostMemory,
            vk::Result::ERROR_INITIALIZATION_FAILED => InstanceCreateError::InitializationFailed,
            vk::Result::ERROR_LAYER_NOT_PRESENT => InstanceCreateError::LayerNotPresent,
            vk::Result::ERROR_EXTENSION_NOT_PRESENT => InstanceCreateError::ExtensionNotPresent,
            vk::Result::ERROR_INCOMPATIBLE_DRIVER => InstanceCreateError::IncompatibeDriver,
            _ => panic!("Unknown error {}", value),
        }
    }
}

impl From<ash::LoadingError> for InstanceCreateError {
    fn from(value: ash::LoadingError) -> Self {
        match value {
            ash::LoadingError::LibraryLoadFailure(_) => InstanceCreateError::LibraryLoadingFailed,
            ash::LoadingError::MissingEntryPoint(_) => InstanceCreateError::EntryPointNotFound,
        }
    }
}

#[derive(Debug)]
pub enum PhysicalDeviceError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
}

impl From<vk::Result> for PhysicalDeviceError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => PhysicalDeviceError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => PhysicalDeviceError::OutOfHostMemory,
            vk::Result::ERROR_INITIALIZATION_FAILED => PhysicalDeviceError::InitializationFailed,
            _ => panic!("Unknown error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum SwapchainError {
    NoSuitableSurfaceFormat,
    Incomplete,
    WaitForSurface,
    OutOfDeviceMemory,
    OutOfHostMemory,
    SurfaceLost,
    DeviceLost,
    NativeWindowInUse,
    InitializationFailed,
}

impl From<vk::Result> for SwapchainError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => SwapchainError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => SwapchainError::OutOfHostMemory,
            vk::Result::ERROR_SURFACE_LOST_KHR => SwapchainError::SurfaceLost,
            vk::Result::INCOMPLETE => SwapchainError::Incomplete,
            vk::Result::ERROR_DEVICE_LOST => SwapchainError::DeviceLost,
            vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR => SwapchainError::NativeWindowInUse,
            vk::Result::ERROR_INITIALIZATION_FAILED => SwapchainError::InitializationFailed,
            _ => panic!("Unknown error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum WaitError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    DeviceLost,
}

impl From<vk::Result> for WaitError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => WaitError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => WaitError::OutOfHostMemory,
            vk::Result::ERROR_DEVICE_LOST => WaitError::DeviceLost,
            _ => panic!("Unknown error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum ResetError {
    OutOfDeviceMemory,
}

impl From<ResetError> for WaitError {
    fn from(value: ResetError) -> Self {
        match value {
            ResetError::OutOfDeviceMemory => WaitError::OutOfDeviceMemory,
        }
    }
}

impl From<vk::Result> for ResetError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => ResetError::OutOfDeviceMemory,
            _ => panic!("Unknown error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum ShaderCreateError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    ReflectionError(rspirv_reflect::ReflectError),
}

impl From<vk::Result> for ShaderCreateError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => ShaderCreateError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => ShaderCreateError::OutOfHostMemory,
            _ => panic!("Unknown error {}", value),
        }
    }
}

impl From<rspirv_reflect::ReflectError> for ShaderCreateError {
    fn from(value: rspirv_reflect::ReflectError) -> Self {
        ShaderCreateError::ReflectionError(value)
    }
}

#[derive(Debug)]
pub enum DeviceCreateError {
    NoSuitableQueues,
    NoExtension(String),
    OutOfDeviceMemory,
    OutOfHostMemory,
    InitializationFailed,
    ExtensionNotPresent,
    FeatureNotPresent,
    TooManyObjects,
    DeviceLost,
}

impl From<CreateError> for DeviceCreateError {
    fn from(value: CreateError) -> Self {
        match value {
            CreateError::OutOfDeviceMemory => DeviceCreateError::OutOfDeviceMemory,
            CreateError::OutOfHostMemory => DeviceCreateError::OutOfHostMemory,
        }
    }
}

impl From<vk::Result> for DeviceCreateError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => DeviceCreateError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => DeviceCreateError::OutOfHostMemory,
            vk::Result::ERROR_INITIALIZATION_FAILED => DeviceCreateError::InitializationFailed,
            vk::Result::ERROR_EXTENSION_NOT_PRESENT => DeviceCreateError::ExtensionNotPresent,
            vk::Result::ERROR_FEATURE_NOT_PRESENT => DeviceCreateError::FeatureNotPresent,
            vk::Result::ERROR_TOO_MANY_OBJECTS => DeviceCreateError::TooManyObjects,
            vk::Result::ERROR_DEVICE_LOST => DeviceCreateError::DeviceLost,
            _ => panic!("Unknown error {}", value),
        }
    }
}
