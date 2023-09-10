use crate::vulkan::{CreateError, MapError, ResetError, ResourceCreateError, WaitError};

#[derive(Debug)]
pub enum StagingError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    NoCompatibleMemory,
    TooManyObjects,
    DeviceLost,
    MapFailed,
}

impl From<ResourceCreateError> for StagingError {
    fn from(value: ResourceCreateError) -> Self {
        match value {
            ResourceCreateError::NoCompatibleMemory => Self::NoCompatibleMemory,
            ResourceCreateError::OutOfDeviceMemory => Self::OutOfHostMemory,
            ResourceCreateError::OutOfHostMemory => Self::OutOfHostMemory,
            ResourceCreateError::TooManyObjects => Self::TooManyObjects,
        }
    }
}

impl From<CreateError> for StagingError {
    fn from(value: CreateError) -> Self {
        match value {
            CreateError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            CreateError::OutOfHostMemory => Self::OutOfHostMemory,
        }
    }
}

impl From<WaitError> for StagingError {
    fn from(value: WaitError) -> Self {
        match value {
            WaitError::OutOfHostMemory => Self::OutOfHostMemory,
            WaitError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            WaitError::DeviceLost => Self::DeviceLost,
        }
    }
}

impl From<ResetError> for StagingError {
    fn from(value: ResetError) -> Self {
        match value {
            ResetError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
        }
    }
}

impl From<MapError> for StagingError {
    fn from(value: MapError) -> Self {
        match value {
            gpu_alloc::MapError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            gpu_alloc::MapError::OutOfHostMemory => Self::OutOfHostMemory,
            gpu_alloc::MapError::MapFailed => Self::MapFailed,
            _ => panic!("Unexpected error {}", value),
        }
    }
}

#[derive(Debug)]
pub enum DescriptorError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    OutOfPoolMemory,
    Fragmentation,
    OutOfUniformSpace,
    MapFailed,
    NoCompatibleMemory,
    TooManyObjects,
}

impl From<gpu_descriptor::AllocationError> for DescriptorError {
    fn from(value: gpu_descriptor::AllocationError) -> Self {
        match value {
            gpu_descriptor::AllocationError::Fragmentation => DescriptorError::Fragmentation,
            gpu_descriptor::AllocationError::OutOfDeviceMemory => {
                DescriptorError::OutOfDeviceMemory
            }
            gpu_descriptor::AllocationError::OutOfHostMemory => DescriptorError::OutOfHostMemory,
        }
    }
}

impl From<gpu_descriptor::CreatePoolError> for DescriptorError {
    fn from(value: gpu_descriptor::CreatePoolError) -> Self {
        match value {
            gpu_descriptor::CreatePoolError::Fragmentation => DescriptorError::Fragmentation,
            gpu_descriptor::CreatePoolError::OutOfDeviceMemory => {
                DescriptorError::OutOfDeviceMemory
            }
            gpu_descriptor::CreatePoolError::OutOfHostMemory => DescriptorError::OutOfHostMemory,
        }
    }
}

impl From<gpu_descriptor::DeviceAllocationError> for DescriptorError {
    fn from(value: gpu_descriptor::DeviceAllocationError) -> Self {
        match value {
            gpu_descriptor::DeviceAllocationError::FragmentedPool => DescriptorError::Fragmentation,
            gpu_descriptor::DeviceAllocationError::OutOfDeviceMemory => {
                DescriptorError::OutOfDeviceMemory
            }
            gpu_descriptor::DeviceAllocationError::OutOfHostMemory => {
                DescriptorError::OutOfHostMemory
            }
            gpu_descriptor::DeviceAllocationError::OutOfPoolMemory => {
                DescriptorError::OutOfPoolMemory
            }
        }
    }
}

impl From<CreateError> for DescriptorError {
    fn from(value: CreateError) -> Self {
        match value {
            CreateError::OutOfDeviceMemory => DescriptorError::OutOfDeviceMemory,
            CreateError::OutOfHostMemory => DescriptorError::OutOfHostMemory,
        }
    }
}

impl From<UniformAllocateError> for DescriptorError {
    fn from(value: UniformAllocateError) -> Self {
        match value {
            UniformAllocateError::OutOfSpace => DescriptorError::OutOfUniformSpace,
        }
    }
}

impl From<UniformCreateError> for DescriptorError {
    fn from(value: UniformCreateError) -> Self {
        match value {
            UniformCreateError::OutOfDeviceMemory => DescriptorError::OutOfHostMemory,
            UniformCreateError::MapFailed => DescriptorError::MapFailed,
            UniformCreateError::NoCompatibleMemory => DescriptorError::NoCompatibleMemory,
            UniformCreateError::OutOfHostMemory => DescriptorError::OutOfHostMemory,
            UniformCreateError::TooManyObjects => DescriptorError::TooManyObjects,
        }
    }
}

#[derive(Debug)]
pub enum UniformCreateError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    NoCompatibleMemory,
    TooManyObjects,
    MapFailed,
}

#[derive(Debug)]

pub enum UniformAllocateError {
    OutOfSpace,
}

impl From<ResourceCreateError> for UniformCreateError {
    fn from(value: ResourceCreateError) -> Self {
        match value {
            ResourceCreateError::NoCompatibleMemory => UniformCreateError::NoCompatibleMemory,
            ResourceCreateError::OutOfDeviceMemory => UniformCreateError::OutOfDeviceMemory,
            ResourceCreateError::OutOfHostMemory => UniformCreateError::OutOfHostMemory,
            ResourceCreateError::TooManyObjects => UniformCreateError::TooManyObjects,
        }
    }
}

impl From<MapError> for UniformCreateError {
    fn from(value: MapError) -> Self {
        match value {
            gpu_alloc::MapError::MapFailed => UniformCreateError::MapFailed,
            gpu_alloc::MapError::OutOfHostMemory => UniformCreateError::OutOfHostMemory,
            gpu_alloc::MapError::OutOfDeviceMemory => UniformCreateError::OutOfDeviceMemory,
            _ => panic!("Unexpected error {}", value),
        }
    }
}
