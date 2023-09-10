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
            _ => panic!("Unknown error {}", value),
        }
    }
}

pub enum DescriptorError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    OutOfPoolMemory,
    Fragmentation,
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
