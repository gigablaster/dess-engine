use ash::vk;
use gpu_alloc::AllocationError;

pub enum BufferAllocationError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    NoCompatibleMemory,
    TooManyObjects,
    TooManyChunks,
}

impl From<vk::Result> for BufferAllocationError {
    fn from(value: vk::Result) -> Self {
        match value {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => BufferAllocationError::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => BufferAllocationError::OutOfHostMemory,
            vk::Result::ERROR_TOO_MANY_OBJECTS => BufferAllocationError::TooManyObjects,
            err => panic!("Unexpected error {:?}", err),
        }
    }
}

impl From<AllocationError> for BufferAllocationError {
    fn from(value: AllocationError) -> Self {
        match value {
            AllocationError::NoCompatibleMemoryTypes => BufferAllocationError::NoCompatibleMemory,
            AllocationError::OutOfDeviceMemory => BufferAllocationError::OutOfDeviceMemory,
            AllocationError::OutOfHostMemory => BufferAllocationError::OutOfHostMemory,
            AllocationError::TooManyObjects => BufferAllocationError::TooManyObjects,
        }
    }
}
