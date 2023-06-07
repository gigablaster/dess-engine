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
