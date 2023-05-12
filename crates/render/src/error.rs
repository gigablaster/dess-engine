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

use ash::{vk, LoadingError};
use gpu_alloc::{AllocationError, MapError};
use rspirv_reflect::ReflectError;

#[derive(Debug)]
pub enum BackendError {
    Loading(LoadingError),
    Vulkan(vk::Result),
    Allocation(AllocationError),
    Mapping(MapError),
    NoExtension(String),
    Other(String),
    Reflection(ReflectError),
    RecreateSwapchain,
    WaitForSurface,
}

impl From<LoadingError> for BackendError {
    fn from(value: ash::LoadingError) -> Self {
        BackendError::Loading(value)
    }
}

impl From<vk::Result> for BackendError {
    fn from(value: vk::Result) -> Self {
        BackendError::Vulkan(value)
    }
}

impl From<String> for BackendError {
    fn from(value: String) -> Self {
        BackendError::Other(value)
    }
}

impl From<AllocationError> for BackendError {
    fn from(value: AllocationError) -> Self {
        BackendError::Allocation(value)
    }
}

impl From<ReflectError> for BackendError {
    fn from(value: ReflectError) -> Self {
        BackendError::Reflection(value)
    }
}

impl From<MapError> for BackendError {
    fn from(value: MapError) -> Self {
        BackendError::Mapping(value)
    }
}

pub type BackendResult<T> = std::result::Result<T, BackendError>;
