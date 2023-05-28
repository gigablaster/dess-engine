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

use dess_render_backend::BackendError;
use gpu_descriptor::AllocationError;

#[derive(Debug)]
pub enum RenderError {
    Backend(BackendError),
    DescriptorAllocation(AllocationError),
    DeviceNotFound,
    RecreateBuffers,
    OutOfMemory,
    WrongBufferSize,
}

impl From<BackendError> for RenderError {
    fn from(value: BackendError) -> Self {
        RenderError::Backend(value)
    }
}

impl From<AllocationError> for RenderError {
    fn from(value: AllocationError) -> Self {
        RenderError::DescriptorAllocation(value)
    }
}

pub type RenderResult<T> = Result<T, RenderError>;
