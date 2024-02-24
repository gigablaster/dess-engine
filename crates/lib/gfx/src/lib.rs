// Copyright (C) 2024 gigablaster

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

mod draw_stream;
mod renderer;
mod staging;
mod temp;
mod temp_images;

use std::{
    io,
    marker::PhantomData,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    slice,
    sync::Arc,
};

use ash::vk;
use dess_backend::{Buffer, Image, Program, RenderPass};
use dess_common::{HotColdPool, Pool, SentinelPoolStrategy};
pub use draw_stream::*;
pub use renderer::*;
pub use temp_images::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferSlice(BufferHandle, u32);

impl Default for BufferSlice {
    fn default() -> Self {
        Self(BufferHandle::invalid(), 0)
    }
}

impl BufferSlice {
    pub fn new(handle: BufferHandle, offset: u32) -> Self {
        Self(handle, offset)
    }

    pub fn handle(&self) -> BufferHandle {
        self.0
    }

    pub fn offset(&self) -> u32 {
        self.1
    }
}

#[derive(Debug)]
pub(crate) struct GpuBufferWriterImpl<T: Sized + Copy> {
    ptr: NonNull<u8>,
    offset: usize,
    count: usize,
    cursor: usize,
    _marker: PhantomData<T>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Out of space in descriptor pool")]
    OutOfSpace,
    #[error("Backend error: {0}")]
    BackendError(dess_backend::Error),
    #[error("IO operation failed: {0}")]
    Io(io::Error),
    #[error("Shader parsing failed: {0}")]
    ShaderParsingFailed(String),
    #[error("Shader compilation failed:\n{0}")]
    ShaderCompilationFailed(String),
    #[error("Image is too big for staging")]
    ImageTooBig,
    #[error("Handle isn't valid")]
    InvalidHandle,
    #[error("Descriptor binding {0} not found")]
    BindingNotFound(usize),
    #[error("Descrptor set isn't fully initialized")]
    InvalidDescriptorSet,
}

impl From<dess_backend::Error> for Error {
    fn from(value: dess_backend::Error) -> Self {
        Self::BackendError(value)
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<vk::Result> for Error {
    fn from(value: vk::Result) -> Self {
        dess_backend::Error::from(value).into()
    }
}

impl From<gpu_descriptor::AllocationError> for Error {
    fn from(value: gpu_descriptor::AllocationError) -> Self {
        match value {
            gpu_descriptor::AllocationError::OutOfDeviceMemory => {
                Self::BackendError(dess_backend::Error::OutOfDeviceMemory)
            }
            gpu_descriptor::AllocationError::OutOfHostMemory => {
                Self::BackendError(dess_backend::Error::OutOfHostMemory)
            }
            gpu_descriptor::AllocationError::Fragmentation => {
                Self::BackendError(dess_backend::Error::Fragmentation)
            }
        }
    }
}

impl<T: Sized + Copy> GpuBufferWriterImpl<T> {
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn write(&mut self, data: &T) -> Result<(), Error> {
        if self.count == self.cursor {
            return Err(Error::OutOfSpace);
        }
        let size = self.count * mem::size_of::<T>();
        unsafe {
            copy_nonoverlapping(
                slice::from_ref(data).as_ptr() as *const u8,
                self.ptr.as_ptr().add(self.offset + size * self.cursor),
                size,
            )
        }
        self.cursor += 1;

        Ok(())
    }
}

#[derive(Debug)]
pub struct GpuBuferWriter<T: Sized + Copy> {
    handle: BufferHandle,
    writer: GpuBufferWriterImpl<T>,
}

impl<T: Sized + Copy> GpuBuferWriter<T> {
    pub fn write(&mut self, data: &T) -> Result<(), Error> {
        self.writer.write(data)
    }

    pub fn finish(self) -> BufferSlice {
        BufferSlice::new(self.handle, self.writer.offset() as _)
    }
}

type ImagePool = Pool<Arc<Image>>;
type BufferPool = HotColdPool<vk::Buffer, Arc<Buffer>, SentinelPoolStrategy<vk::Buffer>>;
type ProgramPool = Vec<Arc<Program>>;
type RenderPassPool = Vec<Arc<RenderPass>>;
