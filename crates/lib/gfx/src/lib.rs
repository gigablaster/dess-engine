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

mod resource_manager;
mod staging;
mod temp;

use std::{
    marker::PhantomData,
    mem,
    ptr::{copy_nonoverlapping, NonNull},
    slice,
};

pub use resource_manager::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferSlice(BufferHandle, u32);

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

pub enum Error {
    OutOfSpace,
    BackendEror(dess_backend::Error),
}

impl From<dess_backend::Error> for Error {
    fn from(value: dess_backend::Error) -> Self {
        Self::BackendEror(value)
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
