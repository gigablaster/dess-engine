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

use std::{collections::HashMap, sync::Arc};

use dess_backend::{
    BackendError, BackendResult, BufferCreateDesc, BufferHandle, BufferSlice, BufferUsage, Device,
};
use dess_common::DynamicAllocator;
use parking_lot::Mutex;

const CHUNK_SIZE: usize = 256 * 1024 * 1024;

/// Keeps all static geometry
///
/// All static geometry suballocated in one or few really big buffers. This way engine
/// guarentee that buffers aren't changed that often.
pub struct BufferPool {
    device: Arc<Device>,
    buffers: Mutex<HashMap<BufferHandle, DynamicAllocator>>,
}

impl BufferPool {
    pub fn new(device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            buffers: Mutex::default(),
        })
    }

    /// Suballocate buffer and copy data there
    ///
    /// Allocate new chunk if there's not enough space in already allocated
    /// memory.
    pub fn allocate<T: Sized>(&self, data: &[T]) -> BackendResult<BufferSlice> {
        let size = std::mem::size_of_val(data);
        let mut buffers = self.buffers.lock();
        let slice = if let Some(slice) = buffers.iter_mut().find_map(|(handle, allocator)| {
            allocator
                .allocate(size)
                .map(|offset| BufferSlice::new(*handle, offset as _))
        }) {
            slice
        } else {
            let buffer = self.device.create_buffer(BufferCreateDesc::gpu(
                CHUNK_SIZE,
                BufferUsage::Vertex
                    | BufferUsage::Index
                    | BufferUsage::Storage
                    | BufferUsage::Destination,
            ))?;
            let mut allocator = DynamicAllocator::new(CHUNK_SIZE, 1024);
            let offset = allocator.allocate(size).ok_or(BackendError::TooBig)?;
            buffers.insert(buffer, allocator);
            BufferSlice::new(buffer, offset as _)
        };
        self.device.upload_buffer(slice, data)?;

        Ok(slice)
    }

    /// Free previosly allocated buffer
    ///
    /// Panics if client attempt to deallocate buffer that wasn't allocated from
    /// same buffer pool.
    pub fn deallocate(&self, buffer: BufferSlice) {
        let mut buffers = self.buffers.lock();
        let allocator = buffers
            .get_mut(&buffer.handle)
            .expect("Only slices allocated from buffer pool can be deallocated there");
        allocator.deallocate(buffer.offset as _);
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        let buffers = self.buffers.lock();
        buffers.keys().for_each(|x| self.device.destroy_buffer(*x));
    }
}
