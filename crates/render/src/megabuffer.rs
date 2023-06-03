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

use std::sync::Arc;

use ash::vk;
use dess_common::memory::{BlockAllocator, DynamicAllocator};
use dess_render_backend::{Buffer, BufferDesc, BufferView, Device};


use crate::{RenderError, RenderResult};

#[derive(Debug, Clone, Copy)]
enum Allocation {
    Dynamic(u64),
    Block(u64, usize),
}

const BLOCK_CHUNK_SIZE: u64 = 256 * 1024;
const BUFFER_SIZE: usize = 64 * 1024 * 1024;
const BLOCK_SIZE_TYPES: usize = 8;

#[derive(Debug)]
struct Block {
    offset: u64,
    block_size: u64,
    min_size: u64,
    allocator: BlockAllocator,
}

#[derive(Debug, Clone, Copy)]
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    allocation: Allocation,
}

impl BufferView for AllocatedBuffer {
    fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn offset(&self) -> u64 {
        self.offset
    }

    fn size(&self) -> u64 {
        self.size
    }
}

impl Block {
    pub fn new(offset: u64, size: u64, min_size: u64, block_size: u64) -> Self {
        Self {
            offset,
            min_size,
            block_size,
            allocator: BlockAllocator::new(block_size, (size / block_size) as _),
        }
    }

    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        if size > self.block_size || size < self.min_size {
            None
        } else if let Some(offset) = self.allocator.allocate() {
            Some(offset + self.offset)
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, offset: u64) {
        self.allocator.dealloc(offset);
    }
}

pub struct MegaBuffer {
    device: Arc<Device>,
    pub(crate) buffer: Buffer,
    allocator: DynamicAllocator,
    blocks: Vec<Block>,
    min_dynamic_size: u64,
}

impl MegaBuffer {
    pub fn new(device: &Arc<Device>) -> RenderResult<Self> {
        let buffer = Buffer::graphics(
            device,
            BufferDesc::gpu_only(
                BUFFER_SIZE,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            ),
            Some("megabuffer"),
        )?;
        let min_dynamic_size = device
            .pdevice
            .properties
            .limits
            .min_uniform_buffer_offset_alignment
            << BLOCK_SIZE_TYPES;
        Ok(Self {
            device: device.clone(),
            buffer,
            allocator: DynamicAllocator::new(
                BUFFER_SIZE as _,
                device
                    .pdevice
                    .properties
                    .limits
                    .min_uniform_buffer_offset_alignment,
            ),
            blocks: Vec::new(),
            min_dynamic_size,
        })
    }

    pub fn allocate(&mut self, size: usize) -> RenderResult<AllocatedBuffer> {
        let size = size as u64;
        if size == 0 || size > BUFFER_SIZE as _ {
            return Err(RenderError::WrongBufferSize);
        }
        if size > self.min_dynamic_size {
            if let Some(offset) = self.allocator.allocate(size) {
                return Ok(AllocatedBuffer {
                    buffer: self.buffer.raw,
                    offset,
                    size,
                    allocation: Allocation::Dynamic(offset),
                });
            } else {
                return Err(RenderError::OutOfMemory);
            }
        }

        let allocation = self
            .blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, block)| block.allocate(size).map(|offset| (offset, index)));
        if let Some((offset, index)) = allocation {
            Ok(AllocatedBuffer {
                buffer: self.buffer.raw,
                offset,
                size,
                allocation: Allocation::Block(offset, index),
            })
        } else {
            let (min, max) = self
                .find_block_size(size)
                .expect("We supposed to filter this variant before");
            if let Some(block_memory) = self.allocator.allocate(BLOCK_CHUNK_SIZE) {
                let index = self.blocks.len();
                let mut block = Block::new(block_memory, BLOCK_CHUNK_SIZE, min, max);
                let offset = block
                    .allocate(size)
                    .expect("Fresly allocated BlockAllocator must allocate something");
                self.blocks.push(block);
                Ok(AllocatedBuffer {
                    buffer: self.buffer.raw,
                    offset,
                    size,
                    allocation: Allocation::Block(offset, index),
                })
            } else {
                Err(crate::RenderError::OutOfMemory)
            }
        }
    }

    pub fn deallocate(&mut self, buffer: AllocatedBuffer) {
        match buffer.allocation {
            Allocation::Block(offset, index) => {
                self.blocks[index].deallocate(offset);
            }
            Allocation::Dynamic(offset) => self.allocator.deallocate(offset),
        }
    }

    fn find_block_size(&self, size: u64) -> Option<(u64, u64)> {
        let mut start = 0;
        let mut end = self
            .device
            .pdevice
            .properties
            .limits
            .min_uniform_buffer_offset_alignment;
        for _ in 0..BLOCK_SIZE_TYPES {
            if size >= start && size < end {
                return Some((start, end));
            }
            let new_end = end << 1;
            start = end + 1;
            end = new_end;
        }

        None
    }
}