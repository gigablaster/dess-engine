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

use crate::{vulkan::{BufferSlice, DescriptorHandle, PipelineHandle, FrameContext, ExecutionContext}, DeferedPass, BackendResult};

pub(crate) const MAX_VERTEX_STREAMS: u32 = 3;
pub(crate) const MAX_DESCRIPTOR_SETS: u32 = 3;
pub(crate) const MAX_DYNAMIC_OFFSETS: u32 = 2;

#[derive(Debug, Clone, Copy)]
struct Draw {
    pipeline: PipelineHandle,
    vertex_buffers: [BufferSlice; MAX_VERTEX_STREAMS as usize],
    index_buffer: BufferSlice,
    descriptors: [DescriptorHandle; MAX_DESCRIPTOR_SETS as usize],
    dynamic_offsets: [u32; MAX_DYNAMIC_OFFSETS as usize],
    first_index: u32,
    instance_count: u32,
    triangle_count: u32,
}

impl Default for Draw {
    fn default() -> Self {
        Self {
            pipeline: PipelineHandle::default(),
            vertex_buffers: [
                BufferSlice::default(),
                BufferSlice::default(),
                BufferSlice::default(),
            ],
            index_buffer: BufferSlice::default(),
            descriptors: [
                DescriptorHandle::default(),
                DescriptorHandle::default(),
                DescriptorHandle::default(),
            ],
            dynamic_offsets: [u32::MAX, u32::MAX],
            instance_count: u32::MAX,
            first_index: u32::MAX,
            triangle_count: u32::MAX,
        }
    }
}

const PIPELINE: u16 = 1 << 0;
const VERTEX_BUFFER0: u16 = 1 << 1;
const VERTEX_BUFFER1: u16 = 1 << 2;
const VERTEX_BUFFER2: u16 = 1 << 3;
const INDEX_BUFFER: u16 = 1 << 4;
const DS1: u16 = 1 << 5;
const DS2: u16 = 1 << 6;
const DS3: u16 = 1 << 7;
const DYNAMIC_OFFSET0: u16 = 1 << 8;
const DYNAMIC_OFFSET1: u16 = 1 << 9;
const INSTANCE_COUNT: u16 = 1 << 10;
const TRIANGLE_COUNT: u16 = 1 << 11;
const FIRST_INDEX: u16 = 1 << 12;
const MAX_BIT: u16 = 13;

/// Collect draw calls
///
/// Optimize and pack draw calls, remove state changes and provides abstraction
/// for command buffer. All commands are stored in some sort of delta-compressed
/// storage and then decoded during render. Allows engine to store few draw calls per
/// one CPU cache line.
#[derive(Debug)]
pub struct DrawStream {
    pass_descriptor_set: DescriptorHandle,
    stream: Vec<u16>,
    current: Draw,
    mask: u16,
}

impl DrawStream {
    pub fn new(pass_descriptor_set: DescriptorHandle) -> Self {
        Self {
            pass_descriptor_set,
            stream: Vec::with_capacity(1024),
            current: Draw::default(),
            mask: 0,
        }
    }

    pub fn pass_descriptor_set(&self) -> DescriptorHandle {
        self.pass_descriptor_set
    }

    pub fn bind_pipeline(&mut self, handle: PipelineHandle) {
        if self.current.pipeline != handle {
            self.mask |= PIPELINE;
            self.current.pipeline = handle;
        }
    }

    pub fn bind_vertex_buffer(&mut self, slot: u32, buffer: Option<BufferSlice>) {
        debug_assert!(slot < MAX_VERTEX_STREAMS);
        let slot = slot as usize;
        let buffer = buffer.unwrap_or_default();
        if self.current.vertex_buffers[slot] != buffer {
            self.mask |= VERTEX_BUFFER0 << slot;
            self.current.vertex_buffers[slot] = buffer;
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: BufferSlice) {
        if self.current.index_buffer != buffer {
            self.mask |= INDEX_BUFFER;
            self.current.index_buffer = buffer;
        }
    }

    pub fn bind_descriptor_set(&mut self, slot: u32, ds: Option<DescriptorHandle>) {
        debug_assert!((1..=MAX_DESCRIPTOR_SETS).contains(&slot));
        let slot = (slot - 1) as usize;
        let ds = ds.unwrap_or_default();
        if self.current.descriptors[slot] != ds {
            self.mask |= DS1 << slot;
            self.current.descriptors[slot] = ds;
        }
    }

    pub fn set_dynamic_buffer_offset(&mut self, slot: u32, offset: Option<u32>) {
        assert!(slot < MAX_DYNAMIC_OFFSETS);
        let slot = slot as usize;
        let offset = offset.unwrap_or(u32::MAX);
        if self.current.dynamic_offsets[slot] != offset {
            self.mask |= DYNAMIC_OFFSET0 << slot;
            self.current.dynamic_offsets[slot] = offset;
        }
    }

    pub fn set_instance_count(&mut self, instance_count: u32) {
        debug_assert!(instance_count >= 1);
        if self.current.instance_count != instance_count {
            self.mask |= INSTANCE_COUNT;
            self.current.instance_count = instance_count;
        }
    }

    pub fn set_mesh(&mut self, first_idnex: u32, triangle_count: u32) {
        if self.current.first_index != first_idnex {
            self.mask |= FIRST_INDEX;
            self.current.first_index = first_idnex;
        }
        if self.current.triangle_count != triangle_count {
            self.mask |= TRIANGLE_COUNT;
            self.current.triangle_count = triangle_count;
        }
    }

    pub fn draw(&mut self) {
        debug_assert!(
            self.current.pipeline.is_valid(),
            "Pipeline handle must be valid"
        );
        debug_assert!(
            self.current.vertex_buffers[0].is_valid(),
            "First vertex stream must be set"
        );
        debug_assert!(
            self.current.index_buffer.is_valid(),
            "Index buffer must be set"
        );
        debug_assert!(
            self.current.triangle_count > 0,
            "Must draw at least one triangle"
        );
        self.stream.push(self.mask);
        if self.mask & PIPELINE != 0 {
            self.write_u32(self.current.pipeline.into());
        }
        for slot in 0..MAX_VERTEX_STREAMS {
            let slot = slot as usize;
            if self.mask & (VERTEX_BUFFER0 << slot) != 0 {
                self.encode_buffer_slice(self.current.vertex_buffers[slot]);
            }
        }
        if self.mask & INDEX_BUFFER != 0 {
            self.encode_buffer_slice(self.current.index_buffer);
        }
        for slot in 0..MAX_DESCRIPTOR_SETS {
            if self.mask & (DS1 << slot) != 0 {
                let slot = slot as usize;
                self.write_u32(self.current.descriptors[slot].into());
            }
        }
        for slot in 0..MAX_DYNAMIC_OFFSETS {
            if self.mask & (DYNAMIC_OFFSET0 << slot) != 0 {
                let slot = slot as usize;
                self.write_u32(self.current.dynamic_offsets[slot]);
            }
        }
        if self.mask & INSTANCE_COUNT != 0 {
            self.write_u32(self.current.instance_count);
        }
        if self.mask & TRIANGLE_COUNT != 0 {
            self.write_u32(self.current.triangle_count);
        }
        if self.mask & FIRST_INDEX != 0 {
            self.write_u32(self.current.first_index);
        }
        self.mask = 0;
    }

    pub(crate) fn execute(&self, context: &ExecutionContext, cb: vk::CommandBuffer) -> BackendResult<()> {
        todo!()
    }

    fn write_u32(&mut self, value: u32) {
        let (first, second) = (((value & 0xffff0000) >> 16) as u16, (value & 0xffff) as u16);
        self.stream.push(first);
        self.stream.push(second);
    }

    fn encode_buffer_slice(&mut self, buffer: BufferSlice) {
        self.write_u32(buffer.handle.into());
        self.write_u32(buffer.offset);
    }

    pub fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }
}
