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

use arrayvec::ArrayVec;
use ash::vk;

use crate::vulkan::{
    BindGroupHandle, BufferHandle, BufferSlice, ExecutionContext, RasterPipelineHandle,
};

pub(crate) const MAX_VERTEX_STREAMS: usize = 3;
pub(crate) const MAX_DESCRIPTOR_SETS: usize = 3;
pub(crate) const MAX_DYNAMIC_OFFSETS: usize = 4;

#[derive(Debug, Clone, Copy)]
struct Draw {
    pipeline: RasterPipelineHandle,
    vertex_buffers: [BufferSlice; MAX_VERTEX_STREAMS],
    index_buffer: BufferSlice,
    descriptors: [BindGroupHandle; MAX_DESCRIPTOR_SETS],
    dynamic_offsets: [u32; MAX_DYNAMIC_OFFSETS],
    first_index: u32,
    index_count: u32,
    instance_count: u32,
    first_instance: u32,
}

impl Default for Draw {
    fn default() -> Self {
        Self {
            pipeline: RasterPipelineHandle::default(),
            vertex_buffers: [BufferSlice::default(); MAX_VERTEX_STREAMS],
            index_buffer: BufferSlice::default(),
            descriptors: [BindGroupHandle::default(); MAX_DESCRIPTOR_SETS],
            dynamic_offsets: [u32::MAX; MAX_DYNAMIC_OFFSETS],
            first_index: u32::MAX,
            index_count: u32::MAX,
            instance_count: u32::MAX,
            first_instance: u32::MAX,
        }
    }
}

const PIPELINE: u16 = 1;
const VERTEX_BUFFER0: u16 = PIPELINE << 1;
const INDEX_BUFFER: u16 = VERTEX_BUFFER0 << MAX_VERTEX_STREAMS;
const DYNAMIC_OFFSET0: u16 = INDEX_BUFFER << 1;
const DS1: u16 = DYNAMIC_OFFSET0 << MAX_DYNAMIC_OFFSETS;
const INDEX_COUNT: u16 = DS1 << MAX_DESCRIPTOR_SETS;
const FIRST_INDEX: u16 = INDEX_COUNT << 1;
const INSTANCE_COUNT: u16 = FIRST_INDEX << 1;
const FIRST_INSTANCE: u16 = INSTANCE_COUNT << 1;

/// Collect draw calls
///
/// Optimize and pack draw calls, remove state changes and provides abstraction
/// for command buffer. All commands are stored in some sort of delta-compressed
/// storage and then decoded during render. Allows engine to store few draw calls per
/// one CPU cache line.
#[derive(Debug)]
pub struct DrawStream {
    pass_descriptor_set: BindGroupHandle,
    stream: Vec<u16>,
    current: Draw,
    mask: u16,
}

pub enum DrawStreamError {
    EndOfStream,
    InvalidHandle,
}

struct DrawStreamReader<'a> {
    stream: &'a [u16],
    cursor: usize,
}

impl<'a> DrawStreamReader<'a> {
    fn read(&mut self) -> Option<u16> {
        if self.cursor >= self.stream.len() {
            None
        } else {
            let data = self.stream[self.cursor];
            self.cursor += 1;
            Some(data)
        }
    }

    fn read_u32(&mut self) -> Result<u32, DrawStreamError> {
        let a = self.read().ok_or(DrawStreamError::EndOfStream)? as u32;
        let b = self.read().ok_or(DrawStreamError::EndOfStream)? as u32;
        Ok((a << 16) | b)
    }
}

impl DrawStream {
    pub fn new(pass_descriptor_set: BindGroupHandle) -> Self {
        Self {
            pass_descriptor_set,
            stream: Vec::with_capacity(1024),
            current: Draw::default(),
            mask: 0,
        }
    }

    pub fn bind_pipeline(&mut self, handle: RasterPipelineHandle) {
        if self.current.pipeline != handle {
            self.mask |= PIPELINE;
            self.current.pipeline = handle;
        }
    }

    pub fn bind_vertex_buffer(&mut self, slot: usize, buffer: Option<BufferSlice>) {
        debug_assert!(slot < MAX_VERTEX_STREAMS);
        let buffer = buffer.unwrap_or_default();
        if self.current.vertex_buffers[slot] != buffer {
            self.mask |= VERTEX_BUFFER0 << slot;
            self.current.vertex_buffers[slot] = buffer;
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: Option<BufferSlice>) {
        let buffer = buffer.unwrap_or_default();
        if self.current.index_buffer != buffer {
            self.mask |= INDEX_BUFFER;
            self.current.index_buffer = buffer;
        }
    }

    pub fn bind_descriptor_set(&mut self, slot: usize, ds: Option<BindGroupHandle>) {
        debug_assert!((1..=MAX_DESCRIPTOR_SETS).contains(&slot));
        let slot = slot - 1;
        let ds = ds.unwrap_or_default();
        if self.current.descriptors[slot] != ds {
            self.mask |= DS1 << slot;
            self.current.descriptors[slot] = ds;
        }
    }

    pub fn set_dynamic_buffer_offset(&mut self, slot: usize, offset: Option<u32>) {
        assert!(slot < MAX_DYNAMIC_OFFSETS);
        let offset = offset.unwrap_or(u32::MAX);
        if self.current.dynamic_offsets[slot] != offset {
            self.mask |= DYNAMIC_OFFSET0 << slot;
            self.current.dynamic_offsets[slot] = offset;
        }
    }

    pub fn draw(
        &mut self,
        first_index: u32,
        index_count: u32,
        instance_count: u32,
        first_instance: u32,
    ) {
        debug_assert!(
            self.current.pipeline.is_valid(),
            "Pipeline handle must be valid"
        );
        debug_assert!(index_count > 0, "Must draw at least one primitive");
        debug_assert!(instance_count > 0, "Must render at least one instance");
        if self.current.first_index != first_index {
            self.current.first_index = first_index;
            self.mask |= FIRST_INDEX;
        }
        if self.current.index_count != index_count {
            self.current.index_count = index_count;
            self.mask |= INDEX_COUNT;
        }
        if self.current.instance_count != instance_count {
            self.current.instance_count = instance_count;
            self.mask |= INSTANCE_COUNT;
        }
        if self.current.first_instance != first_instance {
            self.current.first_instance = first_instance;
            self.mask |= FIRST_INSTANCE;
        }
        self.stream.push(self.mask);
        if self.mask & PIPELINE != 0 {
            self.write_u32(self.current.pipeline.into());
        }
        for slot in 0..MAX_VERTEX_STREAMS {
            if self.mask & (VERTEX_BUFFER0 << slot) != 0 {
                self.encode_buffer_slice(self.current.vertex_buffers[slot]);
            }
        }
        if self.mask & INDEX_BUFFER != 0 {
            self.encode_buffer_slice(self.current.index_buffer);
        }
        for slot in 0..MAX_DYNAMIC_OFFSETS {
            if self.mask & (DYNAMIC_OFFSET0 << slot) != 0 {
                self.write_u32(self.current.dynamic_offsets[slot]);
            }
        }
        for slot in 0..MAX_DESCRIPTOR_SETS {
            if self.mask & (DS1 << slot) != 0 {
                self.write_u32(self.current.descriptors[slot].into());
            }
        }
        if self.mask & INDEX_COUNT != 0 {
            self.write_u32(self.current.index_count);
        }
        if self.mask & FIRST_INDEX != 0 {
            self.write_u32(self.current.first_index);
        }
        if self.mask & INSTANCE_COUNT != 0 {
            self.write_u32(self.current.instance_count);
        }
        if self.mask & FIRST_INSTANCE != 0 {
            self.write_u32(self.current.first_instance);
        }
        self.mask = 0;
    }

    #[allow(clippy::needless_range_loop)]
    pub(crate) fn execute(
        &self,
        context: &ExecutionContext,
        cb: vk::CommandBuffer,
    ) -> Result<(), DrawStreamError> {
        if self.stream.is_empty() {
            return Ok(());
        }
        puffin::profile_function!();
        let mut stream = DrawStreamReader {
            stream: &self.stream,
            cursor: 0,
        };
        let mut rebind_all_descriptors = true;
        let mut dynamic_offset_changed = false;
        let mut current_layout = vk::PipelineLayout::null();
        let mut descriptors = [vk::DescriptorSet::null(); MAX_DESCRIPTOR_SETS + 1];
        let mut dynamic_offsets = [0u32; MAX_DYNAMIC_OFFSETS];
        let mut first_index = 0u32;
        let mut vertex_count = 0u32;
        let mut instance_count = 0u32;
        let mut first_instance = 0u32;
        descriptors[0] = context
            .descriptors
            .get(self.pass_descriptor_set)
            .copied()
            .ok_or(DrawStreamError::InvalidHandle)?;

        while let Some(mask) = stream.read() {
            if mask & PIPELINE != 0 {
                let handle: RasterPipelineHandle = stream.read_u32()?.into();
                let (pipeline, layout) = context
                    .pipelines
                    .get(handle)
                    .copied()
                    .ok_or(DrawStreamError::InvalidHandle)?;
                if current_layout != layout {
                    rebind_all_descriptors = true;
                    current_layout = layout;
                }
                unsafe {
                    context.device.raw.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    )
                };
            }
            for index in 0..MAX_VERTEX_STREAMS {
                if mask & (VERTEX_BUFFER0 << index) != 0 {
                    let handle: BufferHandle = stream.read_u32()?.into();
                    let offset = stream.read_u32()?;
                    let buffer = if handle.is_valid() {
                        context
                            .buffers
                            .get(handle)
                            .copied()
                            .ok_or(DrawStreamError::InvalidHandle)?
                    } else {
                        vk::Buffer::null()
                    };
                    unsafe {
                        context.device.raw.cmd_bind_vertex_buffers(
                            cb,
                            index as _,
                            &[buffer],
                            &[offset as _],
                        )
                    };
                }
            }
            if mask & INDEX_BUFFER != 0 {
                let handle: BufferHandle = stream.read_u32()?.into();
                let offset = stream.read_u32()?;
                let buffer = if handle.is_valid() {
                    context
                        .buffers
                        .get(handle)
                        .copied()
                        .ok_or(DrawStreamError::InvalidHandle)?
                } else {
                    vk::Buffer::null()
                };
                unsafe {
                    context.device.raw.cmd_bind_index_buffer(
                        cb,
                        buffer,
                        offset as _,
                        vk::IndexType::UINT16,
                    )
                };
            }
            for index in 0..MAX_DYNAMIC_OFFSETS {
                if mask & (DYNAMIC_OFFSET0 << index) != 0 {
                    let offset = stream.read_u32()?;
                    if dynamic_offsets[index] != offset {
                        dynamic_offsets[index] = offset;
                        dynamic_offset_changed = true;
                    }
                }
            }
            for index in 0..MAX_DESCRIPTOR_SETS {
                if mask & (DS1 << index) != 0 {
                    let handle: BindGroupHandle = stream.read_u32()?.into();
                    let index = index + 1;
                    if handle.is_valid() {
                        descriptors[index] = context
                            .descriptors
                            .get(handle)
                            .copied()
                            .ok_or(DrawStreamError::InvalidHandle)?;
                    } else {
                        descriptors[index] = vk::DescriptorSet::null();
                    }
                    if !rebind_all_descriptors {
                        let ds = descriptors[index];
                        if index == 3 {
                            let mut offsets = ArrayVec::<_, MAX_DYNAMIC_OFFSETS>::new();
                            for offset_index in 0..MAX_DYNAMIC_OFFSETS {
                                if dynamic_offsets[offset_index] != u32::MAX {
                                    offsets.push(dynamic_offsets[offset_index]);
                                }
                            }
                            unsafe {
                                context.device.raw.cmd_bind_descriptor_sets(
                                    cb,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    current_layout,
                                    index as _,
                                    &[ds],
                                    &offsets,
                                )
                            }
                            dynamic_offset_changed = false;
                        } else {
                            unsafe {
                                context.device.raw.cmd_bind_descriptor_sets(
                                    cb,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    current_layout,
                                    index as _,
                                    &[ds],
                                    &[],
                                )
                            }
                        }
                    }
                }
            }
            if mask & FIRST_INDEX != 0 {
                first_index = stream.read_u32()?;
            }
            if mask & INDEX_COUNT != 0 {
                vertex_count = stream.read_u32()?
            }
            if mask & INSTANCE_COUNT != 0 {
                instance_count = stream.read_u32()?;
            }
            if mask & FIRST_INSTANCE != 0 {
                first_instance = stream.read_u32()?;
            }
            if rebind_all_descriptors {
                let mut offsets = ArrayVec::<_, MAX_DYNAMIC_OFFSETS>::new();
                for offset_index in 0..MAX_DYNAMIC_OFFSETS {
                    if dynamic_offsets[offset_index] != u32::MAX {
                        offsets.push(dynamic_offsets[offset_index]);
                    }
                }
                unsafe {
                    context.device.raw.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        current_layout,
                        0,
                        &descriptors,
                        &offsets,
                    );
                }
                rebind_all_descriptors = false;
                dynamic_offset_changed = false;
            } else if dynamic_offset_changed {
                let mut offsets = ArrayVec::<_, MAX_DYNAMIC_OFFSETS>::new();
                for offset_index in 0..MAX_DYNAMIC_OFFSETS {
                    if dynamic_offsets[offset_index] != u32::MAX {
                        offsets.push(dynamic_offsets[offset_index]);
                    }
                }
                unsafe {
                    context.device.raw.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        current_layout,
                        3,
                        &[descriptors[3]],
                        &offsets,
                    )
                }
                dynamic_offset_changed = false;
            }
            unsafe {
                context.device.raw.cmd_draw_indexed(
                    cb,
                    vertex_count,
                    instance_count,
                    first_index,
                    0,
                    first_instance,
                )
            }
        }
        Ok(())
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
