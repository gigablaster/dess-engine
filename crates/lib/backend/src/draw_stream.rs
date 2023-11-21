use crate::vulkan::{BufferSlice, DescriptorHandle, PipelineHandle};

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
    instance_count: u32,
    first_index: u32,
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
const FIRST_INDEX: u16 = 1 << 11;
const TRIANGLE_COUNT: u16 = 1 << 12;
const MAX_BIT: u16 = 13;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawCommand {
    BindPipeline(PipelineHandle),
    BindVertexBuffer(u32, BufferSlice),
    UnbindVertexBuffer(u32),
    BindIndexBuffer(BufferSlice),
    SetDynamicBufferOffset(u32, u32),
    BindDescriptorSet(u32, DescriptorHandle),
    UnbindDescriptorSet(u32),
    Draw(u32, u32),
    DrawInstanced(u32, u32, u32),
}

#[derive(Debug)]
pub struct DrawStream {
    pass_ds: DescriptorHandle,
    stream: Vec<u16>,
    current: Draw,
    mask: u16,
}

impl DrawStream {
    pub fn new(pass_ds: DescriptorHandle) -> Self {
        Self {
            pass_ds,
            stream: Vec::with_capacity(1024),
            current: Draw::default(),
            mask: 0,
        }
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
        debug_assert!(slot < MAX_DESCRIPTOR_SETS);
        let slot = slot as usize;
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

    pub fn set_mesh(&mut self, first_index: u32, triangle_count: u32) {
        if self.current.first_index != first_index {
            self.mask |= FIRST_INDEX;
            self.current.first_index = first_index;
        }
        if self.current.triangle_count != triangle_count {
            self.mask |= TRIANGLE_COUNT;
            self.current.triangle_count = triangle_count;
        }
    }

    pub fn pass_descriptor_set(&self) -> DescriptorHandle {
        self.pass_ds
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
        if self.mask & FIRST_INDEX != 0 {
            self.write_u32(self.current.first_index);
        }
        if self.mask & TRIANGLE_COUNT != 0 {
            self.write_u32(self.current.triangle_count);
        }
        self.mask = 0;
    }

    pub fn iter(&self) -> Iter {
        Iter {
            stream: &self.stream,
            bit: 0,
            cursor: 0,
            mask: None,
            start_index: 0,
            triangle_count: 0,
            instance_count: 0,
        }
    }

    fn write_u32(&mut self, value: u32) {
        let (first, second) = (((value & 0xffff0000) >> 16) as u16, (value & 0xffff) as u16);
        self.stream.push(first);
        self.stream.push(second);
    }

    fn encode_buffer_slice(&mut self, buffer: BufferSlice) {
        self.write_u32(buffer.buffer.into());
        self.write_u32(buffer.offset);
    }

    pub fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }
}

pub struct Iter<'a> {
    stream: &'a [u16],
    bit: u16,
    cursor: usize,
    mask: Option<u16>,
    start_index: u32,
    triangle_count: u32,
    instance_count: u32,
}

impl<'a> Iter<'a> {
    fn read(&mut self) -> Result<u16, ()> {
        if self.cursor < self.stream.len() {
            let value = self.stream[self.cursor];
            self.cursor += 1;

            Ok(value)
        } else {
            Err(())
        }
    }

    fn read_u32(&mut self) -> Result<u32, ()> {
        let v1 = self.read()? as u32;
        let v2 = self.read()? as u32;

        Ok((v1 << 16) | v2)
    }

    fn read_buffer_slice(&mut self) -> Result<BufferSlice, ()> {
        let handle = self.read_u32()?.into();
        let offset = self.read_u32()?;

        Ok(BufferSlice::new(handle, offset))
    }

    fn read_vertex_buffer_command(&mut self, index: u32) -> Result<DrawCommand, ()> {
        let vertex_buffer = self.read_buffer_slice()?;
        if vertex_buffer.is_valid() {
            Ok(DrawCommand::BindVertexBuffer(index, vertex_buffer))
        } else {
            Ok(DrawCommand::UnbindVertexBuffer(index))
        }
    }

    fn read_bind_group_command(&mut self, index: u32) -> Result<DrawCommand, ()> {
        let ds: DescriptorHandle = self.read_u32()?.into();
        if ds.is_valid() {
            Ok(DrawCommand::BindDescriptorSet(index, ds))
        } else {
            Ok(DrawCommand::UnbindDescriptorSet(index))
        }
    }

    fn read_index_buffer_command(&mut self) -> Result<DrawCommand, ()> {
        Ok(DrawCommand::BindIndexBuffer(self.read_buffer_slice()?))
    }

    fn read_pipeline_command(&mut self) -> Result<DrawCommand, ()> {
        Ok(DrawCommand::BindPipeline(self.read_u32()?.into()))
    }

    fn advance_next_bit(&mut self) {
        self.bit += 1;
    }

    fn decode(&mut self) -> Option<DrawCommand> {
        if self.mask.is_none() {
            self.mask = self.read().ok();
            self.bit = 0;
        }
        loop {
            if let Some(mask) = self.mask {
                while self.bit < MAX_BIT && (mask & 1 << self.bit) != (1 << self.bit) {
                    self.advance_next_bit();
                }
                let mask = 1 << self.bit;
                match mask {
                    PIPELINE => {
                        self.advance_next_bit();
                        return self.read_pipeline_command().ok();
                    }
                    VERTEX_BUFFER0 => {
                        self.advance_next_bit();
                        return self.read_vertex_buffer_command(0).ok();
                    }
                    VERTEX_BUFFER1 => {
                        self.advance_next_bit();
                        return self.read_vertex_buffer_command(1).ok();
                    }
                    VERTEX_BUFFER2 => {
                        self.advance_next_bit();
                        return self.read_vertex_buffer_command(2).ok();
                    }

                    INDEX_BUFFER => {
                        self.advance_next_bit();
                        return self.read_index_buffer_command().ok();
                    }

                    DS1 => {
                        self.advance_next_bit();
                        return self.read_bind_group_command(0).ok();
                    }
                    DS2 => {
                        self.advance_next_bit();
                        return self.read_bind_group_command(1).ok();
                    }
                    DS3 => {
                        self.advance_next_bit();
                        return self.read_bind_group_command(2).ok();
                    }

                    DYNAMIC_OFFSET0 => {
                        self.advance_next_bit();

                        return self
                            .read_u32()
                            .map(|x| DrawCommand::SetDynamicBufferOffset(0, x))
                            .ok();
                    }
                    DYNAMIC_OFFSET1 => {
                        self.advance_next_bit();

                        return self
                            .read_u32()
                            .map(|x| DrawCommand::SetDynamicBufferOffset(1, x))
                            .ok();
                    }

                    INSTANCE_COUNT => {
                        self.advance_next_bit();

                        if let Ok(count) = self.read_u32() {
                            self.instance_count = count
                        } else {
                            return None;
                        }
                    }
                    FIRST_INDEX => {
                        self.advance_next_bit();

                        if let Ok(index) = self.read_u32() {
                            self.start_index = index;
                        } else {
                            return None;
                        }
                    }
                    TRIANGLE_COUNT => {
                        self.advance_next_bit();

                        if let Ok(count) = self.read_u32() {
                            self.triangle_count = count;
                        } else {
                            return None;
                        }
                    }

                    _ => {
                        return None;
                    }
                }
            } else {
                return None;
            }
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = DrawCommand;

    fn next(&mut self) -> Option<Self::Item> {
        let command = self.decode();
        match (command, self.mask) {
            (Some(command), _) => Some(command),
            (None, Some(_)) => {
                self.mask = None;
                Some(DrawCommand::Draw(self.start_index, self.triangle_count))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use dess_common::Handle;

    use crate::{
        vulkan::{BufferSlice, Index},
        DrawCommand, DrawStream,
    };

    #[test]
    #[should_panic]
    fn fail_default_state() {
        let mut stream = DrawStream::new(Handle::default());
        stream.draw();
    }

    #[test]
    fn basic_usage() {
        let mut stream = DrawStream::new(Handle::default());
        stream.bind_pipeline(Index::new(0));
        stream.bind_vertex_buffer(0, Some(BufferSlice::new(Handle::new(1, 1), 0)));
        stream.bind_index_buffer(BufferSlice::new(Handle::new(2, 2), 0));
        stream.set_mesh(10, 100);
        stream.draw();

        let stream = stream.iter().collect::<Vec<_>>();
        assert_eq!(4, stream.len());
        assert_eq!(DrawCommand::BindPipeline(Index::new(0)), stream[0]);
        assert_eq!(
            DrawCommand::BindVertexBuffer(0, BufferSlice::new(Handle::new(1, 1), 0)),
            stream[1]
        );
        assert_eq!(
            DrawCommand::BindIndexBuffer(BufferSlice::new(Handle::new(2, 2), 0)),
            stream[2]
        );
        assert_eq!(DrawCommand::Draw(10, 100), stream[3]);
    }

    #[test]
    fn full_usage() {
        let mut stream = DrawStream::new(Handle::default());
        stream.bind_pipeline(Index::new(0));
        stream.bind_vertex_buffer(0, Some(BufferSlice::new(Handle::new(1, 1), 0)));
        stream.bind_vertex_buffer(1, Some(BufferSlice::new(Handle::new(2, 2), 0)));
        stream.bind_vertex_buffer(2, Some(BufferSlice::new(Handle::new(99, 99), 100)));
        stream.bind_index_buffer(BufferSlice::new(Handle::new(3, 3), 0));
        stream.bind_descriptor_set(0, Some(Handle::new(4, 4)));
        stream.bind_descriptor_set(1, Some(Handle::new(5, 5)));
        stream.bind_descriptor_set(2, Some(Handle::new(6, 6)));
        stream.set_dynamic_buffer_offset(0, Some(0));
        stream.set_mesh(10, 100);
        stream.draw();
        stream.set_dynamic_buffer_offset(0, Some(64));
        stream.draw();

        let stream = stream.iter().collect::<Vec<_>>();
        assert_eq!(12, stream.len());
        assert_eq!(DrawCommand::BindPipeline(Index::new(0)), stream[0]);
        assert_eq!(
            DrawCommand::BindVertexBuffer(0, BufferSlice::new(Handle::new(1, 1), 0)),
            stream[1]
        );
        assert_eq!(
            DrawCommand::BindVertexBuffer(1, BufferSlice::new(Handle::new(2, 2), 0)),
            stream[2]
        );
        assert_eq!(
            DrawCommand::BindVertexBuffer(2, BufferSlice::new(Handle::new(99, 99), 100)),
            stream[3]
        );
        assert_eq!(
            DrawCommand::BindIndexBuffer(BufferSlice::new(Handle::new(3, 3), 0)),
            stream[4]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(0, Handle::new(4, 4)),
            stream[5]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(1, Handle::new(5, 5)),
            stream[6]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(2, Handle::new(6, 6)),
            stream[7]
        );
        assert_eq!(DrawCommand::SetDynamicBufferOffset(0, 0), stream[8]);
        assert_eq!(DrawCommand::Draw(10, 100), stream[9]);
        assert_eq!(DrawCommand::SetDynamicBufferOffset(0, 64), stream[10]);
        assert_eq!(DrawCommand::Draw(10, 100), stream[11]);
    }

    #[test]
    fn reduce_state_changes() {
        let mut stream = DrawStream::new(Handle::default());
        stream.bind_pipeline(Index::new(0));
        stream.bind_vertex_buffer(0, Some(BufferSlice::new(Handle::new(100, 100), 100)));
        stream.bind_vertex_buffer(1, Some(BufferSlice::new(Handle::new(200, 200), 999)));
        stream.bind_vertex_buffer(2, Some(BufferSlice::new(Handle::new(99, 99), 100)));

        stream.bind_vertex_buffer(0, Some(BufferSlice::new(Handle::new(1, 1), 0)));
        stream.bind_vertex_buffer(1, Some(BufferSlice::new(Handle::new(2, 2), 0)));
        stream.bind_vertex_buffer(2, Some(BufferSlice::new(Handle::new(99, 99), 100)));

        stream.bind_index_buffer(BufferSlice::new(Handle::new(500, 600), 200));
        stream.bind_index_buffer(BufferSlice::new(Handle::new(3, 3), 0));

        stream.bind_descriptor_set(0, Some(Handle::new(400, 400)));
        stream.bind_descriptor_set(1, Some(Handle::new(500, 500)));
        stream.bind_descriptor_set(2, Some(Handle::new(600, 600)));

        stream.bind_descriptor_set(0, Some(Handle::new(4, 4)));
        stream.bind_descriptor_set(1, Some(Handle::new(5, 5)));
        stream.bind_descriptor_set(2, Some(Handle::new(6, 6)));

        stream.set_dynamic_buffer_offset(0, Some(200));

        stream.set_dynamic_buffer_offset(0, Some(0));

        stream.set_mesh(100, 1000);

        stream.set_mesh(10, 100);
        stream.draw();

        stream.set_dynamic_buffer_offset(0, Some(256));

        stream.set_dynamic_buffer_offset(0, Some(64));
        stream.draw();

        let stream = stream.iter().collect::<Vec<_>>();
        assert_eq!(12, stream.len());
        assert_eq!(DrawCommand::BindPipeline(Index::new(0)), stream[0]);
        assert_eq!(
            DrawCommand::BindVertexBuffer(0, BufferSlice::new(Handle::new(1, 1), 0)),
            stream[1]
        );
        assert_eq!(
            DrawCommand::BindVertexBuffer(1, BufferSlice::new(Handle::new(2, 2), 0)),
            stream[2]
        );
        assert_eq!(
            DrawCommand::BindVertexBuffer(2, BufferSlice::new(Handle::new(99, 99), 100)),
            stream[3]
        );
        assert_eq!(
            DrawCommand::BindIndexBuffer(BufferSlice::new(Handle::new(3, 3), 0)),
            stream[4]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(0, Handle::new(4, 4)),
            stream[5]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(1, Handle::new(5, 5)),
            stream[6]
        );
        assert_eq!(
            DrawCommand::BindDescriptorSet(2, Handle::new(6, 6)),
            stream[7]
        );
        assert_eq!(DrawCommand::SetDynamicBufferOffset(0, 0), stream[8]);
        assert_eq!(DrawCommand::Draw(10, 100), stream[9]);
        assert_eq!(DrawCommand::SetDynamicBufferOffset(0, 64), stream[10]);
        assert_eq!(DrawCommand::Draw(10, 100), stream[11]);
    }
}
