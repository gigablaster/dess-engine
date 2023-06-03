use std::marker::PhantomData;

use ash::vk;
use gpu_alloc::UsageFlags;
use vk_sync::AccessType;

mod allocator;
mod error;
mod memory;

pub use allocator::*;
pub use error::*;

const BUFFER_INDEX_BITS: u32 = 8;
const MAX_BUFFERS: u32 = (1 << BUFFER_INDEX_BITS) - 1;
const BUFFER_INDEX_OFFSET: u32 = 32 - 8;
const BUFFER_OFFSET_MASK: u32 = (1 << BUFFER_INDEX_OFFSET) - 1;
const BUFFER_INDEX_MASK: u32 = u32::MAX & !BUFFER_OFFSET_MASK;

pub trait BufferAllocator {
    fn allocate(&mut self, size: u32) -> Option<u32>;
    fn deallocate(&mut self, offset: u32);
}

pub struct BufferHandle<T: BufferType> {
    value: u32,
    _phantom: PhantomData<T>,
}

impl<T: BufferType> BufferHandle<T> {
    pub fn new(buffer_index: u32, offset: u32) -> Self {
        assert_eq!(0, buffer_index >> BUFFER_INDEX_BITS);
        assert_eq!(offset, offset & BUFFER_OFFSET_MASK);
        Self {
            value: (buffer_index << BUFFER_INDEX_OFFSET) | offset,
            _phantom: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        ((self.value & BUFFER_INDEX_MASK) >> BUFFER_INDEX_OFFSET) as usize
    }

    pub fn offset(&self) -> usize {
        (self.value & BUFFER_OFFSET_MASK) as usize
    }
}

pub trait BufferType {
    const MEMORY_TYPE: UsageFlags;
    const ACCESS: &'static [AccessType];

    fn usage() -> vk::BufferUsageFlags;
}

pub struct GeometryBufferType {}

impl BufferType for GeometryBufferType {
    const MEMORY_TYPE: UsageFlags = UsageFlags::FAST_DEVICE_ACCESS;
    const ACCESS: &'static [AccessType] = &[AccessType::VertexBuffer, AccessType::IndexBuffer];

    fn usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
    }
}

pub struct UniformBufferType {}

impl BufferType for UniformBufferType {
    const ACCESS: &'static [AccessType] = &[AccessType::AnyShaderReadUniformBuffer];
    const MEMORY_TYPE: UsageFlags = UsageFlags::UPLOAD;

    fn usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::UNIFORM_BUFFER
    }
}