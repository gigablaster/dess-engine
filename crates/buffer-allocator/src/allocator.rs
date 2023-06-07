use std::marker::PhantomData;

use ash::vk;
use gpu_alloc::{GpuAllocator, MemoryBlock, Request};
use gpu_alloc_ash::AshMemoryDevice;

use crate::{
    error::BufferAllocationError,
    memory::{BlockBufferAllocator, DynamicBufferAllocator},
    Buffer, BufferAllocator, BufferHandle, BufferType, GeometryBufferType, UniformBufferType,
    MAX_BUFFERS,
};

#[derive(Debug)]
struct ChunkData<T: BufferAllocator, U: BufferType> {
    buffer: vk::Buffer,
    allocation: Option<MemoryBlock<vk::DeviceMemory>>,
    allocator: T,
    _phantom: PhantomData<U>,
}

impl<T: BufferAllocator, U: BufferType> ChunkData<T, U> {
    pub fn new(
        device: &ash::Device,
        size: u32,
        allocator: T,
        gpu_allocator: &mut GpuAllocator<vk::DeviceMemory>,
        family_index: u32,
    ) -> Result<Self, BufferAllocationError> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .queue_family_indices(&[family_index])
            .size(size as _)
            .usage(U::usage())
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let request = Request {
            size: requirements.size,
            align_mask: requirements.alignment,
            usage: U::MEMORY_TYPE,
            memory_types: requirements.memory_type_bits,
        };
        let allocation = unsafe { gpu_allocator.alloc(AshMemoryDevice::wrap(device), request) }?;
        unsafe { device.bind_buffer_memory(buffer, *allocation.memory(), allocation.offset()) }?;
        Ok(Self {
            buffer,
            allocation: None,
            allocator,
            _phantom: PhantomData,
        })
    }

    pub fn dispose(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator<vk::DeviceMemory>,
    ) {
        if let Some(allocation) = self.allocation.take() {
            unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), allocation) };
        }
        unsafe { device.destroy_buffer(self.buffer, None) };
    }

    pub fn allocate(&mut self, size: u32) -> Option<(vk::Buffer, u32)> {
        if let Some(offset) = self.allocator.allocate(size) {
            Some((self.buffer, offset))
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, offset: u32) {
        self.allocator.deallocate(offset);
    }
}

#[derive(Debug)]
enum Chunk<T: BufferType> {
    Block(u32, u32, ChunkData<BlockBufferAllocator, T>),
    Dynamic(u32, ChunkData<DynamicBufferAllocator, T>),
}

impl<T: BufferType> Chunk<T> {
    pub fn dynamic(
        min_size: u32,
        device: &ash::Device,
        size: u32,
        align: u32,
        gpu_allocator: &mut GpuAllocator<vk::DeviceMemory>,
        family_index: u32,
    ) -> Result<Chunk<T>, BufferAllocationError> {
        Ok(Self::Dynamic(
            min_size,
            ChunkData::new(
                device,
                size,
                DynamicBufferAllocator::new(size, align),
                gpu_allocator,
                family_index,
            )?,
        ))
    }

    pub fn block(
        min_size: u32,
        max_size: u32,
        device: &ash::Device,
        size: u32,
        gpu_allocator: &mut GpuAllocator<vk::DeviceMemory>,
        family_index: u32,
    ) -> Result<Chunk<T>, BufferAllocationError> {
        Ok(Self::Block(
            min_size,
            max_size,
            ChunkData::new(
                device,
                size,
                BlockBufferAllocator::new(max_size, size / max_size),
                gpu_allocator,
                family_index,
            )?,
        ))
    }

    pub fn allocate(&mut self, size: u32) -> Option<(vk::Buffer, u32)> {
        match self {
            Self::Block(min, max, data) => {
                if size >= *min && size <= *max {
                    data.allocate(size)
                } else {
                    None
                }
            }
            Self::Dynamic(min, data) => {
                if size >= *min {
                    data.allocate(size)
                } else {
                    None
                }
            }
        }
    }

    pub fn deallocate(&mut self, offset: u32) {
        match self {
            Self::Block(_, _, data) => data.deallocate(offset),
            Self::Dynamic(_, data) => data.deallocate(offset),
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        match self {
            Self::Block(_, _, data) => data.buffer,
            Self::Dynamic(_, data) => data.buffer,
        }
    }

    pub fn dispose(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator<vk::DeviceMemory>,
    ) {
        match self {
            Self::Block(_, _, data) => data.dispose(device, allocator),
            Self::Dynamic(_, data) => data.dispose(device, allocator),
        }
    }
}

#[derive(Debug)]
pub struct BufferCacheDesc {
    pub align: u32,
    pub max_block_size: u32,
    pub block_chunk_size: u32,
    pub chunk_size: u32,
    pub family_index: u32,
}

impl Default for BufferCacheDesc {
    fn default() -> Self {
        Self {
            align: 64,
            max_block_size: 16384,
            block_chunk_size: 2 * 1024 * 1024,
            chunk_size: 8 * 1024 * 1024,
            family_index: 0,
        }
    }
}

impl BufferCacheDesc {
    pub fn align(mut self, value: u32) -> Self {
        self.align = value;
        self
    }

    pub fn max_block_size(mut self, value: u32) -> Self {
        self.max_block_size = value;
        self
    }

    pub fn block_chunk_size(mut self, value: u32) -> Self {
        self.block_chunk_size = value;
        self
    }

    pub fn chunk_size(mut self, value: u32) -> Self {
        self.chunk_size = value;
        self
    }

    pub fn family_index(mut self, value: u32) -> Self {
        self.family_index = value;
        self
    }
}

#[derive(Debug, Default)]
pub struct BufferCache<T: BufferType> {
    desc: BufferCacheDesc,
    chunks: Vec<Chunk<T>>,
}

pub type GeometryBufferCache = BufferCache<GeometryBufferType>;
pub type UniformBufferCache = BufferCache<UniformBufferType>;
pub type GeometryBufferHandle = BufferHandle<GeometryBufferType>;
pub type UniformBufferHandle = BufferHandle<UniformBufferType>;

impl<T: BufferType> BufferCache<T> {
    pub fn new(desc: BufferCacheDesc) -> Self {
        Self {
            desc,
            chunks: Vec::with_capacity(8),
        }
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        gpu_allocator: &mut GpuAllocator<vk::DeviceMemory>,
        size: u32,
    ) -> Result<BufferHandle<T>, BufferAllocationError> {
        if let Some((index, offset)) =
            self.chunks
                .iter_mut()
                .enumerate()
                .find_map(|(index, chunk)| {
                    if let Some((_, offset)) = chunk.allocate(size) {
                        Some((index, offset))
                    } else {
                        None
                    }
                })
        {
            Ok(BufferHandle::new(index as _, offset))
        } else if self.chunks.len() as u32 == MAX_BUFFERS {
            Err(BufferAllocationError::TooManyChunks)
        } else {
            let mut new_chunk = if let Some((min, max)) = self.find_block_size(size) {
                Chunk::block(
                    min,
                    max,
                    device,
                    self.desc.block_chunk_size,
                    gpu_allocator,
                    self.desc.family_index,
                )?
            } else {
                Chunk::dynamic(
                    self.desc.max_block_size + 1,
                    device,
                    self.desc.chunk_size,
                    self.desc.align,
                    gpu_allocator,
                    self.desc.family_index,
                )?
            };
            let index = self.chunks.len() as u32;
            if let Some((_, offset)) = new_chunk.allocate(size) {
                self.chunks.push(new_chunk);
                Ok(BufferHandle::new(index, offset))
            } else {
                panic!("Failed to allocate buffer from freshly allocated chunk");
            }
        }
    }

    pub fn deallocate(&mut self, handle: BufferHandle<T>) {
        let index = handle.index();
        let offset = handle.offset();
        if index >= self.chunks.len() {
            panic!("Trying to free buffer with indes out of range");
        }
        self.chunks[index].deallocate(offset as _);
    }

    pub fn resolve(&self, handle: BufferHandle<T>) -> Buffer<T> {
        let index = handle.index();
        let offset = handle.offset();
        if index >= self.chunks.len() {
            panic!("Trying to resolve buffer with indes out of range");
        }
        Buffer {
            buffer: self.chunks[index].buffer(),
            offset: offset as u32,
            _phantom: PhantomData,
        }
    }

    pub fn dispose(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator<vk::DeviceMemory>,
    ) {
        self.chunks
            .iter_mut()
            .for_each(|chunk| chunk.dispose(device, allocator));
    }

    fn find_block_size(&self, size: u32) -> Option<(u32, u32)> {
        let mut start = 0;
        let mut end = self.desc.align;
        while end <= self.desc.max_block_size {
            if size >= start && size <= end {
                return Some((start, end));
            }
            let new_end = end << 1;
            start = end + 1;
            end = new_end;
        }

        None
    }
}
