use ash::vk;

use super::{
    memory::{allocate_vram, DynamicAllocator},
    BackendResult, FreeGpuResource, PhysicalDevice,
};

const CHUNK_SIZE: u64 = 256 * 1024 * 1024;
const SMALL_CHUNK_THRESHOLD: u64 = 2 * 1024 * 1024;
const BIG_CHUNK_THRESHOLD: u64 = 16 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
enum ChunkPurpose {
    Small,
    Normal,
    Big,
}

#[derive(Debug, Copy, Clone)]
pub struct ImageMemory {
    pub memory: vk::DeviceMemory,
    pub chunk: u64,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug)]
struct Chunk {
    memory: vk::DeviceMemory,
    allocator: DynamicAllocator,
    index: u32,
    purpose: ChunkPurpose,
}

impl Chunk {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        size: u64,
        mask: u32,
        purpose: ChunkPurpose,
    ) -> BackendResult<Self> {
        let (index, memory) = allocate_vram(
            device,
            pdevice,
            size,
            mask,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocator =
            DynamicAllocator::new(size, pdevice.properties.limits.buffer_image_granularity);

        Ok(Self {
            memory,
            allocator,
            index,
            purpose,
        })
    }

    pub fn is_suitable(&self, requirement: &vk::MemoryRequirements) -> bool {
        Self::purpose(requirement) == self.purpose
            && (1 << self.index) & requirement.memory_type_bits != 0
    }

    pub fn purpose(requirement: &vk::MemoryRequirements) -> ChunkPurpose {
        if requirement.size <= SMALL_CHUNK_THRESHOLD {
            return ChunkPurpose::Small;
        }
        if requirement.size > BIG_CHUNK_THRESHOLD {
            return ChunkPurpose::Big;
        }
        ChunkPurpose::Normal
    }
}

impl FreeGpuResource for Chunk {
    fn free(&self, device: &ash::Device) {
        unsafe { device.free_memory(self.memory, None) };
    }
}

#[derive(Debug, Default)]
pub struct ImageCache {
    chunks: Vec<Chunk>,
}

impl ImageCache {
    pub fn allocate(
        &mut self,
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        image: vk::Image,
    ) -> BackendResult<ImageMemory> {
        let requirement = unsafe { device.get_image_memory_requirements(image) };
        for index in 0..self.chunks.len() {
            let chunk = &mut self.chunks[index];
            if chunk.is_suitable(&requirement) {
                if let Ok(block) = chunk.allocator.alloc(requirement.size) {
                    return Ok(ImageMemory {
                        memory: chunk.memory,
                        chunk: index as _,
                        offset: block.offset,
                        size: block.size,
                    });
                }
            }
        }
        let index = self.chunks.len();
        self.chunks.push(Chunk::new(
            device,
            pdevice,
            CHUNK_SIZE,
            requirement.memory_type_bits,
            Chunk::purpose(&requirement),
        )?);
        let block = self.chunks[index].allocator.alloc(requirement.size as _)?;
        Ok(ImageMemory {
            memory: self.chunks[index].memory,
            chunk: index as _,
            offset: block.offset,
            size: block.size,
        })
    }

    pub fn free(&mut self, memory: ImageMemory) -> BackendResult<()> {
        self.chunks[memory.chunk as usize]
            .allocator
            .free(memory.offset)?;

        Ok(())
    }
}

impl FreeGpuResource for ImageCache {
    fn free(&self, device: &ash::Device) {
        self.chunks.iter().for_each(|chunk| chunk.free(device));
    }
}
