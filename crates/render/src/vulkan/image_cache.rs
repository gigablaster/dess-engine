use ash::vk;

use super::{
    memory::{allocate_vram, DynamicAllocator},
    BackendResult, Device, FreeGpuResource, PhysicalDevice,
};

#[derive(Debug, Copy, Clone)]
pub struct ImageMemory {
    pub memory: vk::DeviceMemory,
    pub chunk: u32,
    pub offset: u32,
    pub size: u32,
}

struct Chunk {
    pub memory: vk::DeviceMemory,
    pub allocator: DynamicAllocator,
}

impl Chunk {
    pub fn new(device: &ash::Device, pdevice: &PhysicalDevice, size: u64) -> BackendResult<Self> {
        let memory = allocate_vram(device, pdevice, size, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let allocator = DynamicAllocator::new(
            size as _,
            pdevice.properties.limits.buffer_image_granularity as _,
        );

        Ok(Self { memory, allocator })
    }
}

impl FreeGpuResource for Chunk {
    fn free(&self, device: &ash::Device) {
        unsafe { device.free_memory(self.memory, None) };
    }
}

pub struct ImageCache {
    chunks: Vec<Chunk>,
    chunk_size: u64,
}

impl ImageCache {
    pub fn new(
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        chunk_size: u64,
    ) -> BackendResult<Self> {
        let mut chunks = Vec::with_capacity(4);
        chunks.push(Chunk::new(device, pdevice, chunk_size)?);
        Ok(Self { chunk_size, chunks })
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        image: vk::Image,
    ) -> BackendResult<ImageMemory> {
        let requirement = unsafe { device.get_image_memory_requirements(image) };
        for index in 0..self.chunks.len() {
            let chunk = &mut self.chunks[index];
            if let Ok(block) = chunk.allocator.alloc(requirement.size as _) {
                return Ok(ImageMemory {
                    memory: chunk.memory,
                    chunk: index as _,
                    offset: block.offset,
                    size: block.size,
                });
            }
        }
        let index = self.chunks.len();
        self.chunks
            .push(Chunk::new(device, pdevice, self.chunk_size)?);
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
