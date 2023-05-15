use ash::vk;
use log::debug;

use super::{
    memory::{allocate_vram, DynamicAllocator},
    BackendError, BackendResult, FreeGpuResource, PhysicalDevice,
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
    size: u64,
    purpose: ChunkPurpose,
    count: u32,
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

        debug!("Allocated chunk size {} purpose {:?}", size, purpose);

        Ok(Self {
            memory,
            allocator,
            index,
            purpose,
            size,
            count: 0,
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

    pub fn allocate(
        &mut self,
        chunk_index: u64,
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        requirement: &vk::MemoryRequirements,
    ) -> BackendResult<ImageMemory> {
        if self.memory == vk::DeviceMemory::null() {
            let (index, memory) = allocate_vram(
                device,
                pdevice,
                self.size,
                requirement.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            assert_eq!(self.index, index);
            self.memory = memory;
            debug!(
                "Allocated once freed chunk size {} purpose {:?}",
                self.size, self.purpose
            );
        }
        let block = self.allocator.alloc(requirement.size)?;
        self.count += 1;
        Ok(ImageMemory {
            memory: self.memory,
            chunk: chunk_index,
            offset: block.offset,
            size: requirement.size,
        })
    }

    pub fn deallocate(&mut self, device: &ash::Device, memory: ImageMemory) -> BackendResult<()> {
        self.allocator.free(memory.offset)?;
        self.count -= 1;
        if self.count == 0 {
            unsafe { device.free_memory(self.memory, None) };
            self.memory = vk::DeviceMemory::null();
            debug!("Chunk isn't needed for now - free device memory");
        }

        Ok(())
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
                if let Ok(block) = chunk.allocate(index as _, device, pdevice, &requirement) {
                    return Ok(block);
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
        return Ok(self.chunks[index].allocate(index as _, device, pdevice, &requirement)?);
    }

    pub fn deallocate(&mut self, device: &ash::Device, memory: ImageMemory) -> BackendResult<()> {
        self.chunks[memory.chunk as usize].deallocate(device, memory)?;

        Ok(())
    }
}

impl FreeGpuResource for ImageCache {
    fn free(&self, device: &ash::Device) {
        self.chunks.iter().for_each(|chunk| chunk.free(device));
    }
}
