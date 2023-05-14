use ash::vk;

use super::{
    memory::{allocate_vram, DynamicAllocator},
    BackendResult, Buffer, FreeGpuResource, PhysicalDevice,
};

#[derive(Debug, Copy, Clone)]
pub struct GeometryBuffer {
    pub(self) buffer: vk::Buffer,
    pub(self) offset: u64,
    pub(self) size: u64,
}

impl Buffer for GeometryBuffer {
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

/// Giant buffer that used for all static geometry data.
/// Clients are supposed to suballocate buffers by calling alloc and
/// free them by calling free.
pub struct GeometryCache {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocator: DynamicAllocator,
}

impl GeometryCache {
    pub fn new(device: &ash::Device, pdevice: &PhysicalDevice, size: u32) -> BackendResult<Self> {
        let create_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = allocate_vram(
            device,
            pdevice,
            requirement.size,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) };
        let allocator = DynamicAllocator::new(size, 16);

        Ok(Self {
            buffer,
            memory,
            allocator,
        })
    }

    pub fn allocate(&mut self, size: u64) -> BackendResult<GeometryBuffer> {
        let block = self.allocator.alloc(size as u32)?;
        Ok(GeometryBuffer {
            buffer: self.buffer,
            offset: block.offset as _,
            size: block.size as _,
        })
    }

    pub fn free(&mut self, buffer: GeometryBuffer) -> BackendResult<()> {
        self.allocator.free(buffer.offset as _)?;

        Ok(())
    }
}

impl FreeGpuResource for GeometryCache {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}
