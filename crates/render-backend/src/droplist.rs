use ash::vk;
use gpu_alloc::{GpuAllocator, MemoryBlock};
use gpu_alloc_ash::AshMemoryDevice;

const CAPACITY: usize = 128;

#[derive(Debug)]
pub(crate) struct DropList {
    memory_to_free: Vec<MemoryBlock<vk::DeviceMemory>>,
    images_to_free: Vec<vk::Image>,
    image_views_to_free: Vec<vk::ImageView>,
    buffers_to_free: Vec<vk::Buffer>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            memory_to_free: Vec::with_capacity(CAPACITY),
            images_to_free: Vec::with_capacity(CAPACITY),
            image_views_to_free: Vec::with_capacity(CAPACITY),
            buffers_to_free: Vec::with_capacity(CAPACITY),
        }
    }
}

impl DropList {
    pub fn drop_image(&mut self, image: vk::Image) {
        self.images_to_free.push(image);
    }

    pub fn drop_image_view(&mut self, view: vk::ImageView) {
        self.image_views_to_free.push(view);
    }

    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers_to_free.push(buffer);
    }

    pub fn free_memory(&mut self, block: MemoryBlock<vk::DeviceMemory>) {
        self.memory_to_free.push(block);
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator<vk::DeviceMemory>) {
        self.image_views_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_image_view(view, None) };
        });
        self.images_to_free.drain(..).for_each(|image| {
            unsafe { device.destroy_image(image, None) };
        });
        self.buffers_to_free.drain(..).for_each(|buffer| {
            unsafe { device.destroy_buffer(buffer, None) };
        });
        self.memory_to_free.drain(..).for_each(|block| {
            unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), block) };
        });
        self.memory_to_free.shrink_to(CAPACITY);
        self.image_views_to_free.shrink_to(CAPACITY);
        self.images_to_free.shrink_to(CAPACITY);
        self.buffers_to_free.shrink_to(CAPACITY);
    }
}
