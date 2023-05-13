use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;

use crate::{Allocation, Allocator};

const CAPACITY: usize = 32;

#[derive(Debug)]
pub(crate) struct DropList {
    memory_to_free: Vec<Allocation>,
    images_to_free: Vec<vk::Image>,
    image_views_to_free: Vec<vk::ImageView>,
    buffers_to_free: Vec<vk::Buffer>,
    buffer_views_to_free: Vec<vk::BufferView>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            memory_to_free: Vec::with_capacity(CAPACITY * 2),
            images_to_free: Vec::with_capacity(CAPACITY),
            image_views_to_free: Vec::with_capacity(CAPACITY),
            buffers_to_free: Vec::with_capacity(CAPACITY),
            buffer_views_to_free: Vec::with_capacity(CAPACITY),
        }
    }
}

impl DropList {
    pub fn drop_memory(&mut self, memory: Allocation) {
        self.memory_to_free.push(memory);
    }

    pub fn drop_image(&mut self, image: vk::Image) {
        self.images_to_free.push(image);
    }

    pub fn drop_image_view(&mut self, view: vk::ImageView) {
        self.image_views_to_free.push(view);
    }

    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers_to_free.push(buffer);
    }

    pub fn drop_buffer_view(&mut self, view: vk::BufferView) {
        self.buffer_views_to_free.push(view);
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.image_views_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_image_view(view, None) };
        });
        self.buffer_views_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_buffer_view(view, None) };
        });
        self.images_to_free.drain(..).for_each(|image| {
            unsafe { device.destroy_image(image, None) };
        });
        self.buffers_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_buffer(view, None) };
        });
        let device = AshMemoryDevice::wrap(device);
        self.memory_to_free.drain(..).for_each(|mem| {
            unsafe { allocator.dealloc(device, mem) };
        });
        self.image_views_to_free.shrink_to(CAPACITY);
        self.buffer_views_to_free.shrink_to(CAPACITY);
        self.images_to_free.shrink_to(CAPACITY);
        self.buffers_to_free.shrink_to(CAPACITY);
        self.memory_to_free.shrink_to(CAPACITY * 2);
    }
}
