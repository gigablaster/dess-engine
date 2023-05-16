use ash::vk;

use super::memory::{Buffer, BufferCache, ImageCache, ImageMemory};

const CAPACITY: usize = 32;

#[derive(Debug, Copy, Clone)]
struct ImageDrop {
    pub image: vk::Image,
    pub memory: ImageMemory,
}

#[derive(Debug)]
pub(crate) struct DropList {
    images_to_free: Vec<ImageDrop>,
    image_views_to_free: Vec<vk::ImageView>,
    buffers_to_free: Vec<Buffer>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            images_to_free: Vec::with_capacity(CAPACITY),
            image_views_to_free: Vec::with_capacity(CAPACITY),
            buffers_to_free: Vec::with_capacity(CAPACITY),
        }
    }
}

impl DropList {
    pub fn drop_image(&mut self, image: vk::Image, memory: ImageMemory) {
        self.images_to_free.push(ImageDrop { memory, image });
    }

    pub fn drop_image_view(&mut self, view: vk::ImageView) {
        self.image_views_to_free.push(view);
    }

    pub fn drop_buffer(&mut self, buffer: Buffer) {
        self.buffers_to_free.push(buffer);
    }

    pub fn free(
        &mut self,
        device: &ash::Device,
        image_cache: &mut ImageCache,
        geometry_cache: &mut BufferCache,
    ) {
        self.image_views_to_free.drain(..).for_each(|view| {
            unsafe { device.destroy_image_view(view, None) };
        });
        self.images_to_free.drain(..).for_each(|image| {
            image_cache.deallocate(device, image.memory).unwrap();
            unsafe { device.destroy_image(image.image, None) };
        });
        self.buffers_to_free.drain(..).for_each(|buffer| {
            geometry_cache.free(buffer).unwrap();
        });
        self.image_views_to_free.shrink_to(CAPACITY);
        self.images_to_free.shrink_to(CAPACITY);
        self.buffers_to_free.shrink_to(CAPACITY);
    }
}
