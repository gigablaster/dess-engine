use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;

use super::{GpuAllocator, GpuMemory};

const CAPACITY: usize = 128;

/// Список удаления
///
/// Хранит объекты отмеченные на удаление. Осовбождает ресурсы когда они
/// больше не используются и кадр в котором они были отмечены на удаление завершился.
#[derive(Debug)]
pub struct DropList {
    memory_to_free: Vec<GpuMemory>,
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
    /// Отмечает изображение на удаление
    pub fn drop_image(&mut self, image: vk::Image) {
        self.images_to_free.push(image);
    }

    /// Отмечает вид изображения на удаление
    pub fn drop_image_view(&mut self, view: vk::ImageView) {
        self.image_views_to_free.push(view);
    }

    /// Отмечает буфер на удаление
    pub fn drop_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers_to_free.push(buffer);
    }

    /// Отмечает блок памяти на удаление
    pub fn free_memory(&mut self, block: GpuMemory) {
        self.memory_to_free.push(block);
    }

    /// Реально освобождает ресурсы.
    ///
    /// Вызывается когда кадр, в котором объекты были отмечены на удаление.
    pub(crate) fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
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
