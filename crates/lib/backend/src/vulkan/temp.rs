use std::{
    cell::UnsafeCell,
    ptr::{copy_nonoverlapping, NonNull},
    sync::atomic::{AtomicU32, Ordering},
};

use ash::vk::{self};
use gpu_alloc::{MemoryPropertyFlags, Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;

use crate::BackendError;

use super::{GpuAllocator, GpuMemory};

pub struct TempGpuMemory {
    buffer: vk::Buffer,
    size: u32,
    top: AtomicU32,
    mapping: NonNull<u8>,
    memory: UnsafeCell<Option<GpuMemory>>,
}

unsafe impl Send for TempGpuMemory {}
unsafe impl Sync for TempGpuMemory {}

impl TempGpuMemory {
    pub fn new(
        device: &ash::Device,
        queue: u32,
        allocator: &mut GpuAllocator,
        size: u32,
    ) -> Result<Self, BackendError> {
        let buffer_desc = vk::BufferCreateInfo::builder()
            .queue_family_indices(&[queue])
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(size as _)
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_desc, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let request = Request {
            size: requirement.size,
            align_mask: requirement.alignment,
            memory_types: requirement.memory_type_bits,
            usage: UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::HOST_ACCESS,
        };
        let mut allocation = unsafe {
            allocator.alloc_with_dedicated(
                AshMemoryDevice::wrap(device),
                request,
                gpu_alloc::Dedicated::Required,
            )
        }?;
        unsafe { device.bind_buffer_memory(buffer, *allocation.memory(), allocation.offset()) }?;
        let mapping = unsafe { allocation.map(AshMemoryDevice::wrap(device), 0, size as _) }?;

        Ok(Self {
            buffer,
            size,
            top: AtomicU32::new(0),
            mapping,
            memory: UnsafeCell::new(Some(allocation)),
        })
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn cleanup(&self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(allocation) = (*self.memory.get()).take() {
            unsafe {
                allocator.dealloc(AshMemoryDevice::wrap(device), allocation);
                device.destroy_buffer(self.buffer, None);
            }
        }
    }

    pub fn reset(&self) {
        self.top.store(0, Ordering::Release);
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn flush(&self, device: &ash::Device) -> Result<(), BackendError> {
        let size = self.top.load(Ordering::Acquire);
        if size == 0 {
            return Ok(());
        }
        let memory = (*self.memory.get())
            .as_ref()
            .expect("Memory must be allocated in order to flush it to GPU");
        if memory.props().contains(MemoryPropertyFlags::HOST_COHERENT) {
            // No need to flush
            return Ok(());
        }
        let op = vk::MappedMemoryRange::builder()
            .memory(*memory.memory())
            .offset(memory.offset())
            .size(size as _)
            .build();
        unsafe { device.flush_mapped_memory_ranges(&[op]) }?;

        Ok(())
    }

    pub fn push<T: Sized>(&self, data: &[T]) -> Option<u32> {
        let bytes = std::mem::size_of_val(data);
        if let Some(offset) = self.allocate(bytes as _) {
            unsafe {
                copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.mapping.as_ptr().offset(offset as _),
                    bytes,
                );
            }
            Some(offset)
        } else {
            None
        }
    }

    fn allocate(&self, size: u32) -> Option<u32> {
        self.top
            .fetch_update(Ordering::Release, Ordering::SeqCst, |x| {
                let new_top = x + size;
                if new_top <= self.size {
                    Some(new_top)
                } else {
                    None
                }
            })
            .ok()
    }
}
