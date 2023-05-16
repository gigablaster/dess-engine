// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use core::slice;
use std::ptr::copy_nonoverlapping;

use ash::vk;
use log::debug;

use vk_sync::{cmd::pipeline_barrier, AccessType, BufferBarrier};

use crate::vulkan::BackendError;

use super::{BackendResult, FreeGpuResource, PhysicalDevice};

#[derive(Debug, Clone, Copy)]
struct BlockData(u64, u64);

#[derive(Debug, Clone, Copy)]
enum Block {
    Free(BlockData),
    Used(BlockData),
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryBlock {
    pub offset: u64,
    pub size: u64,
}

/// Sub-buffer allocator for all in-game geometry.
/// It's simple free-list allocator
#[derive(Debug)]
pub struct DynamicAllocator {
    size: u64,
    granularity: u64,
    blocks: Vec<Block>,
}

#[derive(Debug)]
pub enum AllocError {
    NotEnoughMemory,
    WrongBlock,
}

fn align(value: u64, align: u64) -> u64 {
    if value == 0 || value % align == 0 {
        value
    } else {
        (value & !(align - 1)) + align
    }
}
impl DynamicAllocator {
    pub fn new(size: u64, granularity: u64) -> Self {
        let mut blocks = Vec::with_capacity(256);
        blocks.push(Block::Free(BlockData(0, size)));
        Self {
            blocks,
            size,
            granularity,
        }
    }

    pub fn alloc(&mut self, size: u64) -> Result<MemoryBlock, AllocError> {
        if let Some(index) = self.find_free_block(size) {
            if let Some(offset) = self.split_and_insert_block(index, size) {
                return Ok(MemoryBlock { offset, size });
            }
        }

        Err(AllocError::NotEnoughMemory)
    }

    pub fn free(&mut self, offset: u64) -> Result<(), AllocError> {
        if let Some(index) = self.find_used_block(offset) {
            if let Block::Used(block) = self.blocks[index] {
                self.blocks[index] = Block::Free(block);
                self.merge_free_blocks(index);
                return Ok(());
            }
        }

        Err(AllocError::WrongBlock)
    }

    fn find_used_block(&self, offset: u64) -> Option<usize> {
        self.blocks.iter().enumerate().find_map(|(index, block)| {
            if let Block::Used(block) = block {
                if block.0 == offset {
                    return Some(index);
                }
            }
            None
        })
    }

    fn find_free_block(&self, size: u64) -> Option<usize> {
        self.blocks.iter().enumerate().find_map(|(index, block)| {
            if let Block::Free(block) = block {
                if block.1 >= size {
                    Some(index)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    fn merge_free_blocks(&mut self, index: usize) {
        let mut index = index;
        loop {
            if index > 0 {
                if let Some(block) = self.blocks.get(index - 1) {
                    if let Block::Free(_) = block {
                        index = index - 1;
                        continue;
                    }
                }
            }
            break;
        }
        loop {
            if let Block::Free(block) = self.blocks[index] {
                if let Some(next) = self.blocks.get(index + 1) {
                    if let Block::Free(next) = next {
                        self.blocks[index] = Block::Free(BlockData(block.0, block.1 + next.1));
                        self.blocks.remove(index + 1);
                        continue;
                    }
                }
            }
            break;
        }
    }

    fn split_and_insert_block(&mut self, index: usize, size: u64) -> Option<u64> {
        let size = align(size, self.granularity);
        if let Some(block) = self.blocks.get(index) {
            let block = *block;
            if let Block::Free(block) = block {
                assert!(size <= block.1);
                let new_size = block.1 - size;
                self.blocks[index] = Block::Used(BlockData(block.0, size));
                if new_size > 0 {
                    self.blocks
                        .insert(index + 1, Block::Free(BlockData(block.0 + size, new_size)));
                }
                return Some(block.0);
            }
        }

        None
    }
}

#[derive(Debug, Copy, Clone)]
struct BufferUploadRequest {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
    pub dst: vk::Buffer,
}

#[derive(Debug, Copy, Clone)]
struct ImageUploadRequest {
    pub src_offset: u64,
    pub dst_offset: vk::Offset3D,
    pub dst_subresource: vk::ImageSubresourceLayers,
    pub dst: vk::Image,
    pub dst_layout: vk::ImageLayout,
}

pub struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    granularity: u64,
    allocator: DynamicAllocator, // We supposed to use bump allocator but it will work in same way.
    upload_buffers: Vec<BufferUploadRequest>,
    upload_images: Vec<BufferUploadRequest>,
    mapping: Option<*mut u8>,
}

#[derive(Debug)]
pub enum StagingError {
    NeedUpload,
    VulkanError(vk::Result),
}

impl From<vk::Result> for StagingError {
    fn from(value: vk::Result) -> Self {
        StagingError::VulkanError(value)
    }
}

impl Staging {
    pub fn new(device: &ash::Device, pdevice: &PhysicalDevice, size: u64) -> BackendResult<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(size)
            .flags(vk::BufferCreateFlags::empty())
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (_, memory) = allocate_vram(
            device,
            pdevice,
            requirement.size,
            requirement.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            None,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self {
            buffer,
            memory,
            size,
            granularity: pdevice.properties.limits.buffer_image_granularity,
            allocator: DynamicAllocator::new(
                size,
                pdevice.properties.limits.buffer_image_granularity,
            ),
            upload_buffers: Vec::with_capacity(128),
            upload_images: Vec::with_capacity(32),
            mapping: None,
        })
    }

    fn map_buffer(&self, device: &ash::Device) -> Result<*mut u8, StagingError> {
        Ok(
            unsafe { device.map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty()) }?
                as *mut u8,
        )
    }

    fn unmap_buffer(&self, device: &ash::Device) {
        unsafe { device.unmap_memory(self.memory) };
    }

    pub fn upload_buffer(
        &mut self,
        device: &ash::Device,
        buffer: &Buffer,
        data: &[u8],
    ) -> Result<(), StagingError> {
        assert!(data.len() as u64 <= self.size);
        assert_eq!(buffer.size, data.len() as u64);
        if self.mapping.is_none() {
            self.mapping = Some(self.map_buffer(device)?);
        }
        let mapping = self.mapping.unwrap();
        if let Ok(block) = self.allocator.alloc(data.len() as _) {
            unsafe {
                copy_nonoverlapping(data.as_ptr(), mapping.add(block.offset as _), data.len())
            }
            let request = BufferUploadRequest {
                dst: buffer.buffer,
                src_offset: block.offset,
                dst_offset: buffer.offset,
                size: buffer.size,
            };
            self.upload_buffers.push(request);
            debug!("Query buffer upload {:?}", request);
        } else {
            debug!("No more space in staging - request upload");
            return Err(StagingError::NeedUpload);
        }

        Ok(())
    }

    pub fn upload(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        transfer_queue_index: u32,
        graphics_queue_index: u32,
    ) {
        if self.upload_buffers.is_empty() && self.upload_images.is_empty() {
            return;
        }
        if self.mapping.is_some() {
            self.unmap_buffer(device);
            self.mapping = None;
        }

        // Move main buffer to transfer queue and keep it there
        let barrier = BufferBarrier {
            previous_accesses: &[AccessType::HostWrite, AccessType::TransferRead],
            next_accesses: &[AccessType::HostWrite, AccessType::TransferRead],
            src_queue_family_index: 0,
            dst_queue_family_index: transfer_queue_index,
            buffer: self.buffer,
            offset: 0,
            size: self.size as _,
        };
        pipeline_barrier(device, cb, None, &[barrier], &[]);

        // Record buffer uploads
        self.move_requests_to_queue(
            &self.upload_buffers,
            device,
            cb,
            graphics_queue_index,
            transfer_queue_index,
        );
        self.copy_buffers(&self.upload_buffers, device, cb);
        self.move_requests_to_queue(
            &self.upload_buffers,
            device,
            cb,
            transfer_queue_index,
            graphics_queue_index,
        );

        self.allocator = DynamicAllocator::new(self.size, self.granularity);
        self.upload_buffers.clear();
    }

    fn move_requests_to_queue(
        &self,
        requests: &[BufferUploadRequest],
        device: &ash::Device,
        cb: vk::CommandBuffer,
        from: u32,
        to: u32,
    ) {
        let barriers = requests
            .iter()
            .map(|request| BufferBarrier {
                previous_accesses: &[
                    AccessType::TransferWrite,
                    AccessType::VertexBuffer,
                    AccessType::IndexBuffer,
                ],
                next_accesses: &[
                    AccessType::TransferWrite,
                    AccessType::VertexBuffer,
                    AccessType::IndexBuffer,
                ],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                buffer: request.dst,
                offset: request.dst_offset as _,
                size: request.size as _,
            })
            .collect::<Vec<_>>();
        pipeline_barrier(device, cb, None, &barriers, &[]);
    }

    fn copy_buffers(
        &self,
        requests: &[BufferUploadRequest],
        device: &ash::Device,
        cb: vk::CommandBuffer,
    ) {
        requests.iter().for_each(|request| {
            let region = vk::BufferCopy {
                src_offset: request.src_offset,
                dst_offset: request.dst_offset,
                size: request.size,
            };
            debug!("Upload request {:?}", request);
            unsafe {
                device.cmd_copy_buffer(cb, self.buffer, request.dst, slice::from_ref(&region))
            };
        });
    }
}

impl FreeGpuResource for Staging {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn allocate_vram(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    size: u64,
    mask: u32,
    desired_flags: vk::MemoryPropertyFlags,
    required_flags: Option<vk::MemoryPropertyFlags>,
) -> BackendResult<(u32, vk::DeviceMemory)> {
    let mut index = find_memory(pdevice, mask, desired_flags);
    if index.is_none() {
        if let Some(required_flags) = required_flags {
            index = find_memory(pdevice, mask, required_flags);
        }
    }

    if let Some(index) = index {
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .build();
        let mem = unsafe { device.allocate_memory(&alloc_info, None) }?;

        debug!(
            "Allocate {} bytes flags {:?}/{:?} type {}",
            size, desired_flags, required_flags, index
        );

        Ok((index, mem))
    } else {
        Err(BackendError::VramTypeNotFund)
    }
}

fn find_memory(pdevice: &PhysicalDevice, mask: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    let memory_prop = pdevice.memory_properties;
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & mask != 0 && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

#[derive(Debug, Copy, Clone)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}

/// Giant buffer that used for all static geometry data.
/// Clients are supposed to suballocate buffers by calling alloc and
/// free them by calling free.
pub struct BufferCache {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocator: DynamicAllocator,
}

impl BufferCache {
    pub fn new(device: &ash::Device, pdevice: &PhysicalDevice, size: u64) -> BackendResult<Self> {
        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (_, memory) = allocate_vram(
            device,
            pdevice,
            requirement.size,
            requirement.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            None,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
        let allocator = DynamicAllocator::new(size, 16);

        Ok(Self {
            buffer,
            memory,
            allocator,
        })
    }

    pub fn allocate(&mut self, size: u64) -> BackendResult<Buffer> {
        let block = self.allocator.alloc(size)?;
        debug!("Allocate buffer size {}", size);
        Ok(Buffer {
            buffer: self.buffer,
            offset: block.offset as _,
            size: block.size as _,
        })
    }

    pub fn free(&mut self, buffer: Buffer) -> BackendResult<()> {
        self.allocator.free(buffer.offset as _)?;

        Ok(())
    }
}

impl FreeGpuResource for BufferCache {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

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
            None,
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

    pub fn is_suitable(&self, requirement: &vk::MemoryRequirements, threshold: u64) -> bool {
        Self::purpose(requirement, threshold) == self.purpose
            && (1 << self.index) & requirement.memory_type_bits != 0
    }

    pub fn purpose(requirement: &vk::MemoryRequirements, threshold: u64) -> ChunkPurpose {
        if requirement.size <= threshold {
            return ChunkPurpose::Small;
        }
        if requirement.size > threshold * 4 {
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
                None,
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

#[derive(Debug)]
pub struct ImageCache {
    chunks: Vec<Chunk>,
    size: u64,
    threshold: u64,
}

impl ImageCache {
    pub fn new(size: u64, threshold: u64) -> Self {
        Self {
            chunks: Vec::new(),
            size,
            threshold,
        }
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
            if chunk.is_suitable(&requirement, self.threshold) {
                if let Ok(block) = chunk.allocate(index as _, device, pdevice, &requirement) {
                    return Ok(block);
                }
            }
        }
        let index = self.chunks.len();
        self.chunks.push(Chunk::new(
            device,
            pdevice,
            self.size,
            requirement.memory_type_bits,
            Chunk::purpose(&requirement, self.threshold),
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

#[cfg(test)]
mod test {
    use super::DynamicAllocator;

    #[test]
    fn alloc() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        assert_eq!(0, block1.offset);
        assert_eq!(100, block1.size);
        assert_eq!(128, block2.offset);
        assert_eq!(200, block2.size);
    }

    #[test]
    fn free() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        allocator.free(block1.offset).unwrap();
        allocator.free(block2.offset).unwrap();
        let block = allocator.alloc(300).unwrap();
        assert_eq!(0, block.offset);
        assert_eq!(300, block.size);
    }

    #[test]
    fn allocate_suitable_block() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let _block2 = allocator.alloc(200).unwrap();
        allocator.free(block1.offset).unwrap();
        let block = allocator.alloc(300).unwrap();
        assert_eq!(384, block.offset);
        assert_eq!(300, block.size);
    }

    #[test]
    fn allocate_small_blocks_in_hole_after_big() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        allocator.alloc(100).unwrap();
        let block = allocator.alloc(200).unwrap();
        allocator.alloc(100).unwrap();
        allocator.free(block.offset).unwrap();
        let block1 = allocator.alloc(50).unwrap();
        let block2 = allocator.alloc(50).unwrap();
        assert_eq!(128, block1.offset);
        assert_eq!(50, block1.size);
        assert_eq!(192, block2.offset);
        assert_eq!(50, block2.size);
    }

    #[test]
    fn not_anough_memory() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        allocator.alloc(500).unwrap();
        allocator.alloc(200).unwrap();
        assert!(allocator.alloc(500).is_err());
    }
}
