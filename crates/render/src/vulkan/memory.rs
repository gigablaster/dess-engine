use ash::vk;
use log::debug;

use crate::vulkan::BackendError;

use super::{BackendResult, PhysicalDevice};

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
/*
pub struct BumpAllocator {
    size: u64,
    top: u64
}

impl BumpAllocator {
    pub fn new(size: u64) -> Self {
        Self {
            size,
            top: 0
        }
    }

    pub fn alloc(&mut self, size: u64, align: u64) -> Result<MemoryBlock, AllocError> {
    }
}
*/
pub fn allocate_vram(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    size: u64,
    mask: u32,
    flags: vk::MemoryPropertyFlags,
) -> BackendResult<(u32, vk::DeviceMemory)> {
    if let Some(index) = find_memory(pdevice, mask, flags) {
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .build();
        let mem = unsafe { device.allocate_memory(&alloc_info, None) }?;

        debug!("Allocate {} bytes flags {:?} type {}", size, flags, index);

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
        let block2 = allocator.alloc(200).unwrap();
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
