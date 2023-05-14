use ash::vk;
use common::Align;

#[derive(Debug, Clone, Copy)]
struct BlockData(u32, u32);

#[derive(Debug, Clone, Copy)]
enum Block {
    Free(BlockData),
    Used(BlockData),
}

#[derive(Debug)]
pub struct MemoryBlock<M: Copy> {
    pub memory: M,
    pub offset: u32,
    pub size: u32,
}

/// Sub-buffer allocator for all in-game geometry.
/// It's simple free-list allocator
pub struct DynamicAllocator<M: Copy> {
    memory: M,
    offset: u32,
    size: u32,
    granularity: u32,
    blocks: Vec<Block>,
}

#[derive(Debug)]
pub enum AllocError {
    NotEnoughMemory,
    WrongBlock,
}

impl<M: Copy> DynamicAllocator<M> {
    pub fn new(memory: M, offset: u32, size: u32, granularity: u32) -> Self {
        let mut blocks = Vec::with_capacity(256);
        blocks.push(Block::Free(BlockData(0, size)));
        Self {
            memory,
            blocks,
            offset,
            size,
            granularity,
        }
    }

    pub fn alloc(&mut self, size: u32) -> Result<MemoryBlock<M>, AllocError> {
        if let Some(index) = self.find_free_block(size) {
            if let Some(offset) = self.split_and_insert_block(index, size) {
                self.debug_dump_memory();
                return Ok(MemoryBlock {
                    memory: self.memory,
                    offset,
                    size,
                });
            }
        }

        Err(AllocError::NotEnoughMemory)
    }

    pub fn free(&mut self, block: MemoryBlock<M>) -> Result<(), AllocError> {
        if let Some(index) = self.find_used_block(block.offset) {
            if let Block::Used(block) = self.blocks[index] {
                self.blocks[index] = Block::Free(block);
                self.merge_free_blocks(index);
                self.debug_dump_memory();
                return Ok(());
            }
        }

        Err(AllocError::WrongBlock)
    }

    fn debug_dump_memory(&self) {
        self.blocks.iter().for_each(|block| {
            println!("{:?}", block);
        })
    }

    fn find_used_block(&self, offset: u32) -> Option<usize> {
        self.blocks.iter().enumerate().find_map(|(index, block)| {
            if let Block::Used(block) = block {
                if block.0 == offset {
                    return Some(index);
                }
            }
            None
        })
    }

    fn find_free_block(&self, size: u32) -> Option<usize> {
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

    fn split_and_insert_block(&mut self, index: usize, size: u32) -> Option<u32> {
        let size = size.max(self.granularity).align(self.granularity);
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

#[cfg(test)]
mod test {
    use super::DynamicAllocator;

    #[test]
    fn alloc() {
        let mut allocator = DynamicAllocator::<()>::new((), 0, 1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        assert_eq!(0, block1.offset);
        assert_eq!(100, block1.size);
        assert_eq!(128, block2.offset);
        assert_eq!(200, block2.size);
    }

    #[test]
    fn free() {
        let mut allocator = DynamicAllocator::<()>::new((), 0, 1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        allocator.free(block1).unwrap();
        allocator.free(block2).unwrap();
        let block = allocator.alloc(300).unwrap();
        assert_eq!(0, block.offset);
        assert_eq!(300, block.size);
    }

    #[test]
    fn allocate_suitable_block() {
        let mut allocator = DynamicAllocator::<()>::new((), 0, 1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        allocator.free(block1).unwrap();
        let block = allocator.alloc(300).unwrap();
        assert_eq!(384, block.offset);
        assert_eq!(300, block.size);
    }

    #[test]
    fn allocate_small_blocks_in_hole_after_big() {
        let mut allocator = DynamicAllocator::<()>::new((), 0, 1024, 64);
        allocator.alloc(100).unwrap();
        let block = allocator.alloc(200).unwrap();
        allocator.alloc(100).unwrap();
        allocator.free(block).unwrap();
        let block1 = allocator.alloc(50).unwrap();
        let block2 = allocator.alloc(50).unwrap();
        assert_eq!(128, block1.offset);
        assert_eq!(50, block1.size);
        assert_eq!(192, block2.offset);
        assert_eq!(50, block2.size);
    }

    #[test]
    fn not_anough_memory() {
        let mut allocator = DynamicAllocator::<()>::new((), 0, 1024, 64);
        allocator.alloc(500).unwrap();
        allocator.alloc(200).unwrap();
        assert!(allocator.alloc(500).is_err());
    }
}
