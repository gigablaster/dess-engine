use crate::BufferAllocator;

#[derive(Debug, Clone, Copy)]
struct BlockData(u32, u32);

#[derive(Debug, Clone, Copy)]
enum Block {
    Free(BlockData),
    Used(BlockData),
}

#[derive(Debug)]
pub struct DynamicBufferAllocator {
    granularity: u32,
    blocks: Vec<Block>,
}

fn align(value: u32, align: u32) -> u32 {
    if value == 0 || value % align == 0 {
        value
    } else {
        (value & !(align - 1)) + align
    }
}

impl DynamicBufferAllocator {
    pub fn new(size: u32, align: u32) -> Self {
        let mut blocks = Vec::with_capacity(256);
        blocks.push(Block::Free(BlockData(0, size)));
        Self {
            blocks,
            granularity: align,
        }
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

    fn find_first_free_block(&self, size: u32) -> Option<usize> {
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
                    match block {
                        Block::Free(_) => {
                            index -= 1;
                            continue;
                        }
                        Block::Used(_) => {}
                    }
                }
            }
            break;
        }
        loop {
            if let Block::Free(block) = self.blocks[index] {
                if let Some(next) = self.blocks.get(index + 1) {
                    match next {
                        Block::Free(next) => {
                            self.blocks[index] = Block::Free(BlockData(block.0, block.1 + next.1));
                            self.blocks.remove(index + 1);
                            continue;
                        }
                        Block::Used(_) => {}
                    }
                }
            }
            break;
        }
    }

    fn split_and_insert_block(&mut self, index: usize, size: u32) -> Option<u32> {
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

impl BufferAllocator for DynamicBufferAllocator {
    fn allocate(&mut self, size: u32) -> Option<u32> {
        if let Some(index) = self.find_first_free_block(size) {
            self.split_and_insert_block(index, size)
        } else {
            None
        }
    }

    fn deallocate(&mut self, offset: u32) {
        if let Some(index) = self.find_used_block(offset) {
            if let Block::Used(block) = self.blocks[index] {
                self.blocks[index] = Block::Free(block);
                self.merge_free_blocks(index);
                return;
            }
        }

        panic!("Attempt to free already freed block or block from different allocator");
    }
}

#[derive(Debug)]
pub struct BlockBufferAllocator {
    chunk_size: u32,
    chunk_count: u32,
    empty: Vec<u32>,
}

impl BlockBufferAllocator {
    pub fn new(chunk_size: u32, chunk_count: u32) -> Self {
        let empty = (0..chunk_count).rev().collect::<Vec<_>>();

        Self {
            chunk_size,
            chunk_count,
            empty,
        }
    }
}

impl BufferAllocator for BlockBufferAllocator {
    fn allocate(&mut self, size: u32) -> Option<u32> {
        assert!(size <= self.chunk_size);
        if let Some(slot) = self.empty.pop() {
            Some((slot as u32) * self.chunk_size)
        } else {
            None
        }
    }

    fn deallocate(&mut self, offset: u32) {
        let index = (offset / self.chunk_size) as u32;
        assert!(index < self.chunk_count && offset % self.chunk_size == 0);
        assert!(!self.empty.contains(&index));
        self.empty.push(index);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        memory::{BlockBufferAllocator, DynamicBufferAllocator},
        BufferAllocator,
    };

    #[test]
    fn alloc() {
        let mut allocator = DynamicBufferAllocator::new(1024, 64);
        let block1 = allocator.allocate(100).unwrap();
        let block2 = allocator.allocate(200).unwrap();
        assert_eq!(0, block1);
        assert_eq!(128, block2);
    }

    #[test]
    fn free() {
        let mut allocator = DynamicBufferAllocator::new(1024, 64);
        let block1 = allocator.allocate(100).unwrap();
        let block2 = allocator.allocate(200).unwrap();
        allocator.deallocate(block1);
        allocator.deallocate(block2);
        let block = allocator.allocate(300).unwrap();
        assert_eq!(0, block);
    }

    #[test]
    fn allocate_suitable_block() {
        let mut allocator = DynamicBufferAllocator::new(1024, 64);
        let block1 = allocator.allocate(100).unwrap();
        let _block2 = allocator.allocate(200).unwrap();
        allocator.deallocate(block1);
        let block = allocator.allocate(300).unwrap();
        assert_eq!(384, block);
    }

    #[test]
    fn allocate_small_blocks_in_hole_after_big() {
        let mut allocator = DynamicBufferAllocator::new(1024, 64);
        allocator.allocate(100).unwrap();
        let block = allocator.allocate(200).unwrap();
        allocator.allocate(100).unwrap();
        allocator.deallocate(block);
        let block1 = allocator.allocate(50).unwrap();
        let block2 = allocator.allocate(50).unwrap();
        assert_eq!(128, block1);
        assert_eq!(192, block2);
    }

    #[test]
    fn not_anough_memory() {
        let mut allocator = DynamicBufferAllocator::new(1024, 64);
        allocator.allocate(500).unwrap();
        allocator.allocate(200).unwrap();
        assert!(allocator.allocate(500).is_none());
    }

    #[test]
    fn block_allocator() {
        let mut allocator = BlockBufferAllocator::new(10, 5);
        assert_eq!(Some(0), allocator.allocate(10));
        assert_eq!(Some(10), allocator.allocate(10));
        assert_eq!(Some(20), allocator.allocate(10));
        assert_eq!(Some(30), allocator.allocate(10));
        assert_eq!(Some(40), allocator.allocate(10));
        assert_eq!(None, allocator.allocate(10));
        allocator.deallocate(20);
        assert_eq!(Some(20), allocator.allocate(10));
        assert_eq!(None, allocator.allocate(10));
    }
}
