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

use std::sync::atomic::{AtomicU64, Ordering};

use crate::Align;

#[derive(Debug, Clone, Copy)]
struct BlockData(u64, u64);

#[derive(Debug, Clone, Copy)]
enum Block {
    Free(BlockData),
    Used(BlockData),
}

/// Sub-buffer allocator for all in-game geometry.
/// It's simple free-list allocator
#[derive(Debug)]
pub struct DynamicAllocator {
    size: u64,
    granularity: u64,
    blocks: Vec<Block>,
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

    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        if let Some(index) = self.find_free_block(size) {
            self.split_and_insert_block(index, size)
        } else {
            None
        }
    }

    pub fn free(&mut self, offset: u64) {
        if let Some(index) = self.find_used_block(offset) {
            if let Block::Used(block) = self.blocks[index] {
                self.blocks[index] = Block::Free(block);
                self.merge_free_blocks(index);
                return;
            }
        }

        panic!("Attempt to free already freed block or block from different allocator");
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

pub struct RingAllocator {
    size: u64,
    aligment: u64,
    head: AtomicU64,
}

impl RingAllocator {
    pub fn new(size: u64, aligment: u64) -> Self {
        Self {
            head: AtomicU64::new(0),
            size,
            aligment,
        }
    }

    pub fn allocate(&self, size: u64) -> u64 {
        assert!(size <= self.size);
        let aligned_size = align(size, self.aligment);
        loop {
            let old_head = self.head.load(Ordering::Acquire);
            let end = old_head + aligned_size;
            let (head, new_head) = if end > self.size {
                (0, aligned_size)
            } else {
                (old_head, end)
            };
            if self
                .head
                .compare_exchange(old_head, new_head, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return head;
            }
        }
    }
}

pub struct BumpAllocator {
    size: u64,
    top: u64,
    aligment: u64,
}

impl BumpAllocator {
    pub fn new(size: u64, aligment: u64) -> Self {
        Self {
            size,
            aligment,
            top: 0,
        }
    }

    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        let base = self.top.align(self.aligment);
        if base + size > self.size {
            None
        } else {
            self.top = base + size;
            Some(base)
        }
    }

    pub fn reset(&mut self) {
        self.top = 0;
    }
}

#[cfg(test)]
mod test {
    use crate::memory::RingAllocator;

    use super::{BumpAllocator, DynamicAllocator};

    #[test]
    fn alloc() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        assert_eq!(0, block1);
        assert_eq!(128, block2);
    }

    #[test]
    fn free() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let block2 = allocator.alloc(200).unwrap();
        allocator.free(block1);
        allocator.free(block2);
        let block = allocator.alloc(300).unwrap();
        assert_eq!(0, block);
    }

    #[test]
    fn allocate_suitable_block() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        let block1 = allocator.alloc(100).unwrap();
        let _block2 = allocator.alloc(200).unwrap();
        allocator.free(block1);
        let block = allocator.alloc(300).unwrap();
        assert_eq!(384, block);
    }

    #[test]
    fn allocate_small_blocks_in_hole_after_big() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        allocator.alloc(100).unwrap();
        let block = allocator.alloc(200).unwrap();
        allocator.alloc(100).unwrap();
        allocator.free(block);
        let block1 = allocator.alloc(50).unwrap();
        let block2 = allocator.alloc(50).unwrap();
        assert_eq!(128, block1);
        assert_eq!(192, block2);
    }

    #[test]
    fn not_anough_memory() {
        let mut allocator = DynamicAllocator::new(1024, 64);
        allocator.alloc(500).unwrap();
        allocator.alloc(200).unwrap();
        assert!(allocator.alloc(500).is_none());
    }

    #[test]
    fn ring_allocator() {
        let buffer = RingAllocator::new(1024, 64);
        assert_eq!(0, buffer.allocate(500));
        assert_eq!(512, buffer.allocate(500));
        assert_eq!(0, buffer.allocate(128));
    }

    #[test]
    fn bump_allocator() {
        let mut allocator = BumpAllocator::new(1024, 64);
        assert_eq!(Some(0), allocator.allocate(500));
        assert_eq!(Some(512), allocator.allocate(100));
        assert_eq!(Some(640), allocator.allocate(100));
        assert_eq!(None, allocator.allocate(500));
        allocator.reset();
        assert_eq!(Some(0), allocator.allocate(500));
        assert_eq!(Some(512), allocator.allocate(100));
        assert_eq!(Some(640), allocator.allocate(100));
        assert_eq!(None, allocator.allocate(500));
    }
}