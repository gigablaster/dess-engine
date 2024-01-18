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

use std::{
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use dess_backend::{Buffer, BufferCreateDesc, Device};
use dess_common::BlockAllocator;

use crate::Error;

const BUCKET_SIZE: usize = 0xFFFF;
const MIN_UNIFORM_SIZE: usize = 0x100;
const MAX_UNIFORM_SIZE: usize = 0x4000;
const UNIFORM_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const BUCKET_COUNT: usize = UNIFORM_BUFFER_SIZE / BUCKET_SIZE;
const SIZE_RANGES: [(usize, usize); 7] = [
    (8192, 16384),
    (4096, 8192),
    (2048, 4096),
    (1024, 2048),
    (512, 1024),
    (256, 512),
    (0, 256),
];

#[derive(Debug, Default)]
struct Bucket {
    pub from: usize,
    pub to: usize,
    pub allocated: usize,
    pub free: usize,
    pub allocator: Option<BlockAllocator>,
}

impl Bucket {
    pub fn alloc(&mut self) -> Option<usize> {
        if let Some(allocator) = &mut self.allocator {
            self.allocated += 1;
            self.free -= 1;
            allocator.allocate()
        } else {
            panic!("Allocator isn't initialized for this bucket")
        }
    }

    pub fn dealloc(&mut self, offset: usize) -> bool {
        if let Some(allocator) = &mut self.allocator {
            self.allocated -= 1;
            self.free += 1;
            allocator.dealloc(offset);
            self.allocated > 0
        } else {
            panic!("Allocator isn't initialized for this bucket")
        }
    }

    pub fn init(&mut self, from: usize, to: usize) {
        assert!(to >= MIN_UNIFORM_SIZE);
        assert!(from < MAX_UNIFORM_SIZE);
        assert!(to <= MAX_UNIFORM_SIZE);
        assert!(to > from);
        self.from = from;
        self.to = to;
        self.allocated = 0;
        self.free = BUCKET_SIZE / to;
        self.allocator = Some(BlockAllocator::new(to, UNIFORM_BUFFER_SIZE / to));
    }

    pub fn is_suitable(&self, size: usize) -> bool {
        self.allocator.is_some() && self.free > 0 && size > self.from && size <= self.to
    }

    pub fn release(&mut self) {
        assert_eq!(0, self.allocated);
        self.allocator = None;
        self.free = 0;
        self.allocated = 0;
        self.to = 0;
        self.from = 0;
    }
}

pub struct UniformPool {
    buffer: Buffer,
    mapping: NonNull<u8>,
    buckets: ArrayVec<Bucket, BUCKET_COUNT>,
    free_buckets: ArrayVec<usize, BUCKET_COUNT>,
}

unsafe impl Send for UniformPool {}
unsafe impl Sync for UniformPool {}

impl UniformPool {
    pub fn new(device: &Arc<Device>) -> Result<Self, Error> {
        let mut buffer = Buffer::new(
            device,
            BufferCreateDesc::shared(UNIFORM_BUFFER_SIZE)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .dedicated(true)
                .name("Uniform bufer"),
        )?;
        let mapping = buffer.map()?;
        Ok(Self {
            buffer,
            mapping,
            buckets: ArrayVec::default(),
            free_buckets: ArrayVec::default(),
        })
    }

    pub fn push_bytes(&mut self, data: &[u8]) -> Result<usize, Error> {
        unsafe { self.push_raw(data.as_ptr(), data.len()) }
    }

    pub fn get(&self) -> &Buffer {
        &self.buffer
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn push_raw(&mut self, data: *const u8, size: usize) -> Result<usize, Error> {
        let mut index = self.find_bucket_index(size);
        if index.is_none() {
            index = self.allocate_bucket(size);
        }
        let index = index.ok_or(Error::OutOfSpace)?;
        let base_offset = index * BUCKET_SIZE;
        let offset = self.buckets[index].alloc().ok_or(Error::OutOfSpace)?;
        let offset = base_offset + offset;
        unsafe { copy_nonoverlapping(data, self.mapping.as_ptr().add(offset), size) }

        Ok(offset)
    }

    pub fn dealloc(&mut self, offset: usize) {
        let index = offset / BUCKET_SIZE;
        let local_offset = offset - index * BUCKET_SIZE;
        let bucket = &mut self.buckets[index];
        if !bucket.dealloc(local_offset) {
            bucket.release();
            self.free_buckets.push(index);
        }
    }

    fn find_bucket_index(&self, size: usize) -> Option<usize> {
        self.buckets.iter().position(|x| x.is_suitable(size))
    }

    fn allocate_bucket(&mut self, size: usize) -> Option<usize> {
        let (from, to) = Self::find_size_range(size);
        if let Some(free) = self.free_buckets.pop() {
            self.buckets[free].init(from, to);
            Some(free)
        } else if self.buckets.len() == BUCKET_COUNT {
            None
        } else {
            let index = self.buckets.len();
            let mut bucket = Bucket::default();
            bucket.init(from, to);
            self.buckets.push(bucket);
            Some(index)
        }
    }

    fn find_size_range(size: usize) -> (usize, usize) {
        SIZE_RANGES
            .iter()
            .find(|(min, max)| size > *min && size <= *max)
            .copied()
            .unwrap()
    }
}
