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
    mem::size_of,
    ptr::{copy_nonoverlapping, NonNull},
    slice,
    sync::Arc,
};

use ash::vk;
use dess_common::memory::BlockAllocator;

use crate::{
    error::{UniformAllocateError, UniformCreateError, UniformSyncError},
    vulkan::{Buffer, BufferDesc, CommandBuffer, CommandPool, Device, Semaphore, SubmitWait},
    GpuResource,
};

const BUCKET_SIZE: u32 = 0xFFFF;
const MIN_UNIFORM_SIZE: u32 = 0x100;
const MAX_UNIFORM_SIZE: u32 = 0x4000;
const UNIFORM_BUFFER_SIZE: u32 = 16 * 1024 * 1024;
const BUCKET_COUNT: u32 = UNIFORM_BUFFER_SIZE / BUCKET_SIZE;
const SIZE_RANGES: [(u32, u32); 7] = [
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
    pub from: u32,
    pub to: u32,
    pub allocated: u32,
    pub free: u32,
    pub allocator: Option<BlockAllocator>,
}

impl Bucket {
    pub fn alloc(&mut self) -> Option<u32> {
        if let Some(allocator) = &mut self.allocator {
            self.allocated += 1;
            self.free -= 1;
            allocator.allocate()
        } else {
            panic!("Allocator isn't initialized for this bucket")
        }
    }

    pub fn dealloc(&mut self, offset: u32) -> bool {
        if let Some(allocator) = &mut self.allocator {
            self.allocated -= 1;
            self.free += 1;
            allocator.dealloc(offset);
            self.allocated > 0
        } else {
            panic!("Allocator isn't initialized for this bucket")
        }
    }

    pub fn init(&mut self, from: u32, to: u32) {
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

    pub fn is_suitable(&self, size: u32) -> bool {
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

pub struct Uniforms {
    device: Arc<Device>,
    buffer: Arc<Buffer>,
    staging: Arc<Buffer>,
    pool: CommandPool,
    cb: CommandBuffer,
    semaphore: Semaphore,
    writes: Vec<vk::BufferCopy>,
    mapping: NonNull<u8>,
    buckets: Vec<Bucket>,
    free_buckets: Vec<usize>,
}

impl Uniforms {
    pub fn new(device: &Arc<Device>) -> Result<Self, UniformCreateError> {
        let buffer = Buffer::new(
            device,
            BufferDesc::gpu_only(
                UNIFORM_BUFFER_SIZE as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .dedicated(true),
        )?;
        let mut staging = Buffer::new(
            device,
            BufferDesc::host_only(UNIFORM_BUFFER_SIZE as _, vk::BufferUsageFlags::TRANSFER_SRC),
        )?;
        buffer.name("Unified uniform buffer");
        staging.name("Uniform buffer staging");
        let mapping = Arc::get_mut(&mut staging).unwrap().map()?;
        let mut pool = CommandPool::new(
            device.raw(),
            device.queue_index(),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .unwrap();
        let cb = pool.get_or_create(device.raw()).unwrap();
        Ok(Self {
            device: device.clone(),
            buffer,
            staging,
            writes: Vec::with_capacity(1024),
            cb,
            pool,
            mapping,
            buckets: Vec::new(),
            free_buckets: Vec::new(),
            semaphore: Semaphore::new(device.raw())?,
        })
    }

    pub fn push<T: Sized>(&mut self, data: &T) -> Result<u32, UniformAllocateError> {
        let size = size_of::<T>() as u32;
        let mut index = self.find_bucket_index(size);
        if index.is_none() {
            index = self.allocate_bucket(size);
        }
        if let Some(index) = index {
            let base_offset = index as u32 * BUCKET_SIZE;
            if let Some(offset) = self.buckets[index].alloc() {
                let offset = base_offset + offset;
                unsafe {
                    copy_nonoverlapping(
                        slice::from_ref(data).as_ptr() as *const u8,
                        self.mapping.as_ptr().add(offset as usize),
                        size as usize,
                    )
                }
                self.writes.push(
                    vk::BufferCopy::builder()
                        .src_offset(offset as _)
                        .dst_offset(offset as _)
                        .size(size as _)
                        .build(),
                );

                return Ok(offset);
            }
        }
        Err(UniformAllocateError::OutOfSpace)
    }

    pub fn dealloc(&mut self, offset: u32) {
        let index = offset / BUCKET_SIZE;
        let local_offset = offset - index * BUCKET_SIZE;
        let bucket = &mut self.buckets[index as usize];
        if !bucket.dealloc(local_offset) {
            bucket.release();
            self.free_buckets.push(index as usize);
        }
    }

    pub fn raw(&self) -> vk::Buffer {
        self.buffer.raw()
    }

    pub fn flush(&mut self) -> Result<Option<SubmitWait>, UniformSyncError> {
        self.cb.wait(self.device.raw())?;
        self.cb.reset(self.device.raw())?;
        if self.writes.is_empty() {
            return Ok(None);
        }
        self.cb.record(self.device.raw(), |recorder| {
            unsafe {
                let _label = self.device.scoped_label(*recorder.cb, "Submiting uniforms");
                recorder.device.cmd_copy_buffer(
                    *recorder.cb,
                    self.staging.raw(),
                    self.buffer.raw(),
                    &self.writes,
                )
            };
        })?;
        self.device.submit(&self.cb, &[], &[self.semaphore])?;
        self.writes.clear();

        Ok(Some(SubmitWait::Transfer(self.semaphore)))
    }

    fn find_bucket_index(&self, size: u32) -> Option<usize> {
        self.buckets.iter().position(|x| x.is_suitable(size))
    }

    fn allocate_bucket(&mut self, size: u32) -> Option<usize> {
        let (from, to) = Self::find_size_range(size);
        if let Some(free) = self.free_buckets.pop() {
            self.buckets[free].init(from, to);
            Some(free)
        } else if self.buckets.len() == BUCKET_COUNT as usize {
            None
        } else {
            let index = self.buckets.len();
            let mut bucket = Bucket::default();
            bucket.init(from, to);
            self.buckets.push(bucket);
            Some(index)
        }
    }

    fn find_size_range(size: u32) -> (u32, u32) {
        SIZE_RANGES
            .iter()
            .find(|(min, max)| size > *min && size <= *max)
            .copied()
            .unwrap()
    }
}

impl Drop for Uniforms {
    fn drop(&mut self) {
        self.cb.free(self.device.raw());
        self.pool.free(self.device.raw());
    }
}
