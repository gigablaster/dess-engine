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

use ash::vk;

// YOLO buffer tbh
pub struct RingBuffer {
    buffer: vk::Buffer,
    size: u64,
    write: AtomicU64,
}

impl RingBuffer {
    pub fn new(buffer: vk::Buffer, size: u64) -> Self {
        Self {
            buffer,
            write: AtomicU64::new(0),
            size,
        }
    }

    pub fn alloc(&self, size: u64, align: u64) -> SubBuffer {
        assert!(size <= self.size as _);
        loop {
            let head = self.write.load(Ordering::Acquire);
            let begin = Self::align(head, align);
            let end = begin + size;
            let begin = if end >= self.size { 0 } else { begin };
            if self
                .write
                .compare_exchange(head, end, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return SubBuffer {
                    raw: self.buffer,
                    offset: begin,
                    size,
                };
            }
        }
    }

    fn align(value: u64, align: u64) -> u64 {
        if value == 0 {
            value
        } else {
            (value & !(align - 1)) + align
        }
    }
}

#[cfg(test)]
mod test {
    use ash::vk;

    use crate::{vulkan::BufferView, RingBuffer};

    #[test]
    fn alloc() {
        let buffer = RingBuffer::new(vk::Buffer::null(), 1024);
        let alloc1 = buffer.alloc(500, 64);
        let alloc2 = buffer.alloc(500, 64);
        let alloc3 = buffer.alloc(128, 64);
        assert_eq!(0, alloc1.offset());
        assert_eq!(512, alloc2.offset());
        assert_eq!(0, alloc3.offset());
    }
}
