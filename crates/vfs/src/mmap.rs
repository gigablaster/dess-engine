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
    cmp::min,
    fs::File,
    io::{self, Error, Read, Seek},
    path::Path,
    ptr::copy_nonoverlapping,
    sync::Arc,
};

use memmap2::{Mmap, MmapOptions};

use crate::Loader;

pub fn map_file(path: &Path) -> io::Result<Arc<Mmap>> {
    let mmap = unsafe { MmapOptions::new().map(&File::open(path)?) }?;

    Ok(Arc::new(mmap))
}

#[derive(Debug)]
pub struct MappedFileReader {
    mmap: Arc<Mmap>,
    from: usize,
    size: usize,
    cursor: usize,
}

impl MappedFileReader {
    pub fn new(mmap: &Arc<Mmap>, from: usize, size: usize) -> Self {
        let from = min(from, mmap.len());
        let last = min(from + size, mmap.len());
        let size = last - from;
        Self {
            mmap: mmap.clone(),
            from,
            size,
            cursor: 0,
        }
    }
}

impl Read for MappedFileReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let to_read = min(buf.len(), self.size - self.cursor);
        if to_read == 0 {
            return Ok(0);
        }
        unsafe {
            copy_nonoverlapping(
                self.mmap.as_ptr().add(self.cursor),
                buf.as_mut_ptr(),
                to_read,
            )
        };
        self.cursor += to_read;

        Ok(to_read)
    }
}

impl Loader for MappedFileReader {
    fn size(&self) -> usize {
        self.mmap.len()
    }
}

impl Seek for MappedFileReader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        todo!()
    }
}
