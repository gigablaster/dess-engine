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

use std::{fs::File, io, path::Path, sync::Arc};

use memmap2::{Mmap, MmapOptions};

use crate::Content;

#[derive(Debug)]
pub(crate) struct MappedFile {
    mmap: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self { mmap })
    }
}

impl Content for MappedFile {
    fn data(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

#[derive(Debug)]
pub(crate) struct MappedFileSlice {
    file: Arc<MappedFile>,
    from: usize,
    to: usize,
}

impl MappedFileSlice {
    pub fn new(file: &Arc<MappedFile>, from: usize, size: usize) -> Self {
        Self {
            file: file.clone(),
            from,
            to: from + size,
        }
    }
}

impl Content for MappedFileSlice {
    fn data(&self) -> &[u8] {
        &self.file.data()[self.from..self.to]
    }
}
