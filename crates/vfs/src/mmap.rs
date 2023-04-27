use std::{fs::File, io, path::Path};

use memmap2::{Mmap, MmapOptions};

#[derive(Debug)]
pub struct MappedFile {
    mmap: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self { mmap })
    }

    pub fn data(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}
