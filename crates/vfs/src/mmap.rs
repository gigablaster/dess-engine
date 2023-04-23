use std::{fs::File, io, path::Path};

use memmap2::{Mmap, MmapOptions};

#[derive(Debug)]
pub struct MappedFile {
    file: File,
    mmap: Mmap,
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self { file, mmap })
    }
}
