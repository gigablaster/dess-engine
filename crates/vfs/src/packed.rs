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

use std::{fmt::Debug, io::Read, path::Path, sync::Arc};

use memmap2::Mmap;

use crate::{
    directory::{load_archive_directory, Directory, FileSize},
    mmap::{map_file, MappedFileReader},
    Archive, Loader, VfsError,
};

#[derive(Debug)]
pub struct PackedArchive {
    file: Arc<Mmap>,
    directory: Directory,
}

impl PackedArchive {
    pub fn open(path: &Path) -> Result<Self, VfsError> {
        let file = map_file(path)?;
        let directory = load_archive_directory(&mut MappedFileReader::new(&file, 0, file.len()))?;

        Ok(Self { file, directory })
    }
}

#[derive(Debug)]
pub struct PackedFile {
    size: usize,
    decoder: lz4_flex::frame::FrameDecoder<MappedFileReader>,
}

impl PackedFile {
    pub fn new(file: MappedFileReader, size: usize) -> Self {
        Self {
            size,
            decoder: lz4_flex::frame::FrameDecoder::new(file),
        }
    }
}

impl Read for PackedFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.decoder.read(buf)
    }
}

impl Loader for PackedFile {
    fn size(&self) -> usize {
        self.size
    }
}

impl Archive for PackedArchive {
    fn open(&self, name: &Path) -> Result<Box<dyn Loader>, VfsError> {
        if let Some(header) = self.directory.get(name.to_str().unwrap()) {
            match header.size {
                FileSize::Raw(size) => Ok(Box::new(MappedFileReader::new(
                    &self.file,
                    header.offset as _,
                    size as _,
                ))),
                FileSize::Compressed(unpacked, packed) => Ok(Box::new(PackedFile::new(
                    MappedFileReader::new(&self.file, header.offset as _, packed as _),
                    unpacked as _,
                ))),
            }
        } else {
            Err(VfsError::NotFound(name.into()))
        }
    }
}
