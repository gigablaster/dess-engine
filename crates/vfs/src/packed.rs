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
    fmt::Debug,
    fs::File,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
};

use zstd::stream;

use crate::{
    directory::{load_archive_directory, Compression, Directory},
    mmap::{MappedFile, MappedFileSlice, MappedFileSliceReader},
    Archive, VfsError,
};

type MmapReader = BufReader<MappedFileSlice>;

struct ZstdDataReader<'a> {
    decoder: stream::Decoder<'a, MmapReader>,
}

impl<'a> ZstdDataReader<'a> {
    fn new(file: MappedFileSlice) -> Result<Self, VfsError> {
        Ok(Self {
            decoder: stream::Decoder::new(file).unwrap(),
        })
    }
}

impl<'a> Read for ZstdDataReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[derive(Debug)]
pub struct PackedArchive {
    file: Arc<MappedFile>,
    directory: Directory,
}

impl PackedArchive {
    pub fn open(path: &Path) -> Result<Self, VfsError> {
        let mut file = File::open(path)?;
        let directory = load_archive_directory(&mut file)?;
        let file = Arc::new(MappedFile::open(path)?);

        Ok(Self { file, directory })
    }
}

impl Archive for PackedArchive {
    fn load(&self, name: &str) -> Result<Box<dyn Read>, VfsError> {
        if let Some(header) = self.directory.get(name) {
            match header.compression {
                Compression::None(size) => Ok(Box::new(MappedFileSliceReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, size as _),
                ))),
                Compression::Zstd(_, compressed) => Ok(Box::new(ZstdDataReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, compressed as _),
                )?)),
            }
        } else {
            Err(VfsError::NotFound(name.into()))
        }
    }
}
