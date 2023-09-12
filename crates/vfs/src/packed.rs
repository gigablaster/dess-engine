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

use std::{fmt::Debug, path::Path, sync::Arc};

use memmap2::Mmap;

use crate::{
    directory::{load_archive_directory, Directory},
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

impl Archive for PackedArchive {
    fn open(&self, name: &str) -> Result<Box<dyn Loader>, VfsError> {
        if let Some(header) = self.directory.get(name) {
            Ok(Box::new(MappedFileReader::new(
                &self.file,
                header.offset as _,
                header.size as _,
            )))
        } else {
            Err(VfsError::NotFound(name.into()))
        }
    }
}
