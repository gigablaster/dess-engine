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

use std::path::{Path, PathBuf};

use crate::{
    mmap::{map_file, MappedFileReader},
    Archive, Loader, VfsError,
};

pub struct RawFsArchive {
    root: PathBuf,
}

impl RawFsArchive {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Archive for RawFsArchive {
    fn open(&self, name: &Path) -> Result<Box<dyn Loader>, VfsError> {
        let path = self.root.join(name);
        let file = map_file(&path)?;

        Ok(Box::new(MappedFileReader::new(&file, 0, file.len())))
    }
}
