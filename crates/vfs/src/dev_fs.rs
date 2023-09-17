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

use std::path::Path;

use crate::{raw_fs::RawFsArchive, Archive};

pub struct DevFsArchive {
    cache: RawFsArchive,
    data: RawFsArchive,
}

impl DevFsArchive {
    pub fn new(root: &Path) -> Self {
        Self {
            cache: RawFsArchive::new(root.join(".cache")),
            data: RawFsArchive::new(root.join("data")),
        }
    }
}

impl Archive for DevFsArchive {
    fn open(&self, name: &str) -> Result<Box<dyn crate::Loader>, crate::VfsError> {
        // Open first from cache, second from data
        self.cache.open(name).or_else(|_| self.data.open(name))
    }

    fn create(&self, name: &str) -> Result<Box<dyn std::io::Write>, crate::VfsError> {
        // Create new files only in cache
        self.cache.create(name)
    }
}
