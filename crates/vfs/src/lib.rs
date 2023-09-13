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

mod dev_fs;
mod directory;
mod error;
mod mmap;
mod packed;
mod raw_fs;
mod vfs;

pub use directory::DirectoryBaker;
pub use error::*;
pub use vfs::*;

use std::io::{self, Read, Write};

pub trait Loader: Read {
    fn size(&self) -> usize;
    fn load(&mut self) -> io::Result<Vec<u8>> {
        let mut data = vec![0; self.size()];
        self.read_exact(&mut data)?;

        Ok(data)
    }
}

pub enum AssetPath<'a> {
    Content(&'a str),
    Cache(&'a str),
    Save(&'a str),
}

impl<'a> ToString for AssetPath<'a> {
    fn to_string(&self) -> String {
        match self {
            AssetPath::Cache(name) => name.to_string(),
            AssetPath::Content(name) => name.to_string(),
            AssetPath::Save(name) => name.to_string(),
        }
    }
}

pub trait Archive: Send + Sync {
    fn open(&self, name: &str) -> Result<Box<dyn Loader>, VfsError>;
    fn create(&self, name: &str) -> Result<Box<dyn Write>, VfsError>;
}
