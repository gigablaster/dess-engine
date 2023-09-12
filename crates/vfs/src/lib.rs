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
use lazy_static::lazy_static;

use std::{
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::vfs::Vfs;

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

pub trait Loader: Read {
    fn size(&self) -> usize;
    fn load(&mut self) -> io::Result<Vec<u8>> {
        let mut data = vec![0; self.size()];
        self.read_exact(&mut data)?;

        Ok(data)
    }
}

pub trait Archive: Send + Sync {
    fn open(&self, name: &Path) -> Result<Box<dyn Loader>, VfsError>;
}

pub fn scan(root: impl Into<PathBuf>) -> Result<(), VfsError> {
    VFS.lock().unwrap().scan(root)
}

pub fn get(name: impl Into<PathBuf>) -> Result<Box<dyn Loader>, VfsError> {
    VFS.lock().unwrap().get(&name.into())
}
