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

mod directory;
mod error;
mod mmap;
mod packed;
mod raw_fs;
mod vfs;

pub use error::*;
use lazy_static::lazy_static;

use std::{path::PathBuf, sync::Mutex};

use crate::vfs::Vfs;

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

pub trait Content {
    fn data(&self) -> &[u8];
}

pub trait Archive: Send + Sync {
    fn load(&self, name: &str) -> Result<Box<dyn Content>, VfsError>;
}

pub fn scan(root: impl Into<PathBuf>) -> Result<(), VfsError> {
    VFS.lock().unwrap().scan(root)
}

pub fn get(name: &str) -> Result<Box<dyn Content>, VfsError> {
    VFS.lock().unwrap().get(name)
}
