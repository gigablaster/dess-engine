mod directory;
mod error;
mod mmap;
mod packed;
mod vfs;
mod raw_fs;

pub use error::*;
use lazy_static::lazy_static;

use std::{io::Read, path::Path, sync::Mutex};

use crate::vfs::Vfs;

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

pub trait Archive: Send + Sync {
    fn load(&self, name: &str) -> Result<Box<dyn Read>, VfsError>;
}

pub fn scan(root: &Path) -> Result<(), VfsError> {
    VFS.lock().unwrap().scan(root)
}

pub fn get(name: &str) -> Result<Box<dyn Read>, VfsError> {
    VFS.lock().unwrap().get(name)
}
