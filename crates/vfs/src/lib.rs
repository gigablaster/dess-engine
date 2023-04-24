mod archive;
mod directory;
mod error;
mod mmap;
mod vfs;

use archive::LoadedData;
pub use error::*;
use lazy_static::lazy_static;
use uuid::Uuid;

use std::{path::Path, sync::Mutex};

use crate::vfs::Vfs;

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

pub fn scan(root: &Path) -> Result<(), VfsError> {
    let mut vfs = VFS.lock().unwrap();
    vfs.scan(root)
}

pub fn named(name: &str) -> Result<Box<dyn LoadedData>, VfsError> {
    VFS.lock().unwrap().named(name)
}

pub fn get(uuid: Uuid) -> Result<Box<dyn LoadedData>, VfsError> {
    VFS.lock().unwrap().get(uuid)
}
