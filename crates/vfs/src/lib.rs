mod directory;
mod error;
mod mmap;
mod packed;
mod vfs;

pub use error::*;
use four_cc::FourCC;
use lazy_static::lazy_static;
use uuid::Uuid;

use std::{collections::HashSet, fmt::Debug, io::Read, path::Path, sync::Mutex};

use crate::vfs::Vfs;

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

pub trait DataTypeId {
    fn type_id(&self) -> FourCC;
}

pub trait DataReader: Read + DataTypeId {}

pub trait Archive: Send + Sync + Debug {
    fn load(&self, uuid: Uuid) -> Result<Box<dyn DataReader>, VfsError>;
    fn named(&self, name: &str) -> Option<Uuid>;
    fn tagged(&self, tag: &str) -> Option<&HashSet<Uuid>>;
}

pub fn scan(root: &Path) -> Result<(), VfsError> {
    let mut vfs = VFS.lock().unwrap();
    vfs.scan(root)
}

pub fn named(name: &str) -> Result<Box<dyn DataReader>, VfsError> {
    VFS.lock().unwrap().named(name)
}

pub fn get(uuid: Uuid) -> Result<Box<dyn DataReader>, VfsError> {
    VFS.lock().unwrap().get(uuid)
}
