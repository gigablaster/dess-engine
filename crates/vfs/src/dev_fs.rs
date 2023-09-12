use std::path::Path;

use crate::{raw_fs::RawFsArchive, Archive};

pub struct DevFsArchive {
    cache: RawFsArchive,
    data: RawFsArchive,
}

impl DevFsArchive {
    pub fn new(root: &Path) -> Self {
        Self {
            cache: RawFsArchive::new(root.join("cache")),
            data: RawFsArchive::new(root.join("data")),
        }
    }
}

impl Archive for DevFsArchive {
    fn open(&self, name: &Path) -> Result<Box<dyn crate::Loader>, crate::VfsError> {
        self.cache.open(name).or_else(|_| self.data.open(name))
    }
}
