use std::{path::PathBuf, fs::File, io::Read};

use crate::{Archive, VfsError};

pub struct RawFsArchive {
    root: PathBuf
}

impl RawFsArchive {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Archive for RawFsArchive {
    fn load(&self, name: &str) -> Result<Box<dyn Read>, VfsError> {
        let path = self.root.join::<PathBuf>(name.into());
        let file = File::open(path)?;

        Ok(Box::new(file))
    }
}
