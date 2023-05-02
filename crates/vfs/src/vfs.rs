use std::{fs, io::Read, path::{Path, PathBuf}};

use log::{error, info};

use crate::{packed::PackedArchive, Archive, VfsError, raw_fs::RawFsArchive};

#[derive(Default)]
pub struct Vfs {
    archives: Vec<Box<dyn Archive>>,
}

impl Vfs {
    pub fn scan(&mut self, root: impl Into<PathBuf>) -> Result<(), VfsError> {
        let root = root.into();
        let raw_data_path = root.join("data");
        if raw_data_path.is_dir() {
            self.archives.push(Box::new(RawFsArchive::new(raw_data_path)));
        }
        let paths = fs::read_dir(root)?;
        for path in paths {
            let path = path?.path();
            if path.is_file() && path.ends_with(".dess") {
                info!("Scanning {:?}", path);
                if let Err(err) = self.add_archive(&path) {
                    error!("Failed to add archive {:?} - {:?}", path, err);
                }
            }
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Box<dyn Read>, VfsError> {
        for archive in &self.archives {
            match archive.load(name) {
                Ok(r) => return Ok(r),
                Err(VfsError::NotFound(_)) => {}
                Err(err) => return Err(err),
            }
        }

        Err(VfsError::NotFound(name.into()))
    }

    fn add_archive(&mut self, path: &Path) -> Result<(), VfsError> {
        let archive = Box::new(PackedArchive::open(path)?);
        self.archives.push(archive);

        Ok(())
    }
}
