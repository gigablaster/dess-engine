use std::{
    fs::{self},
    path::Path,
};

use log::{error, info};

use uuid::Uuid;

use crate::{
    archive::{Archive, LoadedData, PackedArchive},
    VfsError,
};

#[derive(Debug, Default)]
pub struct Vfs {
    archives: Vec<Box<dyn Archive>>,
}

impl Vfs {
    pub fn scan(&mut self, root: &Path) -> Result<(), VfsError> {
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

    pub fn named(&self, name: &str) -> Result<Box<dyn LoadedData>, VfsError> {
        for archive in &self.archives {
            if let Some(uuid) = archive.named(name) {
                return archive.load(uuid);
            }
        }

        Err(VfsError::NameNotFound(name.into()))
    }

    pub fn get(&self, uuid: Uuid) -> Result<Box<dyn LoadedData>, VfsError> {
        for archive in &self.archives {
            let res = archive.load(uuid);
            match res {
                Ok(data) => return Ok(data),
                Err(VfsError::AssetNotFound(_)) => {}
                Err(err) => return Err(err),
            }
        }

        Err(VfsError::AssetNotFound(uuid))
    }

    fn add_archive(&mut self, path: &Path) -> Result<(), VfsError> {
        let archive = Box::new(PackedArchive::open(path)?);
        self.archives.push(archive);

        Ok(())
    }
}
