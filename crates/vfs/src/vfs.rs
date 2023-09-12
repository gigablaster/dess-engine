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

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use log::{error, info};

use crate::{
    dev_fs::DevFsArchive, packed::PackedArchive, raw_fs::RawFsArchive, Archive, Loader, VfsError,
};

#[derive(Default)]
pub struct Vfs {
    archives: Vec<Box<dyn Archive>>,
}

impl Vfs {
    pub fn scan(&mut self, root: impl Into<PathBuf>) -> Result<(), VfsError> {
        let root = root.into();
        self.archives.push(Box::new(DevFsArchive::new(&root)));
        let paths = fs::read_dir(root)?;
        for path in paths {
            let path = path?.path();
            let extension = if let Some(ext) = path.extension() {
                if let Some(ext) = ext.to_str() {
                    ext == "pak"
                } else {
                    false
                }
            } else {
                false
            };
            if path.is_file() && extension {
                info!("Adding archive {:?}", path);
                if let Err(err) = self.add_archive(&path) {
                    error!("Failed to add archive {:?} - {:?}", path, err);
                }
            }
        }

        Ok(())
    }

    pub fn get(&self, name: &Path) -> Result<Box<dyn Loader>, VfsError> {
        for archive in &self.archives {
            match archive.open(name) {
                Ok(content) => return Ok(content),
                Err(VfsError::NotFound(_)) => {}
                Err(err) => return Err(err),
            }
        }

        Err(VfsError::NotFound(name.to_path_buf()))
    }

    fn add_archive(&mut self, path: &Path) -> Result<(), VfsError> {
        let archive = Box::new(PackedArchive::open(path)?);
        self.archives.push(archive);

        Ok(())
    }
}
