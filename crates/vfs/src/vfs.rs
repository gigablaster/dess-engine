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
    fs::{self},
    io::Write,
    path::PathBuf,
};

use directories::ProjectDirs;
use log::{error, info};

use crate::{
    dev_fs::DevFsArchive, packed::PackedArchive, raw_fs::RawFsArchive, Archive, AssetPath, Loader,
    VfsError,
};

pub struct Vfs {
    content: Vec<Box<dyn Archive>>,
    cache: Box<dyn Archive>,
    save: Box<dyn Archive>,
}

impl Vfs {
    pub fn new(organization: &str, application: &str) -> Self {
        let mut content = Vec::<Box<dyn Archive>>::new();
        content.push(Box::new(DevFsArchive::new(&PathBuf::from("."))));
        let dir = fs::read_dir(".").unwrap();
        for path in dir {
            let path = path.unwrap().path();
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
                match PackedArchive::new(&path) {
                    Err(err) => error!("Failed to add archive {:?} - {:?}", path, err),
                    Ok(archive) => content.push(Box::new(archive)),
                }
            }
        }
        let project_dirs = ProjectDirs::from("com", organization, application).unwrap();
        let cache = Box::new(RawFsArchive::new(project_dirs.cache_dir()));
        let save = Box::new(RawFsArchive::new(project_dirs.data_dir()));

        Self {
            content,
            cache,
            save,
        }
    }

    pub fn load(&self, name: AssetPath) -> Result<Box<dyn Loader>, VfsError> {
        match name {
            AssetPath::Content(name) => {
                for archive in &self.content {
                    match archive.open(name) {
                        Ok(content) => return Ok(content),
                        Err(VfsError::NotFound(_)) => {}
                        Err(err) => return Err(err),
                    }
                }
            }
            AssetPath::Cache(name) => return self.cache.open(name),
            AssetPath::Save(name) => return self.save.open(name),
        }

        Err(VfsError::NotFound(name.to_string()))
    }

    pub fn create(&self, name: AssetPath) -> Result<Box<dyn Write>, VfsError> {
        match name {
            AssetPath::Content(name) => {
                for archive in &self.content {
                    match archive.create(name) {
                        Ok(content) => return Ok(content),
                        Err(VfsError::ReadOnly) => {}
                        Err(err) => return Err(err),
                    }
                }
            }
            AssetPath::Cache(name) => return self.cache.create(name),
            AssetPath::Save(name) => return self.save.create(name),
        }

        Err(VfsError::ReadOnly)
    }
}
