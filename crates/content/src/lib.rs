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
    io,
    path::{Path, PathBuf},
};

use image::ImageError;

mod mesh;
mod texture;

pub use mesh::GltfModelImporter;
pub use texture::TextureImporter;

pub trait Content {
    fn save(&self, path: &Path) -> io::Result<()>;
}

#[derive(Debug)]
pub enum ImportError {
    NotSupported,
    ImportFailed(String),
    IO(io::Error),
}

impl From<io::Error> for ImportError {
    fn from(value: io::Error) -> Self {
        Self::IO(value)
    }
}

impl From<ImageError> for ImportError {
    fn from(value: ImageError) -> Self {
        match value {
            ImageError::Unsupported(_) => ImportError::NotSupported,
            err => ImportError::ImportFailed(err.to_string()),
        }
    }
}

impl From<gltf::Error> for ImportError {
    fn from(value: gltf::Error) -> Self {
        Self::ImportFailed(value.to_string())
    }
}
pub trait ContentImporter: Sync {
    fn can_handle(&self, path: &Path) -> bool;
    fn import(&self, path: &Path, context: &ImportContext)
        -> Result<Box<dyn Content>, ImportError>;
    fn target_name(&self, path: &Path) -> PathBuf;
}

pub struct ImportContext<'a> {
    pub source_dir: &'a Path,
    pub destination_dir: &'a Path,
}
