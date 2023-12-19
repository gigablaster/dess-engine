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

mod material;
// mod mesh;
mod pool;
mod resource_manager;

use std::{
    fs::File,
    io::{self},
    path::Path,
    sync::Arc,
};

// pub use asset_cache::*;
use dess_assets::{get_cached_asset_path, Asset, AssetLoad, AssetRef};
use dess_backend::BackendError;
use log::debug;
pub use material::*;
use memmap2::Mmap;
// pub use mesh::*;
pub use pool::*;
pub use resource_manager::*;

#[derive(Debug, Clone)]
pub enum Error {
    Io(Arc<io::Error>),
    ParseError(String),
    BackendError(BackendError),
    InvalidHandle,
    ImportFailed(String),
    LoadingFailed,
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(Arc::new(value))
    }
}

impl From<BackendError> for Error {
    fn from(value: BackendError) -> Self {
        Self::BackendError(value)
    }
}

fn map_file<P: AsRef<Path>>(path: P) -> io::Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    Ok(mmap)
}

pub(crate) fn load_cached_asset<T: Asset + AssetLoad>(asset: AssetRef) -> io::Result<T> {
    let path = get_cached_asset_path(asset);
    if path.exists() {
        debug!("Loading asset {:?}", asset);
        let data = map_file(path)?;
        Ok(T::from_bytes(&data)?)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Asset {:?} not found", asset),
        ))
    }
}
