// Copyright (C) 2023 Vladimir Kuskov

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
    any::Any,
    env,
    fmt::{Debug, Display},
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use speedy::{Readable, Writable};
use uuid::Uuid;

mod image;
mod model;
mod shader;

pub use image::*;
pub use model::*;
pub use shader::*;

pub const ROOT_DATA_PATH: &str = "assets";
pub const ASSET_CACHE_PATH: &str = ".cache";
pub const BUNDLE_DESC_PATH: &str = "bundles";

#[derive(Debug, Clone)]
pub enum Error {
    Io(String),
    ImportFailed(String),
    ProcessingFailed(String),
    BadSourceData,
    EvalFailed,
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Readable, Writable)]
pub struct AssetRef(Uuid);

impl Display for AssetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_hyphenated())
    }
}

impl From<u128> for AssetRef {
    fn from(value: u128) -> Self {
        Self(Uuid::from_u128(value))
    }
}

pub trait ContentSource: Debug + Send + Sync {
    fn get_ref(&self) -> AssetRef;
}

pub trait Asset: Send + Sync {
    fn to_bytes(&self) -> io::Result<Vec<u8>>;
}

pub trait AssetLoad: Sized {
    fn from_bytes(data: &[u8]) -> io::Result<Self>;
}

pub fn get_relative_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    // Is this path relative to data folder? Check this option.
    let path = if !path.as_ref().exists() {
        root.join(path)
    } else {
        path.as_ref().into()
    };
    let path = path.canonicalize()?;

    Ok(path.strip_prefix(root).unwrap().into())
}

pub fn get_absolute_asset_path<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    Ok(root.join(get_relative_asset_path(path.as_ref())?))
}

pub fn get_cached_asset_path(asset: AssetRef) -> PathBuf {
    Path::new(ASSET_CACHE_PATH).join(format!("{}", asset))
}
