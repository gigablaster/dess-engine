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
    fs::{self},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

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

pub trait Asset: Sized + Any {
    const TYPE_ID: Uuid;

    fn serialize<W: Write>(&self, w: &mut W) -> io::Result<()>;
    fn deserialize<R: Read>(r: &mut R) -> io::Result<Self>;
}

pub(crate) fn read_to_end<P>(path: P) -> io::Result<Vec<u8>>
where
    P: AsRef<Path>,
{
    let file = fs::File::open(path.as_ref())?;
    // Allocate one extra byte so the buffer doesn't need to grow before the
    // final `read` call at the end of the file.  Don't worry about `usize`
    // overflow because reading will fail regardless in that case.
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
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
