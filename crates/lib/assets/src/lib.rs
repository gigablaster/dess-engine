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
    fmt::Display,
    fs::File,
    hash::Hasher,
    io::{self, Read, Write},
    mem,
    path::Path,
    slice,
};

use memmap2::{Mmap, MmapOptions};
use siphasher::sip128::Hasher128;
use speedy::{Readable, Writable};
use uuid::Uuid;

mod bundle;
mod image;
mod material;
mod model;
mod shader;

pub use bundle::*;
pub use image::*;
pub use material::*;
pub use model::*;
pub use shader::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Readable, Writable)]
pub struct AssetRef(Uuid);

impl AssetRef {
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn from_u128(value: u128) -> Self {
        Self(Uuid::from_u128(value))
    }

    pub fn from_path(path: &Path) -> Self {
        Self::from_bytes(path.to_str().unwrap().as_bytes())
    }

    pub fn from_path_with<T: Copy>(path: &Path, extra: &T) -> Self {
        let mut hash = siphasher::sip128::SipHasher::default();
        hash.write(path.to_str().unwrap().as_bytes());
        hash.write(unsafe {
            slice::from_raw_parts(
                slice::from_ref(&extra).as_ptr() as *const u8,
                mem::size_of::<T>(),
            )
        });
        Self(Uuid::from_u128(hash.finish128().as_u128()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = siphasher::sip128::SipHasher::default().hash(bytes);
        Self(Uuid::from_u128(hash.as_u128()))
    }

    pub fn from_bytes_with<T: Copy>(bytes: &[u8], extra: &T) -> Self {
        let mut hash = siphasher::sip128::SipHasher::default();
        hash.write(bytes);
        hash.write(unsafe {
            slice::from_raw_parts(
                slice::from_ref(&extra).as_ptr() as *const u8,
                mem::size_of::<T>(),
            )
        });
        Self(Uuid::from_u128(hash.finish128().as_u128()))
    }

    pub fn valid(&self) -> bool {
        !self.0.is_nil()
    }

    pub fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }
}

impl Display for AssetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_hyphenated())
    }
}

pub trait Asset: Sized + Any {
    const TYPE_ID: Uuid;

    fn serialize<W: Write>(&self, w: &mut W) -> io::Result<()>;
    fn deserialize<R: Read>(r: &mut R) -> io::Result<Self>;
}

pub struct MappedFile {
    mmap: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            mmap: unsafe { MmapOptions::new().map(&file) }?,
        })
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap
    }
}
