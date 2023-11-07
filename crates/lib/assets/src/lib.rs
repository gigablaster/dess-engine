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

use std::{fmt::Display, fs::File, io, path::Path};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use memmap2::{Mmap, MmapOptions};
use uuid::Uuid;

mod bundle;
mod gpumesh;
mod gpumodel;
mod image;
mod material;
mod shader;

pub use bundle::*;
pub use gpumesh::*;
pub use gpumodel::*;
pub use image::*;
pub use material::*;
pub use shader::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetRef {
    uuid: Uuid,
}

impl AssetRef {
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self { uuid }
    }

    pub fn from_path(path: &Path) -> Self {
        Self::from_bytes(path.to_str().unwrap().as_bytes())
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = siphasher::sip128::SipHasher::default().hash(bytes);
        Self {
            uuid: Uuid::from_u128(hash.as_u128()),
        }
    }

    pub fn valid(&self) -> bool {
        !self.uuid.is_nil()
    }

    pub fn as_u128(&self) -> u128 {
        self.uuid.as_u128()
    }
}

impl Display for AssetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.uuid.as_hyphenated())
    }
}

impl BinarySerialization for AssetRef {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u128::<LittleEndian>(self.uuid.as_u128())
    }
}

impl BinaryDeserialization for AssetRef {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        Ok(Self {
            uuid: Uuid::from_u128(r.read_u128::<LittleEndian>()?),
        })
    }
}

pub trait Asset: Send + Sync {
    const TYPE_ID: Uuid;
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>);
}

pub trait AssetBundle: Sync + Send {
    fn load(&self, asset: AssetRef, expect_ty: Uuid) -> io::Result<Vec<u8>>;
    fn dependencies(&self, asset: AssetRef) -> Option<&[AssetRef]>;
    fn asset_by_name(&self, name: &str) -> Option<AssetRef>;
    fn contains(&self, asset: AssetRef) -> bool;
}

struct MappedFile {
    mmap: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            mmap: unsafe { MmapOptions::new().map(&file) }?,
        })
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        &self.mmap
    }
}
