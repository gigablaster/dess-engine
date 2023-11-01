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

use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use uuid::Uuid;

mod gpumesh;
mod gpumodel;
mod image;
mod material;

pub use gpumesh::*;
pub use gpumodel::*;
pub use image::*;
pub use material::*;

const CACHE_PATH: &str = ".cache";

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetRef {
    uuid: Uuid,
}

impl AssetRef {
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

    pub fn as_path(&self) -> PathBuf {
        format!("{}/{}", CACHE_PATH, self.uuid.hyphenated()).into()
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
            uuid: Uuid::from_u128_le(r.read_u128::<LittleEndian>()?),
        })
    }
}

pub trait Asset {
    const TYPE_ID: Uuid;
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>);
}
