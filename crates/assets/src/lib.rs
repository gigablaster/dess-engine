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

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use siphasher::sip128::SipHasher;

mod gltf_import;
mod gpumesh;
mod gpumodel;
mod image;
mod material;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AssetRef {
    hash: u128,
}

impl From<String> for AssetRef {
    fn from(value: String) -> Self {
        Self {
            hash: SipHasher::default().hash(value.as_bytes()).as_u128(),
        }
    }
}

impl From<Option<String>> for AssetRef {
    fn from(value: Option<String>) -> Self {
        if let Some(name) = value {
            name.into()
        } else {
            AssetRef::default()
        }
    }
}

impl AssetRef {
    pub fn valid(&self) -> bool {
        self.hash != 0
    }
}

impl BinarySerialization for AssetRef {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u128::<LittleEndian>(self.hash)
    }
}

impl BinaryDeserialization for AssetRef {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let hash = r.read_u128::<LittleEndian>()?;

        Ok(Self { hash })
    }
}

pub trait AssetDependencies {
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>);
}

pub(crate) const ROOT_DATA_PATH: &str = "data";
