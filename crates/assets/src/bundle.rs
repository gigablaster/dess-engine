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
    collections::HashMap,
    io::{self, Cursor, Seek},
    mem::size_of,
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use four_cc::FourCC;
use sorted_vec::SortedSet;
use uuid::Uuid;

use crate::{Asset, AssetBundle, AssetRef, MappedFile};

pub const LOCAL_BUNDLE_ALIGN: u64 = 4096;

#[derive(Debug, Eq, Clone, Copy)]
struct BundleDirectoryEntry {
    id: Uuid,
    ty: Uuid,
    offset: u64,
    size: u32,
    packed: u32,
}

impl BundleDirectoryEntry {
    pub fn new(asset: AssetRef, ty: Uuid, offset: u64, size: u32, packed: u32) -> Self {
        Self {
            id: asset.uuid,
            ty,
            offset,
            size,
            packed,
        }
    }
}

impl PartialEq for BundleDirectoryEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for BundleDirectoryEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BundleDirectoryEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.as_u128().cmp(&other.id.as_u128())
    }
}

impl BinarySerialization for BundleDirectoryEntry {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.id.serialize(w)?;
        self.ty.serialize(w)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u32::<LittleEndian>(self.size)?;
        w.write_u32::<LittleEndian>(self.packed)?;

        Ok(())
    }
}

impl BinaryDeserialization for BundleDirectoryEntry {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let id = Uuid::deserialize(r)?;
        let ty = Uuid::deserialize(r)?;
        let offset = r.read_u64::<LittleEndian>()?;
        let size = r.read_u32::<LittleEndian>()?;
        let packed = r.read_u32::<LittleEndian>()?;

        Ok(Self {
            id,
            ty,
            offset,
            size,
            packed,
        })
    }
}

#[derive(Debug, Default)]
pub struct LocalBundleDesc {
    assets: SortedSet<BundleDirectoryEntry>,
    names: HashMap<String, AssetRef>,
}

pub trait BundleDesc: BinarySerialization + BinaryDeserialization {
    fn add_asset(&mut self, asset: AssetRef, ty: Uuid, offset: u64, size: u32, packed: u32);
    fn set_name(&mut self, asset: AssetRef, name: &str);
}

impl BundleDesc for LocalBundleDesc {
    fn add_asset(&mut self, asset: AssetRef, ty: Uuid, offset: u64, size: u32, packed: u32) {
        self.assets
            .push(BundleDirectoryEntry::new(asset, ty, offset, size, packed));
    }

    fn set_name(&mut self, asset: AssetRef, name: &str) {
        self.names.insert(name.into(), asset);
    }
}

impl BinarySerialization for LocalBundleDesc {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.assets.serialize(w)?;
        self.names.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for LocalBundleDesc {
    fn deserialize(r: &mut impl std::io::Read) -> io::Result<Self> {
        let assets = SortedSet::deserialize(r)?;
        let names = HashMap::deserialize(r)?;

        Ok(Self { assets, names })
    }
}

pub struct LocalBundle {
    file: MappedFile,
    desc: LocalBundleDesc,
}

impl AssetBundle for LocalBundle {
    fn asset_by_name(&self, name: &str) -> Option<AssetRef> {
        self.desc.names.get(name).copied()
    }

    fn load<T: Asset>(&self, asset: AssetRef) -> io::Result<Vec<u8>> {
        if let Ok(index) = self
            .desc
            .assets
            .binary_search_by(|x| x.id.as_u128().cmp(&asset.as_u128()))
        {
            let entry = &self.desc.assets[index];
            if entry.ty != T::TYPE_ID {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Wrong asset type: expected {} got {}", T::TYPE_ID, entry.ty),
                ));
            }
            let size = entry.size as usize;
            let packed = entry.packed as usize;
            let offset = entry.offset as usize;
            let slice = &self.file.as_ref()[offset..offset + packed];
            let data = if packed != size {
                lz4_flex::decompress(slice, size).map_err(|x| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Decompression failed: {:?}", x),
                    )
                })?
            } else {
                Vec::from(slice)
            };

            Ok(data)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Asset id {} isn't found", asset.uuid),
            ))
        }
    }
}

pub const LOCAL_BUNDLE_MAGIC: FourCC = FourCC(*b"BNDL");
pub const LOCAL_BUNDLE_FILE_VERSION: u32 = 1;

impl LocalBundle {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = MappedFile::open(path)?;
        let mut cursor = Cursor::new(file.as_ref());
        let magic = FourCC::deserialize(&mut cursor)?;
        if magic != LOCAL_BUNDLE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::Other, "Wrong bundle header"));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if version != LOCAL_BUNDLE_FILE_VERSION {
            return Err(io::Error::new(io::ErrorKind::Other, "Wrong bundle version"));
        }
        cursor.seek(io::SeekFrom::End(0))?;
        let size = cursor.stream_position()?;
        cursor.seek(io::SeekFrom::Start(size - size_of::<u64>() as u64))?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        cursor.seek(io::SeekFrom::Start(offset as _))?;
        let desc = LocalBundleDesc::deserialize(&mut cursor)?;

        Ok(LocalBundle { file, desc })
    }
}
