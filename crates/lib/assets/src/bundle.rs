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
    io::{self, BufRead, Cursor, Read, Seek},
    mem::size_of,
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use four_cc::FourCC;
use uuid::Uuid;

use crate::{AssetBundle, AssetRef, MappedFile};

pub const LOCAL_BUNDLE_ALIGN: u64 = 4096;
pub const LOCAL_BUNDLE_MAGIC: FourCC = FourCC(*b"BNDL");
pub const LOCAL_BUNDLE_FILE_VERSION: u32 = 1;
pub const LOCAL_BUNDLE_DICT_SIZE: usize = 64536;
pub const LOCAL_BUNDLE_DICT_USAGE_LIMIT: usize = LOCAL_BUNDLE_DICT_SIZE * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundleDirectoryEntry {
    ty: Uuid,
    offset: u64,
    size: u32,
    packed: u32,
}

impl BundleDirectoryEntry {
    pub fn new(ty: Uuid, offset: u64, size: u32, packed: u32) -> Self {
        Self {
            ty,
            offset,
            size,
            packed,
        }
    }
}

impl BinarySerialization for BundleDirectoryEntry {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.ty.serialize(w)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u32::<LittleEndian>(self.size)?;
        w.write_u32::<LittleEndian>(self.packed)?;

        Ok(())
    }
}

impl BinaryDeserialization for BundleDirectoryEntry {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let ty = Uuid::deserialize(r)?;
        let offset = r.read_u64::<LittleEndian>()?;
        let size = r.read_u32::<LittleEndian>()?;
        let packed = r.read_u32::<LittleEndian>()?;

        Ok(Self {
            ty,
            offset,
            size,
            packed,
        })
    }
}

#[derive(Debug, Default)]
pub struct LocalBundleDesc {
    assets: HashMap<AssetRef, BundleDirectoryEntry>,
    dependencies: HashMap<AssetRef, Vec<AssetRef>>,
    names: HashMap<String, AssetRef>,
}

impl LocalBundleDesc {
    pub fn add_asset(&mut self, asset: AssetRef, ty: Uuid, offset: u64, size: u32, packed: u32) {
        self.assets
            .insert(asset, BundleDirectoryEntry::new(ty, offset, size, packed));
    }

    pub fn set_name(&mut self, asset: AssetRef, name: &str) {
        self.names.insert(name.into(), asset);
    }

    pub fn set_dependencies(&mut self, asset: AssetRef, dependencies: &[AssetRef]) {
        self.dependencies.insert(asset, dependencies.to_vec());
    }

    pub fn get_by_name(&self, name: &str) -> Option<AssetRef> {
        self.names.get(name).copied()
    }

    pub fn get_asset(&self, asset: AssetRef) -> Option<BundleDirectoryEntry> {
        self.assets.get(&asset).copied()
    }

    pub fn get_dependencies(&self, asset: AssetRef) -> Option<&[AssetRef]> {
        self.dependencies.get(&asset).map(|x| x.as_ref())
    }
}

impl BinarySerialization for LocalBundleDesc {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.assets.serialize(w)?;
        self.dependencies.serialize(w)?;
        self.names.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for LocalBundleDesc {
    fn deserialize(r: &mut impl std::io::Read) -> io::Result<Self> {
        let assets = HashMap::deserialize(r)?;
        let dependencies = HashMap::deserialize(r)?;
        let names = HashMap::deserialize(r)?;

        Ok(Self {
            assets,
            dependencies,
            names,
        })
    }
}

pub struct LocalBundle {
    file: MappedFile,
    dicts: HashMap<Uuid, Vec<u8>>,
    desc: LocalBundleDesc,
}

impl AssetBundle for LocalBundle {
    fn asset_by_name(&self, name: &str) -> Option<AssetRef> {
        self.desc.names.get(name).copied()
    }

    fn load(&self, asset: AssetRef, expect_ty: Uuid) -> io::Result<Vec<u8>> {
        let entry = self.desc.get_asset(asset).ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Asset id {} isn't found", asset.uuid),
        ))?;
        if entry.ty != expect_ty {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Wrong asset type: expected {} got {}", expect_ty, entry.ty),
            ));
        }
        let size = entry.size as usize;
        let packed = entry.packed as usize;
        let offset = entry.offset as usize;
        let slice = &self.file.as_ref()[offset..offset + packed];
        if packed != size {
            let reader = Cursor::new(slice);
            let mut decoder = self.create_decoder(reader, expect_ty, size)?;
            let mut result = vec![0u8; size];
            decoder.read_exact(&mut result)?;

            Ok(result)
        } else {
            Ok(Vec::from(slice))
        }
    }

    fn dependencies(&self, asset: AssetRef) -> Option<&[AssetRef]> {
        self.desc.dependencies.get(&asset).map(|x| x.as_ref())
    }

    fn contains(&self, asset: AssetRef) -> bool {
        self.desc.assets.contains_key(&asset)
    }
}

impl LocalBundle {
    pub fn load(path: &Path) -> io::Result<Box<dyn AssetBundle>> {
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
        let dicts = HashMap::deserialize(&mut cursor)?;
        let desc = LocalBundleDesc::deserialize(&mut cursor)?;

        Ok(Box::new(LocalBundle { file, dicts, desc }))
    }

    fn create_decoder<R: BufRead>(
        &self,
        r: R,
        ty: Uuid,
        unpacked: usize,
    ) -> io::Result<zstd::Decoder<R>> {
        if unpacked <= LOCAL_BUNDLE_DICT_USAGE_LIMIT {
            zstd::Decoder::with_buffer(r)
        } else {
            let dict = self.dicts.get(&ty).ok_or(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Decompression dictionary doesn't exist for asset type {:?}",
                    ty
                ),
            ))?;
            zstd::Decoder::with_dictionary(r, dict)
        }
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use dess_common::traits::{BinaryDeserialization, BinarySerialization};
    use uuid::Uuid;

    use crate::{AssetRef, BundleDirectoryEntry, LocalBundleDesc};

    #[test]
    fn write_read_bundle_desc() {
        let mut desc = LocalBundleDesc::default();
        let uuid1 = uuid::uuid!("36a400d6-1e50-443e-9e4c-b87bd92364ea");
        let uuid2 = uuid::uuid!("7134edb0-1f41-423a-a00e-1d8596d60460");
        let asset1 = AssetRef::from_uuid(uuid1);
        let asset2 = AssetRef::from_uuid(uuid2);
        desc.add_asset(asset1, uuid2, 0, 100, 50);
        desc.add_asset(asset2, uuid1, 200, 200, 200);
        desc.set_name(asset1, "abc");
        desc.set_dependencies(asset1, &[asset2]);

        let mut target = Vec::<u8>::new();
        let mut writer = Cursor::new(&mut target);

        desc.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(target);
        let desc = LocalBundleDesc::deserialize(&mut reader).unwrap();
        assert_eq!(Some(asset1), desc.get_by_name("abc"));
        assert_eq!(None, desc.get_by_name("fuck"));
        assert_eq!(Some(vec![asset2].as_ref()), desc.get_dependencies(asset1));
        assert_eq!(None, desc.get_dependencies(asset2));
        assert_eq!(
            BundleDirectoryEntry {
                ty: uuid2,
                offset: 0,
                size: 100,
                packed: 50
            },
            desc.get_asset(asset1).unwrap()
        );
        assert_eq!(
            BundleDirectoryEntry {
                ty: uuid1,
                offset: 200,
                size: 200,
                packed: 200
            },
            desc.get_asset(asset2).unwrap()
        );
        assert_eq!(None, desc.get_asset(AssetRef::from_uuid(Uuid::nil())));
    }
}
