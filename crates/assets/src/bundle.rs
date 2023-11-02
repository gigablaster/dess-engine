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

#[derive(Debug, Eq)]
struct Entry {
    pub id: Uuid,
    pub ty: Uuid,
    pub offset: u64,
    pub size: u32,
    pub packed: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.as_u128().cmp(&other.id.as_u128())
    }
}

impl BinarySerialization for Entry {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.id.serialize(w)?;
        self.ty.serialize(w)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u32::<LittleEndian>(self.size)?;
        w.write_u32::<LittleEndian>(self.packed)?;

        Ok(())
    }
}

impl BinaryDeserialization for Entry {
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

struct BundleDesc {
    assets: SortedSet<Entry>,
    names: HashMap<String, AssetRef>,
}

impl BinarySerialization for BundleDesc {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.assets.serialize(w)?;
        self.names.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for BundleDesc {
    fn deserialize(r: &mut impl std::io::Read) -> io::Result<Self> {
        let assets = SortedSet::deserialize(r)?;
        let names = HashMap::deserialize(r)?;

        Ok(Self { assets, names })
    }
}

pub struct LocalBundle {
    file: MappedFile,
    desc: BundleDesc,
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
                    io::ErrorKind::Other,
                    format!("Wrong asset type: expected {} got {}", T::TYPE_ID, entry.ty),
                ));
            }
            let size = entry.size as usize;
            let packed = entry.packed as usize;
            let offset = entry.offset as usize;
            let slice = &self.file.as_ref()[offset..offset + packed];
            let data = lz4_flex::decompress(slice, size).map_err(|x| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Decompression failed: {:?}", x),
                )
            })?;

            Ok(data)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Asset id {} isn't found", asset.uuid),
            ))
        }
    }
}

pub const BUNDLE_MAGIC: FourCC = FourCC(*b"BNDL");
pub const BUNDLE_FILE_VERSION: u32 = 1;

impl LocalBundle {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = MappedFile::open(path)?;
        let mut cursor = Cursor::new(file.as_ref());
        let magic = FourCC::deserialize(&mut cursor)?;
        if magic != BUNDLE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::Other, "Wrong bundle header"));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if version != BUNDLE_FILE_VERSION {
            return Err(io::Error::new(io::ErrorKind::Other, "Wrong bundle version"));
        }
        cursor.seek(io::SeekFrom::End(0))?;
        let size = cursor.stream_position()?;
        cursor.seek(io::SeekFrom::Start(size - size_of::<u64>() as u64))?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        cursor.seek(io::SeekFrom::Start(offset as _))?;
        let desc = BundleDesc::deserialize(&mut cursor)?;

        Ok(LocalBundle { file, desc })
    }
}
