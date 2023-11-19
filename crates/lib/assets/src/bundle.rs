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
    collections::HashMap,
    fs::File,
    io::{self, Cursor, Read},
    path::Path,
};

use speedy::{Context, Readable, Writable};
use uuid::Uuid;

use crate::{AssetBundle, AssetRef, MappedFile};

const LOCAL_BUNDLE_MAGIC: [u8; 4] = *b"BNDL";
const LOCAL_BUNDLE_FILE_VERSION: u32 = 1;
pub const ROOT_ASSET_PATH: &str = "assets";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Readable, Writable)]
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

#[derive(Debug, Default)]
pub struct LocalBundleDesc {
    assets: HashMap<AssetRef, BundleDirectoryEntry>,
    dependencies: HashMap<AssetRef, Vec<AssetRef>>,
    names: HashMap<String, AssetRef>,
}

impl<'a, C: Context> Readable<'a, C> for LocalBundleDesc {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        reader
            .read_value::<LocalBundleHeader>()
            .map(|x| x.is_valid())?;
        Ok(Self {
            assets: reader.read_value()?,
            dependencies: reader.read_value()?,
            names: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for LocalBundleDesc {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_value(&LocalBundleHeader::default())?;
        writer.write_value(&self.assets)?;
        writer.write_value(&self.dependencies)?;
        writer.write_value(&self.names)?;

        Ok(())
    }
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

pub struct LocalBundle {
    file: MappedFile,
    desc: LocalBundleDesc,
}

impl AssetBundle for LocalBundle {
    fn get(&self, name: &str) -> Option<AssetRef> {
        self.desc.names.get(name).copied()
    }

    fn load(&self, ty: Uuid, asset: AssetRef) -> io::Result<Vec<u8>> {
        let entry = self.desc.get_asset(asset).ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Asset id {asset} isn't found"),
        ))?;
        if entry.ty != ty {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Wrong asset type: expected {} got {}", ty, entry.ty),
            ));
        }
        let size = entry.size as usize;
        let packed = entry.packed as usize;
        let offset = entry.offset as usize;
        let slice = &self.file.data()[offset..offset + packed];
        if packed != size {
            let mut result = vec![0u8; size];
            let mut decoder = lz4_flex::frame::FrameDecoder::new(Cursor::new(slice));
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

#[derive(Debug, Readable, Copy, Clone, Writable, PartialEq, Eq)]
struct LocalBundleHeader {
    pub magic: [u8; 4],
    pub version: u32,
}

impl LocalBundleHeader {
    pub fn is_valid(&self) -> bool {
        *self == Self::default()
    }
}

impl Default for LocalBundleHeader {
    fn default() -> Self {
        Self {
            magic: LOCAL_BUNDLE_MAGIC,
            version: LOCAL_BUNDLE_FILE_VERSION,
        }
    }
}

impl LocalBundle {
    pub fn load(name: &str) -> io::Result<Box<dyn AssetBundle>> {
        let index_path = Path::new(ROOT_ASSET_PATH).join(format!("{name}.idx"));
        let desc = LocalBundleDesc::read_from_stream_buffered(File::open(index_path)?)?;

        Ok(Box::new(LocalBundle {
            file: MappedFile::open(&Path::new(ROOT_ASSET_PATH).join(format!("{name}.bin")))?,
            desc,
        }))
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use speedy::{Readable, Writable};
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

        desc.write_to_stream(&mut writer).unwrap();

        let mut reader = Cursor::new(target);
        let desc = LocalBundleDesc::read_from_stream_buffered(&mut reader).unwrap();
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
