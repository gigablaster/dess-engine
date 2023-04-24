use std::{
    collections::{HashMap, HashSet},
    io::{self, Read, Seek, Write},
    mem::size_of,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use common::traits::{BinaryDeserialization, BinarySerialization};
use four_cc::FourCC;
use lz4_flex::compress;
use uuid::Uuid;

use crate::VfsError;

const MAGICK: FourCC = FourCC(*b"dess");
const VERSION: u32 = 1;
const FILE_ALIGN: u64 = 4096;

#[derive(Debug, Copy, Clone)]
pub enum Compression {
    None(usize),
    LZ4(usize, usize),
}

impl BinarySerialization for Compression {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            Compression::None(size) => {
                w.write_u8(0)?;
                w.write_u32::<LittleEndian>(*size as u32)?
            }
            Compression::LZ4(uncompressed, compressed) => {
                w.write_u8(1)?;
                w.write_u32::<LittleEndian>(*uncompressed as u32)?;
                w.write_u32::<LittleEndian>(*compressed as u32)?;
            }
        };

        Ok(())
    }
}

impl BinaryDeserialization for Compression {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let byte = r.read_u8()?;
        let result = match byte {
            0 => Compression::None(r.read_u32::<LittleEndian>()? as usize),
            1 => {
                let uncompressed = r.read_u32::<LittleEndian>()? as usize;
                let compressed = r.read_u32::<LittleEndian>()? as usize;

                Compression::LZ4(uncompressed, compressed)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Compression format isn't supported",
                ))
            }
        };

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct FileHeader {
    pub type_id: FourCC,
    pub offset: u64,
    pub compression: Compression,
}

impl BinaryDeserialization for FileHeader {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            type_id: FourCC::deserialize(r)?,
            offset: r.read_u64::<LittleEndian>()?,
            compression: Compression::deserialize(r)?,
        })
    }
}

impl BinarySerialization for FileHeader {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        self.type_id.serialize(w)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        self.compression.serialize(w)?;

        Ok(())
    }
}

#[derive(Debug)]
struct RootHeader {
    magick: FourCC,
    version: u32,
}

impl BinaryDeserialization for RootHeader {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            magick: FourCC::deserialize(r)?,
            version: r.read_u32::<LittleEndian>()?,
        })
    }
}

impl BinarySerialization for RootHeader {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        self.magick.serialize(w)?;
        w.write_u32::<LittleEndian>(self.version)?;

        Ok(())
    }
}

impl Default for RootHeader {
    fn default() -> Self {
        Self {
            magick: MAGICK,
            version: VERSION,
        }
    }
}

#[derive(Debug)]
pub struct Directory {
    files: HashMap<Uuid, FileHeader>,
    names: HashMap<String, Uuid>,
    tags: HashMap<String, HashSet<Uuid>>,
}

impl Directory {
    pub fn load<R: Read + Seek>(r: &mut R) -> Result<Self, VfsError> {
        let root_header = RootHeader::deserialize(r)?;
        if root_header.magick != MAGICK {
            return Err(VfsError::InvalidFormat);
        }
        if root_header.version > VERSION {
            return Err(VfsError::InvalidVersiom);
        }
        r.seek(io::SeekFrom::End(-(size_of::<u64>() as i64)))?;
        let offset = r.read_u64::<LittleEndian>()?;
        r.seek(io::SeekFrom::Start(offset as _))?;

        let names = HashMap::<String, Uuid>::deserialize(r)?;
        let files = HashMap::<Uuid, FileHeader>::deserialize(r)?;
        let tags = HashMap::<String, HashSet<Uuid>>::deserialize(r)?;

        Ok(Self { names, files, tags })
    }

    pub fn by_uuid(&self, uuid: Uuid) -> Option<&FileHeader> {
        self.files.get(&uuid)
    }

    pub fn by_name(&self, name: &str) -> Option<&Uuid> {
        self.names.get(name)
    }

    pub fn by_tag(&self, tag: &str) -> Option<&HashSet<Uuid>> {
        if let Some(tagged) = self.tags.get(tag) {
            Some(tagged)
        } else {
            None
        }
    }
}

pub struct DirectoryBaker<W: Write + Seek> {
    w: W,
    files: HashMap<Uuid, FileHeader>,
    names: HashMap<String, Uuid>,
    tags: HashMap<String, HashSet<Uuid>>,
}

impl<W: Write + Seek> DirectoryBaker<W> {
    pub fn new(w: W) -> Result<Self, VfsError> {
        let mut w = w;
        RootHeader::default().serialize(&mut w)?;
        Ok(Self {
            w,
            files: HashMap::new(),
            names: HashMap::new(),
            tags: HashMap::new(),
        })
    }

    pub fn write(
        &mut self,
        type_id: FourCC,
        data: &[u8],
        uuid: Uuid,
        packed: bool,
    ) -> Result<(), VfsError> {
        let offset = self.try_align()?;
        let compression = if packed {
            let compressed = compress(data);
            self.w.write_all(&compressed)?;

            Compression::LZ4(data.len(), compressed.len())
        } else {
            self.w.write_all(data)?;

            Compression::None(data.len())
        };

        self.files.insert(
            uuid,
            FileHeader {
                type_id,
                offset,
                compression,
            },
        );

        Ok(())
    }

    fn try_align(&mut self) -> io::Result<u64> {
        let offset = self.w.stream_position()?;
        let offset_align = (offset & !(FILE_ALIGN - 1)) + FILE_ALIGN;

        // Align if possible, current position if not.
        Ok(self
            .w
            .seek(io::SeekFrom::Start(offset_align))
            .unwrap_or(self.w.stream_position()?))
    }

    pub fn name(&mut self, uuid: Uuid, name: &str) -> Result<(), VfsError> {
        if self.files.contains_key(&uuid) {
            self.names.insert(name.into(), uuid);

            Ok(())
        } else {
            Err(VfsError::AssetNotFound(uuid))
        }
    }

    pub fn tag(&mut self, uuid: Uuid, tag: &str) -> Result<(), VfsError> {
        if !self.files.contains_key(&uuid) {
            return Err(VfsError::AssetNotFound(uuid));
        }
        if !self.tags.contains_key(tag) {
            self.tags.insert(tag.into(), HashSet::new());
        }
        self.tags.get_mut(tag).unwrap().insert(uuid);

        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        let offset = self.try_align()?;

        self.names.serialize(&mut self.w)?;
        self.files.serialize(&mut self.w)?;
        self.tags.serialize(&mut self.w)?;

        self.w.write_u64::<LittleEndian>(offset)?;

        Ok(())
    }
}
