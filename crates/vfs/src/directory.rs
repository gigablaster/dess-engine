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
    io::{self, Read, Seek, Write},
    mem::size_of,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use common::traits::{BinaryDeserialization, BinarySerialization};
use four_cc::FourCC;

use crate::VfsError;

const MAGICK: FourCC = FourCC(*b"dess");
const VERSION: u32 = 1;
const FILE_ALIGN: u64 = 4096;

#[derive(Debug, Clone)]
pub struct FileHeader {
    pub offset: u64,
    pub size: u32,
}

impl BinaryDeserialization for FileHeader {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            offset: r.read_u64::<LittleEndian>()?,
            size: r.read_u32::<LittleEndian>()?,
        })
    }
}

impl BinarySerialization for FileHeader {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u32::<LittleEndian>(self.size)?;

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

pub type Directory = HashMap<String, FileHeader>;

pub fn load_archive_directory<R: Read + Seek>(r: &mut R) -> Result<Directory, VfsError> {
    let root_header = RootHeader::deserialize(r)?;
    if root_header.magick != MAGICK {
        return Err(VfsError::InvalidFormat);
    }
    if root_header.version > VERSION {
        return Err(VfsError::InvalidVersiom);
    }
    r.seek(io::SeekFrom::End(0))?;
    let size = r.stream_position()?;
    r.seek(io::SeekFrom::Start(size - size_of::<u64>() as u64))?;
    let offset = r.read_u64::<LittleEndian>()?;
    r.seek(io::SeekFrom::Start(offset as _))?;

    let files = HashMap::<String, FileHeader>::deserialize(r)?;

    Ok(files)
}

pub struct DirectoryBaker<W: Write + Seek> {
    w: W,
    files: HashMap<String, FileHeader>,
}

impl<W: Write + Seek> DirectoryBaker<W> {
    pub fn new(w: W) -> Result<Self, VfsError> {
        let mut w = w;
        RootHeader::default().serialize(&mut w)?;
        Ok(Self {
            w,
            files: HashMap::new(),
        })
    }

    pub fn write(&mut self, name: &str, data: &[u8]) -> Result<(), VfsError> {
        let name = name.replace('\\', "/").to_ascii_lowercase();
        let offset = self.try_align()?;
        self.w.write_all(data)?;

        self.files.insert(
            name,
            FileHeader {
                offset,
                size: data.len() as _,
            },
        );

        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), VfsError> {
        self.try_align()?;
        let offset = self.w.stream_position()?;
        self.files.serialize(&mut self.w)?;
        self.w.write_u64::<LittleEndian>(offset)?;

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
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use crate::DirectoryBaker;

    use super::load_archive_directory;

    #[test]
    fn empty_archive() {
        let mut data = Vec::new();
        let mut builder = DirectoryBaker::new(Cursor::new(&mut data)).unwrap();
        builder.finish().unwrap();
        load_archive_directory(&mut Cursor::new(&data)).unwrap();
    }

    #[test]
    fn write_read() {
        let mut data = Vec::new();
        let mut builder = DirectoryBaker::new(Cursor::new(&mut data)).unwrap();
        let data1 = b"Hello world!";
        let data2 = b"Hi there!";
        builder.write("file1", data1).unwrap();
        builder.write("file2", data2).unwrap();
        builder.finish().unwrap();
        let dir = load_archive_directory(&mut Cursor::new(&data)).unwrap();
        assert_eq!(2, dir.len());
        let file1 = dir.get("file1").unwrap();
        assert_eq!(data1.len() as u32, file1.size);
        let file2 = dir.get("file2").unwrap();
        assert_eq!(data2.len() as u32, file2.size);
    }
}
