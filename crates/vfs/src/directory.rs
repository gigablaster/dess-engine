use std::{
    collections::HashMap,
    io::{self, Read, Seek, Write},
    mem::size_of,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;

use crate::VfsError;

use super::{VfsRead, VfsWrite};

const MAGICK: FourCC = FourCC(*b"dess");
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct FileHeader {
    pub offset: u64,
    pub size: u64,
    pub packed: u64,
}

impl VfsRead for FileHeader {
    fn read(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            offset: r.read_u64::<LittleEndian>()?,
            size: r.read_u64::<LittleEndian>()?,
            packed: r.read_u64::<LittleEndian>()?,
        })
    }
}

impl VfsWrite for FileHeader {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u64::<LittleEndian>(self.size)?;
        w.write_u64::<LittleEndian>(self.packed)?;

        Ok(())
    }
}

#[derive(Debug)]
struct DirectoryHeader {
    pub count: u32,
    pub offset: u64,
}

impl VfsRead for DirectoryHeader {
    fn read(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            count: r.read_u32::<LittleEndian>()?,
            offset: r.read_u64::<LittleEndian>()?,
        })
    }
}

impl VfsWrite for DirectoryHeader {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.count)?;
        w.write_u64::<LittleEndian>(self.offset)?;

        Ok(())
    }
}

#[derive(Debug)]
struct RootHeader {
    magick: FourCC,
    version: u32,
}

impl VfsRead for FourCC {
    fn read(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;

        Ok(FourCC(magic))
    }
}

impl VfsWrite for FourCC {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.0)
    }
}

impl VfsRead for RootHeader {
    fn read(r: &mut impl Read) -> io::Result<Self> {
        Ok(Self {
            magick: FourCC::read(r)?,
            version: r.read_u32::<LittleEndian>()?,
        })
    }
}

impl VfsWrite for RootHeader {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        self.magick.write(w)?;
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

impl RootHeader {
    pub fn file_offset() -> usize {
        size_of::<u64>() + size_of::<u32>()
    }
}

pub fn read_archive_directory<T: Read + Seek>(
    file: &mut T,
) -> Result<HashMap<String, FileHeader>, VfsError> {
    let root_header = RootHeader::read(file)?;
    if root_header.magick != MAGICK {
        return Err(VfsError::InvalidFormat);
    }
    if root_header.version > VERSION {
        return Err(VfsError::InvalidVersiom);
    }
    file.seek(io::SeekFrom::End(-(RootHeader::file_offset() as i64)))?;
    let directory_header = DirectoryHeader::read(file)?;
    file.seek(io::SeekFrom::Start(directory_header.offset as _))?;
    let mut result = HashMap::new();
    for _ in 0..directory_header.count {
        let name = String::read(file)?;
        let header = FileHeader::read(file)?;

        result.insert(name, header);
    }

    Ok(result)
}

pub fn prepare_archive<T: Write>(file: &mut T) -> Result<(), VfsError> {
    Ok(RootHeader::default().write(file)?)
}

pub fn write_archive_directory<T: Write + Seek>(
    file: &mut T,
    directory: &HashMap<String, FileHeader>,
) -> Result<(), VfsError> {
    let start_offset = file.stream_position()?;
    for (name, header) in directory {
        name.write(file)?;
        header.write(file)?;
    }
    let directory_header = DirectoryHeader {
        offset: start_offset,
        count: directory.len() as _,
    };
    directory_header.write(file)?;

    Ok(())
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, io::Cursor};

    use super::{prepare_archive, read_archive_directory, write_archive_directory, FileHeader};

    fn prepare() -> Vec<u8> {
        let mut data = Vec::new();
        let mut file = Cursor::new(&mut data);
        prepare_archive(&mut file).unwrap();
        let mut directory = HashMap::new();
        directory.insert(
            "file1".into(),
            FileHeader {
                offset: 0,
                size: 1,
                packed: 2,
            },
        );
        directory.insert(
            "file2".into(),
            FileHeader {
                offset: 11,
                size: 12,
                packed: 13,
            },
        );
        write_archive_directory(&mut file, &directory).unwrap();

        data
    }

    #[test]
    fn write_directory() {
        let data = prepare();
        assert!(!data.is_empty());
    }

    #[test]
    fn read_directory() {
        let data = prepare();
        let mut file = Cursor::new(&data);
        let directory = read_archive_directory(&mut file).unwrap();
        let file1 = directory.get("file1").unwrap();
        let file2 = directory.get("file2").unwrap();
        assert_eq!(file1.offset, 0);
        assert_eq!(file1.size, 1);
        assert_eq!(file1.packed, 2);
        assert_eq!(file2.offset, 11);
        assert_eq!(file2.size, 12);
        assert_eq!(file2.packed, 13);
    }
}
