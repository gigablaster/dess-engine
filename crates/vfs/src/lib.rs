mod directory;
mod error;
mod mmap;
mod traits;
mod vfs;

pub use error::*;
use lazy_static::lazy_static;
pub use traits::*;

use std::{
    io::{self, Read, Write},
    path::Path,
    sync::Mutex,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::{Crc, CRC_32_CKSUM};

use crate::vfs::Vfs;

const CHECKER: Crc<u32> = Crc::<u32>::new(&CRC_32_CKSUM);

lazy_static! {
    static ref VFS: Mutex<Vfs> = Mutex::new(Vfs::default());
}

impl VfsRead for String {
    fn read(r: &mut impl Read) -> io::Result<Self> {
        let count = r.read_u16::<LittleEndian>()?;
        let mut buffer = vec![0; count as _];
        r.read_exact(&mut buffer)?;
        let crc = r.read_u32::<LittleEndian>()?;
        if crc != CHECKER.checksum(&buffer) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad CRC"));
        }

        String::from_utf8(buffer)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UTF8 coversion failed"))
    }
}

impl VfsWrite for String {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let bytes = self.as_bytes();
        w.write_u16::<LittleEndian>(bytes.len() as _)?;
        w.write_all(bytes)?;
        let crc = CHECKER.checksum(bytes);
        w.write_u32::<LittleEndian>(crc)?;

        Ok(())
    }
}

pub fn scan(root: &Path) -> Result<(), VfsError> {
    let mut vfs = VFS.lock().unwrap();
    vfs.scan(root)
}

pub fn get_asset(path: &Path) -> Result<Box<dyn Read>, VfsError> {
    let vfs = VFS.lock().unwrap();
    vfs.get_asset(path)
}
