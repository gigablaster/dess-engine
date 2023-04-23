mod directory;
mod error;
mod mmap;
mod traits;
mod vfs;

pub use error::*;
pub use traits::*;
pub use vfs::*;

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::{Crc, CRC_32_CKSUM};

const CHECKER: Crc<u32> = Crc::<u32>::new(&CRC_32_CKSUM);

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
