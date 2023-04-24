use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    io::{self, Read, Write},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::{Crc, CRC_32_CKSUM};
use four_cc::FourCC;
use uuid::Uuid;

use crate::traits::{BinaryDeserialization, BinarySerialization};

const CHECKER: Crc<u32> = Crc::<u32>::new(&CRC_32_CKSUM);

impl BinaryDeserialization for String {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
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

impl BinarySerialization for String {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        let bytes = self.as_bytes();
        w.write_u16::<LittleEndian>(bytes.len() as _)?;
        w.write_all(bytes)?;
        let crc = CHECKER.checksum(bytes);
        w.write_u32::<LittleEndian>(crc)?;

        Ok(())
    }
}

impl BinaryDeserialization for FourCC {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;

        Ok(FourCC(magic))
    }
}

impl BinarySerialization for FourCC {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.0)
    }
}

impl BinarySerialization for Uuid {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(self.as_bytes())
    }
}

impl BinaryDeserialization for Uuid {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; 16];
        r.read_exact(&mut buf)?;

        Ok(Self::from_bytes(buf))
    }
}

impl<T> BinarySerialization for Vec<T>
where
    T: BinarySerialization,
{
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.len() as _)?;
        for value in self.iter() {
            value.serialize(w)?;
        }

        Ok(())
    }
}

impl<T> BinaryDeserialization for Vec<T>
where
    T: BinaryDeserialization,
{
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let len = r.read_u32::<LittleEndian>()?;
        let mut vec = Vec::with_capacity(len as _);
        for _ in 0..len {
            vec.push(T::deserialize(r)?);
        }

        Ok(vec)
    }
}

impl<T, U> BinarySerialization for HashMap<T, U>
where
    T: BinarySerialization,
    U: BinarySerialization,
{
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.len() as _)?;
        for (key, value) in self.iter() {
            key.serialize(w)?;
            value.serialize(w)?;
        }

        Ok(())
    }
}

impl<T, U> BinaryDeserialization for HashMap<T, U>
where
    T: BinaryDeserialization + Hash + Eq,
    U: BinaryDeserialization,
{
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let len = r.read_u32::<LittleEndian>()?;
        let mut map = HashMap::with_capacity(len as _);
        for _ in 0..len {
            let key = T::deserialize(r)?;
            let value = U::deserialize(r)?;

            map.insert(key, value);
        }

        Ok(map)
    }
}

impl<T> BinarySerialization for HashSet<T>
where
    T: BinarySerialization,
{
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.len() as _)?;

        for value in self.iter() {
            value.serialize(w)?;
        }

        Ok(())
    }
}

impl<T> BinaryDeserialization for HashSet<T>
where
    T: BinaryDeserialization + Hash + Eq,
{
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let len = r.read_u32::<LittleEndian>()?;
        let mut set = HashSet::with_capacity(len as _);
        for _ in 0..len {
            set.insert(T::deserialize(r)?);
        }

        Ok(set)
    }
}
