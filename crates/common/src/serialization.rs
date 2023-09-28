use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    io::{self, Read, Write},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;
use uuid::Uuid;

use crate::traits::{BinaryDeserialization, BinarySerialization};

impl BinaryDeserialization for String {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let count = r.read_u16::<LittleEndian>()?;
        let mut buffer = vec![0; count as _];
        r.read_exact(&mut buffer)?;

        String::from_utf8(buffer)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UTF8 coversion failed"))
    }
}

impl BinarySerialization for String {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        let bytes = self.as_bytes();
        w.write_u16::<LittleEndian>(bytes.len() as _)?;
        w.write_all(bytes)?;

        Ok(())
    }
}

impl BinaryDeserialization for Option<String> {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let count = r.read_u16::<LittleEndian>()?;
        if count != u16::MAX {
            let mut buffer = vec![0; count as _];
            r.read_exact(&mut buffer)?;

            Ok(Some(String::from_utf8(buffer).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "UTF8 coversion failed")
            })?))
        } else {
            Ok(None)
        }
    }
}

impl BinarySerialization for Option<String> {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            Some(string) => string.serialize(w),
            None => w.write_u16::<LittleEndian>(u16::MAX),
        }
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

impl BinarySerialization for Vec<u8> {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.len() as _)?;
        w.write_all(self)?;

        Ok(())
    }
}

impl BinaryDeserialization for Vec<u8> {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let len = r.read_u32::<LittleEndian>()? as _;
        let mut result = vec![0u8; len];
        r.read_exact(&mut result)?;

        Ok(result)
    }
}

impl BinarySerialization for Vec<u16> {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.len() as _)?;
        for it in self.iter() {
            w.write_u16::<LittleEndian>(*it)?;
        }

        Ok(())
    }
}

impl BinaryDeserialization for Vec<u16> {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let len = r.read_u32::<LittleEndian>()? as _;
        let mut result = vec![0u16; len];
        r.read_u16_into::<LittleEndian>(&mut result)?;

        Ok(result)
    }
}

impl BinarySerialization for f32 {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_f32::<LittleEndian>(*self)
    }
}

impl BinaryDeserialization for f32 {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        r.read_f32::<LittleEndian>()
    }
}

impl BinarySerialization for glam::Vec2 {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_f32::<LittleEndian>(self.x)?;
        w.write_f32::<LittleEndian>(self.y)
    }
}

impl BinaryDeserialization for glam::Vec2 {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let x = r.read_f32::<LittleEndian>()?;
        let y = r.read_f32::<LittleEndian>()?;

        Ok(Self { x, y })
    }
}

impl BinarySerialization for glam::Vec3 {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_f32::<LittleEndian>(self.x)?;
        w.write_f32::<LittleEndian>(self.y)?;
        w.write_f32::<LittleEndian>(self.z)
    }
}

impl BinaryDeserialization for glam::Vec3 {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let x = r.read_f32::<LittleEndian>()?;
        let y = r.read_f32::<LittleEndian>()?;
        let z = r.read_f32::<LittleEndian>()?;

        Ok(Self { x, y, z })
    }
}

impl BinarySerialization for glam::Vec4 {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_f32::<LittleEndian>(self.x)?;
        w.write_f32::<LittleEndian>(self.y)?;
        w.write_f32::<LittleEndian>(self.z)?;
        w.write_f32::<LittleEndian>(self.w)
    }
}

impl BinaryDeserialization for glam::Vec4 {
    fn deserialize(r: &mut impl Read) -> io::Result<Self> {
        let x = r.read_f32::<LittleEndian>()?;
        let y = r.read_f32::<LittleEndian>()?;
        let z = r.read_f32::<LittleEndian>()?;
        let w = r.read_f32::<LittleEndian>()?;

        Ok(Self::new(x, y, z, w))
    }
}
