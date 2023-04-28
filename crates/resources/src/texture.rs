use std::io;

use ash::vk;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use common::traits::{BinaryDeserialization, BinarySerialization};

pub struct TextureMip {
    pub data: Vec<u8>,
}

impl BinarySerialization for TextureMip {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.data.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for TextureMip {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let data = Vec::<u8>::deserialize(r)?;
        Ok(Self { data })
    }
}

pub struct TextureLevel {
    mips: Vec<TextureMip>,
}

impl BinarySerialization for TextureLevel {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.mips.serialize(w)
    }
}

impl BinaryDeserialization for TextureLevel {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        Ok(Self {
            mips: Vec::<TextureMip>::deserialize(r)?,
        })
    }
}

pub struct TextureResource {
    pub format: vk::Format,
    pub width: u16,
    pub height: u16,
    pub levels: Vec<TextureLevel>,
}

impl BinarySerialization for TextureResource {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        w.write_i32::<LittleEndian>(self.format.as_raw())?;
        w.write_u16::<LittleEndian>(self.width)?;
        w.write_u16::<LittleEndian>(self.height)?;
        self.levels.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for TextureResource {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        Ok(Self {
            format: vk::Format::from_raw(r.read_i32::<LittleEndian>()?),
            width: r.read_u16::<LittleEndian>()?,
            height: r.read_u16::<LittleEndian>()?,
            levels: Vec::<TextureLevel>::deserialize(r)?,
        })
    }
}
