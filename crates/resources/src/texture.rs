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
