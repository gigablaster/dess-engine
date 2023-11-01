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
use dess_common::traits::{BinaryDeserialization, BinarySerialization};

use crate::Asset;

#[derive(Debug)]
pub struct GpuImage {
    pub format: vk::Format,
    pub dimensions: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl Asset for GpuImage {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("c2871b90-6b51-427f-b1d8-4cedbedc8993");
    fn collect_dependencies(&self, _deps: &mut Vec<crate::AssetRef>) {}
}
impl BinarySerialization for GpuImage {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        w.write_i32::<LittleEndian>(self.format.as_raw())?;
        w.write_u32::<LittleEndian>(self.dimensions[0])?;
        w.write_u32::<LittleEndian>(self.dimensions[1])?;
        self.mips.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for GpuImage {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let format = vk::Format::from_raw(r.read_i32::<LittleEndian>()?);
        let mut dimensions = [0u32; 2];
        r.read_u32_into::<LittleEndian>(&mut dimensions)?;
        let mips = Vec::<Vec<_>>::deserialize(r)?;

        Ok(Self {
            format,
            dimensions,
            mips,
        })
    }
}
