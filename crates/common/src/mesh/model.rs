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

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;

use crate::traits::{BinaryDeserialization, BinarySerialization};

use super::{Bone, StaticMeshData};

const MAGIC: FourCC = FourCC(*b"MODL");

#[derive(Debug, Default)]
pub struct ModelData {
    pub static_meshes: Vec<StaticMeshData>,
    pub mesh_names: HashMap<String, u32>,
    pub bones: Vec<Bone>,
    pub names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

impl BinarySerialization for (u32, u32) {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.0)?;
        w.write_u32::<LittleEndian>(self.1)?;

        Ok(())
    }
}

impl BinaryDeserialization for (u32, u32) {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let a = r.read_u32::<LittleEndian>()?;
        let b = r.read_u32::<LittleEndian>()?;

        Ok((a, b))
    }
}

impl BinarySerialization for u32 {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(*self)?;

        Ok(())
    }
}

impl BinaryDeserialization for u32 {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        r.read_u32::<LittleEndian>()
    }
}

impl BinarySerialization for ModelData {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        MAGIC.serialize(w)?;
        self.static_meshes.serialize(w)?;
        self.mesh_names.serialize(w)?;
        self.bones.serialize(w)?;
        self.names.serialize(w)?;
        self.node_to_mesh.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for ModelData {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        if MAGIC != FourCC::deserialize(r)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Wrong model header",
            ));
        }
        let static_meshes = Vec::deserialize(r)?;
        let mesh_names = HashMap::deserialize(r)?;
        let bones = Vec::deserialize(r)?;
        let names = HashMap::deserialize(r)?;
        let node_to_mesh = Vec::deserialize(r)?;

        Ok(Self {
            static_meshes,
            mesh_names,
            bones,
            names,
            node_to_mesh,
        })
    }
}
