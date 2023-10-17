use std::io;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::traits::{BinaryDeserialization, BinarySerialization};

use super::CpuMesh;

#[derive(Debug)]
pub struct CpuModelBone {
    pub local_tr: glam::Mat4,
    pub global_tr: glam::Mat4,
    pub parent: Option<usize>,
    pub name: String,
}

impl BinarySerialization for CpuModelBone {
    fn serialize(&self, w: &mut impl io::Write) -> std::io::Result<()> {
        self.local_tr.serialize(w)?;
        self.global_tr.serialize(w)?;
        if let Some(parent) = self.parent {
            w.write_u8(0)?;
            w.write_u32::<LittleEndian>(parent as _)?;
        } else {
            w.write_u8(u8::MAX)?;
        }
        self.name.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for CpuModelBone {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let local_tr = glam::Mat4::deserialize(r)?;
        let global_tr = glam::Mat4::deserialize(r)?;
        let parent = if r.read_u8()? == 0 {
            let parent = r.read_u32::<LittleEndian>()? as usize;
            Some(parent)
        } else {
            None
        };
        let name = String::deserialize(r)?;

        Ok(Self {
            local_tr,
            global_tr,
            parent,
            name,
        })
    }
}

#[derive(Debug, Default)]
pub struct CpuModel {
    pub meshes: Vec<CpuMesh>,
    pub bones: Vec<CpuModelBone>,
}

impl BinarySerialization for CpuModel {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.meshes.serialize(w)?;
        self.bones.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for CpuModel {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let meshes = Vec::deserialize(r)?;
        let bones = Vec::deserialize(r)?;

        Ok(Self { meshes, bones })
    }
}
