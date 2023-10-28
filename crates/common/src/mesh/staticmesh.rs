use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;

use crate::traits::{BinaryDeserialization, BinarySerialization};

use super::{Bone, LightingAttributes, PackedVec3, PackedVec4, Surface};

const MAGICK: FourCC = FourCC(*b"STMS");

#[derive(Debug)]
pub struct StaticMesh {
    pub positions: Vec<PackedVec4>, // w is padding
    pub attributes: Vec<LightingAttributes>,
    pub indices: Vec<u16>,
    pub surfaces: Vec<Surface>,
    pub bones: Vec<Bone>,
    pub bone_names: Vec<String>,
}

impl BinarySerialization for StaticMesh {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        MAGICK.serialize(w)?;
        w.write_u32::<LittleEndian>(self.positions.len() as _)?;
        for pos in &self.positions {
            let pos = PackedVec3 {
                x: pos.x,
                y: pos.y,
                z: pos.z,
            };
            pos.serialize(w)?;
        }
        for attr in &self.attributes {
            attr.serialize(w)?;
        }
        self.indices.serialize(w)?;
        self.surfaces.serialize(w)?;
        w.write_u32::<LittleEndian>(self.bones.len() as _)?;
        for bone in &self.bones {
            bone.serialize(w)?;
        }
        for bone_name in &self.bone_names {
            bone_name.serialize(w)?;
        }

        Ok(())
    }
}

impl BinaryDeserialization for StaticMesh {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let magic = FourCC::deserialize(r)?;
        if magic != MAGICK {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Wrong mesh header",
            ));
        }
        let vertex_count = r.read_u32::<LittleEndian>()?;
        let mut positions = Vec::with_capacity(vertex_count as _);
        let mut attributes = Vec::with_capacity(vertex_count as _);
        for _ in 0..vertex_count {
            let pos = PackedVec3::deserialize(r)?;
            positions.push(PackedVec4 {
                x: pos.x,
                y: pos.y,
                z: pos.z,
                w: 0,
            });
        }
        for _ in 0..vertex_count {
            attributes.push(LightingAttributes::deserialize(r)?);
        }
        let indices = Vec::deserialize(r)?;
        let surfaces = Vec::deserialize(r)?;
        let bone_count = r.read_u32::<LittleEndian>()?;
        let mut bones = Vec::with_capacity(bone_count as _);
        let mut bone_names = Vec::with_capacity(bone_count as _);
        for _ in 0..bone_count {
            bones.push(Bone::deserialize(r)?);
        }
        for _ in 0..bone_count {
            bone_names.push(String::deserialize(r)?);
        }

        Ok(Self {
            positions,
            attributes,
            indices,
            surfaces,
            bones,
            bone_names,
        })
    }
}
