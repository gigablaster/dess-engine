mod builder;
mod cpumesh;

use std::{collections::HashMap, io};

pub use builder::{MeshBuilder, MeshLayoutBuilder};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
pub use cpumesh::*;
use four_cc::FourCC;

use crate::traits::{BinaryDeserialization, BinarySerialization};

const VEC2_HEADER: FourCC = FourCC(*b"VF2\0");
const VEC2_NORMALIZED_HEADER: FourCC = FourCC(*b"VF2N");
const VEC3_HEADER: FourCC = FourCC(*b"VF3\0");
const VEC3_NORMALIZED_HEADER: FourCC = FourCC(*b"VF3N");
const VEC4_HEADER: FourCC = FourCC(*b"VF4\0");
const VEC4_NORMALIZED_HEADER: FourCC = FourCC(*b"VF4N");

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub enum VertexAttribute {
    Vec2,
    Vec2Normalized,
    Vec3,
    Vec3Normalized,
    Vec4,
    Vec4Normalized,
}

impl VertexAttribute {
    pub fn count(&self) -> usize {
        match self {
            VertexAttribute::Vec2 | VertexAttribute::Vec2Normalized => 2,
            VertexAttribute::Vec3 | VertexAttribute::Vec3Normalized => 3,
            VertexAttribute::Vec4 | VertexAttribute::Vec4Normalized => 4,
        }
    }
}

pub type MeshLayout = Vec<(String, VertexAttribute, u32)>;

impl BinarySerialization for (String, VertexAttribute, u32) {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        let (name, attr, offset) = self;
        name.serialize(w)?;
        match attr {
            VertexAttribute::Vec2 => VEC2_HEADER,
            VertexAttribute::Vec2Normalized => VEC2_NORMALIZED_HEADER,
            VertexAttribute::Vec3 => VEC3_HEADER,
            VertexAttribute::Vec3Normalized => VEC3_NORMALIZED_HEADER,
            VertexAttribute::Vec4 => VEC4_HEADER,
            VertexAttribute::Vec4Normalized => VEC4_NORMALIZED_HEADER,
        }
        .serialize(w)?;
        w.write_u32::<LittleEndian>(*offset)?;

        Ok(())
    }
}

impl BinaryDeserialization for (String, VertexAttribute, u32) {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let name = String::deserialize(r)?;
        let attr = match FourCC::deserialize(r)? {
            VEC2_HEADER => VertexAttribute::Vec2,
            VEC2_NORMALIZED_HEADER => VertexAttribute::Vec2Normalized,
            VEC3_HEADER => VertexAttribute::Vec3,
            VEC3_NORMALIZED_HEADER => VertexAttribute::Vec3Normalized,
            VEC4_HEADER => VertexAttribute::Vec4,
            VEC4_NORMALIZED_HEADER => VertexAttribute::Vec4Normalized,
            other => panic!("Unknown vertex attribute {}", other),
        };
        let offset = r.read_u32::<LittleEndian>()?;

        Ok((name, attr, offset))
    }
}

#[derive(Debug)]
pub struct EffectInfo {
    pub name: String,
    pub textures: HashMap<String, String>,
}

impl EffectInfo {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            textures: HashMap::new(),
        }
    }

    pub fn add_texture(&mut self, slot: &str, texture: &str) {
        self.textures.insert(slot.into(), texture.into());
    }
}

impl BinaryDeserialization for EffectInfo {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let name = String::deserialize(r)?;
        let textures = HashMap::deserialize(r)?;

        Ok(Self { name, textures })
    }
}

impl BinarySerialization for EffectInfo {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.name.serialize(w)?;
        self.textures.serialize(w)?;

        Ok(())
    }
}
