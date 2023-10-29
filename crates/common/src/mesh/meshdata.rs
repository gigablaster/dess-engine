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

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::{
    bounds::AABB,
    traits::{BinaryDeserialization, BinarySerialization},
};

use super::{Bone, LightingAttributes, Material};

pub trait Geometry: BinarySerialization + BinaryDeserialization + Copy {}

#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub first: u32,
    pub count: u32,
    pub bounds: AABB,
    pub max_position_value: f32,
    pub max_uv_value: f32,
    pub material: Material,
}

impl BinarySerialization for Surface {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.first)?;
        w.write_u32::<LittleEndian>(self.count)?;
        w.write_f32::<LittleEndian>(self.max_position_value)?;
        w.write_f32::<LittleEndian>(self.max_uv_value)?;
        self.bounds.serialize(w)?;
        self.material.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for Surface {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let first = r.read_u32::<LittleEndian>()?;
        let count = r.read_u32::<LittleEndian>()?;
        let max_position_value = r.read_f32::<LittleEndian>()?;
        let max_uv_value = r.read_f32::<LittleEndian>()?;
        let bounds = AABB::deserialize(r)?;
        let material = Material::deserialize(r)?;

        Ok(Self {
            first,
            count,
            max_position_value,
            max_uv_value,
            bounds,
            material,
        })
    }
}

#[derive(Debug, Default)]
pub struct MeshData<T: Geometry> {
    pub geometry: Vec<T>, // w is padding
    pub attributes: Vec<LightingAttributes>,
    pub indices: Vec<u16>,
    pub surfaces: Vec<Surface>,
}

impl<T: Geometry> BinarySerialization for MeshData<T> {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.geometry.len() as _)?;
        for geo in &self.geometry {
            geo.serialize(w)?;
        }
        for attr in &self.attributes {
            attr.serialize(w)?;
        }
        self.indices.serialize(w)?;
        self.surfaces.serialize(w)?;

        Ok(())
    }
}

impl<T: Geometry> BinaryDeserialization for MeshData<T> {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let vertex_count = r.read_u32::<LittleEndian>()?;
        let mut geometry = Vec::with_capacity(vertex_count as _);
        let mut attributes = Vec::with_capacity(vertex_count as _);
        for _ in 0..vertex_count {
            geometry.push(T::deserialize(r)?);
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
            geometry,
            attributes,
            indices,
            surfaces,
        })
    }
}
