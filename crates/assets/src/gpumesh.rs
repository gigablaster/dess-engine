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
use dess_common::{
    bounds::AABB,
    traits::{BinaryDeserialization, BinarySerialization},
    Transform,
};

use crate::{material::Material, AssetRef};

pub trait Geometry: BinarySerialization + BinaryDeserialization + Copy {}

type PackedVec2 = [i16; 2];
type PackedVec3 = [i16; 3];

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct LightingAttributes {
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [i16; 2],
    _pad: [i16; 2],
}

impl LightingAttributes {
    pub fn new(normal: [i16; 2], tangent: [i16; 2], uv: [i16; 2]) -> Self {
        Self {
            normal,
            tangent,
            uv,
            _pad: [0, 0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bone {
    pub parent: u32,
    pub local_tr: Transform,
}

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
pub struct GpuMesh<T: Geometry> {
    pub geometry: Vec<T>, // w is padding
    pub attributes: Vec<LightingAttributes>,
    pub indices: Vec<u16>,
    pub surfaces: Vec<Surface>,
}

impl<T: Geometry> BinarySerialization for GpuMesh<T> {
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

impl<T: Geometry> BinaryDeserialization for GpuMesh<T> {
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

impl BinarySerialization for LightingAttributes {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.normal.serialize(w)?;
        self.tangent.serialize(w)?;
        self.uv.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for LightingAttributes {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let normal = PackedVec2::deserialize(r)?;
        let tangent = PackedVec2::deserialize(r)?;
        let uv = PackedVec2::deserialize(r)?;

        Ok(Self {
            normal,
            tangent,
            uv,
            _pad: [0, 0],
        })
    }
}

impl BinarySerialization for Bone {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.parent)?;
        self.local_tr.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for Bone {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let parent = r.read_u32::<LittleEndian>()?;
        let local_tr = Transform::deserialize(r)?;

        Ok(Self { parent, local_tr })
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct StaticMeshGeometry {
    pub position: [i16; 3],
    _padding: u16,
}

impl StaticMeshGeometry {
    pub fn new(position: [i16; 3]) -> Self {
        Self {
            position,
            _padding: 0,
        }
    }
}

impl BinarySerialization for StaticMeshGeometry {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.position.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for StaticMeshGeometry {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let position = PackedVec3::deserialize(r)?;

        Ok(Self {
            position,
            _padding: 0,
        })
    }
}

impl Geometry for StaticMeshGeometry {}

pub type StaticGpuMesh = GpuMesh<StaticMeshGeometry>;

impl Surface {
    pub(crate) fn collect_dependencies(&self, deps: &mut Vec<AssetRef>) {
        self.material.collect_dependencies(deps);
    }
}

impl<T:Geometry> GpuMesh<T> {
    pub(crate) fn collect_dependencies(&self, deps: &mut Vec<AssetRef>) {
        self.surfaces.iter().for_each(|x| x.collect_dependencies(deps));
    }
}
