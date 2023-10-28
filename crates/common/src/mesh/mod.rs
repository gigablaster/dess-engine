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

mod material;
mod meshdata;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;
pub use material::*;
pub use meshdata::*;
use numquant::linear::quantize;

use crate::{
    bounds::AABB,
    traits::{BinaryDeserialization, BinarySerialization},
    Transform,
};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, packed)]
pub struct PackedVec2 {
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, packed)]
pub struct PackedVec3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, packed)]
pub struct PackedVec4 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    pub w: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, packed)]
pub struct LightingAttributes {
    pub normal: PackedVec2,
    pub tangent: PackedVec2,
    pub uv: PackedVec2,
    _pad: [u16; 2],
}

impl LightingAttributes {
    pub fn new(normal: PackedVec2, tangent: PackedVec2, uv: PackedVec2) -> Self {
        Self {
            normal,
            tangent,
            uv,
            _pad: [0, 0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub first: u32,
    pub count: u32,
    pub bone: u32,
    pub bounds: AABB,
    pub max_position_value: f32,
    pub max_uv_value: f32,
    pub material: Material,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bone {
    pub parent: u32,
    pub local_tr: Transform,
}

impl BinarySerialization for PackedVec2 {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_i16::<LittleEndian>(self.x)?;
        w.write_i16::<LittleEndian>(self.y)?;

        Ok(())
    }
}

impl BinaryDeserialization for PackedVec2 {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let x = r.read_i16::<LittleEndian>()?;
        let y = r.read_i16::<LittleEndian>()?;

        Ok(Self { x, y })
    }
}

impl BinarySerialization for PackedVec3 {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_i16::<LittleEndian>(self.x)?;
        w.write_i16::<LittleEndian>(self.y)?;
        w.write_i16::<LittleEndian>(self.z)?;

        Ok(())
    }
}

impl BinaryDeserialization for PackedVec3 {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let x = r.read_i16::<LittleEndian>()?;
        let y = r.read_i16::<LittleEndian>()?;
        let z = r.read_i16::<LittleEndian>()?;

        Ok(Self { x, y, z })
    }
}

impl BinarySerialization for PackedVec4 {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_i16::<LittleEndian>(self.x)?;
        w.write_i16::<LittleEndian>(self.y)?;
        w.write_i16::<LittleEndian>(self.z)?;
        w.write_i16::<LittleEndian>(self.w)?;

        Ok(())
    }
}

impl BinaryDeserialization for PackedVec4 {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let x = r.read_i16::<LittleEndian>()?;
        let y = r.read_i16::<LittleEndian>()?;
        let z = r.read_i16::<LittleEndian>()?;
        let w = r.read_i16::<LittleEndian>()?;

        Ok(Self { x, y, z, w })
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

impl BinarySerialization for Surface {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.first)?;
        w.write_u32::<LittleEndian>(self.count)?;
        w.write_u32::<LittleEndian>(self.bone)?;
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
        let bone = r.read_u32::<LittleEndian>()?;
        let max_position_value = r.read_f32::<LittleEndian>()?;
        let max_uv_value = r.read_f32::<LittleEndian>()?;
        let bounds = AABB::deserialize(r)?;
        let material = Material::deserialize(r)?;

        Ok(Self {
            first,
            count,
            bone,
            max_position_value,
            max_uv_value,
            bounds,
            material,
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
#[repr(C, packed)]
pub struct StaticMeshGeometry {
    pub position: PackedVec3,
    _padding: u16,
}

impl StaticMeshGeometry {
    pub fn new(position: PackedVec3) -> Self {
        Self {
            position,
            _padding: 0,
        }
    }
}

impl BinarySerialization for StaticMeshGeometry {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.position.serialize(w)
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

impl Geometry for StaticMeshGeometry {
    const MAGICK: FourCC = FourCC(*b"STMS");
}

pub type StaticMeshData = MeshData<StaticMeshGeometry>;

trait Quantizer<T> {
    const COUNT: usize;
    fn write(&self, out: &mut Vec<f32>);
    fn pack(data: &[i16]) -> T;
}

impl Quantizer<PackedVec2> for [f32; 2] {
    const COUNT: usize = 2;

    fn write(&self, out: &mut Vec<f32>) {
        out.push(self[0]);
        out.push(self[1]);
    }

    fn pack(data: &[i16]) -> PackedVec2 {
        PackedVec2 {
            x: data[0],
            y: data[1],
        }
    }
}

impl Quantizer<PackedVec3> for [f32; 3] {
    const COUNT: usize = 3;

    fn write(&self, out: &mut Vec<f32>) {
        out.push(self[0]);
        out.push(self[1]);
        out.push(self[2]);
    }

    fn pack(data: &[i16]) -> PackedVec3 {
        PackedVec3 {
            x: data[0],
            y: data[1],
            z: data[2],
        }
    }
}

fn quantize_values(data: &[f32]) -> (f32, Vec<i16>) {
    let max = data
        .iter()
        .max_by(|x, y| x.abs().total_cmp(&y.abs()))
        .copied()
        .unwrap_or(0.0) as f64;
    let result = data
        .iter()
        .map(|x| quantize(*x as _, -max..max, i16::MAX))
        .collect::<Vec<_>>();

    (max as f32, result)
}

fn quantize_normalized_values(data: &[f32]) -> Vec<i16> {
    let result = data
        .iter()
        .map(|x| quantize(*x as _, -1.0..1.0, i16::MAX))
        .collect::<Vec<_>>();

    result
}

fn quantize_input<T, U: Quantizer<T>>(input: &[U]) -> (f32, Vec<T>) {
    let mut data = Vec::with_capacity(input.len() * U::COUNT);
    input.iter().for_each(|x| x.write(&mut data));
    let (max, values) = quantize_values(&data);
    let mut result = Vec::with_capacity(input.len());
    for index in 0..values.len() / 3 {
        let start = index * U::COUNT;
        let value = &values[start..start + U::COUNT];
        let vec = U::pack(value);
        result.push(vec);
    }

    (max, result)
}

pub fn quantize_positions(input: &[[f32; 3]]) -> (f32, Vec<PackedVec3>) {
    quantize_input(input)
}

pub fn quantize_uvs(input: &[[f32; 2]]) -> (f32, Vec<PackedVec2>) {
    quantize_input(input)
}

pub fn quantize_normalized(input: &[[f32; 3]]) -> Vec<PackedVec2> {
    let mut data = Vec::with_capacity(input.len() * 3);
    input.iter().for_each(|x| x.write(&mut data));
    let values = quantize_normalized_values(&data);
    let mut result = Vec::with_capacity(input.len());
    for index in 0..values.len() / 3 {
        let start = index * 3;
        let value = &values[start..start + 3];
        let vec = PackedVec2 {
            x: value[0],
            y: value[1],
        };
        result.push(vec);
    }

    result
}
