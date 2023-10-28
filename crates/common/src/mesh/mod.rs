mod meshdata;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;
pub use meshdata::*;
use numquant::linear::quantize;

use crate::{
    bounds::AABB,
    traits::{BinaryDeserialization, BinarySerialization},
    Transform,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct PackedVec2 {
    pub x: i16,
    pub y: i16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct PackedVec3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct PackedVec4 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    pub w: i16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct LightingAttributes {
    pub normal: PackedVec2,
    pub tangent: PackedVec2,
    pub uv: PackedVec2,
    _pad: [u16; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub base: Option<String>,
    pub normal: Option<String>,
    pub metallic_roughness: Option<String>,
    pub occlusion: Option<String>,
    pub emission: Option<String>,
    pub base_color: glam::Vec4,
    pub emission_color: glam::Vec4,
    pub emission_value: f32,
    pub metallic_value: f32,
    pub roughness_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnlitMaterial {
    pub base: Option<String>,
    pub base_color: glam::Vec4,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    Pbr(PbrMaterial),
    Unlit(UnlitMaterial),
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

fn value_to_u8(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    (value * 255.0) as u8
}

fn norm_color(color: glam::Vec4) -> [u8; 4] {
    [
        value_to_u8(color.x),
        value_to_u8(color.y),
        value_to_u8(color.z),
        value_to_u8(color.w),
    ]
}

impl BinarySerialization for PbrMaterial {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.base.serialize(w)?;
        self.normal.serialize(w)?;
        self.metallic_roughness.serialize(w)?;
        self.occlusion.serialize(w)?;
        self.emission.serialize(w)?;
        let base_color = norm_color(self.base_color);
        let emission_color = norm_color(self.emission_color);
        w.write_all(&base_color)?;
        w.write_all(&emission_color)?;
        w.write_f32::<LittleEndian>(self.emission_value)?;
        w.write_u8(value_to_u8(self.metallic_value))?;
        w.write_u8(value_to_u8(self.roughness_value))?;

        Ok(())
    }
}

impl BinaryDeserialization for PbrMaterial {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let base = Option::<String>::deserialize(r)?;
        let normal = Option::<String>::deserialize(r)?;
        let metallic_roughness = Option::<String>::deserialize(r)?;
        let occlusion = Option::<String>::deserialize(r)?;
        let emission = Option::<String>::deserialize(r)?;
        let mut base_color = [0u8; 4];
        let mut emission_color = [0u8; 4];
        r.read_exact(&mut base_color)?;
        r.read_exact(&mut emission_color)?;
        let emission_value = r.read_f32::<LittleEndian>()?;
        let metallic_value = r.read_u8()? as f32 / 255.0;
        let roughness_value = r.read_u8()? as f32 / 255.0;
        let base_color = glam::vec4(
            base_color[0] as f32 / 255.0,
            base_color[1] as f32 / 255.0,
            base_color[2] as f32 / 255.0,
            base_color[3] as f32 / 255.0,
        );
        let emission_color = glam::vec4(
            emission_color[0] as f32 / 255.0,
            emission_color[1] as f32 / 255.0,
            emission_color[2] as f32 / 255.0,
            emission_color[3] as f32 / 255.0,
        );

        Ok(Self {
            base,
            normal,
            metallic_roughness,
            occlusion,
            emission,
            base_color,
            emission_color,
            emission_value,
            metallic_value,
            roughness_value,
        })
    }
}

impl BinarySerialization for UnlitMaterial {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.base.serialize(w)?;
        let base_color = norm_color(self.base_color);
        w.write_all(&base_color)?;

        Ok(())
    }
}

impl BinaryDeserialization for UnlitMaterial {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let base = Option::<String>::deserialize(r)?;
        let mut base_color = [0u8; 4];
        r.read_exact(&mut base_color)?;
        let base_color = glam::vec4(
            base_color[0] as f32 / 255.0,
            base_color[1] as f32 / 255.0,
            base_color[2] as f32 / 255.0,
            base_color[3] as f32 / 255.0,
        );

        Ok(Self { base, base_color })
    }
}

impl BinarySerialization for Material {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            Material::Pbr(pbr) => {
                w.write_u8(0)?;
                pbr.serialize(w)?;
            }
            Material::Unlit(unlit) => {
                w.write_u8(1)?;
                unlit.serialize(w)?;
            }
        }

        Ok(())
    }
}

impl BinaryDeserialization for Material {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let ty = r.read_u8()?;
        match ty {
            0 => Ok(Self::Pbr(PbrMaterial::deserialize(r)?)),
            1 => Ok(Self::Unlit(UnlitMaterial::deserialize(r)?)),
            id => panic!("Unknown material ID {}", id),
        }
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

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct StaticMeshGeometry {
    pub position: PackedVec3,
    _padding: u16,
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
    for index in 0..values.len() / 2 {
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
