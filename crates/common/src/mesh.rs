use std::{collections::HashMap, io};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use four_cc::FourCC;
use numquant::{linear::quantize, IntRange, Quantized};

use crate::traits::{BinaryDeserialization, BinarySerialization};

pub const MAX_ATTRIBUTES: usize = 8;

const SINGLE_HEADER: FourCC = FourCC(*b"SNGL");
const VEC_HEADER: FourCC = FourCC(*b"VEC4");
const STATIC_MESH_HEADER: FourCC = FourCC(*b"DESM");

pub const OBJECT_SCALE_PARAM: &str = "object_scale";
pub const UV_SCALE_PARAM: &str = "uv_scale";
pub const BASE_COLOR_TEXTURE: &str = "base";
pub const NORMAL_MAP_TEXTURE: &str = "nomrmal";
pub const METALLIC_ROUGHNESS_TEXTURE: &str = "metallic_roughness";
pub const OCCLUSION_TEXTURE: &str = "occlusion";

/// Packed vertx to store mesh data on the disk
///
/// We use normalized and quantized values to keep mesh small and to
/// use most of bits and keep precision. We only need two values for
/// normals and tangents because those values are always have length 1
/// and we can just restore z in vertex shader.
#[derive(Debug)]
#[repr(C)]
pub struct PackedVertex {
    pub position: [i16; 3],
    _pad: u16,
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [i16; 2],
}

impl BinarySerialization for PackedVertex {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        for pos in self.position {
            w.write_i16::<LittleEndian>(pos)?;
        }
        for normal in self.normal {
            w.write_i16::<LittleEndian>(normal)?;
        }
        for tangent in self.tangent {
            w.write_i16::<LittleEndian>(tangent)?;
        }
        for uv in self.uv {
            w.write_i16::<LittleEndian>(uv)?;
        }

        Ok(())
    }
}

impl BinaryDeserialization for PackedVertex {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut data = [0i16; 9];
        r.read_i16_into::<LittleEndian>(&mut data)?;

        Ok(Self {
            position: [data[0], data[1], data[2]],
            normal: [data[3], data[4]],
            tangent: [data[5], data[6]],
            uv: [data[7], data[8]],
            _pad: 0,
        })
    }
}

impl PackedVertex {
    fn new(position: [i16; 3], normal: [i16; 2], tangent: [i16; 2], uv: [i16; 2]) -> Self {
        Self {
            position,
            normal,
            tangent,
            uv,
            _pad: 0,
        }
    }
}

#[derive(Debug)]
pub enum EffectValue {
    Single(f32),
    Vec(glam::Vec4),
}

impl BinarySerialization for EffectValue {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        match self {
            Self::Single(value) => {
                SINGLE_HEADER.serialize(w)?;
                w.write_f32::<LittleEndian>(*value)
            }
            Self::Vec(value) => {
                VEC_HEADER.serialize(w)?;
                value.serialize(w)
            }
        }
    }
}

impl BinaryDeserialization for EffectValue {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let ty = FourCC::deserialize(r)?;
        match ty {
            SINGLE_HEADER => Ok(Self::Single(r.read_f32::<LittleEndian>()?)),
            VEC_HEADER => Ok(Self::Vec(glam::Vec4::deserialize(r)?)),
            ty => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Unknown value type {:?}", ty),
            )),
        }
    }
}

#[derive(Debug)]
pub struct EffectInfo {
    pub name: String,
    pub textures: HashMap<String, String>,
    pub values: HashMap<String, EffectValue>,
}

impl EffectInfo {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            textures: HashMap::new(),
            values: HashMap::new(),
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
        let values = HashMap::deserialize(r)?;

        Ok(Self {
            name,
            textures,
            values,
        })
    }
}

impl BinarySerialization for EffectInfo {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.name.serialize(w)?;
        self.textures.serialize(w)?;
        self.values.serialize(w)
    }
}

#[derive(Debug)]
pub struct Surface {
    pub first_vertex: u32,
    pub first_index: u32,
    pub index_count: u32,
    pub effect: EffectInfo,
}

impl BinarySerialization for Surface {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u32::<LittleEndian>(self.first_vertex)?;
        w.write_u32::<LittleEndian>(self.first_index)?;
        w.write_u32::<LittleEndian>(self.index_count)?;
        self.effect.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for Surface {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let first_vertex = r.read_u32::<LittleEndian>()?;
        let first_index = r.read_u32::<LittleEndian>()?;
        let index_count = r.read_u32::<LittleEndian>()?;
        let effect = EffectInfo::deserialize(r)?;

        Ok(Self {
            first_vertex,
            first_index,
            index_count,
            effect,
        })
    }
}

#[derive(Debug, Default)]
pub struct StaticMesh {
    surfaces: Vec<Surface>,
    vertices: Vec<PackedVertex>,
    indices: Vec<u16>,
}

impl BinarySerialization for StaticMesh {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        STATIC_MESH_HEADER.serialize(w)?;
        self.surfaces.serialize(w)?;
        self.vertices.serialize(w)?;
        self.indices.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for StaticMesh {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        if FourCC::deserialize(r)? != STATIC_MESH_HEADER {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Wrong static mesh header",
            ));
        }
        let surfaces = Vec::<Surface>::deserialize(r)?;
        let vertices = Vec::<PackedVertex>::deserialize(r)?;
        let indices = Vec::<u16>::deserialize(r)?;

        Ok(Self {
            surfaces,
            vertices,
            indices,
        })
    }
}

type I16<const MIN: i64, const MAX: i64> = IntRange<i16, 0xffff, MIN, MAX>;

impl StaticMesh {
    pub fn add_surface(
        &mut self,
        positions: &[glam::Vec3],
        normals: &[glam::Vec3],
        tangents: &[glam::Vec3],
        uvs: &[glam::Vec2],
        indices: &[u16],
        effect: EffectInfo,
    ) {
        assert!(!positions.is_empty());
        assert!(indices.len() >= 3);
        assert_eq!(0, indices.len() % 3);
        assert_eq!(positions.len(), normals.len());
        assert_eq!(normals.len(), tangents.len());
        assert_eq!(tangents.len(), uvs.len());

        let max_position = Self::find_biggest_value_vec3(positions) as f64;
        let max_uv = Self::find_biggest_value_vec2(uvs) as f64;
        let first_vertex = self.vertices.len() as _;
        let first_index = self.indices.len() as _;
        let mut vertices = positions
            .iter()
            .zip(normals.iter())
            .zip(tangents.iter())
            .zip(uvs.iter())
            .map(|(((position, normal), tangent), uv)| {
                let px = quantize(position.x as f64, -max_position..max_position, i16::MAX);
                let py = quantize(position.y as f64, -max_position..max_position, i16::MAX);
                let pz = quantize(position.z as f64, -max_position..max_position, i16::MAX);
                let nx = Quantized::<I16<-1, 1>>::from_f32(normal.x).raw();
                let ny = Quantized::<I16<-1, 1>>::from_f32(normal.y).raw();
                let tx = Quantized::<I16<-1, 1>>::from_f32(tangent.x).raw();
                let ty = Quantized::<I16<-1, 1>>::from_f32(tangent.y).raw();
                let u = quantize(uv.x as f64, -max_uv..max_uv, i16::MAX);
                let v = quantize(uv.y as f64, -max_uv..max_uv, i16::MAX);

                PackedVertex::new([px, py, pz], [nx, ny], [tx, ty], [u, v])
            })
            .collect::<Vec<_>>();
        self.vertices.append(&mut vertices);
        self.indices.append(&mut indices.to_vec());
        let mut effect = effect;
        effect.values.insert(
            OBJECT_SCALE_PARAM.into(),
            EffectValue::Vec(glam::Vec4::new(
                max_position as _,
                max_position as _,
                max_position as _,
                0.0,
            )),
        );
        effect.values.insert(
            UV_SCALE_PARAM.into(),
            EffectValue::Vec(glam::Vec4::new(max_uv as _, max_uv as _, 0.0, 0.0)),
        );

        self.surfaces.push(Surface {
            first_vertex,
            first_index,
            index_count: indices.len() as _,
            effect,
        })
    }

    fn find_biggest_value_vec3(data: &[glam::Vec3]) -> f32 {
        data.iter()
            .map(|x| x.abs())
            .map(|x| [x.x, x.y, x.z].into_iter().reduce(Self::biggest).unwrap())
            .reduce(Self::biggest)
            .unwrap()
    }

    fn find_biggest_value_vec2(data: &[glam::Vec2]) -> f32 {
        data.iter()
            .map(|x| x.abs())
            .map(|x| [x.x, x.y].into_iter().reduce(Self::biggest).unwrap())
            .reduce(Self::biggest)
            .unwrap()
    }

    fn biggest(x: f32, y: f32) -> f32 {
        if x > y {
            x
        } else {
            y
        }
    }
}
