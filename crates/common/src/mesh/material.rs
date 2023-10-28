use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::traits::{BinaryDeserialization, BinarySerialization};

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
