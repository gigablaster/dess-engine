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

use crate::traits::{BinaryDeserialization, BinarySerialization};

pub trait MaterialBaseColor {
    fn set_base_texture(&mut self, texture: Option<String>);
    fn set_base_color(&mut self, color: glam::Vec4);
}

pub trait MaterialNormals {
    fn set_normal_texture(&mut self, texture: Option<String>);
}

pub trait MaterialValues {
    fn set_metallic_roughness_texture(&mut self, texture: Option<String>);
    fn set_metallic_value(&mut self, value: f32);
    fn set_roughness_value(&mut self, value: f32);
}

pub trait MaterialOcclusion {
    fn set_occlusion_texture(&mut self, texture: Option<String>);
}

pub trait MaterialEmission {
    fn set_emission_texture(&mut self, texture: Option<String>);
    fn set_emission_color(&mut self, value: glam::Vec3);
    fn set_emission_value(&mut self, value: f32);
}

pub trait MaterialBlend {
    fn set_blend_mode(&mut self, value: BlendMode);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    Opaque,
    AlphaTest(f32),
    AlphaBlend,
}

impl BinarySerialization for BlendMode {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            Self::Opaque => w.write_u8(0)?,
            Self::AlphaTest(cutoff) => {
                w.write_u8(1)?;
                w.write_u8(value_to_u8(*cutoff))?;
            }
            Self::AlphaBlend => w.write_u8(2)?,
        }

        Ok(())
    }
}

impl BinaryDeserialization for BlendMode {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let ty = r.read_u8()?;
        match ty {
            0 => Ok(Self::Opaque),
            1 => Ok(Self::AlphaTest(r.read_u8()? as f32 / 255.0)),
            2 => Ok(Self::AlphaBlend),
            val => panic!("Unknown blend mode {}", val),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub blend: BlendMode,
    pub base: Option<String>,
    pub normal: Option<String>,
    pub metallic_roughness: Option<String>,
    pub occlusion: Option<String>,
    pub emission: Option<String>,
    pub base_color: glam::Vec4,
    pub emission_color: glam::Vec3,
    pub emission_value: f32,
    pub metallic_value: f32,
    pub roughness_value: f32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            blend: BlendMode::Opaque,
            base: None,
            normal: None,
            metallic_roughness: None,
            occlusion: None,
            emission: None,
            base_color: glam::vec4(0.5, 0.5, 0.5, 1.0),
            emission_color: glam::Vec3::ZERO,
            emission_value: 0.0,
            metallic_value: 0.0,
            roughness_value: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnlitMaterial {
    pub blend: BlendMode,
    pub base: Option<String>,
    pub base_color: glam::Vec4,
}

impl Default for UnlitMaterial {
    fn default() -> Self {
        Self {
            blend: BlendMode::Opaque,
            base: None,
            base_color: glam::vec4(0.5, 0.5, 0.5, 1.0),
        }
    }
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

impl MaterialBaseColor for PbrMaterial {
    fn set_base_color(&mut self, color: glam::Vec4) {
        self.base_color = color;
    }

    fn set_base_texture(&mut self, texture: Option<String>) {
        self.base = texture;
    }
}

impl MaterialBaseColor for UnlitMaterial {
    fn set_base_color(&mut self, color: glam::Vec4) {
        self.base_color = color;
    }

    fn set_base_texture(&mut self, texture: Option<String>) {
        self.base = texture;
    }
}

impl MaterialBlend for UnlitMaterial {
    fn set_blend_mode(&mut self, value: BlendMode) {
        self.blend = value;
    }
}

impl MaterialValues for PbrMaterial {
    fn set_metallic_value(&mut self, value: f32) {
        self.metallic_value = value;
    }

    fn set_roughness_value(&mut self, value: f32) {
        self.roughness_value = value;
    }

    fn set_metallic_roughness_texture(&mut self, texture: Option<String>) {
        self.metallic_roughness = texture;
    }
}

impl MaterialNormals for PbrMaterial {
    fn set_normal_texture(&mut self, texture: Option<String>) {
        self.normal = texture;
    }
}

impl MaterialOcclusion for PbrMaterial {
    fn set_occlusion_texture(&mut self, texture: Option<String>) {
        self.occlusion = texture;
    }
}

impl MaterialEmission for PbrMaterial {
    fn set_emission_texture(&mut self, texture: Option<String>) {
        self.emission = texture;
    }

    fn set_emission_color(&mut self, value: glam::Vec3) {
        self.emission_color = value;
    }

    fn set_emission_value(&mut self, value: f32) {
        self.emission_value = value;
    }
}

impl MaterialBlend for PbrMaterial {
    fn set_blend_mode(&mut self, value: BlendMode) {
        self.blend = value;
    }
}

fn norm_color4(color: glam::Vec4) -> [u8; 4] {
    [
        value_to_u8(color.x),
        value_to_u8(color.y),
        value_to_u8(color.z),
        value_to_u8(color.w),
    ]
}

fn norm_color3(color: glam::Vec3) -> [u8; 3] {
    [
        value_to_u8(color.x),
        value_to_u8(color.y),
        value_to_u8(color.z),
    ]
}

impl BinarySerialization for PbrMaterial {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.blend.serialize(w)?;
        self.base.serialize(w)?;
        self.normal.serialize(w)?;
        self.metallic_roughness.serialize(w)?;
        self.occlusion.serialize(w)?;
        self.emission.serialize(w)?;
        let base_color = norm_color4(self.base_color);
        let emission_color = norm_color3(self.emission_color);
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
        let blend = BlendMode::deserialize(r)?;
        let base = Option::<String>::deserialize(r)?;
        let normal = Option::<String>::deserialize(r)?;
        let metallic_roughness = Option::<String>::deserialize(r)?;
        let occlusion = Option::<String>::deserialize(r)?;
        let emission = Option::<String>::deserialize(r)?;
        let mut base_color = [0u8; 4];
        let mut emission_color = [0u8; 3];
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
        let emission_color = glam::vec3(
            emission_color[0] as f32 / 255.0,
            emission_color[1] as f32 / 255.0,
            emission_color[2] as f32 / 255.0,
        );

        Ok(Self {
            blend,
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
        self.blend.serialize(w)?;
        self.base.serialize(w)?;
        let base_color = norm_color4(self.base_color);
        w.write_all(&base_color)?;

        Ok(())
    }
}

impl BinaryDeserialization for UnlitMaterial {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let blend = BlendMode::deserialize(r)?;
        let base = Option::<String>::deserialize(r)?;
        let mut base_color = [0u8; 4];
        r.read_exact(&mut base_color)?;
        let base_color = glam::vec4(
            base_color[0] as f32 / 255.0,
            base_color[1] as f32 / 255.0,
            base_color[2] as f32 / 255.0,
            base_color[3] as f32 / 255.0,
        );

        Ok(Self {
            blend,
            base,
            base_color,
        })
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
