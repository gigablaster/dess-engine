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
use dess_common::traits::{BinaryDeserialization, BinarySerialization};

use crate::AssetRef;

pub trait MaterialBaseColor {
    fn set_base_texture(&mut self, texture: AssetRef);
}

pub trait MaterialNormals {
    fn set_normal_texture(&mut self, texture: AssetRef);
}

pub trait MaterialValues {
    fn set_metallic_roughness_texture(&mut self, texture: AssetRef);
}

pub trait MaterialOcclusion {
    fn set_occlusion_texture(&mut self, texture: AssetRef);
}

pub trait MaterialEmission {
    fn set_emission_texture(&mut self, texture: AssetRef);
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

const BLEND_ID_OPAQUE: u8 = 0;
const BLEND_ID_ALPHA_TEST: u8 = 1;
const BLEND_ID_ALPHA_BLEND: u8 = 2;

impl BinarySerialization for BlendMode {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            Self::Opaque => w.write_u8(BLEND_ID_OPAQUE)?,
            Self::AlphaTest(cutoff) => {
                w.write_u8(BLEND_ID_ALPHA_TEST)?;
                w.write_f32::<LittleEndian>(*cutoff)?;
            }
            Self::AlphaBlend => w.write_u8(BLEND_ID_ALPHA_BLEND)?,
        }

        Ok(())
    }
}

impl BinaryDeserialization for BlendMode {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let ty = r.read_u8()?;
        match ty {
            BLEND_ID_OPAQUE => Ok(Self::Opaque),
            BLEND_ID_ALPHA_TEST => Ok(Self::AlphaTest(r.read_u8()? as f32 / 255.0)),
            BLEND_ID_ALPHA_BLEND => Ok(Self::AlphaBlend),
            val => panic!("Unknown blend mode {}", val),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub blend: BlendMode,
    pub base: AssetRef,
    pub normal: AssetRef,
    pub metallic_roughness: AssetRef,
    pub occlusion: AssetRef,
    pub emission: AssetRef,
    pub emission_value: f32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            blend: BlendMode::Opaque,
            base: AssetRef::default(),
            normal: AssetRef::default(),
            metallic_roughness: AssetRef::default(),
            occlusion: AssetRef::default(),
            emission: AssetRef::default(),
            emission_value: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnlitMaterial {
    pub blend: BlendMode,
    pub base: AssetRef,
}

impl Default for UnlitMaterial {
    fn default() -> Self {
        Self {
            blend: BlendMode::Opaque,
            base: AssetRef::default(),
        }
    }
}

impl PbrMaterial {
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>) {
        if self.base.valid() {
            deps.push(self.base);
        }
        if self.normal.valid() {
            deps.push(self.normal);
        }
        if self.metallic_roughness.valid() {
            deps.push(self.metallic_roughness);
        }
        if self.occlusion.valid() {
            deps.push(self.occlusion);
        }
        if self.emission.valid() {
            deps.push(self.occlusion);
        }
    }
}

impl UnlitMaterial {
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>) {
        if self.base.valid() {
            deps.push(self.base);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    Pbr(PbrMaterial),
    Unlit(UnlitMaterial),
}

impl Material {
    pub fn collect_dependencies(&self, deps: &mut Vec<AssetRef>) {
        match self {
            Self::Pbr(pbr) => pbr.collect_dependencies(deps),
            Self::Unlit(unlit) => unlit.collect_dependencies(deps),
        }
    }
}

impl MaterialBaseColor for PbrMaterial {
    fn set_base_texture(&mut self, texture: AssetRef) {
        self.base = texture;
    }
}

impl MaterialBaseColor for UnlitMaterial {
    fn set_base_texture(&mut self, texture: AssetRef) {
        self.base = texture;
    }
}

impl MaterialBlend for UnlitMaterial {
    fn set_blend_mode(&mut self, value: BlendMode) {
        self.blend = value;
    }
}

impl MaterialValues for PbrMaterial {
    fn set_metallic_roughness_texture(&mut self, texture: AssetRef) {
        self.metallic_roughness = texture;
    }
}

impl MaterialNormals for PbrMaterial {
    fn set_normal_texture(&mut self, texture: AssetRef) {
        self.normal = texture;
    }
}

impl MaterialOcclusion for PbrMaterial {
    fn set_occlusion_texture(&mut self, texture: AssetRef) {
        self.occlusion = texture;
    }
}

impl MaterialEmission for PbrMaterial {
    fn set_emission_texture(&mut self, texture: AssetRef) {
        self.emission = texture;
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

impl BinarySerialization for PbrMaterial {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.blend.serialize(w)?;
        self.base.serialize(w)?;
        self.normal.serialize(w)?;
        self.metallic_roughness.serialize(w)?;
        self.occlusion.serialize(w)?;
        self.emission.serialize(w)?;
        w.write_f32::<LittleEndian>(self.emission_value)?;

        Ok(())
    }
}

impl BinaryDeserialization for PbrMaterial {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let blend = BlendMode::deserialize(r)?;
        let base = AssetRef::deserialize(r)?;
        let normal = AssetRef::deserialize(r)?;
        let metallic_roughness = AssetRef::deserialize(r)?;
        let occlusion = AssetRef::deserialize(r)?;
        let emission = AssetRef::deserialize(r)?;
        let emission_value = r.read_f32::<LittleEndian>()?;

        Ok(Self {
            blend,
            base,
            normal,
            metallic_roughness,
            occlusion,
            emission,
            emission_value,
        })
    }
}

impl BinarySerialization for UnlitMaterial {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.blend.serialize(w)?;
        self.base.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for UnlitMaterial {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let blend = BlendMode::deserialize(r)?;
        let base = AssetRef::deserialize(r)?;

        Ok(Self { blend, base })
    }
}

const MATERIAL_ID_PBR: u8 = 0;
const MATERIAL_ID_UNLIT: u8 = 1;

impl BinarySerialization for Material {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            Material::Pbr(pbr) => {
                w.write_u8(MATERIAL_ID_PBR)?;
                pbr.serialize(w)?;
            }
            Material::Unlit(unlit) => {
                w.write_u8(MATERIAL_ID_UNLIT)?;
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
            MATERIAL_ID_PBR => Ok(Self::Pbr(PbrMaterial::deserialize(r)?)),
            MATERIAL_ID_UNLIT => Ok(Self::Unlit(UnlitMaterial::deserialize(r)?)),
            id => panic!("Unknown material ID {}", id),
        }
    }
}
