// Copyright (C) 2023 Vladimir Kuskov

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

use std::collections::HashSet;

use speedy::{Readable, Writable};

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

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub enum BlendMode {
    Opaque,
    AlphaTest(f32),
    AlphaBlend,
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
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

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
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
    fn collect_dependencies(&self, deps: &mut HashSet<AssetRef>) {
        if self.base.valid() {
            deps.insert(self.base);
        }
        if self.normal.valid() {
            deps.insert(self.normal);
        }
        if self.metallic_roughness.valid() {
            deps.insert(self.metallic_roughness);
        }
        if self.occlusion.valid() {
            deps.insert(self.occlusion);
        }
        if self.emission.valid() {
            deps.insert(self.occlusion);
        }
    }
}

impl UnlitMaterial {
    fn collect_dependencies(&self, deps: &mut HashSet<AssetRef>) {
        if self.base.valid() {
            deps.insert(self.base);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub enum Material {
    Pbr(PbrMaterial),
    Unlit(UnlitMaterial),
}

impl Material {
    pub fn collect_dependencies(&self, deps: &mut HashSet<AssetRef>) {
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
