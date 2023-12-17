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

use core::slice;
use std::{
    collections::HashMap,
    hash::{self, Hash, Hasher},
    path::{Path, PathBuf},
};

use normalize_path::NormalizePath;
use numquant::linear::quantize;
use siphasher::sip128::Hasher128;
use speedy::{Context, Readable, Writable};

use crate::{
    get_absolute_asset_path, get_relative_asset_path, Asset, AssetLoad, AssetRef, ContentSource,
    Error, ImageSource,
};

#[derive(Debug, Clone, Hash)]
pub struct GltfSource {
    pub path: String,
}

impl GltfSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path
                .as_ref()
                .to_str()
                .unwrap()
                .to_ascii_lowercase()
                .to_owned(),
        }
    }
}

impl ContentSource for GltfSource {
    fn get_ref(&self) -> AssetRef {
        let mut hasher = siphasher::sip128::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish128().as_u128().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct LightingAttributes {
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [i16; 2],
    _pad: [i16; 2],
}

impl<'a, C: Context> Readable<'a, C> for LightingAttributes {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(Self {
            normal: reader.read_value::<[i16; 2]>()?,
            tangent: reader.read_value::<[i16; 2]>()?,
            uv: reader.read_value::<[i16; 2]>()?,
            _pad: [0, 0],
        })
    }
}

impl<C: Context> Writable<C> for LightingAttributes {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_value(&self.normal)?;
        writer.write_value(&self.tangent)?;
        writer.write_value(&self.uv)?;

        Ok(())
    }
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

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct PbrMeshMaterialUniform {
    pub emissive_power: f32,
}

#[repr(C, align(16))]
pub struct MeshSurfaceUniform {
    pub position_scale: f32,
    pub uv_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub struct Bone {
    pub parent: Option<u32>,
    pub local_translation: [f32; 3],
    pub local_rotation: [f32; 4],
    pub local_scale: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub struct SubMesh {
    pub first: u32,
    pub count: u32,
    pub bounds: ([f32; 3], [f32; 3]),
    pub max_position_value: f32,
    pub max_uv_value: f32,
    pub material: u32,
}

#[derive(Debug, Default, Readable, Writable)]
pub struct MeshData {
    pub geometry: u32,
    pub attributes: u32,
    pub indices: u32,
    pub surfaces: Vec<SubMesh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub enum BlendMode {
    Opaque,
    AlphaTest(f32),
    AlphaBlend,
}

impl Eq for BlendMode {}

impl Hash for BlendMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            BlendMode::AlphaTest(value) => (((value.clamp(0.0, 1.0)) * 255.0) as u8).hash(state),
            _ => {}
        }
    }
}

pub const MATERIAL_TYPE_PBR: &str = "pbr";
pub const MATERIAL_TYPE_UNLIT: &str = "unlit";

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub struct MeshMaterial {
    pub ty: String,
    pub blend: BlendMode,
    pub images: HashMap<String, AssetRef>,
    pub values: HashMap<String, f32>,
}

impl MeshMaterial {
    pub fn new(ty: &str, blend: BlendMode) -> Self {
        Self {
            ty: ty.to_owned(),
            blend,
            images: HashMap::default(),
            values: HashMap::default(),
        }
    }

    pub fn add_image(&mut self, name: &str, image: AssetRef) {
        self.images.insert(name.to_owned(), image);
    }

    pub fn set_value(&mut self, name: &str, value: f32) {
        self.values.insert(name.to_owned(), value);
    }

    pub fn image(mut self, name: &str, image: AssetRef) -> Self {
        self.add_image(name, image);
        self
    }

    pub fn value(mut self, name: &str, value: f32) -> Self {
        self.set_value(name, value);
        self
    }
}

impl Eq for MeshMaterial {}

impl Hash for MeshMaterial {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.blend.hash(state);
        for (name, image) in self.images.iter() {
            name.hash(state);
            image.hash(state);
        }
        for (name, value) in self.values.iter() {
            name.hash(state);
            ((value * 100000.0) as u64).hash(state);
        }
    }
}
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct StaticMeshGeometry {
    pub position: [i16; 3],
    _padding: u16,
}

impl<'a, C: Context> Readable<'a, C> for StaticMeshGeometry {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(Self {
            position: reader.read_value()?,
            _padding: 0,
        })
    }
}

impl<C: Context> Writable<C> for StaticMeshGeometry {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> Result<(), <C as Context>::Error> {
        writer.write_value(&self.position)
    }
}

impl StaticMeshGeometry {
    pub fn new(position: [i16; 3]) -> Self {
        Self {
            position,
            _padding: 0,
        }
    }
}

#[derive(Debug, Default, Readable, Writable)]
pub struct SceneAsset {
    pub static_meshes: Vec<MeshData>,
    pub mesh_names: HashMap<String, u32>,
    pub bones: Vec<Bone>,
    pub names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

#[derive(Debug, Default, Readable, Writable)]
pub struct ModelAsset {
    pub static_geometry: Vec<StaticMeshGeometry>,
    pub attributes: Vec<LightingAttributes>,
    pub indices: Vec<u16>,
    pub materials: Vec<MeshMaterial>,
    pub scenes: HashMap<String, SceneAsset>,
}

impl Asset for ModelAsset {
    fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        Ok(self.write_to_vec()?)
    }
}

impl AssetLoad for ModelAsset {
    fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(data)?)
    }
}
