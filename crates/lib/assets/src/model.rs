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

use std::{
    collections::HashMap,
    hash::{self, Hash},
    path::Path,
};

use bytes::Bytes;
use siphasher::sip128::Hasher128;
use speedy::{Readable, Writable};

use crate::{Asset, AssetLoad, AssetRef, ContentSource};

#[derive(Debug, Clone, Hash)]
pub struct GltfSource {
    pub path: String,
}

impl GltfSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_str().unwrap().to_owned(),
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

#[derive(Debug, Clone, Copy, PartialEq, Default, Readable, Writable)]
#[repr(C)]
pub struct StaticMeshVertex {
    pub position: [f32; 3],
}

impl StaticMeshVertex {
    pub fn new(position: [f32; 3]) -> Self {
        Self { position }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Readable, Writable)]
#[repr(C)]
pub struct MeshVertexAttributes {
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
}

impl MeshVertexAttributes {
    pub fn new(normal: [f32; 3], tangent: [f32; 3], uv1: [f32; 2], uv2: [f32; 2]) -> Self {
        Self {
            normal,
            tangent,
            uv1,
            uv2,
        }
    }
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
    pub first_index: u32,
    pub index_count: u32,
    pub bounds: ([f32; 3], [f32; 3]),
    pub material: u32,
}

#[derive(Debug, Default, Readable, Writable)]
pub struct MeshData {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub submeshes: Vec<SubMesh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub enum MeshBlendMode {
    Opaque,
    AlphaTest(f32),
    AlphaBlend,
}

impl Eq for MeshBlendMode {}

impl Hash for MeshBlendMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        if let MeshBlendMode::AlphaTest(value) = self {
            (((value.clamp(0.0, 1.0)) * 255.0) as u8).hash(state)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub struct MeshMaterial {
    pub blend: MeshBlendMode,
    pub base: AssetRef,
    pub normals: AssetRef,
    pub metallic_roughness: AssetRef,
    pub occlusion: AssetRef,
    pub emissive: AssetRef,
    pub emission_power: f32,
}

impl Eq for MeshMaterial {}

impl Hash for MeshMaterial {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.blend.hash(state);
        self.base.hash(state);
        self.normals.hash(state);
        self.metallic_roughness.hash(state);
        self.occlusion.hash(state);
        self.emissive.hash(state);
        ((self.emission_power * 1000.0) as u64).hash(state);
    }
}

#[derive(Debug, Default, Readable, Writable)]
pub struct ModelAsset {
    pub static_meshes: Vec<MeshData>,
    pub mesh_names: HashMap<String, u32>,
    pub bones: Vec<Bone>,
    pub bone_names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

#[derive(Debug, Default, Readable, Writable)]
pub struct ModelCollectionAsset {
    pub vertices: Vec<StaticMeshVertex>,
    pub attributes: Vec<MeshVertexAttributes>,
    pub indices: Vec<u16>,
    pub materials: Vec<MeshMaterial>,
    pub models: HashMap<String, ModelAsset>,
}

impl Asset for ModelCollectionAsset {
    fn to_bytes(&self) -> std::io::Result<Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl AssetLoad for ModelCollectionAsset {
    fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::read_from_buffer(data)?)
    }
}
