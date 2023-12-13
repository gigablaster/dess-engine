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

use std::{collections::HashMap, sync::Arc};

use dess_backend::vulkan::{BufferSlice, DescriptorHandle};

use crate::{AssetHandle, EngineAsset, RenderMaterial};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct ObjectScaleUniform {
    pub position_scale: f32,
    pub uv_scale: f32,
    _pad: [f32; 2],
}

/// Single primitive to draw
#[derive(Debug, Clone, Copy)]
pub struct SubMesh {
    pub first_index: u32,
    pub index_count: u32,
    pub bounds: (glam::Vec3, glam::Vec3),
    pub object_ds: DescriptorHandle,
    pub material_index: usize,
}

/// Reperesentation of single mesh
#[derive(Debug, Default)]
pub struct StaticRenderMesh {
    pub geometry: BufferSlice,
    pub attributes: BufferSlice,
    pub indices: BufferSlice,
    pub submeshes: Vec<SubMesh>,
    pub materials: Vec<AssetHandle<RenderMaterial>>,
    pub resolved_materials: Vec<Arc<RenderMaterial>>,
}

impl EngineAsset for StaticRenderMesh {
    fn is_ready<T: crate::AssetCacheFns>(&self, asset_cache: &T) -> bool {
        self.materials
            .iter()
            .all(|x| asset_cache.is_material_loaded(*x))
    }

    fn resolve<T: crate::AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), crate::Error> {
        self.resolved_materials.clear();
        for handle in self.materials.iter() {
            self.resolved_materials
                .push(asset_cache.resolve_material(*handle)?);
        }

        Ok(())
    }
}

/// Representation of single gltf scene
#[derive(Debug, Default)]
pub struct RenderScene {
    // bone transformations
    pub bones: Vec<glam::Affine3A>,
    // Bone parents, u32::MAX - no parent
    pub bone_parents: Vec<u32>,
    // Bone names
    pub bone_names: HashMap<String, u32>,
    // Actual meshes
    pub meshes: Vec<StaticRenderMesh>,
    // Mesh names, name->index
    pub mesh_names: HashMap<String, u32>,
    // Which mesh located where, bone->mesh
    pub instances: Vec<(u32, u32)>,
}

impl EngineAsset for RenderScene {
    fn is_ready<T: crate::AssetCacheFns>(&self, asset_cache: &T) -> bool {
        self.meshes.iter().all(|x| x.is_ready(asset_cache))
    }

    fn resolve<T: crate::AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), crate::Error> {
        for mesh in self.meshes.iter_mut() {
            mesh.resolve(asset_cache)?;
        }

        Ok(())
    }
}

/// Representation of single gltf file
///
/// Contains multiple models (scenes).
#[derive(Debug, Default)]
pub struct RenderModel {
    pub static_geometry: BufferSlice,
    pub attributes: BufferSlice,
    pub indices: BufferSlice,
    pub scenes: HashMap<String, RenderScene>,
}

impl EngineAsset for RenderModel {
    fn is_ready<T: crate::AssetCacheFns>(&self, asset_cache: &T) -> bool {
        self.scenes.iter().all(|(_, x)| x.is_ready(asset_cache))
    }

    fn resolve<T: crate::AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), crate::Error> {
        for (_, model) in self.scenes.iter_mut() {
            model.resolve(asset_cache)?;
        }

        Ok(())
    }
}
