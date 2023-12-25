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

use std::{collections::HashMap, fmt::Debug, slice, sync::Arc};

use bytes::{BufMut, Bytes, BytesMut};
use dess_assets::{MeshData, MeshMaterial, ModelAsset, ModelCollectionAsset, ShaderSource};
use dess_backend::{BindGroupHandle, BindGroupLayoutDesc, BindType, BufferSlice, Device};

use dess_common::any_as_u8_slice;
use smol_str::SmolStr;

use crate::{
    BufferPool, Error, Material, Resource, ResourceContext, ResourceDependencies, ResourceHandle,
    ResourceLoader,
};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct ObjectScaleUniform {
    pub position_range: (f32, f32),
    pub uv_ranges: [(f32, f32); 2],
    _pad: [f32; 2],
}

/// Single primitive to draw
#[derive(Debug, Clone, Copy)]
pub struct SubMesh {
    pub first_index: u32,
    pub index_count: u32,
    pub bounds: (glam::Vec3, glam::Vec3),
    pub object_bind_group: BindGroupHandle,
    pub material_index: usize,
}

/// Reperesentation of single mesh
#[derive(Debug, Default)]
pub struct StaticMesh {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
    pub submeshes: Vec<SubMesh>,
    pub materials: Vec<ResourceHandle<Material>>,
    pub resolved_materials: Vec<Arc<Material>>,
}

impl StaticMesh {
    pub(crate) fn new(
        device: &Device,
        asset: MeshData,
        vertices: BufferSlice,
        indices: BufferSlice,
        materials: &[ResourceHandle<Material>],
    ) -> Self {
        let object_bind_layout =
            BindGroupLayoutDesc::default().bind(0, "object", BindType::UniformBuffer, 1);
        let geometry = vertices.part(asset.vertex_offset);
        let indices = indices.part(asset.index_offset);
        let submeshes = asset
            .submeshes
            .iter()
            .enumerate()
            .map(|(index, submesh)| SubMesh {
                first_index: submesh.first_index,
                index_count: submesh.index_count,
                bounds: (
                    glam::Vec3::from_array(submesh.bounds.0),
                    glam::Vec3::from_array(submesh.bounds.1),
                ),
                object_bind_group: device
                    .create_bind_group_from_desc(&object_bind_layout)
                    .unwrap(),
                material_index: index,
            })
            .collect::<Vec<_>>();
        let materials = asset
            .submeshes
            .iter()
            .map(|submesh| materials[submesh.material as usize])
            .collect::<Vec<_>>();
        device
            .with_bind_groups(|ctx| {
                for i in 0..asset.submeshes.len() {
                    let uniform = ObjectScaleUniform {
                        position_range: asset.submeshes[i].position_range,
                        uv_ranges: asset.submeshes[i].uv_ranges,
                        _pad: [0.0; 2],
                    };
                    ctx.bind_uniform(submeshes[i].object_bind_group, 0, &uniform)?;
                }

                Ok(())
            })
            .unwrap();

        Self {
            vertices: geometry,
            indices,
            submeshes,
            resolved_materials: Vec::with_capacity(materials.len()),
            materials,
        }
    }

    fn dispose(&self, ctx: &ResourceContext<'_>) {
        self.submeshes.iter().for_each(|x| {
            ctx.device.destroy_bind_group(x.object_bind_group);
        })
    }
}

impl ResourceDependencies for StaticMesh {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.materials.iter().all(|x| ctx.is_material_finished(*x))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), crate::Error> {
        self.resolved_materials.clear();
        for handle in self.materials.iter() {
            self.resolved_materials.push(ctx.resolve_material(*handle)?);
        }

        Ok(())
    }
}

/// Representation of single gltf scene
#[derive(Debug, Default)]
pub struct Model {
    // bone transformations
    pub bones: Vec<glam::Affine3A>,
    // Bone parents, u32::MAX - no parent
    pub bone_parents: Vec<u32>,
    // Bone names
    pub bone_names: HashMap<String, u32>,
    // Actual meshes
    pub static_meshes: Vec<StaticMesh>,
    // Mesh names, name->index
    pub mesh_names: HashMap<String, u32>,
    // Which mesh located where, bone->mesh
    pub instances: Vec<(u32, u32)>,
}

impl ResourceDependencies for Model {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.static_meshes.iter().all(|x| x.is_finished(ctx))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), crate::Error> {
        for mesh in self.static_meshes.iter_mut() {
            mesh.resolve(ctx)?;
        }

        Ok(())
    }
}

impl Model {
    pub fn new(
        devce: &Device,
        asset: ModelAsset,
        vertices: BufferSlice,
        indices: BufferSlice,
        materials: &[ResourceHandle<Material>],
    ) -> Self {
        let bones = asset
            .bones
            .iter()
            .map(|bone| {
                glam::Affine3A::from_scale_rotation_translation(
                    glam::Vec3::from_array(bone.local_scale),
                    glam::Quat::from_array(bone.local_rotation),
                    glam::Vec3::from_array(bone.local_translation),
                )
            })
            .collect::<Vec<_>>();
        let parents = asset
            .bones
            .iter()
            .map(|bone| bone.parent.unwrap_or(u32::MAX))
            .collect::<Vec<_>>();
        let static_meshes = asset
            .static_meshes
            .into_iter()
            .map(|mesh| StaticMesh::new(devce, mesh, vertices, indices, materials))
            .collect::<Vec<_>>();
        Self {
            bones,
            bone_parents: parents,
            bone_names: asset.bone_names,
            static_meshes,
            mesh_names: asset.mesh_names,
            instances: asset.node_to_mesh,
        }
    }

    fn dispose(&self, ctx: &ResourceContext) {
        self.static_meshes.iter().for_each(|x| x.dispose(ctx));
    }
}

/// Representation of single gltf file
///
/// Contains multiple models (scenes).
#[derive(Debug, Default)]
pub struct ModelCollection {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
    pub models: HashMap<SmolStr, Model>,
}

impl ResourceDependencies for ModelCollection {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.models.iter().all(|(_, x)| x.is_finished(ctx))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), crate::Error> {
        for (_, model) in self.models.iter_mut() {
            model.resolve(ctx)?;
        }

        Ok(())
    }
}

pub trait MeshMaterialFactory: Debug + Send + Sync {
    fn create_material(
        &self,
        loader: &dyn ResourceLoader,
        material: &MeshMaterial,
    ) -> Option<ResourceHandle<Material>>;
}

impl ModelCollection {
    pub fn new<T: ResourceLoader, F: MeshMaterialFactory>(
        loader: &T,
        asset: ModelCollectionAsset,
        buffers: &BufferPool,
        material_factory: &F,
    ) -> Result<Self, Error> {
        let vertices = buffers.allocate(&asset.vertices)?;
        let indices = buffers.allocate(&asset.indices)?;
        let materials = asset
            .materials
            .iter()
            .map(|material| material_factory.create_material(loader, material).unwrap())
            .collect::<Vec<_>>();
        let models = asset
            .models
            .into_iter()
            .map(|(name, model)| {
                (
                    name.into(),
                    Model::new(loader.render_device(), model, vertices, indices, &materials),
                )
            })
            .collect::<HashMap<_, _>>();

        Ok(Self {
            vertices,
            indices,
            models,
        })
    }
}

impl Resource for ModelCollection {
    fn dispose(&self, ctx: &ResourceContext) {
        self.models.iter().for_each(|(_, x)| x.dispose(ctx));
        ctx.buffers.deallocate(self.vertices);
        ctx.buffers.deallocate(self.indices);
    }
}

#[derive(Debug)]
pub struct BasicPbrMaterialFactory;

#[derive(Debug)]
pub struct BasicUnlitMaterialFactory;

#[derive(Debug, Default)]
pub struct MaterialFactoryCollection {
    factories: Vec<Box<dyn MeshMaterialFactory>>,
}

impl MeshMaterialFactory for MaterialFactoryCollection {
    fn create_material(
        &self,
        loader: &dyn ResourceLoader,
        material: &MeshMaterial,
    ) -> Option<ResourceHandle<Material>> {
        self.factories
            .iter()
            .find_map(|factory| factory.create_material(loader, material))
    }
}

impl MaterialFactoryCollection {
    pub fn register(&mut self, factory: Box<dyn MeshMaterialFactory>) {
        self.factories.insert(0, factory);
    }
}

impl MeshMaterialFactory for BasicPbrMaterialFactory {
    fn create_material(
        &self,
        loader: &dyn ResourceLoader,
        material: &MeshMaterial,
    ) -> Option<ResourceHandle<Material>> {
        if material.ty == "pbr" {
            if let Ok(program) = loader.get_or_load_program(&[
                ShaderSource::vertex("shaders/pbr_vs.hlsl"),
                ShaderSource::fragment("shaders/pbr_ps.hlsl"),
            ]) {
                let images = material
                    .images
                    .iter()
                    .map(|(x, y)| (x.clone(), *y))
                    .collect::<Vec<_>>();
                let mut uniform = BytesMut::new();
                uniform.put_f32(
                    material
                        .values
                        .get("emissive_power")
                        .copied()
                        .unwrap_or(0.0),
                );
                uniform.put_f32(0.0);
                uniform.put_f32(0.0);
                uniform.put_f32(0.0);
                return Some(loader.request_material(program, &images, uniform.into()));
            }
        }
        None
    }
}

impl MeshMaterialFactory for BasicUnlitMaterialFactory {
    fn create_material(
        &self,
        loader: &dyn ResourceLoader,
        material: &MeshMaterial,
    ) -> Option<ResourceHandle<Material>> {
        if let Ok(program) = loader.get_or_load_program(&[
            ShaderSource::vertex("shaders/unlit_vs.hlsl"),
            ShaderSource::fragment("shaders/unlit_ps.hlsl"),
        ]) {
            let images = material
                .images
                .iter()
                .map(|(x, y)| (x.clone(), *y))
                .collect::<Vec<_>>();
            return Some(loader.request_material(program, &images, Bytes::default()));
        }
        None
    }
}
