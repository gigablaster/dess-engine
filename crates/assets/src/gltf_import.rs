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

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use dess_common::{bounds::AABB, Transform};
use gltf::{
    material::{AlphaMode, NormalTexture, OcclusionTexture, PbrMetallicRoughness},
    texture::{self, Info},
    Mesh,
};
use turbosloth::{Lazy, LazyWorker, RunContext};

use crate::{
    gpumesh::{
        quantize_normalized, quantize_positions, quantize_uvs, Bone, LightingAttributes,
        StaticGpuMesh, StaticMeshGeometry, Surface,
    },
    gpumodel::GpuModel,
    image::ImagePurpose,
    material::{
        BlendMode, Material, MaterialBaseColor, MaterialBlend, MaterialEmission, MaterialNormals,
        MaterialOcclusion, MaterialValues, PbrMaterial, UnlitMaterial,
    },
    AssetProcessingContext, AssetRef,
};

pub struct LoadGltf {
    path: PathBuf,
}

impl LoadGltf {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

pub struct LoadedGltf {
    path: PathBuf,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
}

#[async_trait]
impl LazyWorker for LoadGltf {
    type Output = anyhow::Result<LoadedGltf>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let (document, buffers, _) = gltf::import(&self.path)?;

        Ok(LoadedGltf {
            path: self.path,
            document,
            buffers,
        })
    }
}

struct ModelImportContext {
    pub model: GpuModel,
    pub processed_meshes: HashMap<usize, u32>, // mesh.index -> index in model
}

pub struct CreateGpuModel {
    pub gltf: Lazy<LoadedGltf>,
    pub context: Arc<AssetProcessingContext>,
}

impl CreateGpuModel {
    pub fn new(gltf: Lazy<LoadedGltf>, context: &Arc<AssetProcessingContext>) -> Self {
        Self {
            gltf,
            context: context.clone(),
        }
    }

    fn set_normal_texture(
        &self,
        material: &mut impl MaterialNormals,
        normal: &Option<NormalTexture>,
    ) {
        if let Some(texture) = normal {
            material
                .set_normal_texture(self.import_texture(texture.texture(), ImagePurpose::Normals));
        }
    }

    fn set_occlusion_texture(
        &self,
        material: &mut impl MaterialOcclusion,
        occlusion: &Option<OcclusionTexture>,
    ) {
        if let Some(texture) = occlusion {
            material.set_occlusion_texture(
                self.import_texture(texture.texture(), ImagePurpose::NonColor),
            );
        }
    }

    fn set_base_color(&self, material: &mut impl MaterialBaseColor, pbr: &PbrMetallicRoughness) {
        if let Some(texture) = pbr.base_color_texture() {
            material.set_base_texture(self.import_texture(texture.texture(), ImagePurpose::Color));
        }
        material.set_base_color(glam::Vec4::from_array(pbr.base_color_factor()));
    }

    fn set_material_values(&self, material: &mut impl MaterialValues, pbr: &PbrMetallicRoughness) {
        if let Some(texture) = pbr.metallic_roughness_texture() {
            material.set_metallic_roughness_texture(
                self.import_texture(texture.texture(), ImagePurpose::NonColor),
            );
        }
        material.set_metallic_value(pbr.metallic_factor());
        material.set_roughness_value(pbr.roughness_factor());
    }

    fn set_emission_color(
        &self,
        material: &mut impl MaterialEmission,
        emission: &Option<Info>,
        color: [f32; 3],
        value: Option<f32>,
    ) {
        if let Some(texture) = emission {
            material.set_emission_texture(
                self.import_texture(texture.texture(), ImagePurpose::NonColor),
            );
        }
        material.set_emission_color(glam::Vec3::from_array(color));
        material.set_emission_value(value.unwrap_or(0.0));
    }

    fn import_texture(&self, texture: texture::Texture, purpose: ImagePurpose) -> AssetRef {
        if let gltf::image::Source::Uri { uri, .. } = texture.source().source() {
            self.context.import_image(Path::new(uri), purpose)
        } else {
            AssetRef::default()
        }
    }

    fn process_node(
        &self,
        context: &mut ModelImportContext,
        parent_index: u32,
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
    ) {
        let (translation, rotation, scale) = node.transform().decomposed();
        let translation = glam::Vec3::from_array(translation);
        let rotation = glam::Quat::from_array(rotation);
        let scale = glam::Vec3::from_array(scale);
        let bone = Bone {
            parent: parent_index,
            local_tr: Transform {
                translation,
                rotation,
                scale,
            },
        };
        let current_bone_index = context.model.bones.len() as u32;
        context.model.bones.push(bone);
        context.model.names.insert(
            node.name()
                .unwrap_or(&format!("GltfNode_{}", current_bone_index))
                .into(),
            current_bone_index,
        );

        if let Some(mesh) = node.mesh() {
            if let Some(index) = context.processed_meshes.get(&mesh.index()) {
                context
                    .model
                    .node_to_mesh
                    .push((current_bone_index, *index));
            } else {
                self.process_mesh(context, mesh, current_bone_index, buffers);
            }
        }
        node.children()
            .for_each(|node| self.process_node(context, current_bone_index, &node, buffers));
    }

    fn create_material(&self, material: gltf::Material) -> Material {
        if material.unlit() {
            Material::Unlit(self.create_unlit_material(material))
        } else {
            Material::Pbr(self.create_pbr_material(material))
        }
    }

    fn set_blend_mode(&self, target: &mut impl MaterialBlend, material: &gltf::Material) {
        match material.alpha_mode() {
            AlphaMode::Opaque => target.set_blend_mode(BlendMode::Opaque),
            AlphaMode::Mask => {
                target.set_blend_mode(BlendMode::AlphaTest(material.alpha_cutoff().unwrap_or(0.0)))
            }
            AlphaMode::Blend => target.set_blend_mode(BlendMode::AlphaBlend),
        }
    }

    fn create_unlit_material(&self, material: gltf::Material) -> UnlitMaterial {
        let mut target = UnlitMaterial::default();
        self.set_blend_mode(&mut target, &material);
        self.set_base_color(&mut target, &material.pbr_metallic_roughness());

        target
    }

    fn create_pbr_material(&self, material: gltf::Material) -> PbrMaterial {
        let mut target = PbrMaterial::default();
        self.set_blend_mode(&mut target, &material);
        self.set_base_color(&mut target, &material.pbr_metallic_roughness());
        self.set_material_values(&mut target, &material.pbr_metallic_roughness());
        self.set_normal_texture(&mut target, &material.normal_texture());
        self.set_occlusion_texture(&mut target, &material.occlusion_texture());
        self.set_emission_color(
            &mut target,
            &material.emissive_texture(),
            material.emissive_factor(),
            material.emissive_strength(),
        );

        target
    }

    fn process_attributes(
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        tangents: &[[f32; 3]],
    ) -> (f32, Vec<LightingAttributes>) {
        let mut result = Vec::with_capacity(uvs.len());
        let (max_uv, uvs) = quantize_uvs(uvs);
        let normals = quantize_normalized(normals);
        let tangents = quantize_normalized(tangents);
        for index in 0..uvs.len() {
            let attr = LightingAttributes::new(normals[index], tangents[index], uvs[index]);
            result.push(attr);
        }

        (max_uv, result)
    }

    fn process_static_geometry(positons: &[[f32; 3]]) -> (f32, Vec<StaticMeshGeometry>) {
        let (max_position, positions) = quantize_positions(positons);
        let positions = positions
            .iter()
            .copied()
            .map(StaticMeshGeometry::new)
            .collect::<Vec<_>>();

        (max_position, positions)
    }

    fn process_mesh(
        &self,
        context: &mut ModelImportContext,
        mesh: Mesh,
        bone: u32,
        buffers: &[gltf::buffer::Data],
    ) {
        let mut target = StaticGpuMesh::default();
        let mut indices_collect = Vec::new();
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions = if let Some(positions) = reader.read_positions() {
                positions.collect::<Vec<_>>()
            } else {
                return;
            };
            let (uvs, has_uvs) = if let Some(texcoord) = reader.read_tex_coords(0) {
                (texcoord.into_f32().collect::<Vec<_>>(), true)
            } else {
                (vec![[0.0, 0.0]; positions.len()], false)
            };
            assert_eq!(positions.len(), uvs.len());
            let (normals, has_normals) = if let Some(normals) = reader.read_normals() {
                (normals.collect::<Vec<_>>(), true)
            } else {
                (vec![[1.0, 0.0, 0.0]; positions.len()], false)
            };
            assert_eq!(positions.len(), normals.len());
            let (mut tangents, has_tangents) = if let Some(tangents) = reader.read_tangents() {
                (tangents.collect::<Vec<_>>(), true)
            } else {
                (vec![[0.0, 1.0, 0.0, 0.0]; positions.len()], false)
            };
            assert_eq!(positions.len(), tangents.len());

            let indices = if let Some(indices) = reader.read_indices() {
                indices.into_u32().collect::<Vec<_>>()
            } else {
                return;
            };

            if has_uvs && has_normals && !has_tangents {
                mikktspace::generate_tangents(&mut TangentCalcContext {
                    indices: &indices,
                    positions: &positions,
                    normals: &normals,
                    uvs: &uvs,
                    tangents: &mut tangents,
                });
            };

            let bounds = prim.bounding_box();
            let bounds = AABB::from_min_max(
                glam::Vec3::from_array(bounds.min),
                glam::Vec3::from_array(bounds.max),
            );

            let tangents = tangents
                .iter()
                .map(|x| [x[0], x[1], x[2]])
                .collect::<Vec<_>>();
            let (max_position_value, geometry) = Self::process_static_geometry(&positions);
            let (max_uv_value, attributes) = Self::process_attributes(&normals, &uvs, &tangents);
            assert_eq!(positions.len(), attributes.len());

            let (total_vertex_count, remap) = meshopt::generate_vertex_remap_multi::<u8>(
                geometry.len(),
                &[
                    meshopt::VertexStream::new(geometry.as_ptr()),
                    meshopt::VertexStream::new(attributes.as_ptr()),
                ],
                Some(&indices),
            );
            let mut geometry = meshopt::remap_vertex_buffer(&geometry, total_vertex_count, &remap);
            let mut attributes =
                meshopt::remap_vertex_buffer(&attributes, total_vertex_count, &remap);
            let mut indices =
                meshopt::remap_index_buffer(Some(&indices), total_vertex_count, &remap);
            meshopt::optimize_vertex_cache_in_place(&indices, geometry.len());

            let first = indices_collect.len() as u32;
            let count = indices.len() as u32;
            indices_collect.append(&mut indices);
            let material = self.create_material(prim.material());
            target.geometry.append(&mut geometry);
            target.attributes.append(&mut attributes);
            target.surfaces.push(Surface {
                first,
                count,
                bounds,
                max_position_value,
                max_uv_value,
                material,
            });
        }
        let remap = meshopt::optimize_vertex_fetch_remap(&indices_collect, target.geometry.len());
        target.geometry =
            meshopt::remap_vertex_buffer(&target.geometry, target.geometry.len(), &remap);
        target.attributes =
            meshopt::remap_vertex_buffer(&target.attributes, target.attributes.len(), &remap);
        let mesh_index = context.model.static_meshes.len() as u32;
        let mesh_name = mesh.name().unwrap_or(&format!("GltfMesh_{}", bone)).into();
        target.indices = indices_collect
            .iter()
            .map(|x| *x as u16)
            .collect::<Vec<_>>();
        context.model.static_meshes.push(target);
        context.model.mesh_names.insert(mesh_name, mesh_index);
        context.model.node_to_mesh.push((bone, mesh_index));
        context.processed_meshes.insert(mesh.index(), mesh_index);
    }
}

struct TangentCalcContext<'a> {
    indices: &'a [u32],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tangents: &'a mut [[f32; 4]],
}

impl<'a> mikktspace::Geometry for TangentCalcContext<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
    }
}

#[async_trait]
impl LazyWorker for CreateGpuModel {
    type Output = anyhow::Result<GpuModel>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let gltf = self.gltf.eval(&ctx).await?;

        let mut import_context = ModelImportContext {
            model: GpuModel::default(),
            processed_meshes: HashMap::new(),
        };
        gltf.document
            .nodes()
            .for_each(|node| self.process_node(&mut import_context, 0, &node, &gltf.buffers));

        Ok(import_context.model)
    }
}
