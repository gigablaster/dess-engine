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
    hash::Hash,
    path::{Path, PathBuf},
};

use normalize_path::NormalizePath;
use numquant::linear::quantize;
use speedy::{Context, Readable, Writable};

use crate::{
    get_absolute_asset_path, get_relative_asset_path, Asset, Error, ImagePurpose, ImageSource,
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

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub struct Bone {
    pub parent: Option<u32>,
    pub local_translation: [f32; 3],
    pub local_rotation: [f32; 4],
    pub local_scale: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub struct Surface {
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
    pub surfaces: Vec<Surface>,
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
    }
}

#[derive(Debug, Clone, Copy, Readable, Writable)]
#[repr(C)]
pub struct PbrMeshMaterial {
    pub blend: BlendMode,
    pub base: u32,
    pub metallic_roughness: u32,
    pub normal: u32,
    pub occlusion: u32,
    pub emissive: u32,
    pub emissive_power: f32,
}

impl Eq for PbrMeshMaterial {}

impl PartialEq for PbrMeshMaterial {
    fn eq(&self, other: &Self) -> bool {
        self.blend == other.blend
            && self.base == other.base
            && self.metallic_roughness == other.metallic_roughness
            && self.normal == other.normal
            && self.occlusion == other.occlusion
            && self.emissive == other.emissive
            && self.emissive_power == other.emissive_power
    }
}

impl Hash for PbrMeshMaterial {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.blend.hash(state);
        self.base.hash(state);
        self.metallic_roughness.hash(state);
        self.normal.hash(state);
        self.occlusion.hash(state);
        self.emissive.hash(state);
        let dirty_hack = (self.emissive_power * 1000.0) as u32;
        dirty_hack.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Readable, Writable)]
pub enum MeshMaterial {
    Pbr(PbrMeshMaterial),
    Unlit(BlendMode, u32),
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
    pub images: Vec<ImageSource>,
    pub scenes: HashMap<String, SceneAsset>,
}

impl Asset for SceneAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("7b229650-8f34-4d5a-b140-8e5d9ce599aa");

    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}

#[derive(Debug)]
pub struct GltfContent {
    base: PathBuf,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    _images: Vec<gltf::image::Data>,
}

impl GltfSource {
    pub fn import(&self) -> Result<GltfContent, Error> {
        let path = Path::new(&self.path).to_owned();
        let (document, buffers, images) = gltf::import(get_absolute_asset_path(&path)?)
            .map_err(|err| Error::ImportFailed(err.to_string()))?;
        let base = get_relative_asset_path(&path)?.parent().unwrap().into();
        Ok(GltfContent {
            document,
            buffers,
            _images: images,
            base,
        })
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

pub struct ProcessGltfAsset {}

struct SceneProcessingContext<'a> {
    model: &'a mut ModelAsset,
    scene: &'a mut SceneAsset,
    buffers: &'a Vec<gltf::buffer::Data>,
    base: &'a Path,
    // Index in gltf -> index in asset
    processed_meshes: HashMap<u32, u32>,
    unique_materials: HashMap<MeshMaterial, u32>,
    unique_images: HashMap<ImageSource, u32>,
}

fn quantize_values(data: &[f32]) -> (f32, Vec<i16>) {
    let max = data
        .iter()
        .max_by(|x, y| x.abs().total_cmp(&y.abs()))
        .copied()
        .unwrap_or(0.0) as f64;
    let result = data
        .iter()
        .map(|x| quantize(*x as _, -max..max, i16::MAX))
        .collect::<Vec<_>>();

    (max as f32, result)
}

fn quantize_input<const N: usize>(input: &[[f32; N]]) -> (f32, Vec<[i16; N]>) {
    let mut data = Vec::with_capacity(input.len() * N);
    input
        .iter()
        .for_each(|x| x.iter().for_each(|x| data.push(*x)));
    let (max, values) = quantize_values(&data);
    let mut result = Vec::with_capacity(input.len());
    for index in 0..values.len() / N {
        let start = index * N;
        let mut value: [i16; N] = [0i16; N];
        let src = &values[start..start + N];
        (0..N).for_each(|i| {
            value[i] = src[i];
        });
        result.push(value);
    }

    (max, result)
}

pub(crate) fn quantize_positions(input: &[[f32; 3]]) -> (f32, Vec<[i16; 3]>) {
    quantize_input(input)
}

pub(crate) fn quantize_uvs(input: &[[f32; 2]]) -> (f32, Vec<[i16; 2]>) {
    quantize_input(input)
}

pub(crate) fn quantize_normalized(input: &[[f32; 3]]) -> Vec<[i16; 2]> {
    let (_, quantized) = quantize_input(input);
    quantized.iter().map(|x| [x[0], x[1]]).collect()
}

impl ProcessGltfAsset {
    fn process_scene(&self, ctx: &mut SceneProcessingContext, scene: gltf::Scene) {
        for node in scene.nodes() {
            self.process_node(ctx, "", None, node);
        }
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

    fn get_or_create_unique_image(
        &self,
        ctx: &mut SceneProcessingContext,
        image: ImageSource,
    ) -> u32 {
        if let Some(index) = ctx.unique_images.get(&image) {
            *index
        } else {
            let index = ctx.model.images.len();
            ctx.model.images.push(image.clone());
            ctx.unique_images.insert(image, index as u32);
            index as u32
        }
    }

    fn process_texture(
        &self,
        ctx: &mut SceneProcessingContext,
        texture: &gltf::texture::Texture,
        purpose: ImagePurpose,
    ) -> u32 {
        match texture.source().source() {
            gltf::image::Source::Uri { uri, .. } => {
                let image_path = ctx.base.join(uri).normalize();
                let source = ImageSource::from_file(image_path, purpose);
                self.get_or_create_unique_image(ctx, source)
            }
            _ => panic!(),
        }
    }

    fn process_placeholder(
        &self,
        ctx: &mut SceneProcessingContext,
        color: [f32; 4],
        purpose: ImagePurpose,
    ) -> u32 {
        let source = ImageSource::from_color(color, purpose);
        self.get_or_create_unique_image(ctx, source)
    }

    fn process_blend(&self, material: &gltf::Material) -> BlendMode {
        match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => BlendMode::Opaque,
            gltf::material::AlphaMode::Mask => {
                BlendMode::AlphaTest(material.alpha_cutoff().unwrap_or(0.0))
            }
            gltf::material::AlphaMode::Blend => BlendMode::AlphaBlend,
        }
    }

    fn create_unlit_material(
        &self,
        ctx: &mut SceneProcessingContext,
        material: &gltf::Material,
    ) -> MeshMaterial {
        let base = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::Color)
        } else {
            self.process_placeholder(
                ctx,
                material.pbr_metallic_roughness().base_color_factor(),
                ImagePurpose::Color,
            )
        };
        MeshMaterial::Unlit(self.process_blend(material), base)
    }

    fn create_pbr_material(
        &self,
        ctx: &mut SceneProcessingContext,
        material: &gltf::Material,
    ) -> MeshMaterial {
        let base = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::Color)
        } else {
            self.process_placeholder(
                ctx,
                material.pbr_metallic_roughness().base_color_factor(),
                ImagePurpose::Color,
            )
        };

        let metallic_roughness = if let Some(texture) = material
            .pbr_metallic_roughness()
            .metallic_roughness_texture()
        {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::NonColor)
        } else {
            self.process_placeholder(
                ctx,
                [
                    material.pbr_metallic_roughness().metallic_factor(),
                    material.pbr_metallic_roughness().roughness_factor(),
                    0.0,
                    1.0,
                ],
                ImagePurpose::NonColor,
            )
        };

        let occlusion = if let Some(texture) = material.occlusion_texture() {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::NonColor)
        } else {
            self.process_placeholder(ctx, [0.0, 0.0, 1.0, 1.0], ImagePurpose::NonColor)
        };

        let normal = if let Some(texture) = material.normal_texture() {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::Normals)
        } else {
            self.process_placeholder(ctx, [0.0, 0.0, 1.0, 1.0], ImagePurpose::Normals)
        };

        let emissive = if let Some(texture) = material.emissive_texture() {
            self.process_texture(ctx, &texture.texture(), ImagePurpose::NonColor)
        } else {
            let emissive_color = material.emissive_factor();
            self.process_placeholder(
                ctx,
                [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
                ImagePurpose::NonColor,
            )
        };

        MeshMaterial::Pbr(PbrMeshMaterial {
            blend: self.process_blend(material),
            base,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
            emissive_power: material.emissive_strength().unwrap_or(0.0),
        })
    }

    fn process_material(&self, ctx: &mut SceneProcessingContext, material: &gltf::Material) -> u32 {
        let material = if material.unlit() {
            self.create_unlit_material(ctx, material)
        } else {
            self.create_pbr_material(ctx, material)
        };
        if let Some(index) = ctx.unique_materials.get(&material) {
            *index
        } else {
            let index = ctx.model.materials.len();
            ctx.model.materials.push(material);
            ctx.unique_materials.insert(material, index as u32);
            index as u32
        }
    }

    fn process_mesh(&self, ctx: &mut SceneProcessingContext, mesh: &gltf::Mesh) {
        let mut target = MeshData::default();
        let mut mesh_indices = Vec::new();
        let mut mesh_geometry = Vec::new();
        let mut mesh_attributes = Vec::new();
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| Some(&ctx.buffers[buffer.index()]));
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
            let bounds = (bounds.min, bounds.max);

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

            let first = mesh_indices.len() as u32;
            let count = indices.len() as u32;
            let material = self.process_material(ctx, &prim.material());
            mesh_indices.append(&mut indices);
            mesh_geometry.append(&mut geometry);
            mesh_attributes.append(&mut attributes);
            target.surfaces.push(Surface {
                first,
                count,
                bounds,
                max_position_value,
                max_uv_value,
                material,
            });
        }
        let remap = meshopt::optimize_vertex_fetch_remap(&mesh_indices, mesh_geometry.len());
        mesh_geometry = meshopt::remap_vertex_buffer(&mesh_geometry, mesh_geometry.len(), &remap);
        mesh_attributes =
            meshopt::remap_vertex_buffer(&mesh_attributes, mesh_attributes.len(), &remap);
        let mut mesh_indices = mesh_indices.iter().map(|x| *x as u16).collect::<Vec<_>>();
        target.geometry = ctx.model.static_geometry.len() as u32;
        target.attributes = ctx.model.attributes.len() as u32;
        target.indices = ctx.model.indices.len() as u32;
        ctx.model.static_geometry.append(&mut mesh_geometry);
        ctx.model.attributes.append(&mut mesh_attributes);
        ctx.model.indices.append(&mut mesh_indices);
        ctx.scene.static_meshes.push(target);
    }

    fn process_node(
        &self,
        ctx: &mut SceneProcessingContext,
        parent_name: &str,
        parent: Option<usize>,
        node: gltf::Node,
    ) {
        let (local_translation, local_rotation, local_scale) = node.transform().decomposed();
        let last = ctx.scene.bones.len();
        ctx.scene.bones.push(Bone {
            parent: parent.map(|x| x as u32),
            local_translation,
            local_rotation,
            local_scale,
        });
        let name = node.name().unwrap_or(&format!("{}", last)).to_owned();
        let node_name = format!("{}/{}", parent_name, name);
        ctx.scene.names.insert(node_name.clone(), last as u32);
        if let Some(mesh) = node.mesh() {
            if let Some(instance_index) = ctx.processed_meshes.get(&(mesh.index() as u32)) {
                ctx.scene.node_to_mesh.push((last as u32, *instance_index));
            } else {
                let mesh_index = ctx.scene.static_meshes.len();
                self.process_mesh(ctx, &mesh);
                ctx.scene
                    .node_to_mesh
                    .push((last as u32, mesh_index as u32));
                ctx.scene.mesh_names.insert(
                    mesh.name().unwrap_or(&format!("{}", mesh_index)).to_owned(),
                    mesh_index as u32,
                );
            }
        }

        for child in node.children() {
            self.process_node(ctx, &node_name, Some(last), child);
        }
    }
}

impl ProcessGltfAsset {
    pub fn process(&self, gltf: GltfContent) -> ModelAsset {
        let mut model = ModelAsset::default();

        for (index, scene) in gltf.document.scenes().enumerate() {
            let name = scene.name().unwrap_or(&format!("{}", index)).to_string();
            let mut result = SceneAsset::default();
            let mut ctx = SceneProcessingContext {
                model: &mut model,
                scene: &mut result,
                base: &gltf.base,
                buffers: &gltf.buffers,
                processed_meshes: HashMap::default(),
                unique_images: HashMap::default(),
                unique_materials: HashMap::default(),
            };
            self.process_scene(&mut ctx, scene);
            model.scenes.insert(name, result);
        }

        model
    }
}
