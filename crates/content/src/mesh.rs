use std::{
    collections::HashMap,
    fs::File,
    hash::BuildHasher,
    path::{Path, PathBuf},
};

use dess_common::{
    mesh::{
        CpuMesh, CpuModel, EffectInfo, MeshBuilder, MeshLayoutBuilder, VertexAttribute,
        BASE_COLOR_TEXTURE, METALLIC_ROUGHNESS_TEXTURE, NORMAL_MAP_TEXTURE, OCCLUSION_TEXTURE,
        VERTEX_NORNAL_CHANNEL, VERTEX_POSITION_CHANNEL, VERTEX_TANGENT_CHANNEL, VERTEX_UV_CHANNEL,
    },
    traits::BinarySerialization,
};
use gltf::{
    image,
    material::{NormalTexture, OcclusionTexture},
    mesh::Mode,
    texture, Document,
};
use normalize_path::NormalizePath;

use crate::{Content, ContentImporter, ImportContext, ImportError};

impl Content for CpuModel {
    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.serialize(&mut File::create(path)?)
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

const PBR_OPAQUE_EFFECT: &str = "bdrf";

#[derive(Debug, Default)]
pub struct GltfModelImporter {}

impl GltfModelImporter {
    fn set_texture(effect: &mut EffectInfo, root: &Path, name: &str, info: &Option<texture::Info>) {
        if let Some(texture) = info {
            Self::bind_texture(effect, root, name, texture.texture());
        }
    }

    fn set_normal_texture(effect: &mut EffectInfo, root: &Path, normal: &Option<NormalTexture>) {
        if let Some(texture) = normal {
            Self::bind_texture(effect, root, NORMAL_MAP_TEXTURE, texture.texture());
        }
    }

    fn set_occlusion_texture(
        effect: &mut EffectInfo,
        root: &Path,
        occlusion: &Option<OcclusionTexture>,
    ) {
        if let Some(texture) = occlusion {
            Self::bind_texture(effect, root, OCCLUSION_TEXTURE, texture.texture());
        }
    }

    fn bind_texture(effect: &mut EffectInfo, root: &Path, name: &str, texture: texture::Texture) {
        if let image::Source::Uri { uri, .. } = texture.source().source() {
            let mut path: PathBuf = root
                .join(Path::new(uri))
                .normalize()
                .as_os_str()
                .to_ascii_lowercase()
                .into();
            path.set_extension("dds");
            effect.add_texture(name, path.to_str().unwrap());
        }
    }

    fn collect_effects(
        document: &Document,
        context: &ImportContext,
    ) -> HashMap<String, EffectInfo> {
        let mut effects = HashMap::new();
        document.materials().for_each(|mat| {
            if let Some(name) = mat.name() {
                let mut effect = EffectInfo::new(PBR_OPAQUE_EFFECT);
                Self::set_texture(
                    &mut effect,
                    context.destination_dir,
                    BASE_COLOR_TEXTURE,
                    &mat.pbr_metallic_roughness().base_color_texture(),
                );
                Self::set_texture(
                    &mut effect,
                    context.destination_dir,
                    METALLIC_ROUGHNESS_TEXTURE,
                    &mat.pbr_metallic_roughness().metallic_roughness_texture(),
                );
                Self::set_normal_texture(
                    &mut effect,
                    context.destination_dir,
                    &mat.normal_texture(),
                );
                Self::set_occlusion_texture(
                    &mut effect,
                    context.destination_dir,
                    &mat.occlusion_texture(),
                );

                effects.insert(name.into(), effect);
            }
        });

        effects
    }

    fn process_node(target: &mut CpuModel, node: &gltf::Node, buffers: &[gltf::buffer::Data]) {
        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions = if let Some(positions) = reader.read_positions() {
                    positions.collect::<Vec<_>>()
                } else {
                    return;
                };
                let layout = MeshLayoutBuilder::default()
                    .channel(VERTEX_POSITION_CHANNEL, VertexAttribute::Vec3);
                let (uvs, layout) = if let Some(texcoord) = reader.read_tex_coords(0) {
                    (
                        texcoord.into_f32().collect::<Vec<_>>(),
                        layout.channel(VERTEX_UV_CHANNEL, VertexAttribute::Vec2),
                    )
                } else {
                    (vec![[0.0, 0.0]; positions.len()], layout)
                };
                let (normals, layout) = if let Some(normals) = reader.read_normals() {
                    (
                        normals.collect::<Vec<_>>(),
                        layout.channel(VERTEX_NORNAL_CHANNEL, VertexAttribute::Vec3Normalized),
                    )
                } else {
                    (vec![[1.0, 0.0, 0.0]; positions.len()], layout)
                };
                let (mut tangents, layout) = if let Some(tangents) = reader.read_tangents() {
                    (
                        tangents.collect::<Vec<_>>(),
                        layout.channel(VERTEX_TANGENT_CHANNEL, VertexAttribute::Vec3Normalized),
                    )
                } else {
                    (vec![[0.0, 1.0, 0.0, 0.0]; positions.len()], layout)
                };

                let indices = if let Some(indices) = reader.read_indices() {
                    indices.into_u32().collect::<Vec<_>>()
                } else {
                    return;
                };
                let layout = if layout.has_channel(VERTEX_UV_CHANNEL)
                    && layout.has_channel(VERTEX_NORNAL_CHANNEL)
                    && !layout.has_channel(VERTEX_TANGENT_CHANNEL)
                {
                    mikktspace::generate_tangents(&mut TangentCalcContext {
                        indices: &indices,
                        positions: &positions,
                        normals: &normals,
                        uvs: &uvs,
                        tangents: &mut tangents,
                    });
                    layout.channel(VERTEX_TANGENT_CHANNEL, VertexAttribute::Vec3Normalized)
                } else {
                    layout
                };
                let has_normals = layout.has_channel(VERTEX_NORNAL_CHANNEL);
                let has_uvs = layout.has_channel(VERTEX_UV_CHANNEL);
                let has_tangents = layout.has_channel(VERTEX_TANGENT_CHANNEL);
                let mut builder = MeshBuilder::new(layout);
                for index in 0..positions.len() {
                    builder.vertex();
                    builder.push(positions[index]);
                    if has_uvs {
                        builder.push(uvs[index]);
                    }
                    if has_normals {
                        builder.push(normals[index]);
                    }
                    if has_tangents {
                        let tangent = [tangents[index][0], tangents[index][1], tangents[index][2]];
                        builder.push(tangent);
                    }
                }
                let cpumesh = CpuMesh::build(builder, EffectInfo::new(PBR_OPAQUE_EFFECT));
                let name = if let Some(name) = node.name() {
                    name
                } else {
                    "none"
                };
                let (translate, rotation, scale) = node.transform().decomposed();
                let translate = glam::Mat4::from_translation(glam::Vec3::from_array(translate));
                let rotation = glam::Mat4::from_quat(glam::Quat::from_array(rotation));
                let scale = glam::Mat4::from_scale(glam::Vec3::from_array(scale));
                let transform = translate * rotation * scale;
                target.add_mesh(cpumesh, name, &transform);
            }
        }
    }
}

impl ContentImporter for GltfModelImporter {
    fn can_handle(&self, path: &std::path::Path) -> bool {
        path.extension()
            .map(|x| {
                x.to_ascii_lowercase()
                    .to_str()
                    .map(|x| x == "gltf")
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn target_name(&self, path: &std::path::Path) -> std::path::PathBuf {
        let mut path = path.to_path_buf();
        path.set_extension("mesh");
        path
    }

    fn import(
        &self,
        path: &std::path::Path,
        context: &ImportContext,
    ) -> Result<Box<dyn Content>, ImportError> {
        let (document, buffers, images) = gltf::import(path)?;

        let effects = Self::collect_effects(&document, context);
        let mut model = CpuModel::default();
        document
            .nodes()
            .for_each(|node| Self::process_node(&mut model, &node, &buffers));
        Ok(Box::new(model))
    }
}
