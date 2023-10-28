use std::{
    fs::File,
    path::{Path, PathBuf},
};

use dess_common::{
    bounds::AABB,
    mesh::{
        quantize_normalized, quantize_positions, quantize_uvs, BlendMode, Bone, LightingAttributes,
        Material, MaterialBaseColor, MaterialBlend, MaterialEmission, MaterialNormals,
        MaterialOcclusion, MaterialValues, PbrMaterial, StaticMeshData, StaticMeshGeometry,
        Surface, UnlitMaterial,
    },
    traits::BinarySerialization,
    Transform,
};
use gltf::{
    image,
    material::{AlphaMode, NormalTexture, OcclusionTexture, PbrMetallicRoughness},
    texture::{self, Info},
    Mesh,
};
use normalize_path::NormalizePath;

use crate::{Content, ContentImporter, ImportContext, ImportError};

struct TangentCalcContext<'a> {
    indices: &'a [u16],
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
    fn set_normal_texture(
        material: &mut impl MaterialNormals,
        root: &Path,
        normal: &Option<NormalTexture>,
    ) {
        if let Some(texture) = normal {
            material.set_normal_texture(Self::texture_path(root, texture.texture()));
        }
    }

    fn set_occlusion_texture(
        material: &mut impl MaterialOcclusion,
        root: &Path,
        occlusion: &Option<OcclusionTexture>,
    ) {
        if let Some(texture) = occlusion {
            material.set_occlusion_texture(Self::texture_path(root, texture.texture()));
        }
    }

    fn set_base_color(
        material: &mut impl MaterialBaseColor,
        root: &Path,
        pbr: &PbrMetallicRoughness,
    ) {
        if let Some(texture) = pbr.base_color_texture() {
            material.set_base_texture(Self::texture_path(root, texture.texture()));
        }
        material.set_base_color(glam::Vec4::from_array(pbr.base_color_factor()));
    }

    fn set_material_values(
        material: &mut impl MaterialValues,
        root: &Path,
        pbr: &PbrMetallicRoughness,
    ) {
        if let Some(texture) = pbr.metallic_roughness_texture() {
            material.set_metallic_roughness_texture(Self::texture_path(root, texture.texture()));
        }
        material.set_metallic_value(pbr.metallic_factor());
        material.set_roughness_value(pbr.roughness_factor());
    }

    fn set_emission_color(
        material: &mut impl MaterialEmission,
        root: &Path,
        emission: &Option<Info>,
        color: [f32; 3],
        value: Option<f32>,
    ) {
        if let Some(texture) = emission {
            material.set_emission_texture(Self::texture_path(root, texture.texture()));
        }
        material.set_emission_color(glam::Vec3::from_array(color));
        material.set_emission_value(value.unwrap_or(0.0));
    }

    fn texture_path(root: &Path, texture: texture::Texture) -> Option<String> {
        if let image::Source::Uri { uri, .. } = texture.source().source() {
            let mut path: PathBuf = root
                .join(Path::new(uri))
                .normalize()
                .as_os_str()
                .to_ascii_lowercase()
                .into();
            path.set_extension("dds");
            path.to_str().map(|x| x.into())
        } else {
            None
        }
    }

    fn process_node(
        target: &mut StaticMeshData,
        parent_index: u32,
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        root: &Path,
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
        let current_bone_index = target.bones.len() as u32;
        target.bones.push(bone);
        target.bone_names.push(
            node.name()
                .unwrap_or(&format!("GltfNode_{}", current_bone_index))
                .into(),
        );

        if let Some(mesh) = node.mesh() {
            Self::process_mesh(target, mesh, current_bone_index, buffers, root);
        }
        node.children()
            .for_each(|node| Self::process_node(target, current_bone_index, &node, buffers, root));
    }

    fn create_material(material: gltf::Material, root: &Path) -> Material {
        if material.unlit() {
            Material::Unlit(Self::create_unlit_material(material, root))
        } else {
            Material::Pbr(Self::create_pbr_material(material, root))
        }
    }

    fn set_blend_mode(target: &mut impl MaterialBlend, material: &gltf::Material) {
        match material.alpha_mode() {
            AlphaMode::Opaque => target.set_blend_mode(BlendMode::Opaque),
            AlphaMode::Mask => {
                target.set_blend_mode(BlendMode::AlphaTest(material.alpha_cutoff().unwrap_or(0.0)))
            }
            AlphaMode::Blend => target.set_blend_mode(BlendMode::AlphaBlend),
        }
    }

    fn create_unlit_material(material: gltf::Material, root: &Path) -> UnlitMaterial {
        let mut target = UnlitMaterial::default();
        Self::set_blend_mode(&mut target, &material);
        Self::set_base_color(&mut target, root, &material.pbr_metallic_roughness());

        target
    }

    fn create_pbr_material(material: gltf::Material, root: &Path) -> PbrMaterial {
        let mut target = PbrMaterial::default();
        Self::set_blend_mode(&mut target, &material);
        Self::set_base_color(&mut target, root, &material.pbr_metallic_roughness());
        Self::set_material_values(&mut target, root, &material.pbr_metallic_roughness());
        Self::set_normal_texture(&mut target, root, &material.normal_texture());
        Self::set_occlusion_texture(&mut target, root, &material.occlusion_texture());
        Self::set_emission_color(
            &mut target,
            root,
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
        target: &mut StaticMeshData,
        mesh: Mesh,
        bone: u32,
        buffers: &[gltf::buffer::Data],
        root: &Path,
    ) {
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
            let (normals, has_normals) = if let Some(normals) = reader.read_normals() {
                (normals.collect::<Vec<_>>(), true)
            } else {
                (vec![[1.0, 0.0, 0.0]; positions.len()], false)
            };
            let (mut tangents, has_tangents) = if let Some(tangents) = reader.read_tangents() {
                (tangents.collect::<Vec<_>>(), true)
            } else {
                (vec![[0.0, 1.0, 0.0, 0.0]; positions.len()], false)
            };

            let mut indices = if let Some(indices) = reader.read_indices() {
                indices.into_u32().map(|x| x as u16).collect::<Vec<_>>()
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
            let (max_position_value, mut geometry) = Self::process_static_geometry(&positions);
            let (max_uv_value, mut attributes) =
                Self::process_attributes(&normals, &uvs, &tangents);
            let first = target.geometry.len() as u32;
            let count = geometry.len() as u32;
            let material = Self::create_material(prim.material(), root);
            target.geometry.append(&mut geometry);
            target.attributes.append(&mut attributes);
            target.indices.append(&mut indices);
            target.surfaces.push(Surface {
                first,
                count,
                bone,
                bounds,
                max_position_value,
                max_uv_value,
                material,
            })
        }
    }
}

impl Content for StaticMeshData {
    fn save(&self, path: &Path) -> std::io::Result<()> {
        self.serialize(&mut File::create(path)?)
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

        let mut model = StaticMeshData::default();
        document.nodes().for_each(|node| {
            Self::process_node(&mut model, 0, &node, &buffers, context.destination_dir)
        });
        Ok(Box::new(model))
    }
}
