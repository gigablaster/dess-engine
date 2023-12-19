use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use dess_assets::{
    get_absolute_asset_path, get_relative_asset_path, AssetRef, BlendMode, Bone, GltfSource,
    ImageSource, ImageSourceDesc, LightingAttributes, MeshData, MeshMaterial, ModelAsset,
    SceneAsset, StaticMeshGeometry, SubMesh, MATERIAL_TYPE_PBR, MATERIAL_TYPE_UNLIT,
};
use normalize_path::NormalizePath;
use numquant::linear::quantize;

use crate::{is_asset_changed, AssetImporter, Error, ImportContext};

#[derive(Debug)]
pub struct GltfContent {
    base: PathBuf,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    _images: Vec<gltf::image::Data>,
}

fn import_gltf(source: &GltfSource) -> Result<GltfContent, Error> {
    let path = Path::new(&source.path).to_owned();
    let (document, buffers, images) = gltf::import(get_absolute_asset_path(&path)?)
        .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
    let base = get_relative_asset_path(&path)?.parent().unwrap().into();
    Ok(GltfContent {
        document,
        buffers,
        _images: images,
        base,
    })
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

struct SceneProcessingContext<'a> {
    ctx: &'a dyn ImportContext,
    model: &'a mut ModelAsset,
    scene: &'a mut SceneAsset,
    buffers: &'a Vec<gltf::buffer::Data>,
    base: &'a Path,
    // Index in gltf -> index in asset
    processed_meshes: HashMap<u32, u32>,
    unique_materials: HashMap<MeshMaterial, u32>,
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

fn process_scene(ctx: &mut SceneProcessingContext, scene: gltf::Scene) {
    for node in scene.nodes() {
        process_node(ctx, "", None, node);
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

fn process_texture(
    ctx: &mut SceneProcessingContext,
    texture: &gltf::texture::Texture,
    desc: ImageSourceDesc,
) -> AssetRef {
    match texture.source().source() {
        gltf::image::Source::Uri { uri, .. } => {
            let image_path = ctx.base.join(uri).normalize();
            ctx.ctx
                .import(Box::new(ImageSource::from_file(image_path, desc)))
        }
        _ => panic!(),
    }
}

fn process_placeholder(
    ctx: &mut SceneProcessingContext,
    color: [f32; 4],
    desc: ImageSourceDesc,
) -> AssetRef {
    ctx.ctx
        .import(Box::new(ImageSource::from_color(color, desc)))
}

fn process_blend(material: &gltf::Material) -> BlendMode {
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => BlendMode::Opaque,
        gltf::material::AlphaMode::Mask => {
            BlendMode::AlphaTest(material.alpha_cutoff().unwrap_or(0.0))
        }
        gltf::material::AlphaMode::Blend => BlendMode::AlphaBlend,
    }
}

fn create_unlit_material(
    ctx: &mut SceneProcessingContext,
    material: &gltf::Material,
) -> MeshMaterial {
    let base = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::color())
    } else {
        process_placeholder(
            ctx,
            material.pbr_metallic_roughness().base_color_factor(),
            ImageSourceDesc::color(),
        )
    };
    MeshMaterial::new(MATERIAL_TYPE_UNLIT, process_blend(material)).image("base", base)
}

fn create_pbr_material(
    ctx: &mut SceneProcessingContext,
    material: &gltf::Material,
) -> MeshMaterial {
    let base = if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::color())
    } else {
        process_placeholder(
            ctx,
            material.pbr_metallic_roughness().base_color_factor(),
            ImageSourceDesc::color(),
        )
    };

    let metallic_roughness = if let Some(texture) = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
    {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::non_color())
    } else {
        process_placeholder(
            ctx,
            [
                material.pbr_metallic_roughness().metallic_factor(),
                material.pbr_metallic_roughness().roughness_factor(),
                0.0,
                1.0,
            ],
            ImageSourceDesc::non_color(),
        )
    };

    let occlusion = if let Some(texture) = material.occlusion_texture() {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::non_color())
    } else {
        process_placeholder(ctx, [0.0, 0.0, 1.0, 1.0], ImageSourceDesc::non_color())
    };

    let normal = if let Some(texture) = material.normal_texture() {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::normals())
    } else {
        process_placeholder(ctx, [0.0, 0.0, 1.0, 1.0], ImageSourceDesc::normals())
    };

    let emissive = if let Some(texture) = material.emissive_texture() {
        process_texture(ctx, &texture.texture(), ImageSourceDesc::non_color())
    } else {
        let emissive_color = material.emissive_factor();
        process_placeholder(
            ctx,
            [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
            ImageSourceDesc::non_color(),
        )
    };
    MeshMaterial::new(MATERIAL_TYPE_PBR, process_blend(material))
        .image("base", base)
        .image("metallic_roughness", metallic_roughness)
        .image("normal", normal)
        .image("occlusion", occlusion)
        .image("emissive", emissive)
        .value(
            "emissive_power",
            material.emissive_strength().unwrap_or(0.0),
        )
}

fn process_material(ctx: &mut SceneProcessingContext, material: &gltf::Material) -> u32 {
    let material = if material.unlit() {
        create_unlit_material(ctx, material)
    } else {
        create_pbr_material(ctx, material)
    };
    if let Some(index) = ctx.unique_materials.get(&material) {
        *index
    } else {
        let index = ctx.model.materials.len();
        ctx.model.materials.push(material.clone());
        ctx.unique_materials.insert(material, index as u32);
        index as u32
    }
}

fn process_mesh(ctx: &mut SceneProcessingContext, mesh: &gltf::Mesh) {
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
        let (max_position_value, geometry) = process_static_geometry(&positions);
        let (max_uv_value, attributes) = process_attributes(&normals, &uvs, &tangents);
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
        let mut attributes = meshopt::remap_vertex_buffer(&attributes, total_vertex_count, &remap);
        let mut indices = meshopt::remap_index_buffer(Some(&indices), total_vertex_count, &remap);
        meshopt::optimize_vertex_cache_in_place(&indices, geometry.len());

        let first = mesh_indices.len() as u32;
        let count = indices.len() as u32;
        let material = process_material(ctx, &prim.material());
        mesh_indices.append(&mut indices);
        mesh_geometry.append(&mut geometry);
        mesh_attributes.append(&mut attributes);
        target.surfaces.push(SubMesh {
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
    mesh_attributes = meshopt::remap_vertex_buffer(&mesh_attributes, mesh_attributes.len(), &remap);
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
            process_mesh(ctx, &mesh);
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
        process_node(ctx, &node_name, Some(last), child);
    }
}

fn process_model(gltf: GltfContent, ctx: &dyn ImportContext) -> ModelAsset {
    let mut model = ModelAsset::default();

    for (index, scene) in gltf.document.scenes().enumerate() {
        let name = scene.name().unwrap_or(&format!("{}", index)).to_string();
        let mut result = SceneAsset::default();
        let mut ctx = SceneProcessingContext {
            ctx,
            model: &mut model,
            scene: &mut result,
            base: &gltf.base,
            buffers: &gltf.buffers,
            processed_meshes: HashMap::default(),
            unique_materials: HashMap::default(),
        };
        process_scene(&mut ctx, scene);
        model.scenes.insert(name, result);
    }

    model
}

impl AssetImporter for GltfSource {
    fn import(&self, ctx: &dyn ImportContext) -> Result<Arc<dyn dess_assets::Asset>, Error> {
        let content = import_gltf(self)?;
        Ok(Arc::new(process_model(content, ctx)))
    }

    fn is_changed(&self, timestamp: std::time::SystemTime) -> bool {
        is_asset_changed(&self.path, timestamp)
    }
}