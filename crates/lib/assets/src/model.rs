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
    fmt::format,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use speedy::{Context, Readable, Writable};
use turbosloth::{Lazy, LazyWorker, RunContext};

use crate::{
    get_absolute_asset_path, get_relative_asset_path, Asset, AssetRef, AssetRefProvider, Error,
    ImageSource,
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

#[derive(Debug, Clone, Copy, Readable, Writable)]
pub enum MeshMaterial {
    Pbr(PbrMeshMaterial),
    Unlit(u32),
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

#[async_trait]
impl LazyWorker for GltfSource {
    type Output = Result<GltfContent, Error>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
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

pub struct ProcessGltfAsset {
    gltf: Lazy<GltfContent>,
}

struct SceneProcessingContext<'a> {
    model: &'a mut ModelAsset,
    scene: &'a mut SceneAsset,
    buffers: &'a Vec<gltf::buffer::Data>,
    document: &'a gltf::Document,
    // Index in gltf -> index in asset
    processed_meshes: HashMap<u32, u32>,
}

impl ProcessGltfAsset {
    fn process_scene(&self, ctx: &mut SceneProcessingContext, scene: gltf::Scene) {
        for node in scene.nodes() {
            self.process_node(ctx, "", None, node);
        }
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

        for child in node.children() {
            self.process_node(ctx, &node_name, Some(last), child);
        }
    }
}

#[async_trait]
impl LazyWorker for ProcessGltfAsset {
    type Output = Result<SceneAsset, Error>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let gltf = self.gltf.eval(&ctx).await?;

        let mut model = ModelAsset::default();

        for (index, scene) in gltf.document.scenes().enumerate() {
            let name = scene.name().unwrap_or(&format!("{}", index)).to_string();
            let mut result = SceneAsset::default();
            let mut ctx = SceneProcessingContext {
                model: &mut model,
                scene: &mut result,
                buffers: &gltf.buffers,
                document: &gltf.document,
                processed_meshes: HashMap::default(),
            };
            self.process_scene(&mut ctx, scene);
            model.scenes.insert(name, result);
        }

        todo!()
    }
}
