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
    fmt::Debug,
    fs::File,
    future::Future,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
};

use bevy_tasks::{block_on, IoTaskPool, Task};
use dess_assets::{
    import_effect, process_image, process_model, Asset, EffectAsset, EffectSource, GltfSource,
    ImageAsset, ImageSource, MeshMaterial, ModelAsset, ASSET_CACHE_PATH, get_absolute_asset_path,
};
use dess_backend::vulkan::{
    DescriptorHandle, Device, ImageCreateDesc, ImageHandle, ImageSubresourceData, ProgramHandle,
    ShaderDesc,
};
use dess_common::{Handle, Pool};
use log::{debug, warn};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{
    BufferPool, Error, RenderEffect, RenderEffectTechinque, RenderMaterial, RenderModel,
    RenderScene, StaticRenderMesh, SubMesh,
};

/// Trait to check asset dependencies
pub trait EngineAsset: Send + Sync + 'static {
    fn is_ready<T: AssetCacheFns>(&self, asset_cache: &T) -> bool;
    fn resolve<T: AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), Error>;
}

pub type AssetHandle<T> = Handle<AssetState<T>, ()>;

#[derive(Debug)]
pub enum AssetState<T: EngineAsset> {
    Pending(Task<Result<T, Error>>),
    Preparing(Arc<T>),
    Loaded(Arc<T>),
    Failed(Error),
}

impl<T: EngineAsset> AssetState<T> {
    fn maintain<U: AssetCacheFns>(&mut self, asset_cache: &U) {
        match self {
            AssetState::Pending(task) => {
                if task.is_finished() {
                    match block_on(task) {
                        Ok(mut result) => {
                            if result.is_ready(asset_cache) {
                                match result.resolve(asset_cache) {
                                    Ok(()) => *self = AssetState::Loaded(Arc::new(result)),
                                    Err(err) => *self = AssetState::Failed(err),
                                }
                            } else {
                                *self = AssetState::Preparing(Arc::new(result));
                            }
                        }
                        Err(err) => *self = AssetState::Failed(err),
                    }
                }
            }
            AssetState::Preparing(result) => {
                if result.is_ready(asset_cache) {
                    match Arc::get_mut(result).unwrap().resolve(asset_cache) {
                        Ok(()) => *self = AssetState::Loaded(result.clone()),
                        Err(err) => *self = AssetState::Failed(err),
                    }
                }
            }
            _ => {}
        }
    }

    fn resolve<U: AssetCacheFns>(&mut self, asset_cache: &U) -> Result<Arc<T>, Error> {
        match self {
            AssetState::Pending(task) => match block_on(task) {
                Ok(mut result) => match result.resolve(asset_cache) {
                    Ok(()) => {
                        let result = Arc::new(result);
                        *self = AssetState::Loaded(result.clone());
                        Ok(result)
                    }
                    Err(err) => {
                        *self = AssetState::Failed(err.clone());
                        Err(err)
                    }
                },
                Err(err) => {
                    *self = AssetState::Failed(err.clone());
                    Err(err)
                }
            },
            AssetState::Preparing(result) => {
                match Arc::get_mut(result).unwrap().resolve(asset_cache) {
                    Ok(()) => {
                        let result = result.clone();
                        *self = AssetState::Loaded(result.clone());
                        Ok(result)
                    }
                    Err(err) => {
                        *self = AssetState::Failed(err.clone());
                        Err(err)
                    }
                }
            }
            AssetState::Loaded(result) => Ok(result.clone()),
            AssetState::Failed(err) => Err(err.clone()),
        }
    }

    fn is_finished<U: AssetCacheFns>(&self, asset_cache: &U) -> bool {
        match self {
            AssetState::Pending(_) => false,
            AssetState::Preparing(asset) => asset.is_ready(asset_cache),
            AssetState::Loaded(_) => true,
            AssetState::Failed(_) => true,
        }
    }
}

impl EngineAsset for ImageHandle {
    fn is_ready<T: AssetCacheFns>(&self, _asset_cache: &T) -> bool {
        true
    }

    fn resolve<T: AssetCacheFns>(&mut self, _asset_cache: &T) -> Result<(), Error> {
        Ok(())
    }
}

#[derive(Debug, Default)]
struct SingleTypeAssetCache<T: EngineAsset> {
    handles: RwLock<HashMap<u64, AssetHandle<T>>>,
    assets: Mutex<Pool<AssetState<T>, ()>>,
}

impl<T: EngineAsset> SingleTypeAssetCache<T> {
    fn request<S, F>(&self, source: &S, loading: F) -> AssetHandle<T>
    where
        S: Hash,
        F: Future<Output = Result<T, Error>> + Send + Sync + 'static,
    {
        let handles = self.handles.upgradable_read();
        let mut hasher = siphasher::sip::SipHasher::default();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(handle) = handles.get(&key) {
            *handle
        } else {
            let mut handles = RwLockUpgradableReadGuard::upgrade(handles);
            let mut assets = self.assets.lock();
            if let Some(handle) = handles.get(&key) {
                *handle
            } else {
                let task = IoTaskPool::get().spawn(loading);
                let handle = assets.push(AssetState::Pending(task), ());
                handles.insert(key, handle);
                handle
            }
        }
    }

    fn resolve<U: AssetCacheFns>(
        &self,
        handle: AssetHandle<T>,
        asset_cache: &U,
    ) -> Result<Arc<T>, Error> {
        self.assets
            .lock()
            .get_hot_mut(handle)
            .ok_or(Error::InvalidHandle)?
            .resolve(asset_cache)
    }

    fn is_finished<U: AssetCacheFns>(&self, handle: AssetHandle<T>, asset_cache: &U) -> bool {
        match self.assets.lock().get_hot(handle) {
            Some(asset) => asset.is_finished(asset_cache),
            None => false,
        }
    }

    fn are_finished<U: AssetCacheFns>(&self, handles: &[AssetHandle<T>], asset_cache: &U) -> bool {
        let assets = self.assets.lock();
        handles.iter().all(|handle| match assets.get_hot(*handle) {
            Some(asset) => asset.is_finished(asset_cache),
            None => false,
        })
    }

    pub(crate) fn maintain<U: AssetCacheFns>(&self, asset_cache: &U) {
        let mut assets = self.assets.lock();
        assets.for_each_mut(|asset, _| asset.maintain(asset_cache));
    }
}

pub struct AssetCache {
    device: Arc<Device>,
    buffers: Arc<BufferPool>,
    programs: RwLock<HashMap<u64, ProgramHandle>>,
    images: SingleTypeAssetCache<ImageHandle>,
    effects: SingleTypeAssetCache<RenderEffect>,
    materials: SingleTypeAssetCache<RenderMaterial>,
    models: SingleTypeAssetCache<RenderModel>,
}

pub trait AssetCacheFns {
    fn request_image(&self, source: &ImageSource) -> AssetHandle<ImageHandle>;
    fn resolve_image(&self, handle: AssetHandle<ImageHandle>) -> Result<Arc<ImageHandle>, Error>;
    fn is_image_loaded(&self, handle: AssetHandle<ImageHandle>) -> bool;
    fn are_images_loaded(&self, handles: &[AssetHandle<ImageHandle>]) -> bool;
    fn request_effect(&self, name: &str) -> AssetHandle<RenderEffect>;
    fn resolve_effect(&self, handle: AssetHandle<RenderEffect>)
        -> Result<Arc<RenderEffect>, Error>;
    fn request_material(&self, material: &MeshMaterial) -> AssetHandle<RenderMaterial>;
    fn is_material_loaded(&self, handle: AssetHandle<RenderMaterial>) -> bool;
    fn resolve_material(
        &self,
        handle: AssetHandle<RenderMaterial>,
    ) -> Result<Arc<RenderMaterial>, Error>;
    fn is_effect_loaded(&self, handle: AssetHandle<RenderEffect>) -> bool;
    fn request_model(&self, name: &str) -> AssetHandle<RenderModel>;
    fn resolve_model(&self, handle: AssetHandle<RenderModel>) -> Result<Arc<RenderModel>, Error>;
    fn is_model_loaded(&self, handle: AssetHandle<RenderModel>) -> bool;
    fn maintain(&self);
    fn render_device(&self) -> &Device;
    fn buffer_pool(&self) -> &BufferPool;
    fn get_or_create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error>;
}

impl AssetCache {
    pub fn new(device: &Arc<Device>, buffers: &Arc<BufferPool>) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            buffers: buffers.clone(),
            programs: RwLock::default(),
            images: SingleTypeAssetCache::default(),
            effects: SingleTypeAssetCache::default(),
            materials: SingleTypeAssetCache::default(),
            models: SingleTypeAssetCache::default(),
        })
    }

    async fn load_from_cache_or_import<
        T: Asset,
        S: Hash + Debug,
        I: Future<Output = Result<T, Error>> + Send + Sync + 'static,
    >(
        import: I,
        source: S,
        ext: &str,
    ) -> Result<T, Error> {
        let path = get_cached_asset_path(&source, ext);
        if let Ok(mut file) = File::open(&path) {
            if let Ok(asset) = T::deserialize(&mut file) {
                debug!("Loaded cached {:?}", source);
                return Ok(asset);
            }
        }
        debug!("Import asset {:?}", source);
        let asset = match import.await {
            Ok(asset) => asset,
            Err(err) => {
                warn!("Failed to load asset {:?} - {:?}", source, err);
                return Err(err);
            }
        };
        if let Ok(mut file) = File::create(path) {
            if asset.serialize(&mut file).is_ok() {
                debug!("Written to cache {:?}", source);
            }
        }

        Ok(asset)
    }

    async fn load_image(device: Arc<Device>, source: ImageSource) -> Result<ImageHandle, Error> {
        let name = source.to_string();
        let asset =
            Self::load_from_cache_or_import(Self::import_image(source.clone()), source, "image")
                .await?;
        let mips = asset
            .mips
            .iter()
            .enumerate()
            .map(|(index, mip)| ImageSubresourceData {
                data: mip,
                row_pitch: (asset.dimensions[0] as usize >> index).max(1) * 4,
            })
            .collect::<Vec<_>>();
        Ok(device.create_image(
            ImageCreateDesc::texture(asset.format, asset.dimensions)
                .mip_levels(asset.mips.len())
                .name(&name)
                .initial_data(&mips),
        )?)
    }

    async fn import_image(source: ImageSource) -> Result<ImageAsset, Error> {
        Ok(process_image(source.import()?)?)
    }

    async fn load_effect<T: AssetCacheFns>(
        asset_cache: T,
        source: String,
    ) -> Result<RenderEffect, Error> {
        let asset =
            Self::load_from_cache_or_import(Self::import_effect(source.clone()), source, "effect")
                .await?;
        let mut techinques = HashMap::new();
        for (name, tech) in asset.techniques {
            let shaders = tech
                .shaders
                .iter()
                .map(|x| ShaderDesc::new(x.stage.into(), &x.code))
                .collect::<Vec<_>>();
            let program = asset_cache.get_or_create_program(&shaders)?;
            techinques.insert(
                name.into(),
                RenderEffectTechinque {
                    program,
                    pipeline_desc: tech.pipeline_desc.into(),
                },
            );
        }
        Ok(RenderEffect { techinques })
    }

    async fn import_effect(source: String) -> Result<EffectAsset, Error> {
        let effect: EffectSource = serde_json::from_reader(File::open(get_absolute_asset_path(source)?)?)?;
        Ok(import_effect(effect)?)
    }

    async fn import_material<T: AssetCacheFns>(
        asset_cache: T,
        source: MeshMaterial,
    ) -> Result<RenderMaterial, Error> {
        Ok(RenderMaterial {
            effect: asset_cache.request_effect(&source.effect),
            uniform: source.uniform,
            images: source
                .images
                .iter()
                .map(|(name, image)| (name.into(), asset_cache.request_image(image)))
                .collect::<HashMap<_, _>>(),
            descriptors: HashMap::default(),
        })
    }

    async fn create_model<T: AssetCacheFns>(
        source: String,
        asset_cache: T,
    ) -> Result<RenderModel, Error> {
        let asset = AssetCache::load_from_cache_or_import(
            Self::import_model(source.clone()),
            source,
            "model",
        )
        .await?;
        let mut model = RenderModel {
            static_geometry: asset_cache.buffer_pool().allocate(&asset.static_geometry)?,
            attributes: asset_cache.buffer_pool().allocate(&asset.attributes)?,
            indices: asset_cache.buffer_pool().allocate(&asset.indices)?,
            ..Default::default()
        };
        for (name, scene) in asset.scenes {
            let mut render_scene = RenderScene {
                bones: scene
                    .bones
                    .iter()
                    .map(|x| {
                        glam::Affine3A::from_scale_rotation_translation(
                            glam::Vec3::from_array(x.local_scale),
                            glam::Quat::from_array(x.local_rotation),
                            glam::Vec3::from_array(x.local_translation),
                        )
                    })
                    .collect(),
                bone_parents: scene
                    .bones
                    .iter()
                    .map(|x| x.parent.unwrap_or(u32::MAX))
                    .collect(),
                bone_names: scene.names.clone(),
                mesh_names: scene.mesh_names.clone(),
                instances: scene.node_to_mesh.clone(),
                ..Default::default()
            };
            render_scene.bone_parents = scene
                .bones
                .iter()
                .map(|x| x.parent.unwrap_or(u32::MAX))
                .collect();
            render_scene.bone_names = scene.names;
            render_scene.mesh_names = scene.mesh_names;
            render_scene.instances = scene.node_to_mesh;
            for mesh in scene.static_meshes {
                let mut render_mesh = StaticRenderMesh {
                    geometry: model.static_geometry.part(mesh.geometry),
                    attributes: model.attributes.part(mesh.attributes),
                    indices: model.static_geometry.part(mesh.indices),
                    ..Default::default()
                };
                let mut mesh_materials = HashMap::new();
                for surface in mesh.surfaces {
                    let material = &asset.materials[surface.material as usize];
                    let material_index = *mesh_materials.entry(material).or_insert({
                        let index = render_mesh.materials.len();
                        render_mesh
                            .materials
                            .push(asset_cache.request_material(material));
                        index
                    });
                    let surface = SubMesh {
                        first_index: surface.first,
                        index_count: surface.count,
                        bounds: (
                            glam::Vec3::from_array(surface.bounds.0),
                            glam::Vec3::from_array(surface.bounds.1),
                        ),
                        object_ds: DescriptorHandle::default(),
                        material_index,
                    };
                    render_mesh.submeshes.push(surface);
                }
                render_scene.meshes.push(render_mesh);
            }
            model.scenes.insert(name, render_scene);
        }

        Ok(model)
    }

    async fn import_model(source: String) -> Result<ModelAsset, Error> {
        Ok(process_model(GltfSource { path: source }.import()?))
    }
}

impl AssetCacheFns for Arc<AssetCache> {
    fn request_image(&self, source: &ImageSource) -> AssetHandle<ImageHandle> {
        self.images.request(
            source,
            AssetCache::load_image(self.device.clone(), source.clone()),
        )
    }

    fn resolve_image(&self, handle: AssetHandle<ImageHandle>) -> Result<Arc<ImageHandle>, Error> {
        self.images.resolve(handle, self)
    }

    fn is_image_loaded(&self, handle: AssetHandle<ImageHandle>) -> bool {
        self.images.is_finished(handle, self)
    }

    fn are_images_loaded(&self, handles: &[AssetHandle<ImageHandle>]) -> bool {
        self.images.are_finished(handles, self)
    }

    fn request_effect(&self, name: &str) -> AssetHandle<RenderEffect> {
        let name = name.to_owned();
        self.effects
            .request(&name.clone(), AssetCache::load_effect(self.clone(), name))
    }

    fn maintain(&self) {
        self.images.maintain(self);
        self.effects.maintain(self);
        self.materials.maintain(self);
        self.models.maintain(self);
    }

    fn render_device(&self) -> &Device {
        &self.device
    }

    fn resolve_effect(
        &self,
        handle: AssetHandle<RenderEffect>,
    ) -> Result<Arc<RenderEffect>, Error> {
        self.effects.resolve(handle, self)
    }

    fn is_effect_loaded(&self, handle: AssetHandle<RenderEffect>) -> bool {
        self.effects.is_finished(handle, self)
    }

    fn get_or_create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error> {
        let mut hasher = siphasher::sip::SipHasher::default();
        shaders.hash(&mut hasher);
        let key = hasher.finish();
        let programs = self.programs.upgradable_read();
        if let Some(program) = programs.get(&key) {
            Ok(*program)
        } else {
            let mut programs = RwLockUpgradableReadGuard::upgrade(programs);
            if let Some(program) = programs.get(&key) {
                Ok(*program)
            } else {
                let program = self.device.create_program(shaders)?;
                programs.insert(key, program);
                Ok(program)
            }
        }
    }

    fn request_material(&self, material: &MeshMaterial) -> AssetHandle<RenderMaterial> {
        self.materials.request(
            material,
            AssetCache::import_material(self.clone(), material.clone()),
        )
    }

    fn is_material_loaded(&self, handle: AssetHandle<RenderMaterial>) -> bool {
        self.materials.is_finished(handle, self)
    }

    fn resolve_material(
        &self,
        handle: AssetHandle<RenderMaterial>,
    ) -> Result<Arc<RenderMaterial>, Error> {
        self.materials.resolve(handle, self)
    }

    fn request_model(&self, name: &str) -> AssetHandle<RenderModel> {
        self.models.request(
            &name.to_owned(),
            AssetCache::create_model(name.to_owned(), self.clone()),
        )
    }

    fn resolve_model(&self, handle: AssetHandle<RenderModel>) -> Result<Arc<RenderModel>, Error> {
        self.models.resolve(handle, self)
    }

    fn is_model_loaded(&self, handle: AssetHandle<RenderModel>) -> bool {
        self.models.is_finished(handle, self)
    }

    fn buffer_pool(&self) -> &BufferPool {
        &self.buffers
    }
}

fn get_cached_asset_path<S: Hash>(source: &S, ext: &str) -> PathBuf {
    let mut hasher = siphasher::sip::SipHasher::default();
    source.hash(&mut hasher);
    let key = hasher.finish();
    Path::new(ASSET_CACHE_PATH).join(format!("{:x}.{}", key, ext))
}
