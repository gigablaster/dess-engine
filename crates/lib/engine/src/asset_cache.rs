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
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use bevy_tasks::{block_on, IoTaskPool, Task};
use dess_assets::{
    import_effect, Asset, EffectAsset, EffectSource, ImageAsset, ImageSource, ProcessImageAsset,
    ASSET_CACHE_PATH,
};
use dess_backend::vulkan::{
    Device, ImageCreateDesc, ImageHandle, ImageSubresourceData, ProgramHandle, ShaderDesc,
};
use dess_common::{Handle, Pool};
use log::debug;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{Error, RenderEffect, RenderEffectTechinque};

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

impl EngineAssetKey for String {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
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

pub trait EngineAssetKey: Debug {
    fn key(&self) -> u64;
}

#[derive(Debug, Default)]
struct SingleTypeAssetCache<T: EngineAsset> {
    handles: RwLock<HashMap<u64, AssetHandle<T>>>,
    assets: Mutex<Pool<AssetState<T>, ()>>,
}

impl<T: EngineAsset> SingleTypeAssetCache<T> {
    fn request<S, F>(&self, asset: &S, loading: F) -> AssetHandle<T>
    where
        S: EngineAssetKey,
        F: Future<Output = Result<T, Error>> + Send + Sync + 'static,
    {
        let handles = self.handles.upgradable_read();
        let key = asset.key();
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

impl<'a> EngineAssetKey for &[ShaderDesc<'a>] {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.iter().for_each(|x| x.hash(&mut hasher));
        hasher.finish()
    }
}

pub struct AssetCache {
    device: Arc<Device>,
    programs: RwLock<HashMap<u64, ProgramHandle>>,
    images: SingleTypeAssetCache<ImageHandle>,
    effects: SingleTypeAssetCache<RenderEffect>,
}

impl EngineAssetKey for ImageSource {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.purpose.hash(&mut hasher);
        mem::discriminant(&self.source).hash(&mut hasher);
        match &self.source {
            dess_assets::ImageDataSource::File(path) => path.hash(&mut hasher),
            dess_assets::ImageDataSource::Bytes(data) => data.hash(&mut hasher),
            dess_assets::ImageDataSource::Placeholder(pixel) => pixel.hash(&mut hasher),
        }
        hasher.finish()
    }
}

pub trait AssetCacheFns {
    fn request_image(&self, source: &ImageSource) -> AssetHandle<ImageHandle>;
    fn resolve_image(&self, handle: AssetHandle<ImageHandle>) -> Result<Arc<ImageHandle>, Error>;
    fn is_image_loaded(&self, handle: AssetHandle<ImageHandle>) -> bool;
    fn are_images_loaded(&self, handles: &[AssetHandle<ImageHandle>]) -> bool;
    fn request_effect(&self, name: &str) -> AssetHandle<RenderEffect>;
    fn resolve_effect(&self, handle: AssetHandle<RenderEffect>)
        -> Result<Arc<RenderEffect>, Error>;
    fn is_effect_loaded(&self, handle: AssetHandle<RenderEffect>) -> bool;
    fn maintain(&self);
    fn render_device(&self) -> &Device;
    fn get_or_create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error>;
}

impl AssetCache {
    pub fn new(device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            images: SingleTypeAssetCache::default(),
            effects: SingleTypeAssetCache::default(),
            programs: RwLock::default(),
        })
    }

    async fn load_from_cache_or_import<
        T: Asset,
        S: EngineAssetKey,
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
        let asset = import.await?;
        if let Ok(mut file) = File::create(path) {
            if asset.serialize(&mut file).is_ok() {
                debug!("Written to cache {:?}", source);
            }
        }

        Ok(asset)
    }

    async fn load_image(device: Arc<Device>, source: ImageSource) -> Result<ImageHandle, Error> {
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
                .initial_data(&mips),
        )?)
    }

    async fn import_image(source: ImageSource) -> Result<ImageAsset, Error> {
        Ok(ProcessImageAsset::import(source.import()?)?)
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
        let effect: EffectSource = serde_json::from_reader(File::open(source)?)?;
        Ok(import_effect(effect)?)
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
        let key = shaders.key();
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
}

fn get_cached_asset_path<S: EngineAssetKey>(source: &S, ext: &str) -> PathBuf {
    Path::new(ASSET_CACHE_PATH).join(format!("{:x}.{}", source.key(), ext))
}
