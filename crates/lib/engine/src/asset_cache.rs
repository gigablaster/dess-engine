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
    fmt::Display,
    fs::File,
    future::Future,
    hash::{Hash, Hasher},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use bevy_tasks::{block_on, IoTaskPool, Task};
use dess_assets::{Asset, ImageAsset, ImageSource, ProcessImageAsset, ASSET_CACHE_PATH};
use dess_backend::vulkan::{Device, ImageCreateDesc, ImageHandle, ImageSubresourceData};
use dess_common::{Handle, Pool};
use log::debug;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::Error;

/// Trait to check asset dependencies
pub trait EngineAsset: Default + Send + Sync + 'static {
    fn is_ready(&self, asset_cache: &AssetCache) -> bool;
    fn resolve(&mut self, asset_cache: &AssetCache) -> Result<(), Error>;
}

type AssetHandle<T> = Handle<AssetState<T>, ()>;

#[derive(Debug)]
pub enum AssetState<T: EngineAsset> {
    Pending(Task<Result<T, Error>>),
    Preparing(Arc<T>),
    Loaded(Arc<T>),
    Failed(Error),
}

impl<T: EngineAsset> AssetState<T> {
    fn maintain(&mut self, asset_cache: &AssetCache) {
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
                        Ok(()) => *self = AssetState::Loaded(std::mem::take(result)),
                        Err(err) => *self = AssetState::Failed(err),
                    }
                }
            }
            _ => {}
        }
    }

    fn resolve(&mut self, asset_cache: &AssetCache) -> Result<Arc<T>, Error> {
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

    fn is_finished(&self, asset_cache: &AssetCache) -> bool {
        match self {
            AssetState::Pending(_) => false,
            AssetState::Preparing(asset) => asset.is_ready(asset_cache),
            AssetState::Loaded(_) => true,
            AssetState::Failed(_) => true,
        }
    }
}

impl EngineAsset for ImageHandle {
    fn is_ready(&self, _asset_cache: &AssetCache) -> bool {
        true
    }

    fn resolve(&mut self, _asset_cache: &AssetCache) -> Result<(), Error> {
        Ok(())
    }
}

pub trait EngineAssetKey: Display {
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

    fn resolve(&self, handle: AssetHandle<T>, asset_cache: &AssetCache) -> Result<Arc<T>, Error> {
        self.assets
            .lock()
            .get_hot_mut(handle)
            .ok_or(Error::InvalidHandle)?
            .resolve(asset_cache)
    }

    fn is_finished(&self, handle: AssetHandle<T>, asset_cache: &AssetCache) -> bool {
        match self.assets.lock().get_hot(handle) {
            Some(asset) => asset.is_finished(asset_cache),
            None => false,
        }
    }

    pub(crate) fn maintain(&self, asset_cache: &AssetCache) {
        let mut assets = self.assets.lock();
        assets.for_each_mut(|asset, _| asset.maintain(asset_cache));
    }
}

pub struct AssetCache {
    device: Arc<Device>,
    images: SingleTypeAssetCache<ImageHandle>,
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

impl AssetCache {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: device.clone(),
            images: SingleTypeAssetCache::default(),
        }
    }
    pub fn request_image(&self, source: &ImageSource) -> AssetHandle<ImageHandle> {
        self.images.request(
            source,
            Self::load_image(self.device.clone(), source.clone()),
        )
    }

    pub fn resolve_image(
        &self,
        handle: AssetHandle<ImageHandle>,
    ) -> Result<Arc<ImageHandle>, Error> {
        self.images.resolve(handle, self)
    }

    pub fn is_image_loaded(&self, handle: AssetHandle<ImageHandle>) -> bool {
        self.images.is_finished(handle, self)
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
                debug!("Loaded cached {}", source);
                return Ok(asset);
            }
        }
        debug!("Import asset {}", source);
        let asset = import.await?;
        if let Ok(mut file) = File::create(path) {
            if asset.serialize(&mut file).is_ok() {
                debug!("Written to cache {}", source);
            }
        }

        Ok(asset)
    }

    pub fn maintain(&self) {
        self.images.maintain(self);
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
}

fn get_cached_asset_path<S: EngineAssetKey>(source: &S, ext: &str) -> PathBuf {
    Path::new(ASSET_CACHE_PATH).join(format!("{:x}.{}", source.key(), ext))
}
