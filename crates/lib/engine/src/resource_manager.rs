use std::{collections::HashMap, future::Future, marker::PhantomData, sync::Arc};

use bevy_tasks::{block_on, IoTaskPool, Task};
use bytes::Bytes;
use dess_assets::{Asset, AssetRef, ContentSource, ShaderAsset, ShaderSource};
use dess_backend::vulkan::Device;
use dess_common::{Handle, Pool};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::{load_cached_asset, Error};

pub trait Resource: Send + Sync + 'static {
    fn is_finished<T: ResourceContext>(&self, ctx: &T) -> bool;
    fn resolve<T: ResourceContext>(&self, ctx: &T) -> Result<(), Error>;
}

pub trait ResourceContext {}

#[derive(Debug)]
enum ResourceState<T: Resource> {
    Pending(Task<Result<Arc<T>, Error>>),
    Resolving(Arc<T>),
    Loaded(Arc<T>),
    Failed(Error),
}

impl<T: Resource> ResourceState<T> {
    pub fn resolve<U: ResourceContext>(&mut self, ctx: &U) -> Result<Arc<T>, Error> {
        match self {
            ResourceState::Pending(task) => match block_on(task) {
                Ok(resource) => match resource.resolve(ctx) {
                    Ok(()) => {
                        *self = ResourceState::Loaded(resource.clone());
                        Ok(resource.clone())
                    }
                    Err(err) => {
                        *self = ResourceState::Failed(err.clone());
                        Err(err)
                    }
                },
                Err(err) => {
                    *self = ResourceState::Failed(err.clone());
                    Err(err)
                }
            },
            ResourceState::Resolving(resource) => match resource.resolve(ctx) {
                Ok(()) => {
                    let resource = resource.clone();
                    *self = ResourceState::Loaded(resource.clone());
                    Ok(resource)
                }
                Err(err) => {
                    *self = ResourceState::Failed(err.clone());
                    Err(err)
                }
            },
            ResourceState::Loaded(resource) => Ok(resource.clone()),
            ResourceState::Failed(err) => Err(err.clone()),
        }
    }

    pub fn is_finished<U: ResourceContext>(&self, ctx: &U) -> bool {
        match self {
            ResourceState::Pending(_) => false,
            ResourceState::Resolving(resource) => resource.is_finished(ctx),
            ResourceState::Loaded(_) => true,
            ResourceState::Failed(_) => true,
        }
    }

    pub fn maintain<U: ResourceContext>(&mut self, ctx: &U) {
        match self {
            ResourceState::Pending(task) => {
                if task.is_finished() {
                    match block_on(task) {
                        Ok(resource) => *self = ResourceState::Resolving(resource),
                        Err(err) => *self = ResourceState::Failed(err),
                    }
                }
            }
            ResourceState::Resolving(resource) => {
                if resource.is_finished(ctx) {
                    match resource.resolve(ctx) {
                        Ok(_) => *self = ResourceState::Loaded(resource.clone()),
                        Err(err) => *self = ResourceState::Failed(err),
                    }
                }
            }
            _ => {}
        }
    }
}

type ResourceHandle<T> = Handle<ResourceState<T>, ()>;

#[derive(Debug, Default)]
struct SingleTypeResourceCache<T: Resource, S: ContentSource> {
    handles: RwLock<HashMap<AssetRef, ResourceHandle<T>>>,
    assets: Mutex<Pool<ResourceState<T>, ()>>,
    _marker: PhantomData<S>,
}

impl<T: Resource, S: ContentSource> SingleTypeResourceCache<T, S> {
    pub fn request<F>(&self, asset: AssetRef, load: F) -> ResourceHandle<T>
    where
        F: Future<Output = Result<Arc<T>, Error>> + Send + Sync + 'static,
    {
        let handles = self.handles.upgradable_read();
        if let Some(handle) = handles.get(&asset) {
            *handle
        } else {
            let mut handles = RwLockUpgradableReadGuard::upgrade(handles);
            if let Some(handle) = handles.get(&asset) {
                *handle
            } else {
                let task = IoTaskPool::get().spawn(load);
                let handle = self.assets.lock().push(ResourceState::Pending(task), ());
                handles.insert(asset, handle);
                handle
            }
        }
    }
}

pub struct ResourceManager {
    device: Arc<Device>,
    shaders: RwLock<HashMap<ShaderSource, Bytes>>,
}

impl ResourceManager {
    pub fn new(device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            shaders: RwLock::default(),
        })
    }

    pub fn maintain(&self) {
        todo!()
    }

    fn try_resolve(&self, asset: AssetRef) -> Option<Result<Arc<dyn Asset>, Error>> {
        todo!()
    }

    pub fn get_or_load_shader_code(&self, source: ShaderSource) -> Result<Bytes, Error> {
        // Shaders are relatively small and fast to load. So we load all shaders at start of the
        // game and don't bother with it anymore.
        let shaders = self.shaders.upgradable_read();
        if let Some(code) = shaders.get(&source) {
            Ok(code.clone())
        } else {
            let mut shaders = RwLockUpgradableReadGuard::upgrade(shaders);
            if let Some(code) = shaders.get(&source) {
                Ok(code.clone())
            } else {
                let shader: ShaderAsset = load_cached_asset(&source)?;
                shaders.insert(source, shader.code.clone());
                Ok(shader.code)
            }
        }
    }
}
