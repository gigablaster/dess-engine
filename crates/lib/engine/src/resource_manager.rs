use std::{collections::HashMap, fmt::Debug, future::Future, hash::Hash, sync::Arc};

use arrayvec::ArrayVec;
use bevy_tasks::{block_on, IoTaskPool, Task};
use bytes::Bytes;
use dess_assets::{
    AssetRef, ContentSource, ImageAsset, ModelCollectionAsset, ShaderAsset, ShaderSource,
};
use dess_backend::vulkan::{
    Device, ImageCreateDesc, ImageHandle, ImageSubresourceData, ProgramHandle, ShaderDesc,
    MAX_SHADERS,
};
use dess_common::{Handle, Pool};
use log::{debug, error};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use siphasher::sip128::Hasher128;

use crate::{
    load_cached_asset, BasicPbrMaterialFactory, BasicUnlitMaterialFactory, BufferPool, Error,
    Material, MaterialFactoryCollection, ModelCollection,
};

pub trait ResourceDependencies: Send + Sync + Debug + 'static {
    fn is_finished(&self, ctx: &ResourceContext) -> bool;
    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), Error>;
}

pub trait Resource: ResourceDependencies {}

impl ResourceDependencies for ImageHandle {
    fn is_finished(&self, _ctx: &ResourceContext) -> bool {
        true
    }

    fn resolve(&mut self, _ctx: &ResourceContext) -> Result<(), Error> {
        Ok(())
    }
}

impl Resource for ImageHandle {}

pub struct ResourceContext<'a> {
    pub device: &'a Device,
    pub buffers: &'a BufferPool,
    images: &'a SingleTypeResourceCache<ImageHandle>,
    materials: &'a SingleTypeResourceCache<Material>,
}

impl<'a> ResourceContext<'a> {
    pub fn resolve_image(
        &self,
        handle: ResourceHandle<ImageHandle>,
    ) -> Result<Arc<ImageHandle>, Error> {
        self.images.resolve(handle, self)
    }

    pub fn is_image_finished(&self, handle: ResourceHandle<ImageHandle>) -> bool {
        self.images.is_finished(handle, self)
    }

    pub fn resolve_material(
        &self,
        handle: ResourceHandle<Material>,
    ) -> Result<Arc<Material>, Error> {
        self.materials.resolve(handle, self)
    }

    pub fn is_material_finished(&self, handle: ResourceHandle<Material>) -> bool {
        self.materials.is_finished(handle, self)
    }
}

pub trait ResourceLoader {
    fn request_image(&self, asset: AssetRef) -> ResourceHandle<ImageHandle>;
    fn request_material(
        &self,
        program: ProgramHandle,
        images: &[(String, AssetRef)],
        uniform: Bytes,
    ) -> ResourceHandle<Material>;
    fn request_model(&self, asset: AssetRef) -> ResourceHandle<ModelCollection>;
    fn get_or_load_program(&self, shaders: &[ShaderSource]) -> Result<ProgramHandle, Error>;
}

#[derive(Debug)]
pub enum ResourceState<T: Resource> {
    Pending(Task<Result<Arc<T>, Error>>),
    Resolving(Arc<T>),
    Loaded(Arc<T>),
    Failed(Error),
}

impl<T: Resource> ResourceState<T> {
    pub fn resolve(&mut self, ctx: &ResourceContext) -> Result<Arc<T>, Error> {
        match self {
            ResourceState::Pending(task) => match block_on(task) {
                Ok(mut resource) => match Arc::get_mut(&mut resource).unwrap().resolve(ctx) {
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
            ResourceState::Resolving(resource) => {
                match Arc::get_mut(resource).unwrap().resolve(ctx) {
                    Ok(()) => {
                        let resource = resource.clone();
                        *self = ResourceState::Loaded(resource.clone());
                        Ok(resource)
                    }
                    Err(err) => {
                        *self = ResourceState::Failed(err.clone());
                        Err(err)
                    }
                }
            }
            ResourceState::Loaded(resource) => Ok(resource.clone()),
            ResourceState::Failed(err) => Err(err.clone()),
        }
    }

    pub fn is_finished(&self, ctx: &ResourceContext) -> bool {
        match self {
            ResourceState::Pending(_) => false,
            ResourceState::Resolving(resource) => resource.is_finished(ctx),
            ResourceState::Loaded(_) => true,
            ResourceState::Failed(_) => true,
        }
    }

    pub fn maintain(&mut self, ctx: &ResourceContext) {
        match self {
            ResourceState::Pending(task) => {
                if task.is_finished() {
                    match block_on(task) {
                        Ok(resource) => *self = ResourceState::Resolving(resource),
                        Err(err) => {
                            error!("Failed to load asset: {:?}", err);
                            *self = ResourceState::Failed(err);
                        }
                    }
                }
            }
            ResourceState::Resolving(resource) => {
                if resource.is_finished(ctx) {
                    match Arc::get_mut(resource).unwrap().resolve(ctx) {
                        Ok(_) => *self = ResourceState::Loaded(resource.clone()),
                        Err(err) => {
                            error!("Failed to resolve asset: {:?}", err);
                            *self = ResourceState::Failed(err);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

pub type ResourceHandle<T> = Handle<ResourceState<T>>;

#[derive(Debug, Default)]
struct SingleTypeResourceCache<T: Resource> {
    handles: RwLock<HashMap<AssetRef, ResourceHandle<T>>>,
    assets: Mutex<Pool<ResourceState<T>>>,
}

impl<T: Resource> SingleTypeResourceCache<T> {
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
                let handle = self.assets.lock().push(ResourceState::Pending(task));
                handles.insert(asset, handle);
                handle
            }
        }
    }

    pub fn is_finished(&self, handle: ResourceHandle<T>, ctx: &ResourceContext) -> bool {
        if let Some(state) = self.assets.lock().get(handle) {
            state.is_finished(ctx)
        } else {
            false
        }
    }

    pub fn resolve(
        &self,
        handle: ResourceHandle<T>,
        ctx: &ResourceContext,
    ) -> Result<Arc<T>, Error> {
        if let Some(state) = self.assets.lock().get_mut(handle) {
            state.resolve(ctx)
        } else {
            Err(Error::InvalidHandle)
        }
    }

    pub fn maintain(&self, ctx: &ResourceContext) {
        self.assets.lock().for_each_mut(|hot| hot.maintain(ctx))
    }
}

type ProgramSource = ArrayVec<ShaderSource, MAX_SHADERS>;

pub struct ResourceManager {
    device: Arc<Device>,
    buffers: Arc<BufferPool>,
    material_factory: Arc<MaterialFactoryCollection>,
    shaders: RwLock<HashMap<ShaderSource, Bytes>>,
    programs: RwLock<HashMap<ProgramSource, ProgramHandle>>,
    images: SingleTypeResourceCache<ImageHandle>,
    materials: SingleTypeResourceCache<Material>,
    models: SingleTypeResourceCache<ModelCollection>,
}

impl ResourceManager {
    pub fn new(device: &Arc<Device>, buffers: &Arc<BufferPool>) -> Arc<Self> {
        let mut material_factory = MaterialFactoryCollection::default();
        material_factory.register(Box::new(BasicUnlitMaterialFactory));
        material_factory.register(Box::new(BasicPbrMaterialFactory));
        Arc::new(Self {
            buffers: buffers.clone(),
            device: device.clone(),
            shaders: RwLock::default(),
            programs: RwLock::default(),
            images: SingleTypeResourceCache::default(),
            materials: SingleTypeResourceCache::default(),
            models: SingleTypeResourceCache::default(),
            material_factory: Arc::new(material_factory),
        })
    }

    pub fn maintain(&self) {
        let context = ResourceContext {
            device: &self.device,
            buffers: &self.buffers,
            images: &self.images,
            materials: &self.materials,
        };
        self.images.maintain(&context);
        self.materials.maintain(&context);
        self.models.maintain(&context);
    }

    pub fn get_or_load_shader_code(&self, source: &ShaderSource) -> Result<Bytes, Error> {
        // Shaders are relatively small and fast to load. So we load all shaders at start of the
        // game and don't bother with it anymore.
        let shaders = self.shaders.upgradable_read();
        if let Some(code) = shaders.get(source) {
            Ok(code.clone())
        } else {
            let mut shaders = RwLockUpgradableReadGuard::upgrade(shaders);
            if let Some(code) = shaders.get(source) {
                Ok(code.clone())
            } else {
                let shader: ShaderAsset = load_cached_asset(source.get_ref())?;
                debug!("Loaded shader: {:?}", source);
                shaders.insert(source.clone(), shader.code.clone());
                Ok(shader.code)
            }
        }
    }

    async fn load_image(device: Arc<Device>, asset: AssetRef) -> Result<Arc<ImageHandle>, Error> {
        let asset: ImageAsset = load_cached_asset(asset)?;
        let data = asset
            .mips
            .iter()
            .map(|x| ImageSubresourceData {
                data: x,
                row_pitch: 0,
            })
            .collect::<Vec<_>>();
        let image = device.create_image(
            ImageCreateDesc::texture(asset.format, asset.dimensions)
                .mip_levels(asset.mips.len())
                .initial_data(&data),
        )?;
        Ok(Arc::new(image))
    }

    async fn load_material(
        loader: Arc<ResourceManager>,
        program: ProgramHandle,
        images: Vec<(String, AssetRef)>,
        uniform: Bytes,
    ) -> Result<Arc<Material>, Error> {
        Ok(Arc::new(Material::new(&loader, program, &images, uniform)))
    }

    async fn load_model(
        loader: Arc<ResourceManager>,
        asset: AssetRef,
        material_factory: Arc<MaterialFactoryCollection>,
    ) -> Result<Arc<ModelCollection>, Error> {
        let asset: ModelCollectionAsset = load_cached_asset(asset)?;
        Ok(Arc::new(ModelCollection::new(
            &loader,
            asset,
            &loader.buffers,
            material_factory.as_ref(),
        )?))
    }
}

impl ResourceLoader for Arc<ResourceManager> {
    fn request_image(&self, asset: AssetRef) -> ResourceHandle<ImageHandle> {
        self.images.request(
            asset,
            ResourceManager::load_image(self.device.clone(), asset),
        )
    }

    fn request_material(
        &self,
        program: ProgramHandle,
        images: &[(String, AssetRef)],
        uniform: Bytes,
    ) -> ResourceHandle<Material> {
        let mut hasher = siphasher::sip128::SipHasher::default();
        program.hash(&mut hasher);
        images.hash(&mut hasher);
        uniform.hash(&mut hasher);
        let asset = hasher.finish128().as_u128().into();
        self.materials.request(
            asset,
            ResourceManager::load_material(self.clone(), program, images.to_vec(), uniform),
        )
    }

    fn request_model(&self, asset: AssetRef) -> ResourceHandle<ModelCollection> {
        self.models.request(
            asset,
            ResourceManager::load_model(self.clone(), asset, self.material_factory.clone()),
        )
    }

    fn get_or_load_program(&self, shaders: &[ShaderSource]) -> Result<ProgramHandle, Error> {
        let key = shaders.iter().cloned().collect::<ProgramSource>();
        let programs = self.programs.upgradable_read();
        if let Some(program) = programs.get(&key) {
            Ok(*program)
        } else {
            let mut programs = RwLockUpgradableReadGuard::upgrade(programs);
            if let Some(program) = programs.get(&key) {
                Ok(*program)
            } else {
                let mut loaded = ArrayVec::<_, MAX_SHADERS>::new();
                for shader in shaders {
                    loaded.push(self.get_or_load_shader_code(shader)?);
                }
                let shaders = shaders
                    .iter()
                    .zip(loaded.iter())
                    .map(|(source, code)| ShaderDesc {
                        stage: source.stage.into(),
                        entry: "main",
                        code,
                    })
                    .collect::<ArrayVec<_, MAX_SHADERS>>();
                let program = self.device.create_program(&shaders)?;
                programs.insert(key, program);
                Ok(program)
            }
        }
    }
}
