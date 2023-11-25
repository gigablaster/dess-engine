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
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    io::{self, BufWriter, Read},
    path::{Path, PathBuf},
};

use desc::ShaderSource;
use dess_assets::{Asset, AssetRef, ImageAsset, ModelAsset, ShaderAsset};
use log::info;
use parking_lot::Mutex;

mod bundler;
pub mod desc;
mod gltf_import;
mod image_import;
mod import_shader;
mod pipeline;

pub use bundler::*;
pub use gltf_import::*;
pub use image_import::*;
pub use import_shader::*;
pub use pipeline::*;
use speedy::{Readable, Writable};
use uuid::Uuid;

#[derive(Debug)]
pub enum Error {
    OutOfDataPath(PathBuf),
    ImportFailed(String),
    Unsupported,
    WrongDependency,
    ProcessingFailed(String),
    BadSourceData,
    Fail,
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error::ImportFailed(value.to_string())
    }
}

pub const ROOT_DATA_PATH: &str = "assets";
pub const ASSET_CACHE_PATH: &str = ".cache";
pub const BUNDLE_DESC_PATH: &str = "bundles";

#[derive(Debug, Default)]
struct AssetProcessingContextImpl {
    /// All currently imported models.
    models: HashMap<PathBuf, AssetRef>,
    /// All currently imported images.
    images: HashMap<PathBuf, AssetRef>,
    /// All currently imported shaders
    shaders: HashMap<PathBuf, AssetRef>,
    /// Asset names. Only named assets can be requested.
    names: HashMap<String, AssetRef>,
    /// Models that should be imported.
    models_to_process: HashMap<AssetRef, GltfSource>,
    /// Images that should be imported.
    images_to_process: HashMap<AssetRef, ImageSource>,
    /// Shaders that should be imported
    shaders_to_process: HashMap<AssetRef, ShaderSource>,
    /// Hashes for all inline assets, to catch and reuse same resources
    assets: HashSet<AssetInfo>,
    /// Asset ownership. Every asset have source, direct or indirect.
    owners: HashMap<AssetRef, String>,
    /// Asset dependencies
    dependencies: HashMap<AssetRef, HashSet<AssetRef>>,
}

unsafe impl Sync for AssetProcessingContextImpl {}

impl AssetProcessingContextImpl {
    pub fn from_database(asset_database: &AssetDatabase) -> Self {
        Self {
            names: asset_database.names.clone(),
            assets: asset_database.assets.clone(),
            owners: asset_database.owners.clone(),
            dependencies: asset_database.dependencies.clone(),
            ..Default::default()
        }
    }

    pub fn to_database(&self) -> AssetDatabase {
        AssetDatabase {
            assets: self.assets.clone(),
            dependencies: self.dependencies.clone(),
            names: self.names.clone(),
            owners: self.owners.clone(),
        }
    }

    pub fn import_model(&mut self, model: &GltfSource) -> AssetRef {
        let path = get_relative_asset_path(&model.path).unwrap();
        if let Some(asset) = self.models.get(&path) {
            *asset
        } else {
            info!("Requested model import {:?}", path);
            let asset = AssetRef::from_path(&model.path);
            self.models.insert(path.clone(), asset);
            self.models_to_process.insert(asset, model.clone());
            self.assets.insert(AssetInfo::new::<ModelAsset>(asset));
            self.add_source(asset, &path);

            asset
        }
    }

    pub fn import_image(&mut self, image: &ImageSource, owner: Option<&Path>) -> AssetRef {
        match &image.source {
            ImageDataSource::File(path) => {
                let path = get_relative_asset_path(path).unwrap();
                if let Some(asset) = self.images.get(&path) {
                    *asset
                } else {
                    let asset = AssetRef::from_path_with(&path, &image.purpose);
                    info!("Requested image import {:?} ref: {}", path, asset);
                    self.images.insert(path.clone(), asset);
                    self.images_to_process.insert(asset, image.clone());
                    self.assets.insert(AssetInfo::new::<ImageAsset>(asset));
                    let owner = owner.unwrap_or(&path);
                    self.add_source(asset, owner);

                    asset
                }
            }
            ImageDataSource::Bytes(bytes) => {
                let asset = AssetRef::from_bytes(bytes);
                if self.assets.contains(&AssetInfo::new::<ImageAsset>(asset)) {
                    asset
                } else {
                    let owner =
                        get_relative_asset_path(owner.expect("Can't add image without owner"))
                            .unwrap();
                    info!("Added image from blob ref {} owner {:?}", asset, owner);
                    self.assets.insert(AssetInfo::new::<ImageAsset>(asset));
                    self.images_to_process.insert(asset, image.clone());
                    self.add_source(asset, &owner);

                    asset
                }
            }
            ImageDataSource::Placeholder(value) => {
                let asset = AssetRef::from_bytes_with(value, &image.purpose);
                if self.assets.contains(&AssetInfo::new::<ImageAsset>(asset)) {
                    asset
                } else {
                    let owner =
                        get_relative_asset_path(owner.expect("Can't add image without owner"))
                            .unwrap();
                    info!("Added image placeholder {} owner {:?}", asset, owner);
                    self.assets.insert(AssetInfo::new::<ImageAsset>(asset));
                    self.images_to_process.insert(asset, image.clone());
                    self.add_source(asset, &owner);

                    asset
                }
            }
        }
    }

    pub fn import_shader(&mut self, shader: &ShaderSource) -> AssetRef {
        let path = get_relative_asset_path(Path::new(&shader.source)).unwrap();
        if let Some(asset) = self.shaders.get(&path) {
            *asset
        } else {
            let asset = AssetRef::from_path(&path);
            info!("Requested shader import {:?} ref: {}", shader, asset);
            self.assets.insert(AssetInfo::new::<ShaderAsset>(asset));
            self.shaders_to_process.insert(asset, shader.clone());
            self.add_source(asset, &path);

            asset
        }
    }

    pub fn clear_dependencies(&mut self, asset: AssetRef) {
        self.dependencies.remove(&asset);
    }

    pub fn add_dependency(&mut self, from: AssetRef, to: AssetRef) {
        self.dependencies.entry(from).or_default().insert(to);
    }

    pub fn set_name(&mut self, asset: AssetRef, name: &str) {
        self.names.insert(name.replace('\\', "/"), asset);
    }

    pub fn drain_models_to_process(&mut self) -> Vec<(AssetRef, GltfSource)> {
        self.models_to_process.drain().collect()
    }

    pub fn drain_images_to_process(&mut self) -> Vec<(AssetRef, ImageSource)> {
        self.images_to_process.drain().collect()
    }

    pub fn drain_shaders_to_process(&mut self) -> Vec<(AssetRef, ShaderSource)> {
        self.shaders_to_process.drain().collect()
    }

    pub fn all_assets(&self) -> Vec<AssetInfo> {
        self.assets.iter().copied().collect()
    }

    pub fn all_names(&self) -> Vec<(String, AssetRef)> {
        self.names
            .iter()
            .map(|(x, y)| (x.clone(), *y))
            .collect::<Vec<_>>()
    }

    pub fn get_owner(&self, asset: AssetRef) -> Option<PathBuf> {
        self.owners.get(&asset).map(|x| Path::new(x).to_owned())
    }

    pub fn add_source(&mut self, asset: AssetRef, owner: &Path) {
        self.owners
            .insert(asset, owner.to_str().unwrap().to_owned());
    }

    pub fn get_model_id(&self, path: &Path) -> Option<AssetRef> {
        self.models
            .get(&get_relative_asset_path(path).unwrap())
            .copied()
    }

    pub fn get_dependencies(&self, asset: AssetRef) -> Option<Vec<AssetRef>> {
        self.dependencies
            .get(&asset)
            .map(|deps| deps.iter().copied().collect())
    }
}

#[derive(Debug, Default)]
pub struct AssetProcessingContext {
    inner: Mutex<AssetProcessingContextImpl>,
}

unsafe impl Sync for AssetProcessingContext {}

impl AssetProcessingContext {
    pub fn from_database(asset_database: &AssetDatabase) -> Self {
        Self {
            inner: Mutex::new(AssetProcessingContextImpl::from_database(asset_database)),
        }
    }

    pub fn to_database(&self) -> AssetDatabase {
        self.inner.lock().to_database()
    }

    pub fn import_model(&self, model: &GltfSource) -> AssetRef {
        self.inner.lock().import_model(model)
    }

    pub fn import_image(&self, image: &ImageSource, owner: Option<&Path>) -> AssetRef {
        self.inner.lock().import_image(image, owner)
    }

    pub fn import_effect(&self, effect: &ShaderSource) -> AssetRef {
        self.inner.lock().import_shader(effect)
    }

    pub fn drain_models_to_process(&self) -> Vec<(AssetRef, GltfSource)> {
        self.inner.lock().drain_models_to_process()
    }

    pub fn drain_images_to_process(&self) -> Vec<(AssetRef, ImageSource)> {
        self.inner.lock().drain_images_to_process()
    }

    pub fn drain_effects_to_process(&self) -> Vec<(AssetRef, ShaderSource)> {
        self.inner.lock().drain_shaders_to_process()
    }

    pub fn all_assets(&self) -> Vec<AssetInfo> {
        self.inner.lock().all_assets()
    }

    pub fn all_names(&self) -> Vec<(String, AssetRef)> {
        self.inner.lock().all_names()
    }

    pub fn get_owner(&self, asset: AssetRef) -> Option<PathBuf> {
        self.inner.lock().get_owner(asset)
    }

    pub fn get_model_id(&self, path: &Path) -> Option<AssetRef> {
        self.inner.lock().get_model_id(path)
    }

    pub fn clear_dependencies(&self, asset: AssetRef) {
        self.inner.lock().clear_dependencies(asset);
    }

    pub fn add_dependency(&self, from: AssetRef, to: AssetRef) {
        self.inner.lock().add_dependency(from, to);
    }

    pub fn get_dependencies(&self, asset: AssetRef) -> Option<Vec<AssetRef>> {
        self.inner.lock().get_dependencies(asset)
    }

    pub fn set_name(&self, asset: AssetRef, name: &str) {
        self.inner.lock().set_name(asset, name);
    }
}

pub trait Content {}

pub trait ContentImporter<T: Content> {
    fn import(&self) -> Result<T, Error>;
}

pub trait ContentProcessor<T: Content, U: Asset>: Default {
    fn process(
        &self,
        asset: AssetRef,
        context: &AssetProcessingContext,
        content: T,
    ) -> Result<U, Error>;
}

pub(crate) fn get_relative_asset_path(path: &Path) -> Result<PathBuf, Error> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    // Is this path relative to data folder? Check this option.
    let path = if !path.exists() {
        root.join(path)
    } else {
        path.into()
    };
    let path = path.canonicalize()?;

    Ok(path.strip_prefix(root).unwrap().into())
}

pub(crate) fn get_absolute_asset_path(path: &Path) -> Result<PathBuf, Error> {
    let root = env::current_dir()?.canonicalize()?.join(ROOT_DATA_PATH);
    Ok(root.join(get_relative_asset_path(path)?))
}

pub(crate) fn read_to_end<P>(path: P) -> io::Result<Vec<u8>>
where
    P: AsRef<Path>,
{
    let file = fs::File::open(path.as_ref())?;
    // Allocate one extra byte so the buffer doesn't need to grow before the
    // final `read` call at the end of the file.  Don't worry about `usize`
    // overflow because reading will fail regardless in that case.
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}

fn cached_asset_path(asset: AssetRef) -> PathBuf {
    Path::new(ASSET_CACHE_PATH).join(Path::new(&format!("{}.bin", asset)))
}

#[derive(Debug, Default, Readable, Writable)]
pub struct AssetDatabase {
    pub assets: HashSet<AssetInfo>,
    pub dependencies: HashMap<AssetRef, HashSet<AssetRef>>,
    pub names: HashMap<String, AssetRef>,
    pub owners: HashMap<AssetRef, String>,
}

impl AssetDatabase {
    pub fn try_load(name: &str) -> Option<AssetDatabase> {
        if let Ok(file) = File::open(Path::new(ASSET_CACHE_PATH).join(format!("{}.db", name))) {
            if let Ok(database) = AssetDatabase::read_from_stream_buffered(file) {
                return Some(database);
            }
        }
        None
    }

    pub fn save(&self, name: &str) -> io::Result<()> {
        Ok(self.write_to_stream(&mut BufWriter::new(File::create(
            Path::new(ASSET_CACHE_PATH).join(format!("{}.db", name)),
        )?))?)
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Readable, Writable)]
pub struct AssetInfo {
    pub asset: AssetRef,
    pub ty: Uuid,
}

impl AssetInfo {
    pub fn new<T: Asset>(asset: AssetRef) -> Self {
        Self {
            asset,
            ty: T::TYPE_ID,
        }
    }
}
