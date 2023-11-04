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
    fs::{self},
    io::{self, Read},
    path::{Path, PathBuf},
};

use dess_assets::{Asset, AssetRef, GpuImage, GpuModel};
use log::info;
use parking_lot::Mutex;

mod bundler;
mod gltf_import;
mod image_import;
mod piepline;

pub use bundler::*;
pub use gltf_import::*;
pub use image_import::*;
pub use piepline::*;
use uuid::Uuid;

#[derive(Debug)]
pub enum Error {
    OutOfDataPath(PathBuf),
    ImportFailed,
    Unsupported,
    WrongDependency,
    ProcessingFailed,
    BadSourceData,
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::Io(value)
    }
}

pub const ROOT_DATA_PATH: &str = "data";
pub const ASSET_CACHE_PATH: &str = ".cache";

#[derive(Debug, Default)]
struct AssetProcessingContextImpl {
    /// All currently imported models.
    models: HashMap<PathBuf, AssetRef>,
    /// All currently imported images.
    images: HashMap<PathBuf, AssetRef>,
    /// Asset names. Only named assets can be requested.
    names: HashMap<String, AssetRef>,
    /// Models that should be imported.
    models_to_process: HashMap<AssetRef, GltfSource>,
    /// Images that should be imported.
    images_to_process: HashMap<AssetRef, ImageSource>,
    /// Hashes for all inline assets, to catch and reuse same resources
    assets: HashSet<(AssetRef, Uuid)>,
    /// Asset ownership. Every asset have source, direct or indirect.
    owners: HashMap<AssetRef, PathBuf>,
}

unsafe impl Sync for AssetProcessingContextImpl {}

impl AssetProcessingContextImpl {
    pub fn import_model(&mut self, model: &GltfSource) -> AssetRef {
        let path = get_relative_asset_path(&model.path).unwrap();
        if let Some(asset) = self.models.get(&path) {
            *asset
        } else {
            info!("Requested model import {:?}", path);
            let asset = AssetRef::from_path(&model.path);
            self.set_name(
                asset,
                path.as_os_str().to_ascii_lowercase().to_str().unwrap(),
            );
            self.models.insert(path.clone(), asset);
            self.models_to_process.insert(asset, model.clone());
            self.assets.insert((asset, GpuModel::TYPE_ID));
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
                    let asset = AssetRef::from_path(&path);
                    info!("Requested image import {:?} ref: {}", path, asset);
                    self.images.insert(path.clone(), asset);
                    self.images_to_process.insert(asset, image.clone());
                    self.assets.insert((asset, GpuImage::TYPE_ID));
                    let owner = owner.unwrap_or(&path);
                    self.add_source(asset, owner);

                    asset
                }
            }
            ImageDataSource::Bytes(bytes) => {
                let asset = AssetRef::from_bytes(bytes);
                if self.assets.contains(&(asset, GpuImage::TYPE_ID)) {
                    asset
                } else {
                    let owner =
                        get_relative_asset_path(owner.expect("Can't add image without owner"))
                            .unwrap();
                    info!("Added image from blob ref {} owner {:?}", asset, owner);
                    self.assets.insert((asset, GpuImage::TYPE_ID));
                    self.images_to_process.insert(asset, image.clone());
                    self.add_source(asset, &owner);

                    asset
                }
            }
        }
    }

    pub fn set_name(&mut self, asset: AssetRef, name: &str) {
        self.names.insert(name.into(), asset);
    }

    pub fn drain_models_to_process(&mut self) -> Vec<(AssetRef, GltfSource)> {
        self.models_to_process.drain().collect()
    }

    pub fn drain_images_to_process(&mut self) -> Vec<(AssetRef, ImageSource)> {
        self.images_to_process.drain().collect()
    }

    pub fn all_assets(&self) -> Vec<(AssetRef, Uuid)> {
        self.assets.iter().copied().collect()
    }

    pub fn all_names(&self) -> Vec<(String, AssetRef)> {
        self.names
            .iter()
            .map(|(x, y)| (x.clone(), *y))
            .collect::<Vec<_>>()
    }

    pub fn get_owner(&self, asset: AssetRef) -> Option<PathBuf> {
        self.owners.get(&asset).cloned()
    }

    pub fn add_source(&mut self, asset: AssetRef, owner: &Path) {
        self.owners.insert(asset, owner.into());
    }

    pub fn get_model_id(&self, path: &Path) -> Option<AssetRef> {
        self.models
            .get(&get_relative_asset_path(path).unwrap())
            .copied()
    }
}

#[derive(Debug, Default)]
pub struct AssetProcessingContext {
    inner: Mutex<AssetProcessingContextImpl>,
}

unsafe impl Sync for AssetProcessingContext {}

impl AssetProcessingContext {
    pub fn import_model(&self, model: &GltfSource) -> AssetRef {
        self.inner.lock().import_model(model)
    }

    pub fn import_image(&self, image: &ImageSource, owner: Option<&Path>) -> AssetRef {
        self.inner.lock().import_image(image, owner)
    }

    pub fn drain_models_to_process(&self) -> Vec<(AssetRef, GltfSource)> {
        self.inner.lock().drain_models_to_process()
    }

    pub fn drain_images_to_process(&self) -> Vec<(AssetRef, ImageSource)> {
        self.inner.lock().drain_images_to_process()
    }

    pub fn all_assets(&self) -> Vec<(AssetRef, Uuid)> {
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
}

pub trait Content {}

pub trait ContentImporter<T: Content> {
    fn import(&self) -> Result<T, Error>;
}

pub trait ContentProcessor<T: Content, U: Asset>: Default {
    fn process(&self, context: &AssetProcessingContext, content: T) -> Result<U, Error>;
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
