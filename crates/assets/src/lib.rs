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
    env, io,
    path::{Path, PathBuf},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use gltf_import::GltfSource;
use image::{ImageDataSource, ImageSource};
use log::info;
use parking_lot::Mutex;
use uuid::Uuid;

mod assetdb;
mod gltf_import;
mod gpumesh;
mod gpumodel;
mod image;
mod material;

#[derive(Debug)]
pub enum Error {
    OutOfDataPath(PathBuf),
    Import,
    Unsupported,
    WrongDependency,
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::Io(value)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetRef {
    uuid: Uuid,
}

impl AssetRef {
    pub fn create() -> Self {
        Self {
            uuid: Uuid::new_v4(),
        }
    }
    pub fn valid(&self) -> bool {
        !self.uuid.is_nil()
    }

    pub fn as_path(&self) -> PathBuf {
        format!("{}/{}", CACHE_PATH, self.uuid.hyphenated()).into()
    }
}

impl BinarySerialization for AssetRef {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u128::<LittleEndian>(self.uuid.as_u128())
    }
}

impl BinaryDeserialization for AssetRef {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        Ok(Self {
            uuid: Uuid::from_u128_le(r.read_u128::<LittleEndian>()?),
        })
    }
}

pub trait Asset {
    const TYPE_ID: Uuid;
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>);
}

const ROOT_DATA_PATH: &str = "data";
const CACHE_PATH: &str = ".cache";

#[derive(Debug, Default)]
struct AssetProcessingContextImpl {
    /// All currently imported models.
    models: HashMap<PathBuf, AssetRef>,
    /// All currently imported images.
    images: HashMap<PathBuf, AssetRef>,
    /// Asset names. Only named assets can be requested.
    names: HashMap<String, AssetRef>,
    /// Which asset created from which file. Needed to regenerate extracted assets
    /// when original file was changed.
    dependencies: HashMap<AssetRef, PathBuf>,
    /// Models that should be imported.
    models_to_process: HashMap<AssetRef, GltfSource>,
    /// Images that should be imported.
    images_to_process: HashMap<AssetRef, ImageSource>,
}

unsafe impl Send for AssetProcessingContextImpl {}
unsafe impl Sync for AssetProcessingContextImpl {}

impl AssetProcessingContextImpl {
    pub fn import_model(&mut self, model: GltfSource) -> Result<AssetRef, Error> {
        let path = get_relative_asset_path(&model.path)?;
        if let Some(asset) = self.models.get(&path) {
            Ok(*asset)
        } else {
            info!("Requested model import {:?}", path);
            let asset = AssetRef::create();
            self.models.insert(path, asset);
            self.models_to_process.insert(asset, model);
            self.dependencies.insert(asset, path);

            Ok(asset)
        }
    }

    pub fn import_image(
        &mut self,
        image: ImageSource,
        origin: Option<&Path>,
    ) -> Result<AssetRef, Error> {
        match image.source {
            ImageDataSource::File(path) => {
                let path = get_relative_asset_path(&path)?;
                if let Some(asset) = self.images.get(&path) {
                    Ok(*asset)
                } else {
                    info!("Requested image import {:?}", path);
                    let asset = AssetRef::create();
                    self.images.insert(path, asset);
                    self.images_to_process.insert(asset, image);
                    self.dependencies.insert(asset, path);

                    Ok(asset)
                }
            }
            ImageDataSource::Bytes(_) if origin.is_some() => {
                let origin = origin.unwrap().into();
                info!("Added image extracted from {:?}", origin);
                let asset = AssetRef::create();
                self.images_to_process.insert(asset, image);
                self.dependencies.insert(asset, origin);

                Ok(asset)
            }
            ImageDataSource::Bytes(_) if origin.is_none() => {
                let asset = AssetRef::create();
                info!("Added image with no origin");
                self.images_to_process.insert(asset, image);

                Ok(asset)
            }
            _ => unreachable!(),
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
}

#[derive(Debug, Default)]
pub struct AssetProcessingContext {
    inner: Mutex<AssetProcessingContextImpl>,
}

unsafe impl Sync for AssetProcessingContext {}

impl AssetProcessingContext {
    pub fn import_model(&self, model: GltfSource) -> Result<AssetRef, Error> {
        self.inner.lock().import_model(model)
    }

    pub fn import_image(
        &self,
        image: ImageSource,
        origin: Option<&Path>,
    ) -> Result<AssetRef, Error> {
        self.inner.lock().import_image(image, origin)
    }

    pub fn drain_models_to_process(&self) -> Vec<(AssetRef, GltfSource)> {
        self.inner.lock().drain_models_to_process()
    }

    pub fn drain_images_to_process(&self) -> Vec<(AssetRef, ImageSource)> {
        self.inner.lock().drain_images_to_process()
    }
}

pub trait Content {}

pub trait ContentImporter<T: Content> {
    fn import(&self) -> anyhow::Result<T>;
}

pub trait ContentProcessor<T: Content, U: Asset> {
    fn process(&self, content: T) -> anyhow::Result<U>;
}

fn get_relative_asset_path(path: &Path) -> Result<PathBuf, Error> {
    if path.is_absolute() {
        let prefix = env::current_dir()?.join(ROOT_DATA_PATH);
        if !path.starts_with(prefix) {
            return Err(Error::OutOfDataPath(path.into()));
        }
        Ok(path.strip_prefix(prefix).unwrap().into())
    } else {
        Ok(path.strip_prefix(ROOT_DATA_PATH).unwrap().into())
    }
}
