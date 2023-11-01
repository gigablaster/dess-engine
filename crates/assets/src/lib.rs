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
    pub fn from_path(path: &Path) -> Self {
        Self::from_bytes(&path.to_str().unwrap().as_bytes())
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = siphasher::sip128::SipHasher::default().hash(bytes);
        Self {
            uuid: Uuid::from_u128(hash.as_u128()),
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
    /// Models that should be imported.
    models_to_process: HashMap<AssetRef, GltfSource>,
    /// Images that should be imported.
    images_to_process: HashMap<AssetRef, ImageSource>,
    /// Hashes for all inline assets, to catch and reuse same resources
    blobs: HashSet<AssetRef>,
}

unsafe impl Send for AssetProcessingContextImpl {}
unsafe impl Sync for AssetProcessingContextImpl {}

impl AssetProcessingContextImpl {
    pub fn import_model(&mut self, model: GltfSource) -> AssetRef {
        let path = get_relative_asset_path(&model.path).unwrap();
        if let Some(asset) = self.models.get(&path) {
            *asset
        } else {
            info!("Requested model import {:?}", path);
            let asset = AssetRef::from_path(&model.path);
            self.models.insert(path, asset);
            self.models_to_process.insert(asset, model);

            asset
        }
    }

    pub fn import_image(&mut self, image: ImageSource) -> AssetRef {
        match image.source {
            ImageDataSource::File(path) => {
                let path = get_relative_asset_path(&path).unwrap();
                if let Some(asset) = self.images.get(&path) {
                    *asset
                } else {
                    let asset = AssetRef::from_path(&path);
                    info!("Requested image import {:?} ref: {:?}", path, asset);
                    self.images.insert(path, asset);
                    self.images_to_process.insert(asset, image);

                    asset
                }
            }
            ImageDataSource::Bytes(bytes) => {
                let asset = AssetRef::from_bytes(&bytes);
                if self.blobs.contains(&asset) {
                    asset
                } else {
                    info!("Added image from blob ref {:?}", asset);
                    self.blobs.insert(asset);
                    self.images_to_process.insert(asset, image);

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
}

#[derive(Debug, Default)]
pub struct AssetProcessingContext {
    inner: Mutex<AssetProcessingContextImpl>,
}

unsafe impl Sync for AssetProcessingContext {}

impl AssetProcessingContext {
    pub fn import_model(&self, model: GltfSource) -> AssetRef {
        self.inner.lock().import_model(model)
    }

    pub fn import_image(&self, image: ImageSource) -> AssetRef {
        self.inner.lock().import_image(image)
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
