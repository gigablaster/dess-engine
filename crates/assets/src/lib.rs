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
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use image::ImagePurpose;
use log::info;
use parking_lot::Mutex;
use siphasher::sip128::{Hasher128, SipHasher};

mod gltf_import;
mod gpumesh;
mod gpumodel;
mod image;
mod material;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AssetRef {
    hash: u128,
}

impl From<String> for AssetRef {
    fn from(value: String) -> Self {
        Self {
            hash: SipHasher::default().hash(value.as_bytes()).as_u128(),
        }
    }
}

impl From<Option<String>> for AssetRef {
    fn from(value: Option<String>) -> Self {
        if let Some(name) = value {
            name.into()
        } else {
            AssetRef::default()
        }
    }
}

impl From<ImageRef> for AssetRef {
    fn from(value: ImageRef) -> Self {
        let mut hasher = SipHasher::default();
        hasher.write(value.path.to_str().unwrap().as_bytes());
        hasher.write(&[value.purpose as u8]);
        Self {
            hash: hasher.finish128().as_u128(),
        }
    }
}

impl AssetRef {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        path.to_str().unwrap().to_string().into()
    }

    pub fn valid(&self) -> bool {
        self.hash != 0
    }
}

impl BinarySerialization for AssetRef {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_u128::<LittleEndian>(self.hash)
    }
}

impl BinaryDeserialization for AssetRef {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let hash = r.read_u128::<LittleEndian>()?;

        Ok(Self { hash })
    }
}

pub trait AssetDependencies {
    fn collect_dependencies(&self, deps: &mut Vec<AssetRef>);
}

pub(crate) const ROOT_DATA_PATH: &str = "data";

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct ImageRef {
    pub path: PathBuf,
    pub purpose: ImagePurpose,
}

#[derive(Debug, Default)]
struct AssetProcessingContextImpl {
    models: HashMap<PathBuf, AssetRef>,
    images: HashMap<ImageRef, AssetRef>,
    models_to_process: HashSet<PathBuf>,
    images_to_process: HashSet<ImageRef>,
}

unsafe impl Send for AssetProcessingContextImpl {}
unsafe impl Sync for AssetProcessingContextImpl {}

impl AssetProcessingContextImpl {
    fn root_path(path: impl Into<PathBuf>) -> PathBuf {
        path.into()
            .strip_prefix(ROOT_DATA_PATH)
            .unwrap()
            .parent()
            .unwrap()
            .into()
    }

    fn get_path(path: &Path) -> PathBuf {
        let root = Self::root_path(path);
        let name = path.file_name().unwrap();

        root.join(name)
    }

    pub fn import_model(&mut self, path: &Path) -> AssetRef {
        let path = Self::get_path(path);
        if let Some(asset) = self.models.get(&path) {
            *asset
        } else {
            info!("Request model import {:?}", path);
            let asset = AssetRef::from_path(&path);
            self.models.insert(path.clone(), asset);
            self.models_to_process.insert(path);

            asset
        }
    }

    pub fn import_image(&mut self, path: &Path, purpose: ImagePurpose) -> AssetRef {
        let path = Self::get_path(path);
        let image_ref = ImageRef { path, purpose };
        if let Some(asset) = self.images.get(&image_ref) {
            *asset
        } else {
            info!("Request texture import {:?}", image_ref);
            let asset = AssetRef::from_path(&image_ref.path);
            self.images.insert(image_ref.clone(), asset);
            self.images_to_process.insert(image_ref);

            asset
        }
    }

    pub fn drain_models_to_process(&mut self) -> Vec<PathBuf> {
        self.models_to_process.drain().collect()
    }

    pub fn drain_images_to_process(&mut self) -> Vec<ImageRef> {
        self.images_to_process.drain().collect()
    }
}

#[derive(Debug, Default)]
pub struct AssetProcessingContext {
    inner: Mutex<AssetProcessingContextImpl>,
}

unsafe impl Sync for AssetProcessingContext {}

impl AssetProcessingContext {
    pub fn import_model(&self, path: &Path) -> AssetRef {
        self.inner.lock().import_model(path)
    }

    pub fn import_image(&self, path: &Path, purpose: ImagePurpose) -> AssetRef {
        self.inner.lock().import_image(path, purpose)
    }

    pub fn drain_models_to_process(&self) -> Vec<PathBuf> {
        self.inner.lock().drain_models_to_process()
    }

    pub fn drain_images_to_process(&self) -> Vec<ImageRef> {
        self.inner.lock().drain_images_to_process()
    }
}
