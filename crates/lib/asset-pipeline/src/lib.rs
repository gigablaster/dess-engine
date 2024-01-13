mod gltf;
mod image;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs::{self, File},
    io::{self, Read, Write},
    path::Path,
    sync::Arc,
    time::SystemTime,
};

use ::image::ImageError;
use bevy_tasks::AsyncComputeTaskPool;
use dess_assets::{get_absolute_asset_path, get_cached_asset_path, Asset, AssetRef, ContentSource};
pub use gltf::*;
pub use image::*;
use log::info;
use parking_lot::Mutex;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    ProcessingFailed(String),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ImageError> for Error {
    fn from(value: ImageError) -> Self {
        Self::ProcessingFailed(value.to_string())
    }
}

pub trait AssetImporter: ContentSource {
    fn import(&self, ctx: &dyn ImportContext) -> Result<Arc<dyn Asset>, Error>;
    fn is_changed(&self, timestamp: SystemTime) -> bool;
}

pub trait ImportContext {
    fn import(&self, content: Box<dyn AssetImporter>) -> AssetRef;
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

#[derive(Debug, Default)]
pub struct ContentProcessor {
    to_process: Mutex<HashMap<AssetRef, Box<dyn AssetImporter>>>,
    processed: Mutex<HashSet<AssetRef>>,
}

impl ImportContext for ContentProcessor {
    fn import(&self, content: Box<dyn AssetImporter>) -> AssetRef {
        let asset = content.get_ref();
        if self.processed.lock().contains(&asset) {
            return asset;
        }
        let mut to_process = self.to_process.lock();
        if to_process.contains_key(&asset) {
            return asset;
        }
        to_process.insert(asset, content);
        asset
    }
}

impl ContentProcessor {
    pub fn process(&self) {
        loop {
            let mut to_process = self
                .to_process
                .lock()
                .drain()
                .map(|(_, x)| x)
                .collect::<Vec<_>>();
            if to_process.is_empty() {
                break;
            }
            AsyncComputeTaskPool::get().scope(|s| {
                for content in to_process.drain(..) {
                    if Self::need_rebuild(content.as_ref()) {
                        s.spawn(self.do_process(content));
                    }
                }
            });
        }
    }

    pub fn need_rebuild(content: &dyn AssetImporter) -> bool {
        let path = get_cached_asset_path(content.get_ref());
        if path.exists() {
            if let Ok(metadata) = fs::metadata(path) {
                if let Ok(modified) = metadata.modified() {
                    return content.is_changed(modified);
                }
                if let Ok(created) = metadata.created() {
                    return content.is_changed(created);
                }
            }
        }
        true
    }

    async fn do_process(&self, content: Box<dyn AssetImporter>) {
        match self.do_process_impl(content.as_ref()).await {
            Ok(_) => info!("Processed {:?}", content),
            Err(err) => match err {
                Error::Io(io) => eprintln!("IO error: {:?}", io),
                Error::ProcessingFailed(desc) => eprintln!("{}", desc),
            },
        }
    }

    async fn do_process_impl(&self, content: &dyn AssetImporter) -> Result<(), Error> {
        let asset = content.get_ref();
        let data = content.import(self)?.to_bytes()?;
        File::create(get_cached_asset_path(asset))?.write_all(&data)?;
        self.processed.lock().insert(asset);
        Ok(())
    }
}

pub(crate) fn is_asset_changed<P: AsRef<Path>>(path: P, timestamp: SystemTime) -> bool {
    if let Ok(path) = get_absolute_asset_path(path) {
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                return modified > timestamp;
            }
            if let Ok(created) = metadata.created() {
                return created > timestamp;
            }
        }
    }
    false
}
