use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Local};
use common::traits::BinarySerialization;
use log::info;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{content::Content, BuildError};

pub trait ContentBuilder<T: BinarySerialization> {
    fn build(content: Content, context: &mut BuildContext) -> Result<T, BuildError>;
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheEntry {
    pub main: PathBuf,
    pub timestamp: DateTime<Local>,
    pub sources: HashMap<PathBuf, DateTime<Local>>,
    pub dependencies: HashSet<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildCache {
    files: HashMap<PathBuf, Uuid>,
    cache: HashMap<Uuid, CacheEntry>,
    named: HashMap<String, Uuid>,
}

impl BuildCache {
    pub fn get_or_create_uuid(&mut self, path: &Path) -> Uuid {
        if let Some(uuid) = self.files.get(path) {
            *uuid
        } else {
            let uuid = Uuid::new_v4();
            self.files.insert(path.into(), uuid);
            uuid
        }
    }

    pub fn must_rebuild(&self, uuid: Uuid, asset: &AssetInfo) -> Result<bool, BuildError> {
        if let Some(assigned_uuid) = self.files.get(&asset.main) {
            if *assigned_uuid != uuid {
                return Err(BuildError::WrongUuid(asset.main.clone(), *assigned_uuid));
            }
        }
        if let Some(entry) = self.cache.get(&uuid) {
            // Main must be the same
            if asset.main != entry.main {
                return Ok(true);
            }
            // Main must not be never
            let modified = DateTime::<Local>::from(asset.main.metadata()?.modified()?);
            if modified > entry.timestamp {
                return Ok(true);
            }
            // All sources must present
            if entry.sources.len() != asset.sources.len() {
                return Ok(true);
            }
            // Sources must be same
            if asset.sources.iter().any(|x| !entry.sources.contains_key(x)) {
                return Ok(true);
            }
            // Sources mut not be newer
            for (path, timestamp) in &entry.sources {
                let modified = DateTime::<Local>::from(path.metadata()?.modified()?);
                if modified > *timestamp {
                    return Ok(true);
                }
            }

            Ok(false)
        } else {
            // File must be rebuild if it isn't in cache
            Ok(true)
        }
    }

    pub fn resource_built(&mut self, uuid: Uuid, asset: AssetInfo) -> Result<(), BuildError> {
        if let Some(assigned_uuid) = self.files.get(&asset.main) {
            if *assigned_uuid != uuid {
                return Err(BuildError::WrongUuid(asset.main.clone(), *assigned_uuid));
            }
        }
        let sources: HashMap<PathBuf, DateTime<Local>> = asset
            .sources
            .iter()
            .map(|path| {
                let timestamp =
                    DateTime::<Local>::from(path.metadata().unwrap().modified().unwrap());
                (PathBuf::from(path), timestamp)
            })
            .collect();
        let timestamp = DateTime::<Local>::from(asset.main.metadata()?.modified()?);
        self.cache.insert(
            uuid,
            CacheEntry {
                main: asset.main,
                timestamp,
                sources,
                dependencies: asset.dependencies,
            },
        );

        Ok(())
    }

    pub fn set_named(&mut self, named: &HashMap<String, Uuid>) {
        self.named = named.clone();
    }
}

#[derive(Debug, Clone)]
pub struct AssetInfo {
    main: PathBuf,
    sources: HashSet<PathBuf>,
    dependencies: HashSet<Uuid>,
}

impl AssetInfo {
    pub fn new(main: &Path) -> Self {
        Self {
            main: main.into(),
            sources: HashSet::new(),
            dependencies: HashSet::new(),
        }
    }

    pub fn add_source(mut self, path: &Path) -> Self {
        self.sources.insert(path.into());
        self
    }

    pub fn add_dependency(mut self, dependency: Uuid) -> Self {
        self.dependencies.insert(dependency);
        self
    }
}

#[derive(Debug)]
pub struct BuildContext {
    cache_dir: PathBuf,
    cache: BuildCache,
    root_assets: HashMap<String, Uuid>,
    assets: HashMap<Uuid, AssetInfo>,
}

impl BuildContext {
    pub fn new(cache: &Path) -> Result<Self, BuildError> {
        let cache_dir = PathBuf::from(cache);
        if !cache_dir.is_dir() {
            return Err(BuildError::WrongCache);
        }
        let cache_file = File::open(cache.join("cache.json"))?;
        let cache = serde_json::from_reader(cache_file)?;

        Ok(Self {
            cache_dir,
            cache,
            root_assets: HashMap::new(),
            assets: HashMap::new(),
        })
    }

    pub fn flush(&mut self) -> Result<(), BuildError> {
        self.cache.set_named(&self.root_assets);
        let mut cache_file = File::create(self.cache_dir.join("cache.json"))?;
        serde_json::to_writer(&mut cache_file, &self.cache)?;

        Ok(())
    }

    pub fn uuid(&mut self, path: &Path) -> Uuid {
        self.cache.get_or_create_uuid(path)
    }

    pub fn name(&mut self, uuid: Uuid, name: &str) -> Result<(), BuildError> {
        if self.root_assets.contains_key(name) {
            Err(BuildError::NameIsUsed(name.into()))
        } else {
            self.root_assets.insert(name.into(), uuid);
            Ok(())
        }
    }

    pub fn build<T: BinarySerialization, F: FnOnce() -> Result<T, BuildError>>(
        &mut self,
        uuid: Uuid,
        asset: AssetInfo,
        cb: F,
    ) -> Result<(), BuildError> {
        if self.cache.must_rebuild(uuid, &asset)? {
            info!("Build asset {:?} -> {}", asset.main, uuid);
            let data = cb()?;
            let file = self.cache_dir.join(uuid.as_braced().to_string());
            let mut file = File::create(file)?;
            data.serialize(&mut file)?;
            self.cache.resource_built(uuid, asset)?;
        }

        Ok(())
    }

    pub fn process<T: BinarySerialization, B: ContentBuilder<T>>(
        &mut self,
        content: Content,
    ) -> Result<(), BuildError> {
        B::build(content, self)?;

        Ok(())
    }
}
