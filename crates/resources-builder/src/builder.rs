use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use chrono::{DateTime, Local};

use log::{error, info};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    content::{ContentImporterFactory, LoadedContent},
    BuildError,
};

pub trait Context {
    fn uuid(&self, path: &Path) -> Uuid;
    fn name(&self, uuid: Uuid, name: &str) -> Result<(), BuildError>;
    fn build<T: ProcessedResource, F: FnOnce() -> Result<T, BuildError>>(
        &self,
        uuid: Uuid,
        asset: AssetInfo,
        cb: F,
    ) -> Result<(), BuildError>;
    fn process(
        &self,
        content: LoadedContent,
        builder: &dyn ContentBuilder,
    ) -> Result<(), BuildError>;
    fn import(&self, path: &Path) -> Option<LoadedContent>;
    fn builder(
        &self,
        content: &LoadedContent,
        prefered: Option<&str>,
    ) -> Option<Box<dyn ContentBuilder>>;
}

pub trait ProcessedResource {
    fn write(&self, file: File) -> io::Result<()>;
}

pub trait ContentBuilderFactory: Debug {
    fn builder(&self, content: &LoadedContent) -> Option<Box<dyn ContentBuilder>>;
}

pub trait ContentBuilder: Debug {
    fn build(
        &self,
        content: LoadedContent,
        context: &Arc<Mutex<BuildContext>>,
    ) -> Result<Box<dyn ProcessedResource>, BuildError>;
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
            // Main must not be newer
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
    pub main: PathBuf,
    pub sources: HashSet<PathBuf>,
    pub dependencies: HashSet<Uuid>,
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
    importers: Vec<Box<dyn ContentImporterFactory>>,
    builders: HashMap<String, Box<dyn ContentBuilderFactory>>,
}

impl BuildContext {
    pub fn new(cache: &Path) -> Result<Arc<Mutex<Self>>, BuildError> {
        let cache_dir = PathBuf::from(cache);
        if !cache_dir.is_dir() {
            return Err(BuildError::WrongCache);
        }
        let cache_file = File::open(cache.join("cache.json"))?;
        let cache = serde_json::from_reader(cache_file)?;

        Ok(Arc::new(Mutex::new(Self {
            cache_dir,
            cache,
            root_assets: HashMap::new(),
            assets: HashMap::new(),
            importers: Vec::new(),
            builders: HashMap::new(),
        })))
    }

    pub fn flush(&mut self) -> Result<(), BuildError> {
        self.cache.set_named(&self.root_assets);
        let mut cache_file = File::create(self.cache_dir.join("cache.json"))?;
        serde_json::to_writer(&mut cache_file, &self.cache)?;

        Ok(())
    }

    pub fn add_importer(&mut self, importer: Box<dyn ContentImporterFactory>) {
        self.importers.push(importer);
    }

    pub fn add_processor(&mut self, name: &str, processor: Box<dyn ContentBuilderFactory>) {
        self.builders.insert(name.into(), processor);
    }
}

impl Context for Arc<Mutex<BuildContext>> {
    fn uuid(&self, path: &Path) -> Uuid {
        let mut builder = self.lock().unwrap();
        builder.cache.get_or_create_uuid(path)
    }

    fn name(&self, uuid: Uuid, name: &str) -> Result<(), BuildError> {
        let mut builder = self.lock().unwrap();
        if builder.root_assets.contains_key(name) {
            Err(BuildError::NameIsUsed(name.into()))
        } else {
            builder.root_assets.insert(name.into(), uuid);
            Ok(())
        }
    }

    fn build<T: ProcessedResource, F: FnOnce() -> Result<T, BuildError>>(
        &self,
        uuid: Uuid,
        asset: AssetInfo,
        cb: F,
    ) -> Result<(), BuildError> {
        let must_rebuild = {
            let builder = self.lock().unwrap();
            builder.cache.must_rebuild(uuid, &asset)?
        };
        if must_rebuild {
            info!("Build asset {:?} -> {}", asset.main, uuid);
            let data = cb()?;
            let file = {
                let builder = self.lock().unwrap();
                builder.cache_dir.join(uuid.as_braced().to_string())
            };
            let file = File::create(file)?;
            data.write(file)?;
            let mut builder = self.lock().unwrap();
            builder.cache.resource_built(uuid, asset)?;
        }

        Ok(())
    }

    fn process(
        &self,
        content: LoadedContent,
        builder: &dyn ContentBuilder,
    ) -> Result<(), BuildError> {
        builder.build(content, self)?;

        Ok(())
    }

    fn import(&self, path: &Path) -> Option<LoadedContent> {
        let importer = {
            let builder = self.lock().unwrap();
            builder
                .importers
                .iter()
                .find_map(|factory| factory.importer(path))
        };
        if let Some(importer) = importer {
            match importer.import(path) {
                Ok(content) => Some(content),
                Err(err) => {
                    error!("Failed to import {:?} - {:?}", path, err);
                    None
                }
            }
        } else {
            error!("No importer for {:?}", path);
            None
        }
    }

    fn builder(
        &self,
        content: &LoadedContent,
        prefered: Option<&str>,
    ) -> Option<Box<dyn ContentBuilder>> {
        let builder = self.lock().unwrap();
        if let Some(prefered) = prefered {
            if let Some(factory) = builder.builders.get(prefered) {
                factory.builder(content)
            } else {
                None
            }
        } else {
            builder
                .builders
                .iter()
                .find_map(|(_, builder)| builder.builder(content))
        }
    }
}
