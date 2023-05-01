mod text;
mod texture;

use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

pub use text::*;
pub use texture::*;

use crate::ImportError;

#[derive(Debug, Clone)]
pub enum Content {
    Image(image::DynamicImage),
    Text(String),
}

#[derive(Debug, Clone)]
pub struct LoadedContent {
    pub path: PathBuf,
    pub content: Content,
}

pub trait ContentImporter: Debug {
    fn import(&self, path: &Path) -> Result<LoadedContent, ImportError>;
}

pub trait ContentImporterFactory: Debug + Sync + Send {
    fn importer(&self, path: &Path) -> Option<Box<dyn ContentImporter>>;
}
