mod texture;

use std::path::{Path, PathBuf};

pub use texture::*;

use crate::ImportError;

#[derive(Debug, Clone)]
pub enum Content {
    Image(image::DynamicImage),
}

pub struct LoadedContent {
    pub path: PathBuf,
    pub content: Content,
}

pub trait ContentImporter {
    fn import(&self, path: &Path) -> Result<LoadedContent, ImportError>;
}
