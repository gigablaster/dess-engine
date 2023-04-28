mod texture;

use std::path::Path;

pub use texture::*;

use crate::ImportError;

pub enum Content {
    ColorTexture(Texture),
    NormalTexture(Texture),
    GrayscaleTexture(Texture),
    NotColorTexture(Texture),
    SpriteTexture(Texture),
}

pub trait ContentImporter {
    fn import(&self, path: &Path) -> Result<Content, ImportError>;
}
