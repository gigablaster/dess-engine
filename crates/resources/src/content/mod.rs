mod error;
mod texture;

use std::path::Path;

pub use error::*;
pub use texture::*;

pub enum Content {
    Texture(Texture),
    Text(String),
    Binary(Vec<u8>),
}

pub trait ContentImporter {
    fn import(path: &Path) -> Result<Content, ContentError>;
}
