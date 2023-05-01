use std::{io, path::PathBuf};

use image::ImageError;
use uuid::Uuid;

mod builder;
mod content;

#[derive(Debug)]
pub enum ImportError {
    IO(io::Error),
    Unsupported,
}

#[derive(Debug)]
pub enum BuildError {
    IO(io::Error),
    BadJson(serde_json::Error),
    WrongCache,
    WrongUuid(PathBuf, Uuid),
    NameIsUsed(String),
}

impl From<ImageError> for ImportError {
    fn from(value: ImageError) -> Self {
        match value {
            ImageError::IoError(err) => Self::IO(err),
            _ => Self::Unsupported,
        }
    }
}

impl From<io::Error> for ImportError {
    fn from(value: io::Error) -> Self {
        Self::IO(value)
    }
}

impl From<io::Error> for BuildError {
    fn from(value: io::Error) -> Self {
        Self::IO(value)
    }
}

impl From<serde_json::Error> for BuildError {
    fn from(value: serde_json::Error) -> Self {
        BuildError::BadJson(value)
    }
}
