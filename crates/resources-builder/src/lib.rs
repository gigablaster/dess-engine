use std::io;

use image::ImageError;

mod content;

pub enum ImportError {
    IO(io::Error),
    Unsupported,
}

impl From<ImageError> for ImportError {
    fn from(value: ImageError) -> Self {
        match value {
            ImageError::IoError(err) => Self::IO(err),
            _ => Self::Unsupported,
        }
    }
}
