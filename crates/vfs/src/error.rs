use std::io;

use lz4_flex::frame;
use uuid::Uuid;

#[derive(Debug)]
pub enum VfsError {
    IO(io::Error),
    InvalidVersiom,
    InvalidFormat,
    AssetNotFound(Uuid),
    NameNotFound(String),
    Compression(frame::Error),
}

impl From<io::Error> for VfsError {
    fn from(value: io::Error) -> Self {
        VfsError::IO(value)
    }
}

impl From<frame::Error> for VfsError {
    fn from(value: frame::Error) -> Self {
        VfsError::Compression(value)
    }
}
