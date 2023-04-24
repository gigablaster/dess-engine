use std::io;

use lz4_flex::block::{CompressError, DecompressError};
use uuid::Uuid;

#[derive(Debug)]
pub enum VfsError {
    IO(io::Error),
    InvalidVersiom,
    InvalidFormat,
    AssetNotFound(Uuid),
    NameNotFound(String),
    Compression(CompressError),
    Decompression(DecompressError),
}

impl From<io::Error> for VfsError {
    fn from(value: io::Error) -> Self {
        VfsError::IO(value)
    }
}

impl From<CompressError> for VfsError {
    fn from(value: CompressError) -> Self {
        VfsError::Compression(value)
    }
}

impl From<DecompressError> for VfsError {
    fn from(value: DecompressError) -> Self {
        VfsError::Decompression(value)
    }
}
