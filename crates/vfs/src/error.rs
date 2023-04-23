use std::{io, path::PathBuf};

#[derive(Debug)]
pub enum VfsError {
    IO(io::Error),
    InvalidVersiom,
    InvalidFormat,
    NotFound(PathBuf),
}

impl From<io::Error> for VfsError {
    fn from(value: io::Error) -> Self {
        VfsError::IO(value)
    }
}
