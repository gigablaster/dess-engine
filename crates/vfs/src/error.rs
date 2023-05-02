use std::io;

#[derive(Debug)]
pub enum VfsError {
    IO(io::Error),
    InvalidVersiom,
    InvalidFormat,
    NotFound(String),
}

impl From<io::Error> for VfsError {
    fn from(value: io::Error) -> Self {
        VfsError::IO(value)
    }
}
