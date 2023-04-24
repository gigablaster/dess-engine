use std::io;

pub enum ContentError {
    NotSupported,
    IO(io::Error),
    ImageError(image::ImageError),
}

impl From<io::Error> for ContentError {
    fn from(value: io::Error) -> Self {
        ContentError::IO(value)
    }
}

impl From<image::ImageError> for ContentError {
    fn from(value: image::ImageError) -> Self {
        ContentError::ImageError(value)
    }
}
