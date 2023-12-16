mod image;

use std::{
    fs,
    io::{self, Read},
    path::Path,
};

pub use image::*;

use ::image::ImageError;
use dess_assets::{Asset, AssetRef, ContentSource};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    BadSourceData,
    ProcessingFailed(String),
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ImageError> for Error {
    fn from(value: ImageError) -> Self {
        Self::ProcessingFailed(value.to_string())
    }
}

pub trait AsseetImporter: ContentSource {
    fn import(&self, ctx: &dyn ImportContext) -> Result<Box<dyn Asset>, Error>;
}

pub trait ImportContext {
    fn import(&self, content: Box<dyn AsseetImporter>) -> AssetRef;
}

pub(crate) fn read_to_end<P>(path: P) -> io::Result<Vec<u8>>
where
    P: AsRef<Path>,
{
    let file = fs::File::open(path.as_ref())?;
    // Allocate one extra byte so the buffer doesn't need to grow before the
    // final `read` call at the end of the file.  Don't worry about `usize`
    // overflow because reading will fail regardless in that case.
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data)?;
    Ok(data)
}
