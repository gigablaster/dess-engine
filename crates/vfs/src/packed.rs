use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
};

use zstd::stream;

use crate::{
    directory::{load_archive_directory, Compression, Directory},
    mmap::{MappedFile, MappedFileSlice, MappedFileSliceReader},
    Archive, VfsError,
};

type MmapReader = BufReader<MappedFileSlice>;

struct ZstdDataReader<'a> {
    decoder: stream::Decoder<'a, MmapReader>,
}

impl<'a> ZstdDataReader<'a> {
    fn new(file: MappedFileSlice) -> Result<Self, VfsError> {
        Ok(Self {
            decoder: stream::Decoder::new(file).unwrap(),
        })
    }
}

impl<'a> Read for ZstdDataReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[derive(Debug)]
pub struct PackedArchive {
    file: Arc<MappedFile>,
    directory: Directory,
}

impl PackedArchive {
    pub fn open(path: &Path) -> Result<Self, VfsError> {
        let mut file = File::open(path)?;
        let directory = load_archive_directory(&mut file)?;
        let file = Arc::new(MappedFile::open(path)?);

        Ok(Self { file, directory })
    }
}

impl Archive for PackedArchive {
    fn load(&self, name: &str) -> Result<Box<dyn Read>, VfsError> {
        if let Some(header) = self.directory.get(name) {
            match header.compression {
                Compression::None(size) => Ok(Box::new(MappedFileSliceReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, size as _),
                ))),
                Compression::Zstd(_, compressed) => Ok(Box::new(ZstdDataReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, compressed as _),
                )?)),
            }
        } else {
            Err(VfsError::NotFound(name.into()))
        }
    }
}
