use std::{
    collections::HashSet,
    fmt::Debug,
    io::{Cursor, Read},
    path::Path,
    sync::Arc,
};

use four_cc::FourCC;
use lz4_flex::frame::FrameDecoder;
use uuid::Uuid;

use crate::{
    directory::{Compression, Directory},
    mmap::{MappedFile, MappedFileSlice, MappedFileSliceReader},
    Archive, DataReader, DataTypeId, VfsError,
};

#[derive(Debug)]
struct Lz4DataReader {
    type_id: FourCC,
    decoder: FrameDecoder<MappedFileSlice>,
}

impl DataTypeId for Lz4DataReader {
    fn type_id(&self) -> FourCC {
        self.type_id
    }
}

impl Read for Lz4DataReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.decoder.read(buf)
    }
}

impl DataReader for Lz4DataReader {}

impl Lz4DataReader {
    fn new(file: MappedFileSlice, type_id: FourCC) -> Self {
        Self {
            type_id,
            decoder: FrameDecoder::new(file),
        }
    }
}

#[derive(Debug)]
pub struct PackedArchive {
    file: Arc<MappedFile>,
    directory: Directory,
}

impl PackedArchive {
    pub fn open(path: &Path) -> Result<Self, VfsError> {
        let file = Arc::new(MappedFile::open(path)?);
        let directory = Directory::load(&mut Cursor::new(&file.data()))?;

        Ok(Self { file, directory })
    }
}

impl Archive for PackedArchive {
    fn load(&self, uuid: Uuid) -> Result<Box<dyn DataReader>, VfsError> {
        if let Some(header) = self.directory.by_uuid(uuid) {
            match header.compression {
                Compression::None(size) => Ok(Box::new(MappedFileSliceReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, size as _),
                    header.type_id,
                ))),
                Compression::LZ4(_, compressed) => Ok(Box::new(Lz4DataReader::new(
                    MappedFileSlice::new(&self.file, header.offset as _, compressed as _),
                    header.type_id,
                ))),
            }
        } else {
            Err(VfsError::AssetNotFound(uuid))
        }
    }

    fn named(&self, name: &str) -> Option<Uuid> {
        self.directory.by_name(name).copied()
    }

    fn tagged(&self, tag: &str) -> Option<&HashSet<Uuid>> {
        self.directory.by_tag(tag)
    }
}
