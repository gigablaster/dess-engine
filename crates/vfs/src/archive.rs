use std::{collections::HashSet, fmt::Debug, io::Cursor, path::Path, sync::Arc};

use lz4_flex::decompress_into;
use uuid::Uuid;

use crate::{
    directory::{Compression, Directory},
    mmap::MappedFile,
    VfsError,
};

pub trait LoadedData {
    fn data(&self) -> &[u8];
}

impl LoadedData for MappedFile {
    fn data(&self) -> &[u8] {
        self.as_ref()
    }
}

pub trait Archive: Send + Sync + Debug {
    fn load(&self, uuid: Uuid) -> Result<Box<dyn LoadedData>, VfsError>;
    fn named(&self, name: &str) -> Option<Uuid>;
    fn tagged(&self, tag: &str) -> Option<&HashSet<Uuid>>;
}

struct MappedFileSlice {
    file: Arc<MappedFile>,
    from: usize,
    to: usize,
}

impl LoadedData for MappedFileSlice {
    fn data(&self) -> &[u8] {
        &self.file.data()[self.from..self.to]
    }
}

impl MappedFileSlice {
    pub fn new(file: &Arc<MappedFile>, from: usize, size: usize) -> Self {
        Self {
            file: file.clone(),
            from,
            to: from + size,
        }
    }
}

struct UnpackedData {
    data: Vec<u8>,
}

impl UnpackedData {
    pub fn unpack(packed: &[u8], size: usize) -> Result<Self, VfsError> {
        let mut data = Vec::with_capacity(size);
        decompress_into(packed, &mut data)?;

        Ok(Self { data })
    }
}

impl LoadedData for UnpackedData {
    fn data(&self) -> &[u8] {
        &self.data
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
    fn load(&self, uuid: Uuid) -> Result<Box<dyn LoadedData>, VfsError> {
        if let Some(header) = self.directory.by_uuid(uuid) {
            match header.compression {
                Compression::None(size) => Ok(Box::new(MappedFileSlice::new(
                    &self.file,
                    header.offset as _,
                    size,
                ))),
                Compression::LZ4(uncompressed, compressed) => Ok(Box::new(UnpackedData::unpack(
                    MappedFileSlice::new(&self.file, header.offset as _, compressed).data(),
                    uncompressed,
                )?)),
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
