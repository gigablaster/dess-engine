use std::{collections::HashSet, fmt::Debug, io::Cursor, path::Path, sync::Arc};

use four_cc::FourCC;
use lz4_flex::decompress_into;
use uuid::Uuid;

use crate::{
    directory::{Compression, Directory},
    mmap::MappedFile,
    VfsError,
};

pub trait LoadedData {
    fn type_id(&self) -> FourCC;
    fn data(&self) -> &[u8];
}

pub trait Archive: Send + Sync + Debug {
    fn load(&self, uuid: Uuid) -> Result<Box<dyn LoadedData>, VfsError>;
    fn named(&self, name: &str) -> Option<Uuid>;
    fn tagged(&self, tag: &str) -> Option<&HashSet<Uuid>>;
}

struct MappedFileSlice {
    file: Arc<MappedFile>,
    type_id: FourCC,
    from: usize,
    to: usize,
}

impl LoadedData for MappedFileSlice {
    fn type_id(&self) -> FourCC {
        self.type_id
    }

    fn data(&self) -> &[u8] {
        &self.file.data()[self.from..self.to]
    }
}

impl MappedFileSlice {
    pub fn new(file: &Arc<MappedFile>, type_id: FourCC, from: usize, size: usize) -> Self {
        Self {
            file: file.clone(),
            type_id,
            from,
            to: from + size,
        }
    }
}

struct UnpackedData {
    type_id: FourCC,
    data: Vec<u8>,
}

impl UnpackedData {
    pub fn unpack(packed: &[u8], type_id: FourCC, size: usize) -> Result<Self, VfsError> {
        let mut data = Vec::with_capacity(size);
        decompress_into(packed, &mut data)?;

        Ok(Self { type_id, data })
    }
}

impl LoadedData for UnpackedData {
    fn data(&self) -> &[u8] {
        &self.data
    }

    fn type_id(&self) -> FourCC {
        self.type_id
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
                    header.type_id,
                    header.offset as _,
                    size,
                ))),
                Compression::LZ4(uncompressed, compressed) => Ok(Box::new(UnpackedData::unpack(
                    MappedFileSlice::new(
                        &self.file,
                        header.type_id,
                        header.offset as _,
                        compressed,
                    )
                    .data(),
                    header.type_id,
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
