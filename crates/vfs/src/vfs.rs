use std::{
    collections::HashMap,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use log::{error, info};

use crate::{
    directory::{read_archive_directory, FileHeader},
    mmap::MappedFile,
    VfsError,
};

#[derive(Debug)]
struct ArchiveReference {
    file: Arc<MappedFile>,
    location: FileHeader,
}

impl ArchiveReference {
    fn new(file: &Arc<MappedFile>, location: FileHeader) -> Self {
        Self {
            file: file.clone(),
            location,
        }
    }
}

#[derive(Debug)]
enum FileReference {
    File(PathBuf),
    Archive(ArchiveReference),
}

#[derive(Debug, Default)]
pub struct Vfs {
    catalog: HashMap<PathBuf, FileReference>,
}

impl Vfs {
    pub fn scan(&mut self, root: &Path) -> Result<(), VfsError> {
        let paths = fs::read_dir(root)?;
        for path in paths {
            let path = path?.path();
            if path.is_file() && path.ends_with(".dess") {
                info!("Scanning {:?}", path);
                if let Err(err) = self.add_archive(&path) {
                    error!("Failed to add archive {:?} - {:?}", path, err);
                }
            }
            if path.is_dir() && path.ends_with("/data") {
                info!("Scanning plain data directory {:?}", path);
                if let Err(err) = self.add_directory("/", &path) {
                    error!("Failed to add data directory {:?} - {:?}", path, err);
                }
            }
        }

        Ok(())
    }

    fn add_archive(&mut self, path: &Path) -> Result<(), VfsError> {
        let mapped_file = Arc::new(MappedFile::open(path)?);
        let directory = read_archive_directory(&mut Cursor::new(mapped_file.as_ref()))?;
        directory.into_iter().for_each(|(name, header)| {
            self.catalog.insert(
                name.into(),
                FileReference::Archive(ArchiveReference::new(&mapped_file, header)),
            );
        });

        Ok(())
    }

    fn add_directory(
        &mut self,
        root: impl Into<PathBuf>,
        data_root: &Path,
    ) -> Result<(), VfsError> {
        let root = root.into();
        let scan_root = data_root.join(&root);
        let paths = fs::read_dir(&scan_root)?;

        for path in paths {
            let path = path?.path();
            let full_path = root.join(&path);

            if path.is_dir() {
                self.add_directory(&full_path, data_root)?;
                continue;
            }

            if path.is_symlink() || path.starts_with(".") || path.ends_with(".meta") {
                continue;
            }

            self.catalog.insert(
                full_path.clone(),
                FileReference::File(data_root.join(full_path)),
            );
        }
        Ok(())
    }
}
