use crate::ImportError;

use super::{Content, ContentImporter, ContentImporterFactory, LoadedContent};

#[derive(Debug, Default)]
pub struct ImageImporter {}

#[derive(Debug, Default)]
pub struct ImageImporterFactory {}

impl ContentImporter for ImageImporter {
    fn import(&self, path: &std::path::Path) -> Result<LoadedContent, ImportError> {
        let image = image::open(path)?;
        Ok(LoadedContent {
            path: path.into(),
            content: Content::Image(image),
        })
    }
}

impl ContentImporterFactory for ImageImporterFactory {
    fn importer(&self, path: &std::path::Path) -> Option<Box<dyn ContentImporter>> {
        if let Some(ext) = path.extension() {
            if let Some(ext) = ext.to_str() {
                match ext {
                    "jpg" | "jpeg" | "png" | "tiff" | "tga" => {
                        Some(Box::<ImageImporter>::default())
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

