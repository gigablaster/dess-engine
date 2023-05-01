use std::fs::{read_to_string, File};

use crate::ImportError;

use super::{Content, ContentImporter, ContentImporterFactory, LoadedContent};

#[derive(Debug, Default)]
pub struct TextImporter {}

#[derive(Debug, Default)]
pub struct TextImporterFactory {}

impl ContentImporterFactory for TextImporterFactory {
    fn importer(&self, path: &std::path::Path) -> Option<Box<dyn super::ContentImporter>> {
        if let Some(ext) = path.extension() {
            if let Some(ext) = ext.to_str() {
                return match ext {
                    "txt" | "xml" | "json" | "toml" => Some(Box::<TextImporter>::default()),
                    _ => None,
                };
            }
        }

        None
    }
}

impl ContentImporter for TextImporter {
    fn import(&self, path: &std::path::Path) -> Result<LoadedContent, ImportError> {
        let data = read_to_string(path)?;
        Ok(LoadedContent {
            path: path.into(),
            content: Content::Text(data),
        })
    }
}
