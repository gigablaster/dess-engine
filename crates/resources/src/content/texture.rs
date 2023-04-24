use image::{DynamicImage, ImageError};

use super::{Content, ContentError, ContentImporter};

pub struct Texture {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

pub struct TextureImporter {}

impl ContentImporter for TextureImporter {
    fn import(path: &std::path::Path) -> Result<Content, ContentError> {
        let img = image::open(path);
        match img {
            Ok(img) => Ok(Content::Texture(Self::import_image(img))),
            Err(err) => {
                if let ImageError::Unsupported(_) = err {
                    Err(ContentError::NotSupported)
                } else {
                    Err(ContentError::ImageError(err))
                }
            }
        }
    }
}

impl TextureImporter {
    fn import_image(image: DynamicImage) -> Texture {
        let width = image.width() as _;
        let height = image.height() as _;
        let buffer = image.into_rgba8();
        let mut data = Vec::with_capacity(4 * width * height);
        buffer.rows().for_each(|row| {
            row.for_each(|pixel| {
                data.extend_from_slice(&pixel.0);
            })
        });

        Texture {
            width,
            height,
            data,
        }
    }
}
