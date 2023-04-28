use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};

use crate::ImportError;

use super::{Content, ContentImporter};

pub enum TextureData {
    Rgb(RgbImage),
    Rgba(RgbaImage),
    Gray(GrayImage),
}

pub struct Texture {
    pub width: u16,
    pub height: u16,
    pub data: TextureData,
}

pub struct TextureImporter {}

impl ContentImporter for TextureImporter {
    fn import(&self, path: &std::path::Path) -> Result<Content, ImportError> {
        let img = image::open(path)?;
        if let Some(name) = path.file_stem() {
            if let Some(name) = name.to_str() {
                let name = name.to_lowercase();
                let content = if name.ends_with("_nm") || name.ends_with("_n") {
                    Content::NormalTexture(process_color_map(img))
                } else if name.ends_with("_e")
                    || name.ends_with("_m")
                    || name.ends_with("_l")
                    || name.ends_with("_mra")
                {
                    Content::NotColorTexture(process_color_map(img))
                } else if name.ends_with("_r")
                    || name.ends_with("_ao")
                    || name.ends_with("_s")
                    || name.ends_with("_mt")
                {
                    Content::GrayscaleTexture(process_grayscale_map(img))
                } else if name.ends_with("_base") || name.ends_with("_d") {
                    Content::ColorTexture(process_color_map(img))
                } else {
                    Content::SpriteTexture(process_color_map(img))
                };

                return Ok(content);
            }
        }

        Err(ImportError::Unsupported)
    }
}

fn process_color_map(image: DynamicImage) -> Texture {
    let width = image.width() as u16;
    let height = image.height() as u16;
    let data = match image {
        DynamicImage::ImageRgb8(data) => TextureData::Rgb(data),
        DynamicImage::ImageRgba8(data) => TextureData::Rgba(data),
        DynamicImage::ImageRgb16(_) => TextureData::Rgb(image.into_rgb8()),
        DynamicImage::ImageRgba16(_) => TextureData::Rgba(image.into_rgba8()),
        DynamicImage::ImageLuma8(_) => TextureData::Rgb(image.into_rgb8()),
        DynamicImage::ImageLumaA8(_) => TextureData::Rgba(image.into_rgba8()),
        DynamicImage::ImageLuma16(_) => TextureData::Rgb(image.into_rgb8()),
        DynamicImage::ImageLumaA16(_) => TextureData::Rgba(image.into_rgba8()),
        DynamicImage::ImageRgb32F(_) => TextureData::Rgb(image.into_rgb8()),
        DynamicImage::ImageRgba32F(_) => TextureData::Rgba(image.into_rgba8()),
        _ => unimplemented!(),
    };

    Texture {
        width,
        height,
        data,
    }
}

fn process_grayscale_map(image: DynamicImage) -> Texture {
    let width = image.width() as u16;
    let height = image.height() as u16;
    let data = if let DynamicImage::ImageLuma8(image) = image {
        TextureData::Gray(image)
    } else {
        TextureData::Gray(image.to_luma8())
    };

    Texture {
        width,
        height,
        data,
    }
}
