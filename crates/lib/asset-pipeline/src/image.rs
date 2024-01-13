use std::{io::Cursor, path::Path, sync::Arc};

use ash::vk;
use bytes::Bytes;
use ddsfile::{Dds, DxgiFormat};
use dess_assets::{
    get_absolute_asset_path, Asset, ImageAsset, ImageDataSource, ImageRgba8Data, ImageSource,
    ImageSourceDesc, ImageType,
};
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgba};
use intel_tex_2::{bc5, bc7};

use crate::{is_asset_changed, read_to_end, AssetImporter, Error, ImportContext};

#[derive(Debug)]
pub enum RawImageData {
    Rgba(ImageRgba8Data),
    Dds(Box<Dds>),
}

#[derive(Debug)]
pub struct ImageContent {
    data: RawImageData,
    desc: ImageSourceDesc,
}

fn import_image(source: &ImageSource) -> Result<ImageContent, Error> {
    let bytes = match &source.source {
        ImageDataSource::Bytes(bytes) => bytes.clone(),
        ImageDataSource::File(path) => read_to_end(get_absolute_asset_path(Path::new(path))?)?,
        ImageDataSource::Placeholder(pixels) => {
            let data = ImageRgba8Data {
                data: Bytes::copy_from_slice(pixels),
                dimensions: [1, 1],
            };
            return Ok(ImageContent {
                data: RawImageData::Rgba(data),
                desc: source.desc,
            });
        }
    };
    if let Ok(dds) = Dds::read(Cursor::new(&bytes)) {
        Ok(ImageContent {
            data: RawImageData::Dds(Box::new(dds)),
            desc: source.desc,
        })
    } else {
        let image = image::load_from_memory(&bytes)?.to_rgba8();
        let dimensions = [image.dimensions().0, image.dimensions().1];

        Ok(ImageContent {
            data: RawImageData::Rgba(ImageRgba8Data {
                data: image.into_raw().into(),
                dimensions,
            }),
            desc: source.desc,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcMode {
    Bc5,
    Bc7,
}

impl BcMode {
    fn block_bytes(self) -> usize {
        match self {
            BcMode::Bc5 => 16,
            BcMode::Bc7 => 16,
        }
    }
}

fn process_dds(image: &Dds) -> Result<ImageAsset, Error> {
    if let Some(format) = get_backend_format(image) {
        let data = image
            .get_data(0)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
        let pitch_height = image.get_pitch_height();
        let mut offset = 0;
        let mips = (0..image.get_num_mipmap_levels())
            .map(|mip| {
                let width = (image.get_width() >> mip).max(pitch_height);
                let height = (image.get_height() >> mip).max(pitch_height);
                let pitch = dds_util::get_pitch(image, width).unwrap();
                let size = dds_util::get_texture_size(pitch, pitch_height, height, 1);
                let mip = &data[offset..offset + size];
                offset += size;

                mip.to_owned()
            })
            .collect::<Vec<_>>();
        Ok(ImageAsset {
            format,
            dimensions: [image.get_width(), image.get_height()],
            mips,
        })
    } else {
        Err(Error::ProcessingFailed(
            "Failed to load DDS file".to_owned(),
        ))
    }
}

fn process_rgba(image: &ImageRgba8Data, desc: ImageSourceDesc) -> Result<ImageAsset, Error> {
    let dimensions = image.dimensions;

    let need_compression = desc.need_compression
        && dimensions[0] >= 4
        && dimensions[1] >= 4
        && is_pow2(dimensions[0])
        && is_pow2(dimensions[1]);

    let format = match desc.ty {
        ImageType::Rgba if need_compression && desc.srgb => vk::Format::BC7_SRGB_BLOCK,
        ImageType::Rgba if need_compression => vk::Format::BC7_UNORM_BLOCK,
        ImageType::Rg if need_compression => vk::Format::BC5_UNORM_BLOCK,
        _ if desc.srgb => vk::Format::R8G8B8A8_SRGB,
        _ => vk::Format::R8G8B8A8_UNORM,
    };

    let mut image = DynamicImage::ImageRgba8(
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
            image.dimensions[0],
            image.dimensions[1],
            image.data.to_vec(),
        )
        .unwrap(),
    );

    let mut current_dimensions = dimensions;
    let mut mips = Vec::new();
    if desc.generate_mips && image.width() >= 4 && image.height() >= 4 {
        while current_dimensions[0] >= 4 && current_dimensions[1] >= 4 {
            mips.push(prepare_image(image.as_rgba8().unwrap(), format));
            current_dimensions[0] >>= 1;
            current_dimensions[1] >>= 1;
            image = image.resize_exact(
                current_dimensions[0],
                current_dimensions[1],
                FilterType::Lanczos3,
            );
        }
    } else {
        mips.push(prepare_image(image.as_rgba8().unwrap(), format));
    }

    Ok(ImageAsset {
        format,
        dimensions,
        mips,
    })
}

fn block_compress(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, bc: BcMode) -> Vec<u8> {
    let block_count = intel_tex_2::divide_up_by_multiple(image.width() * image.height(), 16);

    let needs_alpha = bc == BcMode::Bc7 && image.pixels().any(|px| px.0[3] != 255);

    let block_bytes = bc.block_bytes();

    let surface = intel_tex_2::RgbaSurface {
        width: image.width(),
        height: image.height(),
        stride: image.width() * 4,
        data: image,
    };

    let mut compressed_bytes = vec![0u8; block_count as usize * block_bytes];

    match bc {
        BcMode::Bc5 => bc5::compress_blocks_into(&surface, &mut compressed_bytes),
        BcMode::Bc7 => {
            let settings = if needs_alpha {
                bc7::alpha_basic_settings()
            } else {
                bc7::opaque_basic_settings()
            };
            bc7::compress_blocks_into(&settings, &surface, &mut compressed_bytes)
        }
    }

    compressed_bytes
}

fn prepare_image(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, format: vk::Format) -> Vec<u8> {
    match format {
        vk::Format::BC5_UNORM_BLOCK => block_compress(image, BcMode::Bc5),
        vk::Format::BC7_SRGB_BLOCK | vk::Format::BC7_UNORM_BLOCK => {
            block_compress(image, BcMode::Bc7)
        }
        vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM => image.to_vec(),
        _ => panic!("Unknow format: {:?}", format),
    }
}

fn get_backend_format(image: &Dds) -> Option<vk::Format> {
    if let Some(format) = image.get_dxgi_format() {
        match format {
            DxgiFormat::R8G8B8A8_UNorm => return Some(vk::Format::R8G8B8A8_UNORM),
            DxgiFormat::R8G8B8A8_UNorm_sRGB => return Some(vk::Format::R8G8B8A8_SRGB),
            DxgiFormat::BC1_UNorm => return Some(vk::Format::BC1_RGB_UNORM_BLOCK),
            DxgiFormat::BC1_UNorm_sRGB => return Some(vk::Format::BC1_RGBA_SRGB_BLOCK),
            DxgiFormat::BC2_UNorm => return Some(vk::Format::BC2_UNORM_BLOCK),
            DxgiFormat::BC2_UNorm_sRGB => return Some(vk::Format::BC2_SRGB_BLOCK),
            DxgiFormat::BC3_UNorm => return Some(vk::Format::BC3_SRGB_BLOCK),
            DxgiFormat::BC3_UNorm_sRGB => return Some(vk::Format::BC3_SRGB_BLOCK),
            DxgiFormat::BC4_UNorm => return Some(vk::Format::BC4_UNORM_BLOCK),
            DxgiFormat::BC4_SNorm => return Some(vk::Format::BC4_SNORM_BLOCK),
            DxgiFormat::BC5_UNorm => return Some(vk::Format::BC5_UNORM_BLOCK),
            DxgiFormat::BC5_SNorm => return Some(vk::Format::BC5_SNORM_BLOCK),
            DxgiFormat::BC6H_SF16 => return Some(vk::Format::BC6H_SFLOAT_BLOCK),
            DxgiFormat::BC6H_UF16 => return Some(vk::Format::BC6H_UFLOAT_BLOCK),
            DxgiFormat::BC7_UNorm => return Some(vk::Format::BC7_UNORM_BLOCK),
            DxgiFormat::BC7_UNorm_sRGB => return Some(vk::Format::BC7_UNORM_BLOCK),
            _ => return None,
        }
    }

    None
}

pub fn process_image(content: ImageContent) -> Result<ImageAsset, Error> {
    match &content.data {
        RawImageData::Dds(dds) => process_dds(dds),
        RawImageData::Rgba(image) => process_rgba(image, content.desc),
    }
}

fn is_pow2(x: u32) -> bool {
    x & (x - 1) == 0
}

mod dds_util {
    pub fn get_texture_size(pitch: u32, pitch_height: u32, height: u32, depth: u32) -> usize {
        let row_height = (height + (pitch_height - 1)) / pitch_height;
        pitch as usize * row_height as usize * depth as usize
    }

    pub fn get_pitch(dds: &ddsfile::Dds, width: u32) -> Option<u32> {
        // Try format first
        if let Some(format) = dds.get_format() {
            if let Some(pitch) = format.get_pitch(width) {
                return Some(pitch);
            }
        }

        // Then try to calculate it ourselves
        if let Some(bpp) = dds.get_bits_per_pixel() {
            return Some((bpp * width + 7) / 8);
        }
        None
    }
}

impl AssetImporter for ImageSource {
    fn import(&self, _ctx: &dyn ImportContext) -> Result<Arc<dyn Asset>, Error> {
        let content = import_image(self)?;
        Ok(Arc::new(process_image(content)?))
    }

    fn is_changed(&self, timestamp: std::time::SystemTime) -> bool {
        match &self.source {
            ImageDataSource::File(path) => is_asset_changed(path, timestamp),
            _ => false,
        }
    }
}
