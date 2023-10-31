// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{fs::File, io, path::PathBuf};

use ash::vk;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use bytes::Bytes;
use ddsfile::{Dds, DxgiFormat};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use image::{imageops::FilterType, io::Reader, DynamicImage, GenericImageView, ImageBuffer, Rgba};
use intel_tex_2::{bc5, bc7};

use crate::{Asset, Content, ContentImporter, ContentProcessor};

#[derive(Debug)]
pub struct ImageRgba8Data {
    pub data: Bytes,
    pub dimensions: [u32; 2],
}

#[derive(Debug)]
pub struct LoadImage {
    path: PathBuf,
}

impl LoadImage {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[derive(Debug)]
pub enum RawImage {
    Rgba(ImageRgba8Data),
    Dds(Box<Dds>),
}

impl Content for RawImage {}

#[derive(Debug)]
pub struct GpuImage {
    pub format: vk::Format,
    pub dimensions: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl Asset for GpuImage {
    fn collect_dependencies(&self, deps: &mut Vec<crate::AssetRef>) {}
}

impl ContentImporter<RawImage> for LoadImage {
    fn import(&self) -> anyhow::Result<RawImage> {
        if let Ok(dds) = Dds::read(&mut File::open(&self.path)?) {
            Ok(RawImage::Dds(Box::new(dds)))
        } else {
            let image = Reader::open(&self.path)?.with_guessed_format()?.decode()?;
            let dimensions = [image.dimensions().0, image.dimensions().1];
            let image = image.to_rgba8();

            Ok(RawImage::Rgba(ImageRgba8Data {
                data: image.into_raw().into(),
                dimensions,
            }))
        }
    }
}

pub struct CreatePlaceholderImage {
    data: [u8; 4],
}

impl CreatePlaceholderImage {
    pub fn new(data: [u8; 4]) -> Self {
        Self { data }
    }
}

impl ContentImporter<RawImage> for CreatePlaceholderImage {
    fn import(&self) -> anyhow::Result<RawImage> {
        Ok(RawImage::Rgba(ImageRgba8Data {
            data: Bytes::from(self.data.to_vec()),
            dimensions: [1, 1],
        }))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum ImagePurpose {
    Color,
    NonColor,
    Normals,
    Sprite,
}

pub struct CreateGpuImage {
    purpose: ImagePurpose,
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

impl CreateGpuImage {
    pub fn new(purpose: ImagePurpose) -> Self {
        Self { purpose }
    }

    fn process_dds(image: &Dds) -> anyhow::Result<GpuImage> {
        if let Some(format) = Self::get_vk_format(image) {
            let data = image.get_data(0)?;
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
            Ok(GpuImage {
                format,
                dimensions: [image.get_width(), image.get_height()],
                mips,
            })
        } else {
            Err(anyhow::format_err!("Can't extract format from dds file"))
        }
    }

    fn process_rgba(image: &ImageRgba8Data, purpose: ImagePurpose) -> anyhow::Result<GpuImage> {
        let dimensions = image.dimensions;

        let need_compression = purpose != ImagePurpose::Sprite
            && dimensions[0] >= 4
            && dimensions[1] >= 4
            && is_pow2(dimensions[0])
            && is_pow2(dimensions[1]);

        if !need_compression {
            let format = if purpose == ImagePurpose::Color {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            };
            return Ok(GpuImage {
                format,
                dimensions,
                mips: vec![image.data.to_vec()],
            });
        }

        let format = match purpose {
            ImagePurpose::Color => vk::Format::BC7_SRGB_BLOCK,
            ImagePurpose::NonColor => vk::Format::BC7_UNORM_BLOCK,
            ImagePurpose::Normals => vk::Format::BC5_UNORM_BLOCK,
            _ => unreachable!(),
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
        while current_dimensions[0] >= 4 && current_dimensions[1] >= 4 {
            mips.push(Self::compress_image(image.as_rgba8().unwrap(), purpose));
            current_dimensions[0] >>= 1;
            current_dimensions[1] >>= 1;
            image = image.resize_exact(
                current_dimensions[0],
                current_dimensions[1],
                FilterType::Lanczos3,
            );
        }

        Ok(GpuImage {
            format,
            dimensions,
            mips,
        })
    }

    fn compress_image(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, purpose: ImagePurpose) -> Vec<u8> {
        let block_count = intel_tex_2::divide_up_by_multiple(image.width() * image.height(), 16);

        let needs_alpha = purpose == ImagePurpose::Color && image.pixels().any(|px| px.0[3] != 255);

        let mode = match purpose {
            ImagePurpose::Sprite => unreachable!(),
            ImagePurpose::Color | ImagePurpose::NonColor => BcMode::Bc7,
            ImagePurpose::Normals => BcMode::Bc5,
        };

        let block_bytes = mode.block_bytes();

        let surface = intel_tex_2::RgbaSurface {
            width: image.width(),
            height: image.height(),
            stride: image.width() * 4,
            data: image,
        };

        let mut compressed_bytes = vec![0u8; block_count as usize * block_bytes];

        match mode {
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

    fn get_vk_format(image: &Dds) -> Option<vk::Format> {
        if let Some(format) = image.get_dxgi_format() {
            match format {
                DxgiFormat::R8G8B8A8_UNorm => return Some(vk::Format::R8G8B8A8_UNORM),
                DxgiFormat::R8G8B8A8_UNorm_sRGB => return Some(vk::Format::R8G8B8A8_SRGB),
                DxgiFormat::BC1_UNorm => return Some(vk::Format::BC1_RGB_UNORM_BLOCK),
                DxgiFormat::BC1_UNorm_sRGB => return Some(vk::Format::BC1_RGB_SRGB_BLOCK),
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
}

impl ContentProcessor<RawImage, GpuImage> for CreateGpuImage {
    fn process(&self, content: RawImage) -> anyhow::Result<GpuImage> {
        match content {
            RawImage::Dds(dds) => Self::process_dds(&dds),
            RawImage::Rgba(image) => Self::process_rgba(&image, self.purpose),
        }
    }
}

impl BinarySerialization for GpuImage {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        w.write_i32::<LittleEndian>(self.format.as_raw())?;
        w.write_u32::<LittleEndian>(self.dimensions[0])?;
        w.write_u32::<LittleEndian>(self.dimensions[1])?;
        self.mips.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for GpuImage {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let format = vk::Format::from_raw(r.read_i32::<LittleEndian>()?);
        let mut dimensions = [0u32; 2];
        r.read_u32_into::<LittleEndian>(&mut dimensions)?;
        let mips = Vec::<Vec<_>>::deserialize(r)?;

        Ok(Self {
            format,
            dimensions,
            mips,
        })
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
