use std::{
    fs::File,
    io::{Cursor, Write},
    path::Path,
};

use ::image::{imageops::FilterType, DynamicImage};
use ddsfile::{AlphaMode, D3D10ResourceDimension, Dds, DxgiFormat, NewDxgiParams};
use image::io::Reader;
use texpresso::{Format, Params, COLOUR_WEIGHTS_UNIFORM};

use crate::{Content, ContentImporter, ImportContext, ImportError};

const MIN_MIP_SIZE: u32 = 4;

struct MipLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

struct Layer {
    pub data: Vec<MipLevel>,
}

pub struct TextureContent {
    layers: Vec<Layer>,
    desired_compression: Option<Format>,
    srgb: bool,
}

impl MipLevel {
    pub fn compress(&self, format: Format) -> Vec<u8> {
        let size = format.compressed_size(self.width, self.height);
        let mut result = vec![0; size];
        let mut params = Params::default();
        if format == Format::Bc5 || format == Format::Bc4 {
            params.weights = COLOUR_WEIGHTS_UNIFORM;
        }
        if format == Format::Bc2 {
            params.weigh_colour_by_alpha = true;
        }
        format.compress(
            &self.data,
            self.width,
            self.height,
            Params::default(),
            result.as_mut_slice(),
        );

        result
    }
}

impl Layer {
    pub fn new(image: DynamicImage, generate_mips: bool) -> Self {
        assert!(image.width() >= MIN_MIP_SIZE && image.height() >= MIN_MIP_SIZE);
        let mut image = image;
        let top_mip = MipLevel {
            width: image.width() as _,
            height: image.height() as _,
            data: image.to_rgba8().as_flat_samples().as_slice().into(),
        };
        let mut mips = vec![top_mip];
        if generate_mips {
            while image.width() >= MIN_MIP_SIZE && image.height() >= MIN_MIP_SIZE {
                image = image.resize(image.width() / 2, image.height() / 2, FilterType::Lanczos3);
                mips.push(MipLevel {
                    width: image.width() as _,
                    height: image.height() as _,
                    data: image.to_rgba8().as_flat_samples().as_slice().into(),
                });
            }
        }

        Self { data: mips }
    }
}

fn has_alpha(image: &DynamicImage) -> bool {
    image.to_rgba8().pixels().any(|pixel| pixel.0[3] < 255)
}

impl TextureContent {
    pub fn color_texture(image: DynamicImage) -> Box<Self> {
        let format = if has_alpha(&image) {
            Format::Bc2
        } else {
            Format::Bc1
        };
        Box::new(Self {
            layers: vec![Layer::new(image, true)],
            desired_compression: Some(format),
            srgb: true,
        })
    }

    pub fn normal_map(image: DynamicImage) -> Box<Self> {
        Box::new(Self {
            layers: vec![Layer::new(image, true)],
            desired_compression: Some(Format::Bc5),
            srgb: false,
        })
    }

    pub fn masks(image: DynamicImage) -> Box<Self> {
        Box::new(Self {
            layers: vec![Layer::new(image, true)],
            desired_compression: Some(Format::Bc1),
            srgb: false,
        })
    }

    pub fn grayscale(image: DynamicImage) -> Box<Self> {
        Box::new(Self {
            layers: vec![Layer::new(image, false)],
            desired_compression: Some(Format::Bc4),
            srgb: false,
        })
    }

    pub fn ui(image: DynamicImage) -> Box<Self> {
        Box::new(Self {
            layers: vec![Layer::new(image, false)],
            desired_compression: None,
            srgb: false,
        })
    }

    pub fn other(image: DynamicImage) -> Box<Self> {
        let width = image.width();
        let height = image.height();
        let is_texture = Self::is_power_of_2(width) && Self::is_power_of_2(height);
        let compression = if is_texture {
            if has_alpha(&image) {
                Some(Format::Bc2)
            } else {
                Some(Format::Bc1)
            }
        } else {
            None
        };
        Box::new(Self {
            layers: vec![Layer::new(image, is_texture)],
            desired_compression: compression,
            srgb: false,
        })
    }

    fn is_power_of_2(n: u32) -> bool {
        n & (n - 1) == 0
    }
    fn dxgi(&self) -> NewDxgiParams {
        let format = if self.srgb {
            self.dxgi_format_srgb()
        } else {
            self.dxgi_format()
        };
        NewDxgiParams {
            height: self.layers[0].data[0].height as _,
            width: self.layers[0].data[0].width as _,
            depth: None,
            format,
            mipmap_levels: Some(self.layers[0].data.len() as _),
            array_layers: Some(self.layers.len() as _),
            caps2: None,
            is_cubemap: false,
            resource_dimension: D3D10ResourceDimension::Texture2D,
            alpha_mode: AlphaMode::Unknown,
        }
    }

    fn dxgi_format(&self) -> DxgiFormat {
        match self.desired_compression {
            None => DxgiFormat::R8G8B8A8_UNorm,
            Some(Format::Bc1) => DxgiFormat::BC1_UNorm,
            Some(Format::Bc2) => DxgiFormat::BC2_UNorm,
            Some(Format::Bc3) => DxgiFormat::BC3_UNorm,
            Some(Format::Bc4) => DxgiFormat::BC4_UNorm,
            Some(Format::Bc5) => DxgiFormat::BC5_UNorm,
        }
    }

    fn dxgi_format_srgb(&self) -> DxgiFormat {
        match self.desired_compression {
            None => DxgiFormat::R8G8B8A8_UNorm_sRGB,
            Some(Format::Bc1) => DxgiFormat::BC1_UNorm_sRGB,
            Some(Format::Bc2) => DxgiFormat::BC2_UNorm_sRGB,
            Some(Format::Bc3) => DxgiFormat::BC3_UNorm_sRGB,
            Some(Format::Bc4) => DxgiFormat::BC4_UNorm,
            Some(Format::Bc5) => DxgiFormat::BC5_UNorm,
        }
    }
}

impl Content for TextureContent {
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut dds = Dds::new_dxgi(self.dxgi()).unwrap();
        self.layers.iter().enumerate().for_each(|(index, layer)| {
            let target: &mut [u8] = dds.get_mut_data(index as _).unwrap();
            let mut cursor = Cursor::new(target);
            layer.data.iter().for_each(|mip| {
                if let Some(format) = self.desired_compression {
                    cursor.write_all(&mip.compress(format)).unwrap();
                } else {
                    cursor.write_all(&mip.data).unwrap();
                }
            });
        });

        dds.write(&mut File::create(path)?).unwrap();

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct TextureImporter {}

impl TextureImporter {
    fn has_suffix(path: &Path, values: &[&str]) -> bool {
        path.file_stem()
            .map(|x| {
                x.to_ascii_lowercase()
                    .to_str()
                    .map(|x| values.iter().any(|y| x.ends_with(y)))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn is_color(path: &Path) -> bool {
        Self::has_suffix(
            path,
            &[
                "_d",
                "_diffuse",
                "_color",
                "_c",
                "_albedo",
                "_base",
                "_basecolor",
            ],
        )
    }

    fn is_normal(path: &Path) -> bool {
        Self::has_suffix(path, &["_nm", "_normal"])
    }

    fn is_masks(path: &Path) -> bool {
        Self::has_suffix(path, &["_mra", "_mask"])
    }

    fn is_grayscale(path: &Path) -> bool {
        Self::has_suffix(
            path,
            &["_r", "_roughness", "_m", "_metallic", "_ao", "_occlusion"],
        )
    }

    fn is_ui(path: &Path) -> bool {
        Self::has_suffix(path, &["_ui"])
    }
}

impl ContentImporter for TextureImporter {
    fn import(
        &self,
        path: &Path,
        _context: &ImportContext,
    ) -> Result<Box<dyn Content>, ImportError> {
        let image = Reader::open(path)?.with_guessed_format()?.decode()?;
        if Self::is_color(path) {
            Ok(TextureContent::color_texture(image))
        } else if Self::is_normal(path) {
            Ok(TextureContent::normal_map(image))
        } else if Self::is_masks(path) {
            Ok(TextureContent::masks(image))
        } else if Self::is_ui(path) {
            Ok(TextureContent::ui(image))
        } else if Self::is_grayscale(path) {
            Ok(TextureContent::grayscale(image))
        } else {
            Ok(TextureContent::other(image))
        }
    }

    fn can_handle(&self, path: &Path) -> bool {
        path.extension()
            .map(|x| {
                x.to_ascii_lowercase()
                    .to_str()
                    .map(|x| x == "png" || x == "jpg" || x == "jpeg" || x == "tiff" || x == "tga")
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn target_name(&self, path: &Path) -> std::path::PathBuf {
        let mut path = path.to_path_buf();
        path.set_extension("dds");
        path
    }
}
