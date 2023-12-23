// Copyright (C) 2023 Vladimir Kuskov

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

use std::{hash::Hash, io, path::Path};

use bytes::Bytes;
use dess_backend::Format;
use siphasher::sip128::Hasher128;
use speedy::{Context, Readable, Writable};

use crate::{Asset, AssetLoad, AssetRef, ContentSource};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Readable, Writable)]
pub enum ImageType {
    Rgba,
    Rg,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Readable, Writable)]
pub struct ImageSourceDesc {
    pub ty: ImageType,
    pub generate_mips: bool,
    pub need_compression: bool,
    pub srgb: bool,
}

impl ImageSourceDesc {
    pub fn color() -> Self {
        Self {
            ty: ImageType::Rgba,
            generate_mips: true,
            need_compression: true,
            srgb: true,
        }
    }

    pub fn non_color() -> Self {
        Self {
            ty: ImageType::Rgba,
            generate_mips: true,
            need_compression: true,
            srgb: false,
        }
    }

    pub fn normals() -> Self {
        Self {
            ty: ImageType::Rg,
            generate_mips: true,
            need_compression: true,
            srgb: false,
        }
    }
}

#[derive(Debug)]
pub struct ImageRgba8Data {
    pub data: Bytes,
    pub dimensions: [u32; 2],
}

#[derive(Debug, Clone, Readable, Writable, Hash, Eq, PartialEq)]
pub enum ImageDataSource {
    File(String),
    Bytes(Vec<u8>),
    Placeholder([u8; 4]),
}

#[derive(Debug, Clone, PartialEq, Eq, Readable, Writable, Hash)]
pub struct ImageSource {
    pub source: ImageDataSource,
    pub desc: ImageSourceDesc,
}

impl ContentSource for ImageSource {
    fn get_ref(&self) -> AssetRef {
        let mut hasher = siphasher::sip128::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish128().as_u128().into()
    }
}

impl ImageSource {
    pub fn from_file<P: AsRef<Path>>(path: P, desc: ImageSourceDesc) -> Self {
        Self {
            source: ImageDataSource::File(path.as_ref().to_str().unwrap().to_owned()),
            desc,
        }
    }

    pub fn from_bytes(bytes: &[u8], desc: ImageSourceDesc) -> Self {
        Self {
            source: ImageDataSource::Bytes(bytes.to_vec()),
            desc,
        }
    }

    pub fn from_color(color: [f32; 4], desc: ImageSourceDesc) -> Self {
        Self {
            source: ImageDataSource::Placeholder(color_to_pixles(color)),
            desc,
        }
    }
}

fn color_to_pixles(color: [f32; 4]) -> [u8; 4] {
    [
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        (color[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

#[derive(Debug, Default, Readable, Writable)]
pub struct ImageAsset {
    pub format: Format,
    pub dimensions: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl Asset for ImageAsset {
    fn to_bytes(&self) -> io::Result<Bytes> {
        Ok(self.write_to_vec()?.into())
    }
}

impl AssetLoad for ImageAsset {
    fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self::read_from_buffer(data)?)
    }
}
