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

use std::path::PathBuf;

use ash::vk;
use bytes::Bytes;
use speedy::{Context, Readable, Writable};

use crate::{Asset, AssetRef, AssetRefProvider};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "purpose")]
pub enum ImagePurpose {
    #[serde(rename = "color")]
    Color,
    #[serde(rename = "data")]
    NonColor,
    #[serde(rename = "normals")]
    Normals,
    #[serde(rename = "sprite")]
    Sprite,
}

#[derive(Debug)]
pub struct ImageRgba8Data {
    pub data: Bytes,
    pub dimensions: [u32; 2],
}

#[derive(Debug, Clone)]
pub enum ImageDataSource {
    File(PathBuf),
    Bytes(Bytes),
    Placeholder([u8; 4]),
}

#[derive(Debug, Clone)]
pub struct ImageSource {
    pub source: ImageDataSource,
    pub purpose: ImagePurpose,
}

impl AssetRefProvider for ImageSource {
    fn asset_ref(&self) -> AssetRef {
        match &self.source {
            ImageDataSource::File(path) => AssetRef::from_path_with(&path, &self.purpose),
            ImageDataSource::Bytes(bytes) => AssetRef::from_bytes_with(&bytes, &self.purpose),
            ImageDataSource::Placeholder(pixel) => AssetRef::from_bytes_with(pixel, &self.purpose),
        }
    }
}

impl ImageSource {
    pub fn from_file(path: impl Into<PathBuf>, purpose: ImagePurpose) -> Self {
        Self {
            source: ImageDataSource::File(path.into()),
            purpose,
        }
    }

    pub fn from_bytes(bytes: &[u8], purpose: ImagePurpose) -> Self {
        Self {
            source: ImageDataSource::Bytes(Bytes::copy_from_slice(bytes)),
            purpose,
        }
    }

    pub fn from_color(color: [f32; 4], purpose: ImagePurpose) -> Self {
        Self {
            source: ImageDataSource::Placeholder(color_to_pixles(color)),
            purpose,
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

#[derive(Debug)]
pub struct ImageAsset {
    pub format: vk::Format,
    pub dimensions: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl<'a, C: Context> Readable<'a, C> for ImageAsset {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, <C as Context>::Error> {
        Ok(Self {
            format: vk::Format::from_raw(reader.read_i32()?),
            dimensions: reader.read_value()?,
            mips: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for ImageAsset {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> Result<(), <C as Context>::Error> {
        writer.write_i32(self.format.as_raw())?;
        writer.write_value(&self.dimensions)?;
        writer.write_value(&self.mips)?;

        Ok(())
    }
}

impl Asset for ImageAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("c2871b90-6b51-427f-b1d8-4cedbedc8993");

    fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}
