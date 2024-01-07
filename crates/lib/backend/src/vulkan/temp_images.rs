// Copyright (C) 2023-2024 gigablaster

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
use std::marker::PhantomData;

use crate::{
    BackendResult, Device, Format, FrameContext, ImageCreateDesc, ImageHandle, ImageUsage,
    ImageView, ImageViewDesc, RenderArea,
};

#[derive(Debug, Default, Hash, PartialEq, Eq, Clone, Copy)]
pub enum TemporaryImageDims {
    #[default]
    Backbuffer,
    Divided(u32),
    Absolute([u32; 2]),
}

impl TemporaryImageDims {
    fn to_actual(self, area: RenderArea) -> [u32; 2] {
        match self {
            Self::Backbuffer => [area.width, area.height],
            Self::Divided(a) => [area.width / a, area.height / a],
            Self::Absolute(dims) => dims,
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct TempImageDesc {
    pub format: Format,
    pub usage: ImageUsage,
    pub dims: [u32; 2],
}

#[derive(Debug)]
pub struct TemporaryImage<'frame> {
    device: &'frame Device,
    image: ImageHandle,
    desc: TempImageDesc,
    _marker: PhantomData<&'frame ()>,
}

impl<'frame> TemporaryImage<'frame> {
    pub fn get_or_create_view(&self, desc: ImageViewDesc) -> BackendResult<ImageView> {
        self.device.get_or_create_view(self.image, desc)
    }

    pub fn as_color(&self) -> BackendResult<ImageView> {
        self.get_or_create_view(ImageViewDesc::color())
    }

    pub fn as_depth(&self) -> BackendResult<ImageView> {
        self.get_or_create_view(ImageViewDesc::depth())
    }

    pub fn as_handle(&self) -> ImageHandle {
        self.image
    }
}

impl<'frame> Drop for TemporaryImage<'frame> {
    fn drop(&mut self) {
        self.device.release_temp_image(self.desc, self.image);
    }
}

fn get_free_image(
    images: &mut Vec<(TempImageDesc, ImageHandle)>,
    key: &TempImageDesc,
) -> Option<ImageHandle> {
    images
        .iter()
        .enumerate()
        .find_map(
            |(index, (image_key, _))| {
                if image_key == key {
                    Some(index)
                } else {
                    None
                }
            },
        )
        .map(|index| images.remove(index).1)
}

impl Device {
    pub(crate) fn get_or_create_temp_image(
        &self,
        desc: TempImageDesc,
    ) -> BackendResult<ImageHandle> {
        let mut free_images = self.free_temp_images.lock();
        if let Some(image) = get_free_image(&mut free_images, &desc) {
            Ok(image)
        } else {
            let mut all_images = self.temp_images.lock();
            let image = self.create_image(
                ImageCreateDesc::new(desc.format, desc.dims)
                    .usage(desc.usage)
                    .name(&format!("{} - {:?}", all_images.len(), desc)),
            )?;
            all_images.push((desc, image));
            Ok(image)
        }
    }

    pub(crate) fn release_temp_image(&self, desc: TempImageDesc, image: ImageHandle) {
        let mut free_image = self.free_temp_images.lock();
        free_image.push((desc, image));
    }
}

impl<'device, 'frame> FrameContext<'device, 'frame> {
    pub fn get_temporary_image(
        &self,
        format: Format,
        usage: ImageUsage,
        dims: TemporaryImageDims,
    ) -> BackendResult<TemporaryImage> {
        let dims = dims.to_actual(self.render_area);
        let desc = TempImageDesc {
            format,
            usage,
            dims,
        };
        let image = self.device.get_or_create_temp_image(desc)?;
        Ok(TemporaryImage {
            device: self.device,
            image,
            desc,
            _marker: PhantomData,
        })
    }
}
