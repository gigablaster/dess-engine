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
use std::sync::Arc;

use ash::vk::{self};
use dess_backend::{ImageCreateDesc, ImageViewDesc};
use parking_lot::Mutex;

use crate::{Error, ImageHandle, ResourceManager};

#[derive(Debug, Default, Hash, PartialEq, Eq, Clone, Copy)]
pub enum TemporaryImageDims {
    #[default]
    Backbuffer,
    Divided(u32),
    Absolute([u32; 2]),
}

impl TemporaryImageDims {
    fn to_actual(self, dims: [u32; 2]) -> [u32; 2] {
        match self {
            Self::Backbuffer => dims,
            Self::Divided(a) => [dims[0] / a, dims[1] / a],
            Self::Absolute(dims) => dims,
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct TempImageDesc {
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    dims: [u32; 2],
}

pub struct TemporaryImage<'frame> {
    handle: ImageHandle,
    desc: TempImageDesc,
    pool: &'frame TempImagePool,
}

impl<'frame> TemporaryImage<'frame> {
    pub fn as_handle(&self) -> ImageHandle {
        self.handle
    }

    pub fn as_color(&self) -> Result<vk::ImageView, Error> {
        Ok(self
            .pool
            .resource_manager
            .image(self.handle)
            .ok_or(Error::InvalidHandle)?
            .view(ImageViewDesc::color())?)
    }

    pub fn as_depth(&self) -> Result<vk::ImageView, Error> {
        Ok(self
            .pool
            .resource_manager
            .image(self.handle)
            .ok_or(Error::InvalidHandle)?
            .view(ImageViewDesc::depth())?)
    }
}

impl<'frame> Drop for TemporaryImage<'frame> {
    fn drop(&mut self) {
        self.pool.release(self.handle, self.desc);
    }
}

pub struct TempImagePool {
    resource_manager: Arc<ResourceManager>,
    images: Mutex<Vec<(TempImageDesc, ImageHandle)>>,
    backbuffer_dims: [u32; 2],
}

impl TempImagePool {
    pub fn new(resource_manager: &Arc<ResourceManager>) -> Self {
        Self {
            resource_manager: resource_manager.clone(),
            images: Mutex::default(),
            backbuffer_dims: [0; 2],
        }
    }

    fn attachment(
        &self,
        desc: ImageCreateDesc,
        aspect: vk::ImageAspectFlags,
    ) -> Result<TemporaryImage, Error> {
        let temp_desc = TempImageDesc {
            format: desc.format,
            usage: desc.usage,
            aspect,
            dims: desc.dims,
        };
        let mut images = self.images.lock();
        let handle = if let Some(handle) = Self::find_image(&mut images, &temp_desc) {
            handle
        } else {
            self.resource_manager
                .create_image(desc, ImageViewDesc::new(aspect))?
        };
        Ok(TemporaryImage {
            pool: self,
            desc: temp_desc,
            handle,
        })
    }

    pub fn depth_attachment(
        &self,
        format: vk::Format,
        dims: TemporaryImageDims,
    ) -> Result<TemporaryImage, Error> {
        self.attachment(
            ImageCreateDesc::new(format, dims.to_actual(self.backbuffer_dims)).usage(
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
            vk::ImageAspectFlags::DEPTH,
        )
    }

    pub fn color_attachment(
        &self,
        format: vk::Format,
        dims: TemporaryImageDims,
    ) -> Result<TemporaryImage, Error> {
        self.attachment(
            ImageCreateDesc::new(format, dims.to_actual(self.backbuffer_dims))
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT),
            vk::ImageAspectFlags::COLOR,
        )
    }

    fn release(&self, image: ImageHandle, desc: TempImageDesc) {
        self.images.lock().push((desc, image));
    }

    fn find_image(
        images: &mut Vec<(TempImageDesc, ImageHandle)>,
        desc: &TempImageDesc,
    ) -> Option<ImageHandle> {
        images
            .iter()
            .enumerate()
            .find_map(|(index, (x, _))| (x == desc).then_some(index))
            .map(|index| images.remove(index).1)
    }
}
