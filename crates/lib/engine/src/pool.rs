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

use std::collections::HashMap;

use ash::vk;
use dess_backend::{
    vulkan::{Device, ImageCreateDesc, ImageHandle, ImageViewDesc},
    BackendError, BackendResult,
};
use parking_lot::Mutex;
use smol_str::SmolStr;

/// Resource size
///
/// Related to back buffer or fixed size.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RelativeImageSize {
    /// Same as back buffer
    #[default]
    Backbuffer,
    /// Some part of backbuffer, backbuffer resolution divided by value
    Divied(u32),
    /// Fixed size
    Fixed(u32, u32),
}

impl RelativeImageSize {
    pub fn apply(&self, dims: [u32; 2]) -> [u32; 2] {
        match self {
            RelativeImageSize::Backbuffer => dims,
            RelativeImageSize::Divied(div) => [dims[0] / div, dims[1] / div],
            RelativeImageSize::Fixed(w, h) => [*w, *h],
        }
    }
}

/// Pool resource descriptor
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PoolImageDesc {
    pub format: vk::Format,
    pub aspect_mask: vk::ImageAspectFlags,
    pub usage: vk::ImageUsageFlags,
    pub resolution: RelativeImageSize,
}

/// Image allocated from pool
#[derive(Debug, Default)]
pub struct PoolImage {
    pub handle: ImageHandle,
    pub desc: PoolImageDesc,
    pub view: vk::ImageView,
}

/// Pool for render resources
///
/// Supports only images for now. Should work for any other resources, until their size isn't
/// changing from frame to frame.
///
/// In the future I should analyze render graph for resources that might be aliased.
/// But that's outside of my scope.
pub struct ResourcePool<'a> {
    device: &'a Device,
    /// All free resources. Resource goes there when it isn't needed anymore.
    /// When desired resource isn't needed - it goes back to pool.
    /// Pool is cleaned when swapchain is reacreated.
    pool: Mutex<Vec<PoolImage>>,
    /// All named resources. Resource goes there from pool when resources is going
    /// outside of node, going back to pool when it isn't needed in any future nodes.
    resources: Mutex<HashMap<SmolStr, Option<PoolImage>>>,
}

/// Temporary image
///
/// Only used in current scope. returned to pool after dropping.
pub struct TemporaryImage<'a> {
    pool: &'a ResourcePool<'a>,
    image: PoolImage,
}

impl<'a> TemporaryImage<'a> {
    pub fn view(&self) -> vk::ImageView {
        self.image.view
    }

    pub fn image(&self) -> ImageHandle {
        self.image.handle
    }
}

impl<'a> Drop for TemporaryImage<'a> {
    fn drop(&mut self) {
        self.pool.deallocate(std::mem::take(&mut self.image));
    }
}

impl<'a> ResourcePool<'a> {
    pub fn new(device: &'a Device) -> BackendResult<Self> {
        Ok(Self {
            device,
            pool: Mutex::default(),
            resources: Mutex::default(),
        })
    }

    /// Get or allocate resource
    ///
    /// Resource is allocated if not found in pool.
    /// Returns barrier to convert resource from nothing to desired layout.
    ///
    /// Assume that resources allocated this way are going to be written. No reason
    /// to read disposed image.
    fn allocate(&self, backbuffer_dims: [u32; 2], desc: PoolImageDesc) -> BackendResult<PoolImage> {
        let mut pool = self.pool.lock();
        let image = if let Some(index) =
            pool.iter()
                .enumerate()
                .find_map(|(index, x)| if x.desc == desc { Some(index) } else { None })
        {
            pool.remove(index)
        } else {
            let dims = desc.resolution.apply(backbuffer_dims);
            let name = format!("{:?}", desc);
            let create_desc = ImageCreateDesc::new(desc.format, dims)
                .array_elements(1)
                .mip_levels(1)
                .usage(desc.usage)
                .tiling(vk::ImageTiling::OPTIMAL)
                .ty(vk::ImageType::TYPE_2D)
                .name(&name);
            let image = self.device.create_image(create_desc)?;
            let view = self
                .device
                .get_or_create_view(image, ImageViewDesc::default())?;

            PoolImage {
                handle: image,
                desc,
                view,
            }
        };

        Ok(image)
    }

    /// Returns resouce back to pool
    fn deallocate(&self, image: PoolImage) {
        let pool = &mut self.pool.lock();
        pool.push(image);
    }

    pub fn temp_image(
        &'a self,
        backbuffer_dims: [u32; 2],
        desc: PoolImageDesc,
    ) -> BackendResult<TemporaryImage<'a>> {
        Ok(TemporaryImage {
            pool: self,
            image: self.allocate(backbuffer_dims, desc)?,
        })
    }

    /// Get or allocated named resource
    ///
    /// Calls allocate for newly created resource, otherwise gets it from named resource
    /// pool. In any way returns image and barrier.
    pub fn get_or_allocate(
        &self,
        backbuffer_dims: [u32; 2],
        name: &str,
        desc: PoolImageDesc,
    ) -> BackendResult<PoolImage> {
        let resources = &mut self.resources.lock();
        if let Some(image) = resources.get_mut(name) {
            if let Some(image) = image.take() {
                Ok(image)
            } else {
                Err(BackendError::Fail)
            }
        } else {
            resources.insert(name.into(), None);
            self.allocate(backbuffer_dims, desc)
        }
    }

    /// Stores resource for future use by other nodes
    pub fn put(&self, name: &str, image: PoolImage) {
        self.resources
            .lock()
            .entry(name.into())
            .or_default()
            .replace(image);
    }

    /// Clear all existing resources in pool
    ///
    /// Called when swapchain is recreated or render graph was changed.
    pub fn purge(&self) {
        let mut pool = self.pool.lock();
        let mut resources = self.resources.lock();
        pool.iter_mut()
            .for_each(|x| self.device.destroy_image(x.handle));
        resources.iter_mut().for_each(|(_, image)| {
            self.device.destroy_image(
                image
                    .take()
                    .expect("All named resources must be in pool")
                    .handle,
            )
        });
        pool.clear();
        resources.clear();
    }
}

impl<'a> Drop for ResourcePool<'a> {
    fn drop(&mut self) {
        self.purge();
    }
}
