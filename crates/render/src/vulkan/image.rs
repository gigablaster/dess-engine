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

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use super::{image_cache::ImageMemory, BackendError, BackendResult, Device};
use ash::vk;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ImageType {
    Tex1D,
    Tex1DArray,
    Tex2D,
    Tex2DArray,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub image_type: ImageType,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
    pub format: vk::Format,
    pub extent: [u32; 2],
    pub tiling: vk::ImageTiling,
    pub mip_levels: u32,
    pub array_elements: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub view_type: Option<vk::ImageViewType>,
    pub format: Option<vk::Format>,
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: Option<u32>,
}

impl Default for ImageViewDesc {
    fn default() -> Self {
        Self {
            view_type: None,
            format: None,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: None,
        }
    }
}

impl ImageViewDesc {
    pub fn view_type(mut self, view_type: vk::ImageViewType) -> Self {
        self.view_type = Some(view_type);
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = Some(format);
        self
    }

    pub fn aspect_mask(mut self, aspect_mask: vk::ImageAspectFlags) -> Self {
        self.aspect_mask = aspect_mask;
        self
    }

    pub fn base_mip_level(mut self, base_mip_level: u32) -> Self {
        self.base_mip_level = base_mip_level;
        self
    }

    pub fn level_count(mut self, level_count: u32) -> Self {
        self.level_count = Some(level_count);
        self
    }

    fn build(&self, image: &Image) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo::builder()
            .format(self.format.unwrap_or(image.desc.format))
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                self.view_type
                    .unwrap_or_else(|| Self::convert_image_type_to_view_type(image)),
            )
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect_mask,
                base_mip_level: self.base_mip_level,
                level_count: self.level_count.unwrap_or(image.desc.mip_levels),
                base_array_layer: 0,
                layer_count: 1,
            })
            .build()
    }

    fn convert_image_type_to_view_type(image: &Image) -> vk::ImageViewType {
        match image.desc.image_type {
            ImageType::Tex1D => vk::ImageViewType::TYPE_1D,
            ImageType::Tex1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            ImageType::Tex2D => vk::ImageViewType::TYPE_2D,
            ImageType::Tex2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
        }
    }
}

#[derive(Debug)]
pub enum SubImage {
    All,
    Layer(u32),
    LayerAndMip(u32, u32),
}

#[derive(Debug)]
enum ImageAllocation {
    None,
    External,
    Cache(ImageMemory),
}

#[derive(Debug)]
pub struct Image {
    pub device: Arc<Device>,
    pub raw: vk::Image,
    pub desc: ImageDesc,
    allocation: ImageAllocation,
    views: Mutex<HashMap<ImageViewDesc, vk::ImageView>>,
}

impl Image {
    pub fn texture(
        device: &Arc<Device>,
        image_desc: ImageDesc,
        name: Option<&str>,
    ) -> BackendResult<Self> {
        let image = unsafe { device.raw.create_image(&image_desc.build(), None) }?;
        let allocation = ImageAllocation::Cache(device.allocate_image(image)?);

        if let Some(name) = name {
            device.set_object_name(image, name)?;
        }

        Ok(Self {
            device: device.clone(),
            raw: image,
            desc: image_desc,
            views: Default::default(),
            allocation,
        })
    }

    pub fn external(
        device: &Arc<Device>,
        image: vk::Image,
        image_desc: ImageDesc,
        name: Option<&str>,
    ) -> BackendResult<Self> {
        if let Some(name) = name {
            device.set_object_name(image, name)?
        }

        Ok(Self {
            device: device.clone(),
            raw: image,
            desc: image_desc,
            views: Default::default(),
            allocation: ImageAllocation::External,
        })
    }

    pub fn subresource(
        &self,
        subimage: SubImage,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageSubresourceRange {
        let desc = vk::ImageSubresourceRange::builder().aspect_mask(aspect);
        match subimage {
            SubImage::All => desc
                .base_array_layer(0)
                .layer_count(self.desc.array_elements)
                .base_mip_level(0)
                .level_count(self.desc.mip_levels)
                .build(),
            SubImage::Layer(layer) => desc
                .base_array_layer(layer)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(self.desc.mip_levels)
                .build(),
            SubImage::LayerAndMip(layer, mip) => desc
                .base_array_layer(layer)
                .base_mip_level(mip)
                .layer_count(1)
                .level_count(1)
                .build(),
        }
    }

    pub fn get_or_create_view(&self, view_desc: ImageViewDesc) -> BackendResult<vk::ImageView> {
        let mut views = self.views.lock().unwrap();
        if let Some(view) = views.get(&view_desc) {
            Ok(*view)
        } else {
            if self.desc.format == vk::Format::D32_SFLOAT
                && !view_desc.aspect_mask.contains(vk::ImageAspectFlags::DEPTH)
            {
                return Err(BackendError::Other(
                    "Depth-only resource used without vk::ImageAspectFlags::DEPTH flag".into(),
                ));
            }
            let create_info = vk::ImageViewCreateInfo {
                image: self.raw,
                ..view_desc.build(self)
            };

            let view = unsafe { self.device.raw.create_image_view(&create_info, None) }?;

            views.insert(view_desc, view);

            Ok(view)
        }
    }

    pub(crate) fn destroy_all_views(&self) {
        let mut views = self.views.lock().unwrap();
        views.iter().for_each(|(_, view)| {
            unsafe { self.device.raw.destroy_image_view(*view, None) };
        });
        views.clear();
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        self.device.with_drop_list(|droplist| {
            self.views
                .lock()
                .unwrap()
                .drain()
                .for_each(|(_, view)| droplist.drop_image_view(view));
            if let ImageAllocation::Cache(memory) = self.allocation {
                droplist.drop_image(self.raw, memory);
            }
        })
    }
}

impl ImageDesc {
    pub fn new(format: vk::Format, image_type: ImageType, extent: [u32; 2]) -> Self {
        Self {
            image_type,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            extent,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
        }
    }

    pub fn image_type(mut self, image_type: ImageType) -> Self {
        self.image_type = image_type;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn flags(mut self, flags: vk::ImageCreateFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn extents(mut self, extents: [u32; 2]) -> Self {
        self.extent = extents;
        self
    }

    pub fn tiling(mut self, tiling: vk::ImageTiling) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn array_elements(mut self, array_elements: u32) -> Self {
        self.array_elements = array_elements;
        self
    }

    fn build(&self) -> vk::ImageCreateInfo {
        let (image_type, image_extents, _image_layers) = match self.image_type {
            ImageType::Tex1D => (
                vk::ImageType::TYPE_1D,
                vk::Extent3D {
                    width: self.extent[0],
                    height: 1,
                    depth: 1,
                },
                1,
            ),

            ImageType::Tex1DArray => (
                vk::ImageType::TYPE_1D,
                vk::Extent3D {
                    width: self.extent[0],
                    height: 1,
                    depth: 1,
                },
                self.array_elements,
            ),
            ImageType::Tex2D => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D {
                    width: self.extent[0],
                    height: self.extent[1],
                    depth: 1,
                },
                1,
            ),
            ImageType::Tex2DArray => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D {
                    width: self.extent[0],
                    height: self.extent[1],
                    depth: 1,
                },
                self.array_elements,
            ),
        };

        vk::ImageCreateInfo {
            flags: self.flags,
            image_type,
            format: self.format,
            extent: image_extents,
            mip_levels: self.mip_levels,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: self.tiling,
            usage: self.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        }
    }
}
