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

use ash::vk::{self};
use bitflags::bitflags;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use speedy::{Readable, Writable};

use crate::{BackendError, BackendResult, Format};

use super::{AsVulkan, Device, DropList, GpuMemory, ImageHandle, ImageSubresourceData, ToDrop};

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageType {
    Type1D,
    #[default]
    Type2D,
    Type3D,
}

impl From<ImageType> for vk::ImageType {
    fn from(value: ImageType) -> Self {
        match value {
            ImageType::Type1D => vk::ImageType::TYPE_1D,
            ImageType::Type2D => vk::ImageType::TYPE_2D,
            ImageType::Type3D => vk::ImageType::TYPE_3D,
        }
    }
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
    pub struct ImageUsage: u32 {
        const None = 0;
        const Sampled = 1;
        const Storage = 2;
        const ColorTarget = 4;
        const DepthStencilTarget = 8;
        const Destination = 16;
        const Source = 32;
    }
}
impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(value: ImageUsage) -> Self {
        let mut result = vk::ImageUsageFlags::empty();
        if value.contains(ImageUsage::Sampled) {
            result |= vk::ImageUsageFlags::SAMPLED;
        }
        if value.contains(ImageUsage::Storage) {
            result |= vk::ImageUsageFlags::STORAGE;
        }
        if value.contains(ImageUsage::ColorTarget) {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if value.contains(ImageUsage::DepthStencilTarget) {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if value.contains(ImageUsage::Destination) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if value.contains(ImageUsage::Source) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        result
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub extent: [u32; 2],
    pub ty: ImageType,
    pub usage: ImageUsage,
    pub format: Format,
    pub mip_levels: u32,
    pub array_elements: u32,
    pub(crate) tiling: vk::ImageTiling,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub ty: Option<ImageViewType>,
    pub format: Option<Format>,
    pub aspect: ImageAspect,
    pub base_mip_level: u32,
    pub level_count: Option<u32>,
}

bitflags! {
    #[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
    pub struct ImageAspect: u32 {
        const None = 0;
        const Color = 1;
        const Depth = 2;
        const Stencil = 4;
    }
}

impl From<ImageAspect> for vk::ImageAspectFlags {
    fn from(value: ImageAspect) -> Self {
        let mut result = vk::ImageAspectFlags::empty();
        if value.contains(ImageAspect::Color) {
            result |= vk::ImageAspectFlags::COLOR;
        }
        if value.contains(ImageAspect::Depth) {
            result |= vk::ImageAspectFlags::DEPTH;
        }
        if value.contains(ImageAspect::Stencil) {
            result |= vk::ImageAspectFlags::STENCIL;
        }

        result
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageViewType {
    Type1D,
    Type1DArray,
    #[default]
    Type2D,
    Type2DArray,
    Type3D,
}

impl From<ImageViewType> for vk::ImageViewType {
    fn from(value: ImageViewType) -> Self {
        match value {
            ImageViewType::Type1D => vk::ImageViewType::TYPE_1D,
            ImageViewType::Type1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            ImageViewType::Type2D => vk::ImageViewType::TYPE_2D,
            ImageViewType::Type2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            ImageViewType::Type3D => vk::ImageViewType::TYPE_3D,
        }
    }
}

impl Default for ImageViewDesc {
    fn default() -> Self {
        Self {
            ty: None,
            format: None,
            aspect: ImageAspect::None,
            base_mip_level: 0,
            level_count: None,
        }
    }
}

impl ImageViewDesc {
    pub fn color() -> Self {
        Self {
            ty: None,
            format: None,
            aspect: ImageAspect::Color,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn depth() -> Self {
        Self {
            ty: None,
            format: None,
            aspect: ImageAspect::Depth,
            base_mip_level: 0,
            level_count: None,
        }
    }

    pub fn view_type(mut self, view_type: ImageViewType) -> Self {
        self.ty = Some(view_type);
        self
    }

    pub fn format(mut self, format: Format) -> Self {
        self.format = Some(format);
        self
    }

    pub fn aspect(mut self, aspect: ImageAspect) -> Self {
        self.aspect = aspect;
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
            .format(self.format.unwrap_or(image.desc.format).into())
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                self.ty
                    .unwrap_or_else(|| Self::convert_image_type_to_view_type(image))
                    .into(),
            )
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect.into(),
                base_mip_level: self.base_mip_level,
                level_count: self.level_count.unwrap_or(image.desc.mip_levels),
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image.raw)
            .build()
    }

    fn convert_image_type_to_view_type(image: &Image) -> ImageViewType {
        match image.desc.ty {
            ImageType::Type1D if image.desc.array_elements == 1 => ImageViewType::Type1D,
            ImageType::Type1D => ImageViewType::Type1DArray,
            ImageType::Type2D if image.desc.array_elements == 1 => ImageViewType::Type2D,
            ImageType::Type2D => ImageViewType::Type2DArray,
            ImageType::Type3D => ImageViewType::Type3D,
        }
    }
}

#[derive(Debug)]
pub struct Image {
    pub(crate) raw: vk::Image,
    pub desc: ImageDesc,
    pub(crate) memory: Option<GpuMemory>,
    pub(crate) views: RwLock<HashMap<ImageViewDesc, vk::ImageView>>,
}

impl Image {
    pub fn get_or_create_view(
        &self,
        device: &ash::Device,
        desc: ImageViewDesc,
    ) -> BackendResult<vk::ImageView> {
        let views = self.views.upgradable_read();
        if let Some(view) = views.get(&desc) {
            Ok(*view)
        } else {
            let mut views = RwLockUpgradableReadGuard::upgrade(views);
            if let Some(view) = views.get(&desc) {
                Ok(*view)
            } else {
                let view = unsafe { device.create_image_view(&desc.build(self), None) }?;
                views.insert(desc, view);

                Ok(view)
            }
        }
    }

    pub(crate) fn clear_views(&self, device: &ash::Device) {
        let mut views = self.views.write();
        for (_, view) in views.iter() {
            unsafe { device.destroy_image_view(*view, None) }
        }
        views.clear();
    }

    fn drop_views(&self, drop_list: &mut DropList) {
        let mut views = self.views.write();
        for (_, view) in views.iter() {
            drop_list.drop_image_view(*view);
        }
        views.clear();
    }

    pub fn free(&mut self, device: &Device) {
        self.to_drop(&mut device.current_drop_list.lock());
    }
}

impl ToDrop for Image {
    fn to_drop(&mut self, drop_list: &mut DropList) {
        self.drop_views(drop_list);
        if let Some(memory) = self.memory.take() {
            drop_list.free_memory(memory);
            drop_list.drop_image(self.raw);
        }
    }
}

impl AsVulkan<vk::Image> for Image {
    fn as_vk(&self) -> vk::Image {
        self.raw
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable)]
pub enum ImageMultisampling {
    #[default]
    None,
    MSAA2,
    MSAA4,
    MSAA8,
}

impl From<ImageMultisampling> for vk::SampleCountFlags {
    fn from(value: ImageMultisampling) -> Self {
        match value {
            ImageMultisampling::None => vk::SampleCountFlags::TYPE_1,
            ImageMultisampling::MSAA2 => vk::SampleCountFlags::TYPE_2,
            ImageMultisampling::MSAA4 => vk::SampleCountFlags::TYPE_4,
            ImageMultisampling::MSAA8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageCreateDesc<'a> {
    pub extent: [u32; 2],
    pub ty: ImageType,
    pub usage: ImageUsage,
    pub format: Format,
    pub samples: ImageMultisampling,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
    pub(crate) flags: vk::ImageCreateFlags,
    pub(crate) tiling: vk::ImageTiling,
    pub initial_data: Option<&'a [ImageSubresourceData<'a>]>,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: ImageType::Type2D,
            usage: ImageUsage::empty(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::default(),
            mip_levels: 0,
            array_elements: 0,
            dedicated: false,
            name: None,
            initial_data: None,
        }
    }

    pub fn texture(format: Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: ImageType::Type2D,
            usage: ImageUsage::Sampled | ImageUsage::Destination,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::default(),
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_data: None,
        }
    }

    pub fn cubemap(format: Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: ImageType::Type2D,
            usage: ImageUsage::Sampled | ImageUsage::Destination,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::default(),
            mip_levels: 1,
            array_elements: 6,
            dedicated: false,
            name: None,
            initial_data: None,
        }
    }

    pub fn color_attachment(format: Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: ImageType::Type2D,
            usage: ImageUsage::ColorTarget,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::default(),
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_data: None,
        }
    }

    pub fn depth_stencil_attachment(format: Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: ImageType::Type2D,
            usage: ImageUsage::DepthStencilTarget,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: ImageMultisampling::default(),
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
            initial_data: None,
        }
    }

    pub fn ty(mut self, value: ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn usage(mut self, value: ImageUsage) -> Self {
        self.usage = value;
        self
    }

    pub fn sampled(mut self) -> Self {
        self.usage |= ImageUsage::Sampled;
        self
    }

    pub fn samples(mut self, value: ImageMultisampling) -> Self {
        self.samples = value;
        self
    }

    pub fn mip_levels(mut self, value: usize) -> Self {
        self.mip_levels = value;
        self
    }

    pub fn array_elements(mut self, value: usize) -> Self {
        self.array_elements = value;
        self
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn initial_data(mut self, data: &'a [ImageSubresourceData]) -> Self {
        self.initial_data = Some(data);
        self
    }

    fn build(&self) -> vk::ImageCreateInfo {
        let mut usage = self.usage;
        if self.initial_data.is_some() {
            usage |= ImageUsage::Destination;
        }
        vk::ImageCreateInfo::builder()
            .array_layers(self.array_elements as _)
            .mip_levels(self.mip_levels as _)
            .usage(usage.into())
            .flags(self.flags)
            .format(self.format.into())
            .samples(self.samples.into())
            .image_type(self.ty.into())
            .tiling(self.tiling)
            .extent(self.create_extents())
            .build()
    }

    fn create_extents(&self) -> vk::Extent3D {
        match self.ty {
            ImageType::Type1D => vk::Extent3D {
                width: self.extent[0],
                height: 1,
                depth: 1,
            },
            ImageType::Type2D => vk::Extent3D {
                width: self.extent[0],
                height: self.extent[1],
                depth: 1,
            },
            ImageType::Type3D => vk::Extent3D {
                width: self.extent[0],
                height: self.extent[1],
                depth: self.array_elements as u32,
            },
        }
    }
}

impl Device {
    pub fn create_image(&self, desc: ImageCreateDesc) -> BackendResult<ImageHandle> {
        let image = self.create_image_impl(desc)?;
        Ok(self.image_storage.write().push(image.raw, image))
    }

    pub fn get_or_create_view(
        &self,
        handle: ImageHandle,
        desc: ImageViewDesc,
    ) -> BackendResult<vk::ImageView> {
        self.image_storage
            .read()
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .get_or_create_view(&self.raw, desc)
    }

    pub fn upload_image(
        &self,
        handle: ImageHandle,
        data: &[ImageSubresourceData],
    ) -> BackendResult<()> {
        let images = self.image_storage.read();
        let image = images.get_cold(handle).ok_or(BackendError::InvalidHandle)?;
        self.staging.lock().upload_image(self, image, data)
    }

    fn create_image_impl(&self, desc: ImageCreateDesc) -> BackendResult<Image> {
        let image = unsafe { self.raw.create_image(&desc.build(), None) }?;
        if let Some(name) = desc.name {
            self.set_object_name(image, name);
        }
        let mut requirements = unsafe { self.raw.get_image_memory_requirements(image) };
        // Workaround - gpu_alloc returns wrong offset is wrong when size < aligment.
        requirements.size = requirements.size.max(requirements.alignment);
        let memory = Self::allocate_impl(
            &self.raw,
            &mut self.memory_allocator.lock(),
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            desc.dedicated,
        )?;
        unsafe {
            self.raw
                .bind_image_memory(image, *memory.memory(), memory.offset())
        }?;
        let image = Image {
            raw: image,
            desc: ImageDesc {
                extent: desc.extent,
                ty: desc.ty,
                usage: desc.usage,
                format: desc.format,
                tiling: desc.tiling,
                mip_levels: desc.mip_levels as u32,
                array_elements: desc.array_elements as u32,
            },
            memory: Some(memory),
            views: RwLock::default(),
        };

        if let Some(data) = desc.initial_data {
            self.staging.lock().upload_image(self, &image, data)?;
        }

        Ok(image)
    }
}
