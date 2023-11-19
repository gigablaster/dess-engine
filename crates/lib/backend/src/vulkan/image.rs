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
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

use crate::BackendResult;

use super::{Device, DropList, GpuAllocator, GpuMemory, ImageHandle, Instance, ToDrop};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub extent: [u32; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
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
        match image.desc.ty {
            vk::ImageType::TYPE_1D if image.desc.array_elements == 1 => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D_ARRAY,
            vk::ImageType::TYPE_2D if image.desc.array_elements == 2 => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            ty => panic!("Unknown image type {ty:?}"),
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageCreateDesc<'a> {
    pub extent: [u32; 2],
    pub ty: vk::ImageType,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub mip_levels: usize,
    pub array_elements: usize,
    pub dedicated: bool,
    pub name: Option<&'a str>,
}

impl<'a> ImageCreateDesc<'a> {
    pub fn new(format: vk::Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::empty(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 0,
            array_elements: 0,
            dedicated: false,
            name: None,
        }
    }

    pub fn texture(format: vk::Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn cubemap(format: vk::Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 6,
            dedicated: false,
            name: None,
        }
    }

    pub fn color_attachment(format: vk::Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn depth_stencil_attachment(format: vk::Format, extent: [u32; 2]) -> Self {
        Self {
            extent,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
            dedicated: false,
            name: None,
        }
    }

    pub fn ty(mut self, value: vk::ImageType) -> Self {
        self.ty = value;
        self
    }

    pub fn usage(mut self, value: vk::ImageUsageFlags) -> Self {
        self.usage = value;
        self
    }

    pub fn flags(mut self, value: vk::ImageCreateFlags) -> Self {
        self.flags = value;
        self
    }

    pub fn tiling(mut self, value: vk::ImageTiling) -> Self {
        self.tiling = value;
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

    fn build(&self) -> vk::ImageCreateInfo {
        vk::ImageCreateInfo::builder()
            .array_layers(self.array_elements as _)
            .mip_levels(self.mip_levels as _)
            .usage(self.usage)
            .flags(self.flags)
            .format(self.format)
            .image_type(self.ty)
            .tiling(self.tiling)
            .extent(self.create_extents())
            .build()
    }

    fn create_extents(&self) -> vk::Extent3D {
        match self.ty {
            vk::ImageType::TYPE_1D => vk::Extent3D {
                width: self.extent[0],
                height: 1,
                depth: 1,
            },
            vk::ImageType::TYPE_2D => vk::Extent3D {
                width: self.extent[0],
                height: self.extent[1],
                depth: 1,
            },
            vk::ImageType::TYPE_3D => vk::Extent3D {
                width: self.extent[0],
                height: self.extent[1],
                depth: self.array_elements as u32,
            },
            ty => panic!("Unknown image type {ty:?}"),
        }
    }
}

impl Device {
    pub fn create_image(&self, desc: ImageCreateDesc) -> BackendResult<ImageHandle> {
        let image =
            Self::create_image_impl(&self.instance, &self.raw, &self.memory_allocator, desc)?;
        Ok(self.image_storage.write().push(image.raw, image))
    }

    fn create_image_impl(
        instance: &Instance,
        device: &ash::Device,
        allocator: &Mutex<GpuAllocator>,
        desc: ImageCreateDesc,
    ) -> BackendResult<Image> {
        let image = unsafe { device.create_image(&desc.build(), None) }?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory = Self::allocate_impl(
            device,
            &mut allocator.lock(),
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            desc.dedicated,
        )?;
        unsafe { device.bind_image_memory(image, *memory.memory(), memory.offset()) }?;
        if let Some(name) = desc.name {
            Self::set_object_name_impl(instance, device, image, name);
        }
        Ok(Image {
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
        })
    }
}
