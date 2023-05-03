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

use std::sync::Arc;

use ash::{extensions::khr, vk};
use log::info;

use crate::{
    vulkan::{ImageDesc, ImageType},
    BackendResult,
};

use super::{Device, Image, Surface};

#[derive(Debug, Copy, Clone)]
pub struct SwapchainDesc {
    pub format: vk::SurfaceFormatKHR,
    pub dims: vk::Extent2D,
    pub vsync: bool,
}

pub struct Swapchain {
    device: Arc<Device>,
    pub loader: khr::Swapchain,
    pub raw: vk::SwapchainKHR,
    pub desc: SwapchainDesc,
    pub images: Vec<Arc<Image>>,
    pub acquire_semaphore: Vec<vk::Semaphore>,
    pub rendering_finished_semaphore: Vec<vk::Semaphore>,
    pub next_semaphore: usize,
}

pub struct SwapchainImage {
    pub image: Arc<Image>,
    pub image_index: u32,
    pub acquire_semaphore: vk::Semaphore,
    pub rendering_finished_semaphore: vk::Semaphore,
}

impl Swapchain {
    pub fn enumerate_surface_formats(
        device: &Device,
        surface: &Surface,
    ) -> BackendResult<Vec<vk::SurfaceFormatKHR>> {
        Ok(unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(device.pdevice.raw, surface.raw)
        }?)
    }

    pub fn select_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
        let prefered = [
            vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];

        prefered.into_iter().find(|format| formats.contains(format))
    }

    pub fn new(
        device: &Arc<Device>,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> BackendResult<Self> {
        let surface_capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.pdevice.raw, surface.raw)
        }?;
        let mut desired_image_count = 3.max(surface_capabilities.min_image_count);
        if surface_capabilities.max_image_count != 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        info!("Swapchain image count {}", desired_image_count);

        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => desc.dims,
            _ => surface_capabilities.current_extent,
        };

        if surface_resolution.width == 0 || surface_resolution.height == 0 {
            return Err(crate::BackendError::Other(
                "Swapchain resolution can't be zero".to_owned(),
            ));
        }

        let present_mode_preferences = if desc.vsync {
            vec![vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO]
        } else {
            vec![vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        };

        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(device.pdevice.raw, surface.raw)
        }?;

        info!("Swapchain format: {:?}", desc.format.format);

        let present_mode = present_mode_preferences
            .into_iter()
            .find(|mode| present_modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);

        info!("Presentation mode: {:?}", present_mode);

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.raw)
            .min_image_count(desired_image_count)
            .image_format(desc.format.format)
            .image_color_space(desc.format.color_space)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .build();

        let loader = khr::Swapchain::new(&device.instance.raw, &device.raw);
        let swapchain = unsafe { loader.create_swapchain(&swapchain_create_info, None) }?;
        let images = unsafe { loader.get_swapchain_images(swapchain) }?;
        let images = images
            .iter()
            .map(|image| Image {
                device: device.clone(),
                raw: *image,
                desc: ImageDesc {
                    image_type: ImageType::Tex2D,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    flags: vk::ImageCreateFlags::empty(),
                    format: desc.format.format,
                    extent: [surface_resolution.width, surface_resolution.height],
                    tiling: vk::ImageTiling::OPTIMAL,
                    mip_levels: 1,
                    array_elements: 1,
                },
                allocation: None,
                views: Default::default(),
            })
            .map(Arc::new)
            .collect::<Vec<_>>();

        let acquire_semaphore = (0..images.len())
            .map(|_| unsafe {
                device
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();

        let rendering_finished_semaphore = (0..images.len())
            .map(|_| unsafe {
                device
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();

        Ok(Self {
            device: device.clone(),
            raw: swapchain,
            loader,
            desc: *desc,
            images,
            acquire_semaphore,
            rendering_finished_semaphore,
            next_semaphore: 0,
        })
    }

    pub fn acquire_next_image(&mut self) -> BackendResult<SwapchainImage> {
        let acquire_semaphore = self.acquire_semaphore[self.next_semaphore];
        let rendering_finished_semaphore = self.rendering_finished_semaphore[self.next_semaphore];

        let present_index = unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, acquire_semaphore, vk::Fence::null())
        };

        match present_index {
            Ok((present_index, _)) => {
                assert_eq!(present_index as usize, self.next_semaphore);

                self.next_semaphore = (self.next_semaphore + 1) % self.images.len();
                Ok(SwapchainImage {
                    image: self.images[present_index as usize].clone(),
                    image_index: present_index,
                    acquire_semaphore,
                    rendering_finished_semaphore,
                })
            }
            Err(err)
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                    || err == vk::Result::SUBOPTIMAL_KHR =>
            {
                Err(crate::BackendError::Other(
                    "Swapchain recreation needed".into(),
                ))
            }
            _err => panic!("Shitshow"),
        }
    }

    pub fn present_image(&self, image: SwapchainImage) {
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&[image.rendering_finished_semaphore])
            .swapchains(&[self.raw])
            .image_indices(&[image.image_index])
            .build();

        match unsafe {
            self.loader
                .queue_present(self.device.graphics_queue.raw, &present_info)
        } {
            Ok(_) => (),
            Err(err)
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                    || err == vk::Result::SUBOPTIMAL_KHR => {}
            _err => panic!("Fuckshit!"),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
        for semaphore in &self.acquire_semaphore {
            unsafe { self.device.raw.destroy_semaphore(*semaphore, None) };
        }
        for semaphore in &self.rendering_finished_semaphore {
            unsafe { self.device.raw.destroy_semaphore(*semaphore, None) };
        }
        for image in self.images.iter() {
            image.destroy_all_views(&self.device.raw);
        }
    }
}
