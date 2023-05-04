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

use std::{slice, sync::Arc};

use ash::{extensions::khr, vk};
use log::info;

use crate::{
    vulkan::{ImageDesc, ImageType},
    BackendError, BackendResult,
};

use super::{Device, Image, Surface};

struct SwapchainInner {
    pub raw: vk::SwapchainKHR,
    pub images: Vec<Arc<Image>>,
    pub loader: khr::Swapchain,
    pub acquire_semaphore: Vec<vk::Semaphore>,
    pub rendering_finished_semaphore: Vec<vk::Semaphore>,
    pub next_semaphore: usize,
    pub dims: [u32; 2],
    pub format: vk::Format,
}

impl SwapchainInner {
    pub fn new(device: &Arc<Device>, surface: &Surface) -> BackendResult<Self> {
        let surface_capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.pdevice.raw, surface.raw)
        }?;

        let formats = Self::enumerate_surface_formats(device, surface)?;
        let format = match Self::select_surface_format(&formats) {
            Some(format) => format,
            None => {
                return Err(BackendError::Other(
                    "Can't find suitable surface format".into(),
                ))
            }
        };

        let mut desired_image_count = 3.max(surface_capabilities.min_image_count);
        if surface_capabilities.max_image_count != 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        info!("Swapchain image count {}", desired_image_count);

        let window_resolution = surface.window.size();
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: window_resolution.0,
                height: window_resolution.1,
            },
            _ => surface_capabilities.current_extent,
        };

        if surface_resolution.width == 0 || surface_resolution.height == 0 {
            return Err(crate::BackendError::WaitForSurface);
        }

        let present_mode_preferences = [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO];

        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(device.pdevice.raw, surface.raw)
        }?;

        info!("Swapchain format: {:?}", format.format);

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
            .image_format(format.format)
            .image_color_space(format.color_space)
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
                    format: format.format,
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
            raw: swapchain,
            images,
            acquire_semaphore,
            rendering_finished_semaphore,
            next_semaphore: 0,
            loader,
            format: format.format,
            dims: [surface_resolution.width, surface_resolution.height],
        })
    }

    fn enumerate_surface_formats(
        device: &Device,
        surface: &Surface,
    ) -> BackendResult<Vec<vk::SurfaceFormatKHR>> {
        Ok(unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(device.pdevice.raw, surface.raw)
        }?)
    }

    fn select_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
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

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
        for semaphore in &self.acquire_semaphore {
            unsafe { device.destroy_semaphore(*semaphore, None) };
        }
        for semaphore in &self.rendering_finished_semaphore {
            unsafe { device.destroy_semaphore(*semaphore, None) };
        }
        for image in self.images.iter() {
            image.destroy_all_views(&device);
        }
    }
}

pub struct Swapchain<'a> {
    device: Arc<Device>,
    surface: Surface<'a>,
    inner: SwapchainInner,
}

pub struct SwapchainImage {
    pub image: Arc<Image>,
    pub image_index: u32,
    pub acquire_semaphore: vk::Semaphore,
    pub rendering_finished_semaphore: vk::Semaphore,
}

impl<'a> Swapchain<'a> {
    pub fn new(device: &Arc<Device>, surface: Surface<'a>) -> BackendResult<Self> {
        Ok(Self {
            device: device.clone(),
            inner: SwapchainInner::new(device, &surface)?,
            surface,
        })
    }

    pub fn acquire_next_image(&mut self) -> BackendResult<SwapchainImage> {
        let acquire_semaphore = self.inner.acquire_semaphore[self.inner.next_semaphore];
        let rendering_finished_semaphore =
            self.inner.rendering_finished_semaphore[self.inner.next_semaphore];

        let present_index = unsafe {
            self.inner.loader.acquire_next_image(
                self.inner.raw,
                u64::MAX,
                acquire_semaphore,
                vk::Fence::null(),
            )
        };

        match present_index {
            Ok((present_index, _)) => {
                assert_eq!(present_index as usize, self.inner.next_semaphore);

                self.inner.next_semaphore =
                    (self.inner.next_semaphore + 1) % self.inner.images.len();
                Ok(SwapchainImage {
                    image: self.inner.images[present_index as usize].clone(),
                    image_index: present_index,
                    acquire_semaphore,
                    rendering_finished_semaphore,
                })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                Err(BackendError::RecreateSwapchain)
            }
            Err(err) => Err(BackendError::Vulkan(err)),
        }
    }

    pub fn present_image(&self, image: SwapchainImage) {
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&image.rendering_finished_semaphore))
            .swapchains(slice::from_ref(&self.inner.raw))
            .image_indices(slice::from_ref(&image.image_index))
            .build();

        match unsafe {
            self.inner
                .loader
                .queue_present(self.device.graphics_queue.raw, &present_info)
        } {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            _err => panic!("Can't present image by some reasons"),
        }
    }

    pub fn recreate(&mut self) -> BackendResult<()> {
        self.device.wait();
        info!("Recreate swapchain");
        self.inner.cleanup(&self.device.raw);
        self.inner = SwapchainInner::new(&self.device, &self.surface)?;

        Ok(())
    }

    pub fn backbuffer_format(&self) -> vk::Format {
        self.inner.format
    }
}

impl<'a> Drop for Swapchain<'a> {
    fn drop(&mut self) {
        self.inner.cleanup(&self.device.raw);
    }
}
