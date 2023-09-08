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

use ash::{
    extensions::khr,
    vk::{self},
};
use log::info;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::vulkan::{ImageDesc, ImageType};

use super::{
    AcquireError, CreateError, Device, GpuResource, Image, Instance, Semaphore, SwapchainError,
};

pub struct Surface {
    _instance: Arc<Instance>,
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) loader: khr::Surface,
}

impl Surface {
    pub fn create(
        instance: &Arc<Instance>,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self, CreateError> {
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.raw,
                display_handle,
                window_handle,
                None,
            )
        }?;
        let loader = khr::Surface::new(&instance.entry, &instance.raw);

        Ok(Self {
            _instance: instance.clone(),
            raw: surface,
            loader,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
struct SwapchainInner {
    pub raw: vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub loader: khr::Swapchain,
    pub acquire_semaphore: Vec<Semaphore>,
    pub rendering_finished_semaphore: Vec<Semaphore>,
    pub next_semaphore: usize,
    pub dims: [u32; 2],
    pub format: vk::Format,
}

impl SwapchainInner {
    pub fn new(
        device: &Device,
        surface: &Surface,
        resolution: [u32; 2],
    ) -> Result<Self, SwapchainError> {
        let surface_capabilities = unsafe {
            surface.loader.get_physical_device_surface_capabilities(
                device.physical_device().raw(),
                surface.raw,
            )
        }?;

        let formats = Self::enumerate_surface_formats(device, surface)?;
        let format = match Self::select_surface_format(&formats) {
            Some(format) => format,
            None => return Err(SwapchainError::NoSuitableSurfaceFormat),
        };

        let mut desired_image_count = 3.max(surface_capabilities.min_image_count);
        if surface_capabilities.max_image_count != 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        info!("Swapchain image count {}", desired_image_count);

        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: resolution[0],
                height: resolution[1],
            },
            _ => surface_capabilities.current_extent,
        };

        if surface_resolution.width == 0 || surface_resolution.height == 0 {
            return Err(SwapchainError::WaitForSurface);
        }

        let present_mode_preferences = [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO];

        let present_modes = unsafe {
            surface.loader.get_physical_device_surface_present_modes(
                device.physical_device().raw(),
                surface.raw,
            )
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

        let loader = khr::Swapchain::new(device.instance(), device.raw());
        let swapchain = unsafe { loader.create_swapchain(&swapchain_create_info, None) }?;
        let images = unsafe { loader.get_swapchain_images(swapchain) }?;
        let images = images
            .iter()
            .map(|image| {
                Image::external(
                    *image,
                    ImageDesc {
                        image_type: ImageType::Tex2D,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        flags: vk::ImageCreateFlags::empty(),
                        format: format.format,
                        extent: [surface_resolution.width, surface_resolution.height],
                        tiling: vk::ImageTiling::OPTIMAL,
                        mip_levels: 1,
                        array_elements: 1,
                    },
                )
            })
            .collect::<Vec<_>>();

        images.iter().enumerate().for_each(|(index, image)| {
            device.set_object_name(image.raw(), &format!("swapchain_image_{}", index));
        });

        let acquire_semaphore = (0..images.len())
            .map(|_| Semaphore::new(device.raw()).unwrap())
            .collect();

        let rendering_finished_semaphore = (0..images.len())
            .map(|_| Semaphore::new(device.raw()).unwrap())
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
    ) -> Result<Vec<vk::SurfaceFormatKHR>, SwapchainError> {
        Ok(unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(device.physical_device().raw(), surface.raw)
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
        for semaphore in &mut self.acquire_semaphore {
            semaphore.free(device)
        }
        for semaphore in &mut self.rendering_finished_semaphore {
            semaphore.free(device)
        }
        for image in self.images.iter() {
            image.destroy_all_views(device);
        }
    }
}

pub struct Swapchain {
    device: Arc<Device>,
    surface: Surface,
    inner: SwapchainInner,
}

pub struct SwapchainImage<'a> {
    pub image: &'a Image,
    pub image_index: u32,
    pub acquire_semaphore: Semaphore,
    pub presentation_finished: Semaphore,
}

impl Swapchain {
    pub fn new(
        device: &Arc<Device>,
        surface: Surface,
        resolution: [u32; 2],
    ) -> Result<Self, SwapchainError> {
        Ok(Self {
            device: device.clone(),
            inner: SwapchainInner::new(device, &surface, resolution)?,
            surface,
        })
    }

    pub fn acquire_next_image(&mut self) -> Result<SwapchainImage, AcquireError> {
        puffin::profile_scope!("wait for swapchain");
        let acquire_semaphore = self.inner.acquire_semaphore[self.inner.next_semaphore];
        let rendering_finished_semaphore =
            self.inner.rendering_finished_semaphore[self.inner.next_semaphore];

        let (present_index, _) = unsafe {
            self.inner.loader.acquire_next_image(
                self.inner.raw,
                u64::MAX,
                acquire_semaphore.raw,
                vk::Fence::null(),
            )
        }?;

        assert_eq!(present_index as usize, self.inner.next_semaphore);

        self.inner.next_semaphore = (self.inner.next_semaphore + 1) % self.inner.images.len();
        Ok(SwapchainImage {
            image: &self.inner.images[present_index as usize],
            image_index: present_index,
            acquire_semaphore,
            presentation_finished: rendering_finished_semaphore,
        })
    }

    pub fn present_image(&self, device: &Device, image: SwapchainImage) {
        puffin::profile_scope!("present");
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&image.presentation_finished.raw))
            .swapchains(slice::from_ref(&self.inner.raw))
            .image_indices(slice::from_ref(&image.image_index))
            .build();

        match unsafe {
            self.inner
                .loader
                .queue_present(device.queue().raw, &present_info)
        } {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            _err => panic!("Can't present image by some reasons"),
        }
    }

    pub fn recreate(
        &mut self,
        device: &Device,
        resolution: [u32; 2],
    ) -> Result<(), SwapchainError> {
        device.wait();
        info!("Recreate swapchain");
        self.inner.cleanup(device.raw());
        self.inner = SwapchainInner::new(device, &self.surface, resolution)?;

        Ok(())
    }

    pub fn backbuffer_format(&self) -> vk::Format {
        self.inner.format
    }

    pub fn render_area(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.inner.dims[0],
                height: self.inner.dims[1],
            },
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.inner.cleanup(self.device.raw());
    }
}