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

use std::{collections::HashSet, ffi::CStr, fmt::Debug, os::raw::c_char};

use ash::vk;

use super::{BackendError, BackendResult, Instance, Surface};

#[derive(Debug, Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

impl QueueFamily {
    pub fn is_supported(&self, flags: vk::QueueFlags) -> bool {
        self.properties.queue_flags.contains(flags)
    }
}

#[derive(Clone)]
pub struct PhysicalDevice {
    pub(crate) raw: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) supported_extensions: HashSet<String>,
}

impl PhysicalDevice {
    pub fn is_queue_flag_supported(&self, flags: vk::QueueFlags) -> bool {
        self.queue_families
            .iter()
            .any(|queue_family| queue_family.is_supported(flags))
    }

    pub fn get_queue(&self, flags: vk::QueueFlags) -> BackendResult<QueueFamily> {
        self.queue_families
            .iter()
            .filter(|x| x.is_supported(flags))
            .copied()
            .next()
            .ok_or(BackendError::Other(format!(
                "Can't find queue with {:?} support",
                flags
            )))
    }
}

impl Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PhysicalDevice {{ {:#?} }}", self.properties)
    }
}

impl Instance {
    pub fn enumerate_physical_devices(&self) -> BackendResult<Vec<PhysicalDevice>> {
        unsafe {
            Ok(self
                .raw
                .enumerate_physical_devices()?
                .into_iter()
                .map(|pdevice| {
                    let properties = self.raw.get_physical_device_properties(pdevice);
                    let queue_families = self
                        .raw
                        .get_physical_device_queue_family_properties(pdevice)
                        .into_iter()
                        .enumerate()
                        .map(|(index, properties)| QueueFamily {
                            index: index as _,
                            properties,
                        })
                        .collect();

                    let memory_properties = self.raw.get_physical_device_memory_properties(pdevice);
                    let extension_properties = self
                        .raw
                        .enumerate_device_extension_properties(pdevice)
                        .unwrap();
                    let supported_extensions = extension_properties
                        .iter()
                        .map(|ext| {
                            CStr::from_ptr(ext.extension_name.as_ptr() as *const c_char)
                                .to_string_lossy()
                                .as_ref()
                                .to_owned()
                        })
                        .collect();

                    PhysicalDevice {
                        raw: pdevice,
                        queue_families,
                        properties,
                        memory_properties,
                        supported_extensions,
                    }
                })
                .collect())
        }
    }

    pub fn find_optimal_format(
        &self,
        pdevice: &PhysicalDevice,
        formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        formats.iter().find_map(|format| {
            let props = unsafe {
                self.raw
                    .get_physical_device_format_properties(pdevice.raw, *format)
            };
            if tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features)
            {
                Some(*format)
            } else if tiling == vk::ImageTiling::OPTIMAL
                && props.optimal_tiling_features.contains(features)
            {
                Some(*format)
            } else {
                None
            }
        })
    }
}

pub trait PhysicalDeviceList {
    fn with_support(&self, surface: &Surface, flags: vk::QueueFlags) -> Vec<PhysicalDevice>;
    fn with_device_type(&self, device_type: vk::PhysicalDeviceType) -> Vec<PhysicalDevice>;
    fn find_suitable_device(
        &self,
        surface: &Surface,
        device_types: &[vk::PhysicalDeviceType],
    ) -> Option<PhysicalDevice>;
}

impl PhysicalDeviceList for Vec<PhysicalDevice> {
    fn with_support(&self, surface: &Surface, flags: vk::QueueFlags) -> Vec<PhysicalDevice> {
        self.into_iter()
            .filter_map(|pdevice| {
                let support_flags =
                    pdevice
                        .queue_families
                        .iter()
                        .enumerate()
                        .any(|(index, info)| {
                            info.is_supported(flags)
                                && unsafe {
                                    surface
                                        .loader
                                        .get_physical_device_surface_support(
                                            pdevice.raw,
                                            index as u32,
                                            surface.raw,
                                        )
                                        .unwrap()
                                }
                        });

                if support_flags {
                    Some(pdevice.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn with_device_type(&self, device_type: vk::PhysicalDeviceType) -> Vec<PhysicalDevice> {
        self.into_iter()
            .filter(|pdevice| pdevice.properties.device_type == device_type)
            .cloned()
            .collect()
    }

    fn find_suitable_device(
        &self,
        surface: &Surface,
        device_types: &[vk::PhysicalDeviceType],
    ) -> Option<PhysicalDevice> {
        let with_graphics_support = self.with_support(surface, vk::QueueFlags::GRAPHICS);
        device_types.iter().find_map(|device_type| {
            let suitable = with_graphics_support.with_device_type(*device_type);
            suitable.into_iter().next()
        })
    }
}
