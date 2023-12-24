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
    ffi::{c_void, CStr, CString},
    sync::Arc,
};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self, Bool32, DebugUtilsMessengerEXT},
};
use log::{info, log, Level};
use raw_window_handle::RawDisplayHandle;

use crate::BackendResult;

pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub(crate) raw: ash::Instance,
    debug_utils: Option<DebugUtils>,
    debug_messenger: Option<DebugUtilsMessengerEXT>,
}

#[derive(Debug, Default)]
pub struct InstanceBuilder {
    pub extensions: Vec<&'static CStr>,
    pub debug: bool,
    pub trace: bool,
}

impl InstanceBuilder {
    pub fn add_extensions(mut self, extensions: &[&'static CStr]) -> Self {
        self.extensions.extend_from_slice(extensions);
        self
    }

    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }
}

impl Instance {
    fn generate_extension_names(
        builder: &InstanceBuilder,
        display_handle: RawDisplayHandle,
    ) -> Vec<CString> {
        let mut names = Vec::new();
        if builder.debug {
            names.push(vk::ExtDebugUtilsFn::name().into());
        }
        let window_extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .iter()
            .map(|x| unsafe { CStr::from_ptr(*x).into() })
            .collect::<Vec<_>>();

        names.extend_from_slice(&window_extensions);

        names
    }

    fn generate_layer_names(builder: &InstanceBuilder) -> Vec<CString> {
        let mut names = Vec::new();
        if builder.debug {
            names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        }
        if builder.trace {
            names.push(CString::new("VK_LAYER_LUNARG_api_dump").unwrap());
        }

        names
    }

    pub(crate) fn vulkan_version() -> u32 {
        vk::make_api_version(0, 1, 3, 0)
    }

    pub fn new(builder: InstanceBuilder, display_handle: RawDisplayHandle) -> BackendResult<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let layer_names = Self::generate_layer_names(&builder);
        let layer_names = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let extensions = Self::generate_extension_names(&builder, display_handle);
        let extensions = extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let app_desc = vk::ApplicationInfo::builder()
            .api_version(Self::vulkan_version())
            .build();

        let instance_desc = vk::InstanceCreateInfo::builder()
            .application_info(&app_desc)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extensions)
            .build();

        let instance = unsafe { entry.create_instance(&instance_desc, None)? };
        info!("Created a Vulkan instance");
        let (debug_utils, debug_messenger) = if builder.debug {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(Self::vk_debug))
                .build();
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
            (Some(debug_utils_loader), Some(debug_callback))
        } else {
            (None, None)
        };

        Ok(Self {
            entry,
            raw: instance,
            debug_utils,
            debug_messenger,
        })
    }

    pub(crate) fn get_debug_utils(&self) -> Option<&DebugUtils> {
        if let Some(debug_utils) = &self.debug_utils {
            Some(debug_utils)
        } else {
            None
        }
    }

    fn get_vk_message_type(message_type: vk::DebugUtilsMessageTypeFlagsEXT) -> &'static str {
        match message_type {
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "VK:Performance",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VK:Validation:",
            _ => "VK",
        }
    }

    fn get_vk_message_severity(message_severity: vk::DebugUtilsMessageSeverityFlagsEXT) -> Level {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
            _ => Level::Debug,
        }
    }

    unsafe extern "system" fn vk_debug(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> Bool32 {
        let message = CStr::from_ptr((*data).p_message).to_str().unwrap();
        log!(
            target: Self::get_vk_message_type(message_type),
            Self::get_vk_message_severity(message_severity),
            "{}",
            message
        );

        if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            //panic!("!!!! VULKAN ERROR !!!!");
        }

        vk::FALSE
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(debug_utils) = self.debug_utils.take() {
            if let Some(debug_messenger) = self.debug_messenger.take() {
                unsafe { debug_utils.destroy_debug_utils_messenger(debug_messenger, None) };
            }
        }
        unsafe { self.raw.destroy_instance(None) };
    }
}
