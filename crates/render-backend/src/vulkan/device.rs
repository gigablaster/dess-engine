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
    ffi::CStr,
    fmt::Debug,
    slice,
    sync::{Arc, Mutex},
};

use ash::{extensions::khr, vk};
use gpu_alloc::Config;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::info;

use crate::{Allocator, BackendError, BackendResult, DropList};

use super::{CommandBuffer, FrameContext, Instance, PhysicalDevice, QueueFamily, SwapchainImage};

pub struct Queue {
    pub(crate) raw: vk::Queue,
    pub(crate) family: QueueFamily,
}

pub struct Device {
    pub(crate) instance: Arc<Instance>,
    pub raw: ash::Device,
    pub pdevice: PhysicalDevice,
    pub graphics_queue: Queue,
    pub transfer_queue: Queue,
    pub allocator: Arc<Mutex<Allocator>>,
    setup_cb: Mutex<CommandBuffer>,
    frames: [Mutex<Arc<FrameContext>>; 2],
    drop_lists: [Mutex<DropList>; 2],
}

impl Device {
    pub fn create(instance: &Arc<Instance>, pdevice: &PhysicalDevice) -> BackendResult<Arc<Self>> {
        if !pdevice.is_queue_flag_supported(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER) {
            return Err(BackendError::Other(
                "Device doesn't support graphics and transfer queues".into(),
            ));
        };

        let device_extension_names = vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrImagelessFramebufferFn::name().as_ptr(),
            vk::KhrImageFormatListFn::name().as_ptr(),
        ];

        for ext in &device_extension_names {
            let ext = unsafe { CStr::from_ptr(*ext).to_str() }.unwrap();
            if !pdevice.supported_extensions.contains(ext) {
                return Err(BackendError::NoExtension(ext.into()));
            }
        }

        let graphics_queue = pdevice.get_queue(vk::QueueFlags::GRAPHICS)?;
        let transfer_queue = pdevice.get_queue(vk::QueueFlags::TRANSFER)?;

        let mut imageless_framebuffer_address =
            vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut imageless_framebuffer_address)
            .build();

        unsafe {
            instance
                .raw
                .get_physical_device_features2(pdevice.raw, &mut features)
        };

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue.index)
            .queue_priorities(&priorities)
            .build()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features)
            .build();

        let device = unsafe {
            instance
                .raw
                .create_device(pdevice.raw, &device_create_info, None)?
        };

        info!("Created a Vulkan device");

        let device_properties =
            unsafe { device_properties(&instance.raw, Instance::vulkan_version(), pdevice.raw)? };
        let allocator_config = Config::i_am_potato();
        let allocator = Allocator::new(allocator_config, device_properties);
        let setup_cb = CommandBuffer::new(&device, &transfer_queue)?;

        let frames = [
            Mutex::new(Arc::new(FrameContext::new(&device, &graphics_queue)?)),
            Mutex::new(Arc::new(FrameContext::new(&device, &graphics_queue)?)),
        ];

        let drop_lists = [
            Mutex::new(DropList::default()),
            Mutex::new(DropList::default()),
        ];

        Ok(Arc::new(Self {
            instance: instance.clone(),
            pdevice: pdevice.clone(),
            graphics_queue: Self::create_queue(&device, graphics_queue),
            transfer_queue: Self::create_queue(&device, transfer_queue),
            allocator: Arc::new(Mutex::new(allocator)),
            setup_cb: Mutex::new(setup_cb),
            raw: device,
            frames,
            drop_lists,
        }))
    }

    fn create_queue(device: &ash::Device, queue_family: QueueFamily) -> Queue {
        let queue = unsafe { device.get_device_queue(queue_family.index, 0) };

        Queue {
            raw: queue,
            family: queue_family,
        }
    }

    pub fn begin_frame(&self) -> BackendResult<Arc<FrameContext>> {
        let mut frame0 = self.frames[0].lock().unwrap();
        {
            if let Some(frame0) = Arc::get_mut(&mut frame0) {
                unsafe {
                    self.raw.wait_for_fences(
                        slice::from_ref(&frame0.command_buffer.fence),
                        true,
                        u64::MAX,
                    )
                }?;
                let mut allocator = self.allocator.lock().unwrap();
                self.drop_lists[0]
                    .lock()
                    .unwrap()
                    .free(&self.raw, &mut allocator);
            } else {
                return Err(BackendError::Other(
                    "Unable to begin frame: frame data is being held by user code".into(),
                ));
            }
        }
        Ok(frame0.clone())
    }

    pub fn end_frame(&self, frame: Arc<FrameContext>) -> BackendResult<()> {
        drop(frame);

        let mut frame0 = self.frames[0].lock().unwrap();
        if let Some(frame0) = Arc::get_mut(&mut frame0) {
            let mut frame1 = self.frames[1].lock().unwrap();
            let frame1 = Arc::get_mut(&mut frame1).unwrap();
            std::mem::swap(frame0, frame1);
            std::mem::swap(
                &mut self.drop_lists[0].lock().unwrap(),
                &mut self.drop_lists[1].lock().unwrap(),
            );
            Ok(())
        } else {
            Err(BackendError::Other(
                "Unable to finish frame: frame data is being held by user code".into(),
            ))
        }
    }

    pub fn submit_render(&self, cb: &CommandBuffer, image: &SwapchainImage) -> BackendResult<()> {
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .wait_semaphores(slice::from_ref(&image.acquire_semaphore))
            .signal_semaphores(slice::from_ref(&image.rendering_finished_semaphore))
            .command_buffers(slice::from_ref(&cb.raw))
            .build();
        unsafe {
            self.raw.reset_fences(slice::from_ref(&cb.fence))?;
            self.raw.queue_submit(
                self.graphics_queue.raw,
                slice::from_ref(&submit_info),
                cb.fence,
            )?;
        }

        Ok(())
    }

    pub fn wait(&self) {
        unsafe { self.raw.device_wait_idle().unwrap() };
    }

    pub(crate) fn drop_resources<F: FnOnce(&mut DropList)>(&self, cb: F) {
        let mut list = self.drop_lists[0].lock().unwrap();
        cb(&mut list);
    }

    pub(crate) fn allocate<T, F: FnOnce(&mut Allocator, &AshMemoryDevice) -> T>(&self, cb: F) -> T {
        let mut allocator = self.allocator.lock().unwrap();
        let device = AshMemoryDevice::wrap(&self.raw);
        cb(&mut allocator, device)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut allocator = self.allocator.lock().unwrap();
        self.setup_cb.lock().unwrap().free(&self.raw);
        self.drop_lists.iter().for_each(|list| {
            let mut list = list.lock().unwrap();
            list.free(&self.raw, &mut allocator);
        });
        self.frames.iter().for_each(|frame| {
            let mut frame = frame.lock().unwrap();
            let frame = Arc::get_mut(&mut frame).unwrap();
            frame.free(&self.raw);
        });
        unsafe { allocator.cleanup(AshMemoryDevice::wrap(&self.raw)) };
        unsafe { self.raw.destroy_device(None) };
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Vulkan Device")
    }
}
