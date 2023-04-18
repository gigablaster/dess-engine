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

use std::sync::{Arc, Mutex};

use ash::{extensions::khr, vk};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};
use log::info;

use crate::{BackendError, BackendResult};

use super::{CommandBuffer, GpuResource, Instance, PhysicalDevice, QueueFamily, SwapchainImage};

pub struct Queue {
    pub raw: vk::Queue,
    pub family: QueueFamily,
}

pub struct DeviceFrame {
    pub device: ash::Device,
    pub command_buffer: CommandBuffer,
    pub queue_family: QueueFamily,
}

impl DeviceFrame {
    pub fn new(device: &ash::Device, queue_family: &QueueFamily) -> BackendResult<Self> {
        Ok(Self {
            device: device.clone(),
            command_buffer: CommandBuffer::new(device, queue_family)?,
            queue_family: *queue_family,
        })
    }
}

impl GpuResource for DeviceFrame {
    fn free(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.command_buffer.free(device, allocator);
    }
}

pub struct Device {
    pub instance: Arc<Instance>,
    pub raw: ash::Device,
    pub pdevice: PhysicalDevice,
    pub queue: Queue,
    pub allocator: Arc<Mutex<Allocator>>,
    setup_cb: Mutex<CommandBuffer>,
    frames: [Mutex<Arc<DeviceFrame>>; 2],
}

impl Device {
    pub fn create(instance: &Arc<Instance>, pdevice: &PhysicalDevice) -> BackendResult<Arc<Self>> {
        if !pdevice.is_queue_flag_supported(vk::QueueFlags::GRAPHICS) {}

        let device_extension_names = vec![khr::Swapchain::name().as_ptr()];

        let desired_queue = pdevice
            .queue_families
            .iter()
            .filter(|queue| queue.is_supported(vk::QueueFlags::GRAPHICS))
            .copied()
            .next();

        let desired_queue = if let Some(queue) = desired_queue {
            queue
        } else {
            return Err(BackendError::Other(
                "Can't create device for physica device that doesn't support graphics".into(),
            ));
        };

        let priorities = [1.0];

        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(desired_queue.index)
            .queue_priorities(&priorities)
            .build()];

        let mut physical_device_buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut physical_device_buffer_device_address)
            .build();

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

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw.clone(),
            device: device.clone(),
            physical_device: pdevice.raw,
            debug_settings: AllocatorDebugSettings {
                log_leaks_on_shutdown: true,
                log_memory_information: true,
                log_allocations: true,
                ..Default::default()
            },
            buffer_device_address: false,
        })
        .map_err(|err| err.to_string())?;

        let queue = unsafe { device.get_device_queue(desired_queue.index, 0) };

        let queue = Queue {
            raw: queue,
            family: desired_queue,
        };

        let setup_cb = CommandBuffer::new(&device, &desired_queue)?;

        let frames = [
            Mutex::new(Arc::new(DeviceFrame::new(&device, &desired_queue)?)),
            Mutex::new(Arc::new(DeviceFrame::new(&device, &desired_queue)?)),
        ];

        Ok(Arc::new(Self {
            instance: instance.clone(),
            raw: device,
            pdevice: pdevice.clone(),
            queue,
            allocator: Arc::new(Mutex::new(allocator)),
            setup_cb: Mutex::new(setup_cb),
            frames,
        }))
    }

    pub fn begin_frame(&self) -> BackendResult<Arc<DeviceFrame>> {
        let mut frame0 = self.frames[0].lock().unwrap();
        {
            if let Some(frame0) = Arc::get_mut(&mut frame0) {
                unsafe {
                    self.raw
                        .wait_for_fences(&[frame0.command_buffer.fence], true, u64::MAX)
                }?;
            } else {
                return Err(BackendError::Other(
                    "Unable to begin frame: frame data is being held by user code".into(),
                ));
            }
        }
        Ok(frame0.clone())
    }

    pub fn end_frame(&self, frame: Arc<DeviceFrame>) -> BackendResult<()> {
        drop(frame);

        let mut frame0 = self.frames[0].lock().unwrap();
        if let Some(frame0) = Arc::get_mut(&mut frame0) {
            let mut frame1 = self.frames[1].lock().unwrap();
            let frame1 = Arc::get_mut(&mut frame1).unwrap();
            std::mem::swap(frame0, frame1);
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
            .wait_semaphores(&[image.acquire_semaphore])
            .signal_semaphores(&[image.rendering_finished_semaphore])
            .command_buffers(&[cb.raw])
            .build();
        unsafe {
            self.raw.reset_fences(&[cb.fence])?;
            self.raw
                .queue_submit(self.queue.raw, &[submit_info], cb.fence)?;
        }

        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut allocator = self.allocator.lock().unwrap();
        self.setup_cb
            .lock()
            .unwrap()
            .free(&self.raw, &mut allocator);
        self.frames.iter().for_each(|frame| {
            let mut frame = frame.lock().unwrap();
            let frame = Arc::get_mut(&mut frame).unwrap();
            frame.free(&self.raw, &mut allocator);
        });
        unsafe { self.raw.destroy_device(None) };
    }
}
