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
    sync::{Arc, Mutex}, collections::HashMap,
};

use arrayvec::ArrayVec;
use ash::{extensions::khr, vk};
use gpu_alloc::Config;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::info;

use crate::{Allocator, BackendError, BackendResult, DropList};

use super::{CommandBuffer, FrameContext, Instance, PhysicalDevice, QueueFamily};

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode
}

pub struct Queue {
    pub raw: vk::Queue,
    pub family: QueueFamily,
}

pub struct SubmitWaitDesc {
    pub semaphore: vk::Semaphore,
    pub stage: vk::PipelineStageFlags,
}

pub struct Device {
    pub(crate) instance: Arc<Instance>,
    pub raw: ash::Device,
    pub pdevice: PhysicalDevice,
    pub graphics_queue: Queue,
    pub transfer_queue: Queue,
    pub allocator: Arc<Mutex<Allocator>>,
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    setup_pool: vk::CommandPool,
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

        let device_extension_names = vec![khr::Swapchain::name().as_ptr()];

        for ext in &device_extension_names {
            let ext = unsafe { CStr::from_ptr(*ext).to_str() }.unwrap();
            if !pdevice.supported_extensions.contains(ext) {
                return Err(BackendError::NoExtension(ext.into()));
            }
        }

        let graphics_queue = pdevice.get_queue(vk::QueueFlags::GRAPHICS)?;
        let transfer_queue = pdevice.get_queue(vk::QueueFlags::TRANSFER)?;

        let mut features = vk::PhysicalDeviceFeatures2::builder().build();

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
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(transfer_queue.index)
            .build();

        let setup_pool = unsafe { device.create_command_pool(&pool_create_info, None)? };
        let setup_cb = CommandBuffer::new(&device, setup_pool)?;

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
            samplers: Self::generate_samplers(&device),
            raw: device,
            frames,
            drop_lists,
            setup_pool,
        }))
    }

    fn generate_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [vk::SamplerMipmapMode::NEAREST, vk::SamplerMipmapMode::LINEAR];
        let address_modes = [vk::SamplerAddressMode::REPEAT, vk::SamplerAddressMode::CLAMP_TO_EDGE];
        let mut result = HashMap::new();
        texel_filters.into_iter().for_each(|texel_filter| {
            mipmap_modes.into_iter().for_each(|mipmap_mode| {
                address_modes.into_iter().for_each(|address_mode| {
                    let anisotropy = texel_filter == vk::Filter::LINEAR;
                    let sampler_create_info = vk::SamplerCreateInfo::builder()
                        .mag_filter(texel_filter)
                        .min_filter(texel_filter)
                        .mipmap_mode(mipmap_mode)
                        .address_mode_u(address_mode)
                        .address_mode_v(address_mode)
                        .address_mode_w(address_mode)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .max_anisotropy(16.0)
                        .anisotropy_enable(anisotropy)
                        .build();
                    let sampler = unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };

                    result.insert(SamplerDesc { texel_filter, mipmap_mode, address_mode }, sampler);
                })
            })
        });

        result
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
                        slice::from_ref(&frame0.presentation_cb.fence),
                        true,
                        u64::MAX,
                    )
                }?;
                let mut allocator = self.allocator.lock().unwrap();
                self.drop_lists[0]
                    .lock()
                    .unwrap()
                    .free(&self.raw, &mut allocator);
                frame0.reset()?;
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

    pub fn submit_transfer(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWaitDesc],
        trigger: &[vk::Semaphore],
    ) -> BackendResult<()> {
        let masks = wait.iter().map(|x| x.stage).collect::<ArrayVec<_, 8>>();
        let semaphors = wait.iter().map(|x| x.semaphore).collect::<ArrayVec<_, 8>>();
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&masks)
            .wait_semaphores(&semaphors)
            .signal_semaphores(trigger)
            .command_buffers(slice::from_ref(&cb.raw))
            .build();
        unsafe {
            self.raw.reset_fences(slice::from_ref(&cb.fence))?;
            self.raw.queue_submit(
                self.transfer_queue.raw,
                slice::from_ref(&submit_info),
                cb.fence,
            )?;
        }

        Ok(())
    }

    pub fn submit_render(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWaitDesc],
        trigger: &[vk::Semaphore],
    ) -> BackendResult<()> {
        let masks = wait.iter().map(|x| x.stage).collect::<ArrayVec<_, 8>>();
        let semaphors = wait.iter().map(|x| x.semaphore).collect::<ArrayVec<_, 8>>();
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&masks)
            .wait_semaphores(&semaphors)
            .signal_semaphores(trigger)
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

    pub fn get_sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut allocator = self.allocator.lock().unwrap();
        self.setup_cb.lock().unwrap().free(&self.raw);
        unsafe { self.raw.destroy_command_pool(self.setup_pool, None) };
        self.drop_lists.iter().for_each(|list| {
            let mut list = list.lock().unwrap();
            list.free(&self.raw, &mut allocator);
        });
        self.frames.iter().for_each(|frame| {
            let mut frame = frame.lock().unwrap();
            let frame = Arc::get_mut(&mut frame).unwrap();
            frame.free(&self.raw);
        });
        self.samplers.iter().for_each(|(_, sampler)| unsafe {
            self.raw.destroy_sampler(*sampler, None);
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
