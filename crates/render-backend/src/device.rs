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
    collections::HashMap,
    ffi::{CStr, CString},
    fmt::Debug,
    slice,
    sync::{Arc, Mutex, MutexGuard},
};

use arrayvec::ArrayVec;
use ash::{
    extensions::khr,
    vk::{self, Handle},
};
use gpu_alloc::Config;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::info;

use crate::{BackendError, GpuAllocator, Semaphore};

use super::{
    droplist::DropList, BackendResult, CommandBuffer, FrameContext, FreeGpuResource, Instance,
    PhysicalDevice, QueueFamily,
};

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
}

pub struct Queue {
    pub raw: vk::Queue,
    pub family: QueueFamily,
}

#[derive(Debug)]
pub enum SubmitWait<'a> {
    Transfer(&'a Semaphore),
    ColorAttachmentOutput(&'a Semaphore),
}

impl<'a> SubmitWait<'a> {
    pub fn get_stage_flags(&self) -> vk::PipelineStageFlags {
        match self {
            SubmitWait::ColorAttachmentOutput(_) => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            SubmitWait::Transfer(_) => vk::PipelineStageFlags::TRANSFER,
        }
    }

    pub fn get_semahore(&self) -> vk::Semaphore {
        match self {
            SubmitWait::ColorAttachmentOutput(semaphore) => semaphore.raw,
            SubmitWait::Transfer(semaphore) => semaphore.raw,
        }
    }
}

pub struct Device {
    instance: Instance,
    pub raw: ash::Device,
    pub pdevice: PhysicalDevice,
    pub universal_queue: Queue,
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    frames: [Mutex<Arc<FrameContext>>; 2],
    current_drop_list: Mutex<DropList>,
    drop_lists: [Mutex<DropList>; 2],
    allocator: Mutex<GpuAllocator>,
}

impl Device {
    pub fn create(instance: Instance, pdevice: PhysicalDevice) -> BackendResult<Arc<Self>> {
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

        let universal_queue =
            pdevice.get_queue(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)?;

        let mut features = vk::PhysicalDeviceFeatures2::builder().build();

        unsafe {
            instance
                .raw
                .get_physical_device_features2(pdevice.raw, &mut features)
        };

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(universal_queue.index)
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

        let frames = [
            Mutex::new(Arc::new(FrameContext::new(
                &instance,
                &device,
                &universal_queue,
                "frame 1",
            )?)),
            Mutex::new(Arc::new(FrameContext::new(
                &instance,
                &device,
                &universal_queue,
                "frame 2",
            )?)),
        ];

        let drop_lists = [
            Mutex::new(DropList::default()),
            Mutex::new(DropList::default()),
        ];

        let allocator_config = Config {
            dedicated_threshold: 64 * 1024 * 1024,
            preferred_dedicated_threshold: 32 * 1024 * 1024,
            transient_dedicated_threshold: 32 * 1024 * 1024,
            final_free_list_chunk: 1024 * 1024,
            minimal_buddy_size: 128,
            starting_free_list_chunk: 16 * 1024,
            initial_buddy_dedicated_size: 32 * 1024 * 1024,
        };
        let allocator_props =
            unsafe { device_properties(&instance.raw, Instance::vulkan_version(), pdevice.raw) }?;
        let allocator = GpuAllocator::new(allocator_config, allocator_props);

        let universal_queue = Self::create_queue(&device, universal_queue);

        Self::set_object_name_impl(&instance, &device, universal_queue.raw, "Main queue")?;

        Ok(Arc::new(Self {
            instance,
            pdevice,
            universal_queue,
            samplers: Self::generate_samplers(&device),
            raw: device,
            frames,
            current_drop_list: Mutex::new(DropList::default()),
            drop_lists,
            allocator: Mutex::new(allocator),
        }))
    }

    fn generate_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ];
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
                    let sampler =
                        unsafe { device.create_sampler(&sampler_create_info, None).unwrap() };

                    result.insert(
                        SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_mode,
                        },
                        sampler,
                    );
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
        puffin::profile_scope!("begin frame");
        let mut frame0 = self.frames[0].lock().unwrap();
        {
            if let Some(frame0) = Arc::get_mut(&mut frame0) {
                unsafe {
                    self.raw.wait_for_fences(
                        &[frame0.presentation_cb.fence, frame0.main_cb.fence],
                        true,
                        u64::MAX,
                    )
                }?;
                self.drop_lists[0]
                    .lock()
                    .unwrap()
                    .free(&self.raw, &mut self.allocator());
                frame0.reset(&self.raw)?;
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
            *self.drop_lists[0].lock().unwrap() =
                std::mem::take::<DropList>(&mut self.current_drop_list.lock().unwrap());
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

    pub fn submit(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> BackendResult<()> {
        let masks = wait
            .iter()
            .map(|x| x.get_stage_flags())
            .collect::<ArrayVec<_, 8>>();
        let wait = wait
            .iter()
            .map(|x| x.get_semahore())
            .collect::<ArrayVec<_, 8>>();
        let trigger = trigger.iter().map(|x| x.raw).collect::<ArrayVec<_, 8>>();
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&masks)
            .wait_semaphores(&wait)
            .signal_semaphores(&trigger)
            .command_buffers(slice::from_ref(&cb.raw))
            .build();
        unsafe {
            self.raw
                .wait_for_fences(slice::from_ref(&cb.fence), true, u64::MAX)?;
            self.raw.reset_fences(slice::from_ref(&cb.fence))?;
            self.raw.queue_submit(
                self.universal_queue.raw,
                slice::from_ref(&submit_info),
                cb.fence,
            )?;
        }

        Ok(())
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance.raw
    }

    pub fn wait(&self) {
        unsafe { self.raw.device_wait_idle().unwrap() };
    }

    pub(crate) fn with_drop_list<F: FnOnce(&mut DropList)>(&self, cb: F) {
        let mut list = self.current_drop_list.lock().unwrap();
        cb(&mut list);
    }

    pub fn get_sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
    }

    pub fn set_object_name<T: Handle>(&self, object: T, name: &str) -> BackendResult<()> {
        Self::set_object_name_impl(&self.instance, &self.raw, object, name)
    }

    pub(crate) fn set_object_name_impl<T: Handle>(
        instance: &Instance,
        devcie: &ash::Device,
        object: T,
        name: &str,
    ) -> BackendResult<()> {
        if let Some(debug_utils) = instance.get_debug_utils() {
            let name = CString::new(name).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(object.as_raw())
                .object_name(&name)
                .build();
            unsafe { debug_utils.set_debug_utils_object_name(devcie.handle(), &name_info) }?;
        }

        Ok(())
    }

    pub fn scoped_label(&self, cb: vk::CommandBuffer, label: &str) -> ScopedCommandBufferLabel {
        self.cmd_begin_label(cb, label);
        ScopedCommandBufferLabel { device: self, cb }
    }

    pub fn cmd_begin_label(&self, cb: vk::CommandBuffer, label: &str) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            let label = CString::new(label).unwrap();
            let label = vk::DebugUtilsLabelEXT::builder().label_name(&label).build();
            unsafe { debug_utils.cmd_begin_debug_utils_label(cb, &label) }
        }
    }

    pub fn cmd_end_label(&self, cb: vk::CommandBuffer) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(cb) }
        }
    }

    pub fn marker(&self, cb: vk::CommandBuffer, label: &str) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            let label = CString::new(label).unwrap();
            let label = vk::DebugUtilsLabelEXT::builder().label_name(&label).build();

            unsafe { debug_utils.cmd_insert_debug_utils_label(cb, &label) }
        }
    }

    pub fn allocator(&self) -> MutexGuard<GpuAllocator> {
        self.allocator.lock().unwrap()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut allocator = self.allocator();
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

pub struct ScopedCommandBufferLabel<'a> {
    device: &'a Device,
    cb: vk::CommandBuffer,
}

impl<'a> Drop for ScopedCommandBufferLabel<'a> {
    fn drop(&mut self) {
        self.device.cmd_end_label(self.cb);
    }
}
