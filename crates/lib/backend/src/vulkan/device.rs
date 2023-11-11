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
use gpu_descriptor_ash::AshDescriptorDevice;
use log::info;

use crate::{BackendError, GpuResource};

use super::{
    droplist::DropList, CommandBuffer, DescriptorAllocator, FrameContext, GpuAllocator, Instance,
    PhysicalDevice, QueueFamily, Semaphore,
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
pub enum SubmitWait {
    Transfer(Semaphore),
    ColorAttachmentOutput(Semaphore),
}

impl SubmitWait {
    pub fn stage_flags(&self) -> vk::PipelineStageFlags {
        match self {
            SubmitWait::ColorAttachmentOutput(_) => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            SubmitWait::Transfer(_) => vk::PipelineStageFlags::TRANSFER,
        }
    }

    pub fn semahore(&self) -> vk::Semaphore {
        match self {
            SubmitWait::ColorAttachmentOutput(semaphore) => semaphore.raw,
            SubmitWait::Transfer(semaphore) => semaphore.raw,
        }
    }
}

/// All devices I target has one queue family that supports all operations. But among them we have 2 kinds of devices
/// 1 - devices with one queue for main family that does everything (like integrated Intel GPUs).
/// 2 - devices with multiple queues for main family
/// So we check how many queues we have and then use one of two scenaros - all submits go into single queue or they go
/// into separate queues.
enum SubmitQueue {
    Single(Mutex<Queue>),
    Multiple(Mutex<Queue>, Mutex<Queue>, Mutex<Queue>),
}

impl SubmitQueue {
    pub fn new(device: &ash::Device, queue_family: QueueFamily) -> Self {
        if queue_family.properties.queue_count < 3 {
            Self::Single(Mutex::new(Self::create_queue(device, queue_family, 0)))
        } else {
            Self::Multiple(
                Mutex::new(Self::create_queue(device, queue_family, 0)),
                Mutex::new(Self::create_queue(device, queue_family, 1)),
                Mutex::new(Self::create_queue(device, queue_family, 2)),
            )
        }
    }

    pub fn submit_draw(
        &self,
        device: &ash::Device,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        match self {
            Self::Single(queue) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
            Self::Multiple(queue, ..) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
        }
    }

    pub fn submit_transfer(
        &self,
        device: &ash::Device,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        match self {
            Self::Single(queue) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
            Self::Multiple(_, queue, ..) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
        }
    }

    pub fn submit_compute(
        &self,
        device: &ash::Device,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        match self {
            Self::Single(queue) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
            Self::Multiple(_, _, queue) => {
                Self::submit_impl(device, &queue.lock().unwrap(), cb, wait, trigger)
            }
        }
    }

    pub fn get_present_queue(&self) -> MutexGuard<Queue> {
        match self {
            Self::Single(queue) => queue.lock().unwrap(),
            Self::Multiple(queue, ..) => queue.lock().unwrap(),
        }
    }

    /// Does actual submit job
    fn submit_impl(
        device: &ash::Device,
        queue: &Queue,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        let masks = wait
            .iter()
            .map(|x| x.stage_flags())
            .collect::<ArrayVec<_, 8>>();
        let wait = wait
            .iter()
            .map(|x| x.semahore())
            .collect::<ArrayVec<_, 8>>();
        let trigger = trigger.iter().map(|x| x.raw).collect::<ArrayVec<_, 8>>();
        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&masks)
            .wait_semaphores(&wait)
            .signal_semaphores(&trigger)
            .command_buffers(slice::from_ref(&cb.raw()))
            .build();
        unsafe {
            device.wait_for_fences(slice::from_ref(&cb.fence()), true, u64::MAX)?;
            device.reset_fences(slice::from_ref(&cb.fence()))?;
            device.queue_submit(queue.raw, slice::from_ref(&submit_info), cb.fence())?;
        }

        Ok(())
    }

    fn create_queue(device: &ash::Device, queue_family: QueueFamily, index: u32) -> Queue {
        let queue = unsafe { device.get_device_queue(queue_family.index, index) };

        Queue {
            raw: queue,
            family: queue_family,
        }
    }
}

pub struct Device {
    raw: ash::Device,
    instance: Arc<Instance>,
    pdevice: PhysicalDevice,
    queue: SubmitQueue,
    queue_family_index: u32,
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    frames: [Mutex<Arc<FrameContext>>; 2],
    current_drop_list: Mutex<DropList>,
    drop_lists: [Mutex<DropList>; 2],
    allocator: Mutex<GpuAllocator>,
    descriptor_allocator: Mutex<DescriptorAllocator>,
}

unsafe impl Sync for Device {}

impl Device {
    pub fn create(
        instance: &Arc<Instance>,
        pdevice: PhysicalDevice,
    ) -> Result<Arc<Self>, BackendError> {
        if !pdevice.is_queue_flag_supported(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER | vk::QueueFlags::COMPUTE,
        ) {
            return Err(BackendError::NoSuitableDevice);
        };

        let device_extension_names = vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrImagelessFramebufferFn::name().as_ptr(),
        ];

        for ext in &device_extension_names {
            let ext = unsafe { CStr::from_ptr(*ext).to_str() }.unwrap();
            if !pdevice.is_extensions_sipported(ext) {
                return Err(BackendError::ExtensionNotFound(ext.into()));
            }
        }

        let universal_queue = pdevice
            .get_queue(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER | vk::QueueFlags::COMPUTE,
            )
            .ok_or(BackendError::NoSuitableQueue)?;

        let mut imageless_frame_buffer = vk::PhysicalDeviceImagelessFramebufferFeatures::default();

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut imageless_frame_buffer)
            .build();

        unsafe {
            instance
                .raw
                .get_physical_device_features2(pdevice.raw(), &mut features)
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
                .create_device(pdevice.raw(), &device_create_info, None)?
        };

        info!("Created a Vulkan device");

        let frames = [
            Mutex::new(Arc::new(FrameContext::new(&device, universal_queue.index)?)),
            Mutex::new(Arc::new(FrameContext::new(&device, universal_queue.index)?)),
        ];

        frames.iter().enumerate().for_each(|(index, frame)| {
            let frame = frame.lock().unwrap();
            Self::set_object_name_impl(
                instance,
                &device,
                frame.main_cb.raw(),
                &format!("MainCB #{}", index),
            );
            Self::set_object_name_impl(
                instance,
                &device,
                frame.presentation_cb.raw(),
                &format!("PresentCB #{}", index),
            );
            Self::set_object_name_impl(
                instance,
                &device,
                frame.render_finished.raw,
                &format!("RenderFinished Semaphore #{}", index),
            );
        });

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
            unsafe { device_properties(&instance.raw, Instance::vulkan_version(), pdevice.raw()) }?;
        let allocator = GpuAllocator::new(allocator_config, allocator_props);
        let descriptor_allocator = DescriptorAllocator::new(128);

        let queue = SubmitQueue::new(&device, universal_queue);

        //Self::set_object_name_impl(instance, &device, queue.raw, "Main queue");

        Ok(Arc::new(Self {
            instance: instance.clone(),
            pdevice,
            queue_family_index: universal_queue.index,
            queue,
            samplers: Self::generate_samplers(&device),
            raw: device,
            frames,
            current_drop_list: Mutex::new(DropList::default()),
            drop_lists,
            allocator: Mutex::new(allocator),
            descriptor_allocator: Mutex::new(descriptor_allocator),
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

    pub fn begin_frame(&self) -> Result<Arc<FrameContext>, BackendError> {
        puffin::profile_scope!("begin frame");
        let mut frame0 = self.frames[0].lock().unwrap();
        {
            if let Some(frame0) = Arc::get_mut(&mut frame0) {
                unsafe {
                    self.raw.wait_for_fences(
                        &[frame0.presentation_cb.fence(), frame0.main_cb.fence()],
                        true,
                        u64::MAX,
                    )
                }?;
                self.drop_lists[0].lock().unwrap().free(
                    &self.raw,
                    &mut self.allocator(),
                    &mut self.descriptor_allocator(),
                );
                frame0.reset(&self.raw)?;
            } else {
                panic!("Unable to begin frame: frame data is being held by user code")
            }
        }
        Ok(frame0.clone())
    }

    pub fn end_frame(&self, frame: Arc<FrameContext>) {
        drop(frame);

        let mut frame0 = self.frames[0].lock().unwrap();
        if let Some(frame0) = Arc::get_mut(&mut frame0) {
            let mut frame1 = self.frames[1].lock().unwrap();
            let frame1 = Arc::get_mut(&mut frame1).unwrap();
            std::mem::swap(frame0, frame1);
            *self.drop_lists[0].lock().unwrap() =
                std::mem::take::<DropList>(&mut self.current_drop_list.lock().unwrap());
            std::mem::swap(
                &mut self.drop_lists[0].lock(),
                &mut self.drop_lists[1].lock(),
            );
        } else {
            panic!("Unable to finish frame: frame data is being held by user code",)
        }
    }

    pub fn submit_draw(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        self.queue.submit_draw(&self.raw, cb, wait, trigger)
    }

    pub fn submit_transfer(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        self.queue.submit_transfer(&self.raw, cb, wait, trigger)
    }

    pub fn submit_compute(
        &self,
        cb: &CommandBuffer,
        wait: &[SubmitWait],
        trigger: &[Semaphore],
    ) -> Result<(), BackendError> {
        self.queue.submit_compute(&self.raw, cb, wait, trigger)
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance.raw
    }

    pub fn wait(&self) {
        unsafe { self.raw.device_wait_idle().unwrap() };
    }

    pub fn get_sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
    }

    pub fn set_object_name<T: Handle>(&self, object: T, name: &str) {
        Self::set_object_name_impl(&self.instance, &self.raw, object, name);
    }

    pub(crate) fn set_object_name_impl<T: Handle>(
        instance: &Instance,
        devcie: &ash::Device,
        object: T,
        name: &str,
    ) {
        if let Some(debug_utils) = instance.get_debug_utils() {
            let name = CString::new(name).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(object.as_raw())
                .object_name(&name)
                .build();
            unsafe { debug_utils.set_debug_utils_object_name(devcie.handle(), &name_info) }
                .unwrap();
        }
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

    pub fn descriptor_allocator(&self) -> MutexGuard<DescriptorAllocator> {
        self.descriptor_allocator.lock().unwrap()
    }

    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.pdevice
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn drop_list(&self) -> MutexGuard<DropList> {
        self.current_drop_list.lock().unwrap()
    }

    pub(crate) fn get_present_queue(&self) -> MutexGuard<Queue> {
        self.queue.get_present_queue()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut allocator = self.allocator();
        let mut descriptor_allocator = self.descriptor_allocator();
        self.current_drop_list.lock().unwrap().free(
            &self.raw,
            &mut allocator,
            &mut descriptor_allocator,
        );
        self.drop_lists.iter().for_each(|list| {
            let mut list = list.lock().unwrap();
            list.free(&self.raw, &mut allocator, &mut descriptor_allocator);
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
        unsafe { descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.raw)) };
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