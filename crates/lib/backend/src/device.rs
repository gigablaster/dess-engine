// Copyright (C) 2023-2024 gigablaster

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

use arrayvec::ArrayVec;
use ash::{extensions::khr, vk};
use gpu_alloc::{Dedicated, Request};
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::info;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt::Debug;
use std::sync::Arc;
use std::{mem, slice};

use crate::{AsVulkan, AsVulkanCommandBuffer, Error, Result, SwapchainImage};

use super::frame::FrameContext;
use super::{frame::Frame, DropList, GpuAllocator, GpuMemory, Instance, PhysicalDevice};
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy_level: u32,
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

pub struct Device {
    raw: ash::Device,
    frames: [Mutex<Arc<Frame>>; 2],
    instance: Instance,
    pdevice: PhysicalDevice,
    memory_allocator: Mutex<GpuAllocator>,
    current_drop_list: Mutex<DropList>,
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    universal_queue: Arc<Mutex<vk::Queue>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
    universal_queue_index: u32,
    transfer_queue_index: u32,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", vk::Handle::as_raw(self.raw.handle()))
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn new(instance: Instance, pdevice: PhysicalDevice) -> Result<Arc<Self>> {
        if !pdevice.is_queue_flag_supported(vk::QueueFlags::GRAPHICS) {
            return Err(Error::NoSuitableDevice);
        };

        let device_extension_names = vec![
            khr::Swapchain::name().as_ptr(),
            khr::Maintenance4::name().as_ptr(),
            khr::BufferDeviceAddress::name().as_ptr(),
            vk::ExtDescriptorIndexingFn::name().as_ptr(),
            khr::Synchronization2::name().as_ptr(),
            khr::CopyCommands2::name().as_ptr(),
        ];

        for ext in &device_extension_names {
            let ext = unsafe { CStr::from_ptr(*ext).to_str() }.unwrap();
            if !pdevice.is_extensions_sipported(ext) {
                return Err(Error::ExtensionNotFound(ext.into()));
            }
        }

        let universal_queue_family = pdevice
            .find_queue(vk::QueueFlags::GRAPHICS, &[])
            .ok_or(Error::NoSuitableQueue)?;
        let transfer_queue_family =
            pdevice.find_queue(vk::QueueFlags::TRANSFER, &[universal_queue_family.index]);

        let universal_queue_index = universal_queue_family.index;
        let transfer_queue_index = transfer_queue_family
            .unwrap_or(universal_queue_family)
            .index;

        let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_uniform_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true);

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut buffer_device_address)
            .push_next(&mut maintenance4)
            .push_next(&mut descriptor_indexing)
            .push_next(&mut synchronization2)
            .build();

        let queue_priorities = [1.0];
        let mut queue_info = Vec::new();
        queue_info.push(
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(universal_queue_family.index)
                .queue_priorities(&queue_priorities)
                .build(),
        );
        if let Some(transfer_queue_family) = transfer_queue_family {
            queue_info.push(
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_family.index)
                    .queue_priorities(&queue_priorities)
                    .build(),
            )
        }

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features)
            .build();

        let device = unsafe {
            instance
                .get()
                .create_device(pdevice.get(), &device_create_info, None)?
        };

        let universal_queue = Arc::new(Mutex::new(unsafe {
            device.get_device_queue(universal_queue_index, 0)
        }));
        let transfer_queue = transfer_queue_family
            .map(|x| Arc::new(Mutex::new(unsafe { device.get_device_queue(x.index, 0) })))
            .unwrap_or(universal_queue.clone());

        let allocator_config = gpu_alloc::Config {
            dedicated_threshold: 64 * 1024 * 1024,
            preferred_dedicated_threshold: 16 * 1024 * 1024,
            transient_dedicated_threshold: 32 * 1024 * 1024,
            final_free_list_chunk: 1024 * 1024,
            minimal_buddy_size: 256,
            starting_free_list_chunk: 256 * 1024,
            initial_buddy_dedicated_size: 128 * 1024 * 1024,
        };
        let allocator_props = unsafe {
            device_properties(instance.get(), Instance::vulkan_version(), pdevice.get())
        }?;
        let memory_allocator = Mutex::new(GpuAllocator::new(allocator_config, allocator_props));

        let frames = [
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
            Mutex::new(Arc::new(Frame::new(&device, universal_queue_index)?)),
        ];

        Ok(Arc::new(Self {
            instance,
            samplers: Self::generate_samplers(&device),
            pdevice,
            memory_allocator,
            universal_queue,
            transfer_queue,
            frames,
            current_drop_list: Mutex::default(),
            raw: device,
            universal_queue_index,
            transfer_queue_index,
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
            vk::SamplerAddressMode::MIRRORED_REPEAT,
        ];
        let aniso_levels = [0, 1, 2, 3, 4];
        let mut result = HashMap::new();
        texel_filters.into_iter().for_each(|texel_filter| {
            mipmap_modes.into_iter().for_each(|mipmap_mode| {
                address_modes.into_iter().for_each(|address_mode| {
                    aniso_levels.into_iter().for_each(|aniso_level| {
                        let anisotropy = aniso_level > 0 && texel_filter == vk::Filter::LINEAR;
                        let anisotropy_level = if anisotropy { 1 << aniso_level } else { 0 };
                        let desc = SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_mode,
                            anisotropy_level,
                        };
                        result.entry(desc).or_insert_with(|| {
                            let sampler_create_info = vk::SamplerCreateInfo::builder()
                                .mag_filter(texel_filter)
                                .min_filter(texel_filter)
                                .mipmap_mode(mipmap_mode)
                                .address_mode_u(address_mode)
                                .address_mode_v(address_mode)
                                .address_mode_w(address_mode)
                                .max_lod(vk::LOD_CLAMP_NONE)
                                .max_anisotropy(anisotropy_level as _)
                                .anisotropy_enable(anisotropy)
                                .build();
                            unsafe { device.create_sampler(&sampler_create_info, None).unwrap() }
                        });
                    })
                })
            })
        });

        result
    }

    pub fn frame(&self) -> Result<FrameContext> {
        let mut frame = self.frames[0].lock();
        {
            let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
            puffin::profile_scope!("Wait for submit");
            unsafe {
                self.raw
                    .wait_for_fences(slice::from_ref(&frame.fence()), true, u64::MAX)?
            };
            frame.reset(&self.raw, &mut self.memory_allocator.lock())?;
        }
        Ok(FrameContext {
            device: self,
            frame: Some(frame.clone()),
        })
    }

    pub(crate) fn end_frame(&self, frame: Arc<Frame>) {
        drop(frame);

        let mut frame = self.frames[0].lock();
        let frame = Arc::get_mut(&mut frame).expect("Frame is used by client code");
        let mut next_frame = self.frames[1].lock();
        let next_frame = Arc::get_mut(&mut next_frame).unwrap();
        frame.assign_drop_list(mem::take(&mut self.current_drop_list.lock()));
        mem::swap(frame, next_frame);
    }

    pub fn present(&self, image: SwapchainImage) {
        puffin::profile_scope!("present");
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&image.rendering_finished))
            .swapchains(slice::from_ref(&image.swapchain.as_vk()))
            .image_indices(slice::from_ref(&image.image_index))
            .build();

        match unsafe {
            image
                .swapchain
                .loader()
                .queue_present(*self.universal_queue.lock(), &present_info)
        } {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {}
            Err(err) => panic!("Can't present image: {}", err),
        }
    }

    pub fn submit_graphics<C: AsVulkanCommandBuffer>(
        &self,
        cb: &C,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<()> {
        self.submit(
            *self.universal_queue.lock(),
            cb.command_buffer(),
            cb.fence(),
            wait,
            triggers,
        )
    }

    pub fn submit_transfer<C: AsVulkanCommandBuffer>(
        &self,
        cb: &C,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<()> {
        self.submit(
            *self.transfer_queue.lock(),
            cb.command_buffer(),
            cb.fence(),
            wait,
            triggers,
        )
    }

    fn submit(
        &self,
        queue: vk::Queue,
        cb: vk::CommandBuffer,
        fence: vk::Fence,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags)],
        triggers: &[vk::Semaphore],
    ) -> Result<()> {
        puffin::profile_function!();
        let wait_semaphores = wait.iter().map(|x| x.0).collect::<ArrayVec<_, 8>>();
        let wait_stages = wait.iter().map(|x| x.1).collect::<ArrayVec<_, 8>>();
        let info = vk::SubmitInfo::builder()
            .command_buffers(slice::from_ref(&cb))
            .signal_semaphores(triggers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .build();
        unsafe {
            self.raw.reset_fences(&[fence])?;
            self.raw
                .queue_submit(queue, slice::from_ref(&info), fence)?;
        };

        Ok(())
    }

    pub fn with_drop_list<CB: FnOnce(&mut DropList)>(&self, cb: CB) {
        cb(&mut self.current_drop_list.lock());
    }

    pub(crate) fn cmd_end_label(&self, cb: vk::CommandBuffer) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(cb) }
        }
    }

    pub(crate) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemory> {
        Self::allocate_impl(
            &self.raw,
            &mut self.memory_allocator.lock(),
            requirements,
            location,
            dedicated,
        )
    }

    pub(crate) fn allocate_impl(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> Result<GpuMemory> {
        let request = Request {
            size: requirements.size,
            align_mask: requirements.alignment,
            usage: location,
            memory_types: requirements.memory_type_bits,
        };

        Ok(if dedicated {
            unsafe {
                allocator.alloc_with_dedicated(
                    AshMemoryDevice::wrap(device),
                    request,
                    Dedicated::Required,
                )
            }
        } else {
            unsafe { allocator.alloc(AshMemoryDevice::wrap(device), request) }
        }?)
    }

    pub fn set_object_name<T: vk::Handle, S: AsRef<str>>(&self, object: T, name: S) {
        Self::set_object_name_impl(&self.instance, &self.raw, object, name.as_ref());
    }

    pub fn main_queue_index(&self) -> u32 {
        self.universal_queue_index
    }

    pub fn transfer_queue_index(&self) -> u32 {
        self.transfer_queue_index
    }

    pub fn sampler(&self, desc: &SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(desc).copied()
    }

    pub fn get(&self) -> &ash::Device {
        &self.raw
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.pdevice
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub(crate) fn set_object_name_impl<T: vk::Handle>(
        instance: &Instance,
        device: &ash::Device,
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
            unsafe { debug_utils.set_debug_utils_object_name(device.handle(), &name_info) }
                .unwrap();
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        info!("Cleanup...");
        unsafe { self.raw.device_wait_idle() }.unwrap();
        let mut memory_allocator = self.memory_allocator.lock();

        self.current_drop_list
            .lock()
            .purge(&self.raw, &mut memory_allocator);

        self.frames.iter().for_each(|frame| {
            let mut frame = frame.lock();
            let frame = Arc::get_mut(&mut frame).unwrap();
            frame.free(&self.raw, &mut memory_allocator)
        });

        for (_, sampler) in self.samplers.drain() {
            unsafe { self.raw.destroy_sampler(sampler, None) };
        }

        unsafe {
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.raw));
            self.raw.destroy_device(None);
        }
    }
}
