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

use arrayvec::ArrayVec;
use ash::{extensions::khr, vk};
use crossbeam::channel::{bounded, Receiver, Sender};
use dess_common::{Handle, Pool};
use gpu_alloc::{Dedicated, Request};
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use gpu_descriptor_ash::AshDescriptorDevice;
use log::{info, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::{mem, slice, thread};

use crate::{BackendError, BackendResult};

use super::pipeline_cache::{load_or_create_pipeline_cache, save_pipeline_cache};
use super::{
    frame::Frame, Buffer, DescriptorAllocator, DropList, GpuAllocator, GpuMemory, Image, Instance,
    PhysicalDevice, ToDrop, UniformStorage,
};
use super::{DescriptorHandle, DescriptorStorage, Index, Program, Staging};

pub type ImageHandle = Handle<vk::Image, Image>;
pub type BufferHandle = Handle<vk::Buffer, Buffer>;
pub type ProgramHandle = Index<Program>;
pub type PipelineHandle = Index<(vk::Pipeline, vk::PipelineLayout)>;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BufferSlice {
    pub buffer: BufferHandle,
    pub offset: u32,
}

impl BufferSlice {
    pub fn new(buffer: BufferHandle, offset: u32) -> Self {
        Self { buffer, offset }
    }

    pub fn is_valid(&self) -> bool {
        self.buffer.is_valid()
    }
}

pub(crate) type ImageStorage = Pool<vk::Image, Image>;
pub(crate) type BufferStorage = Pool<vk::Buffer, Buffer>;
pub(crate) type ProgramStorage = Vec<Program>;
pub(crate) type PipelineStorage = Vec<(vk::Pipeline, vk::PipelineLayout)>;

const PIPELINES_IN_FLY: usize = 128;

pub struct CommandBuffer {
    pub raw: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl CommandBuffer {
    pub fn primary(device: &ash::Device, pool: vk::CommandPool) -> BackendResult<Self> {
        let cb = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
        }?[0];
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )?
        };
        Ok(Self { raw: cb, fence })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe {
            device
                .wait_for_fences(slice::from_ref(&self.fence), true, u64::MAX)
                .unwrap();
            device.destroy_fence(self.fence, None);
            // Command buffer itself is freed by pool.
        }
    }
}

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
    frames: [Mutex<Arc<Frame>>; 2],
    pub(crate) instance: Instance,
    pub(crate) raw: Arc<ash::Device>,
    pub(crate) pdevice: PhysicalDevice,
    pub(crate) memory_allocator: Mutex<GpuAllocator>,
    pub(crate) descriptor_allocator: Mutex<DescriptorAllocator>,
    pub(crate) image_storage: RwLock<ImageStorage>,
    pub(crate) buffer_storage: RwLock<BufferStorage>,
    pub(crate) uniform_storage: Mutex<UniformStorage>,
    pub(crate) program_storage: RwLock<ProgramStorage>,
    pub(crate) current_drop_list: Mutex<DropList>,
    pub(crate) descriptor_storage: RwLock<DescriptorStorage>,
    pub(crate) dirty_descriptors: Mutex<HashSet<DescriptorHandle>>,
    pub(crate) samplers: HashMap<SamplerDesc, vk::Sampler>,
    pub(crate) staging: Mutex<Staging>,
    pub(crate) universal_queue: Mutex<vk::Queue>,
    pub(crate) pipelines: RwLock<PipelineStorage>,
    pub(crate) pipeline_cache: vk::PipelineCache,
    pub(crate) pipeline_compiled_sender: Sender<(PipelineHandle, vk::Pipeline, vk::PipelineLayout)>,
    pub(crate) pipeline_compiled_receiver:
        Receiver<(PipelineHandle, vk::Pipeline, vk::PipelineLayout)>,
    pub(crate) pipelines_in_fly: AtomicUsize,
}

impl Device {
    pub fn new(instance: Instance, pdevice: PhysicalDevice) -> BackendResult<Self> {
        if !pdevice.is_queue_flag_supported(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER | vk::QueueFlags::COMPUTE,
        ) {
            return Err(BackendError::NoSuitableDevice);
        };

        let device_extension_names = vec![
            khr::Swapchain::name().as_ptr(),
            vk::KhrDynamicRenderingFn::name().as_ptr(),
            vk::KhrSynchronization2Fn::name().as_ptr(),
            vk::KhrCopyCommands2Fn::name().as_ptr(),
            vk::KhrBufferDeviceAddressFn::name().as_ptr(),
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
        let universal_queue_index = universal_queue.index;

        let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut copy_commands2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchronization2)
            .push_next(&mut copy_commands2)
            .push_next(&mut buffer_device_address)
            .build();

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

        let universal_queue = unsafe { device.get_device_queue(universal_queue_index, 0) };

        let allocator_config = gpu_alloc::Config {
            dedicated_threshold: 64 * 1024 * 1024,
            preferred_dedicated_threshold: 16 * 1024 * 1024,
            transient_dedicated_threshold: 32 * 1024 * 1024,
            final_free_list_chunk: 1024 * 1024,
            minimal_buddy_size: 256,
            starting_free_list_chunk: 256 * 1024,
            initial_buddy_dedicated_size: 128 * 1024 * 1024,
        };
        let allocator_props =
            unsafe { device_properties(&instance.raw, Instance::vulkan_version(), pdevice.raw) }?;
        let mut memory_allocator = GpuAllocator::new(allocator_config, allocator_props);
        let descriptor_allocator = DescriptorAllocator::new(0);

        let frames = [
            Mutex::new(Arc::new(Frame::new(
                &device,
                &mut memory_allocator,
                universal_queue_index,
            )?)),
            Mutex::new(Arc::new(Frame::new(
                &device,
                &mut memory_allocator,
                universal_queue_index,
            )?)),
        ];

        let uniform_storage = UniformStorage::new(&device, &mut memory_allocator)?;

        let (pipeline_compiled_sender, pipeline_compiled_receiver) = bounded(PIPELINES_IN_FLY);
        Ok(Self {
            staging: Mutex::new(Staging::new(
                &instance,
                &device,
                &pdevice,
                &mut memory_allocator,
                universal_queue_index,
            )?),
            instance,
            samplers: Self::generate_samplers(&device),
            pipeline_cache: load_or_create_pipeline_cache(&device, &pdevice)?,
            pdevice,
            memory_allocator: Mutex::new(memory_allocator),
            descriptor_allocator: Mutex::new(descriptor_allocator),
            uniform_storage: Mutex::new(uniform_storage),
            universal_queue: Mutex::new(universal_queue),
            frames,
            image_storage: RwLock::default(),
            buffer_storage: RwLock::default(),
            current_drop_list: Mutex::default(),
            descriptor_storage: RwLock::default(),
            dirty_descriptors: Mutex::default(),
            program_storage: RwLock::default(),
            pipelines: RwLock::default(),
            pipeline_compiled_receiver,
            pipeline_compiled_sender,
            pipelines_in_fly: AtomicUsize::new(0),
            raw: Arc::new(device),
        })
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
            vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
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

    pub fn begin_frame(&self) -> BackendResult<Arc<Frame>> {
        puffin::profile_function!();
        let mut frame0 = self.frames[0].lock();
        {
            let frame0 = Arc::get_mut(&mut frame0)
                .expect("Unable to begin frame: frame data is being held by user code");
            unsafe {
                self.raw.wait_for_fences(
                    &[frame0.present_cb.fence, frame0.main_cb.fence],
                    true,
                    u64::MAX,
                )
            }?;
            let mut memory_allocator = self.memory_allocator.lock();
            let mut descriptor_allocator = self.descriptor_allocator.lock();
            let mut uniforms = self.uniform_storage.lock();
            frame0.reset(
                &self.raw,
                &mut memory_allocator,
                &mut descriptor_allocator,
                &mut uniforms,
            )?;
            frame0
                .drop_list
                .replace(mem::take(&mut self.current_drop_list.lock()));

            // Sync compiled pipelines
            let mut pipelines = self.pipelines.write();
            self.pipeline_compiled_receiver
                .iter()
                .for_each(|(index, pipeline, layout)| {
                    pipelines[index.index()] = (pipeline, layout);
                    self.pipelines_in_fly.fetch_sub(1, Ordering::SeqCst);
                });

            // Update descriptor sets
            self.update_descriptor_sets()?;
        }
        Ok(frame0.clone())
    }

    pub fn end_frame(&self, frame: Arc<Frame>) {
        drop(frame);

        let mut frame0 = self.frames[0].lock();
        {
            let frame0 = Arc::get_mut(&mut frame0)
                .expect("Unable to finish frame: frame data is being held by user code");
            let mut frame1 = self.frames[1].lock();
            let frame1 = Arc::get_mut(&mut frame1).unwrap();
            std::mem::swap(frame0, frame1);
        }
    }

    pub fn submit(
        &self,
        cb: vk::CommandBuffer,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        triggers: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> BackendResult<()> {
        puffin::profile_function!();
        let command_buffers = vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cb)
            .build();
        let wait = wait
            .iter()
            .map(|x| {
                vk::SemaphoreSubmitInfo::builder()
                    .semaphore(x.0)
                    .stage_mask(x.1)
                    .build()
            })
            .collect::<ArrayVec<_, 8>>();
        let triggers = triggers
            .iter()
            .map(|x| {
                vk::SemaphoreSubmitInfo::builder()
                    .semaphore(x.0)
                    .stage_mask(x.1)
                    .build()
            })
            .collect::<ArrayVec<_, 8>>();
        let info = vk::SubmitInfo2::builder()
            .command_buffer_infos(slice::from_ref(&command_buffers))
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&triggers)
            .build();
        let queue = self.universal_queue.lock();
        unsafe {
            self.raw
                .queue_submit2(*queue, slice::from_ref(&info), vk::Fence::null())
        }?;

        Ok(())
    }

    pub fn scoped_label(&self, cb: vk::CommandBuffer, label: &str) -> ScopedCommandBufferLabel {
        self.cmd_begin_label(cb, label);
        ScopedCommandBufferLabel { device: self, cb }
    }

    pub(crate) fn cmd_begin_label(&self, cb: vk::CommandBuffer, label: &str) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            let label = CString::new(label).unwrap();
            let label = vk::DebugUtilsLabelEXT::builder().label_name(&label).build();
            unsafe { debug_utils.cmd_begin_debug_utils_label(cb, &label) }
        }
    }

    pub(crate) fn cmd_end_label(&self, cb: vk::CommandBuffer) {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(cb) }
        }
    }

    pub(crate) fn allocate_impl(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        requirements: vk::MemoryRequirements,
        location: gpu_alloc::UsageFlags,
        dedicated: bool,
    ) -> BackendResult<GpuMemory> {
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

    pub fn set_object_name<T: vk::Handle>(&self, object: T, name: &str) {
        Self::set_object_name_impl(&self.instance, &self.raw, object, name);
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
        unsafe { self.raw.device_wait_idle() }.unwrap();
        info!("Waiting for pipeine compilation to finish...");
        while Arc::get_mut(&mut self.raw).is_none() {
            thread::sleep(Duration::from_millis(20));
        }
        info!("Cleanup...");

        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        let mut uniform_storage = self.uniform_storage.lock();

        let mut drop_list = DropList::default();
        let mut images = self.image_storage.write().drain().collect::<Vec<_>>();
        let mut buffers = self.buffer_storage.write().drain().collect::<Vec<_>>();
        images.iter_mut().for_each(|x| x.1.to_drop(&mut drop_list));
        buffers.iter_mut().for_each(|x| x.1.to_drop(&mut drop_list));
        self.descriptor_storage
            .write()
            .drain()
            .filter_map(|(_, x)| x.descriptor)
            .for_each(|x| drop_list.free_descriptor_set(x));

        drop_list.purge(
            &self.raw,
            &mut memory_allocator,
            &mut descriptor_allocator,
            &mut uniform_storage,
        );

        self.program_storage
            .write()
            .drain(..)
            .for_each(|x| x.free(&self.raw));

        self.frames.iter().for_each(|x| {
            Arc::get_mut(&mut x.lock())
                .expect("Frame data shouldn't be kept by anybody else at this point")
                .free(
                    &self.raw,
                    &mut memory_allocator,
                    &mut descriptor_allocator,
                    &mut uniform_storage,
                )
        });

        uniform_storage.free(&self.raw, &mut memory_allocator);
        self.staging.lock().free(&self.raw, &mut memory_allocator);

        if let Err(err) = save_pipeline_cache(&self.raw, &self.pdevice, self.pipeline_cache) {
            warn!("Failed to save pipeline cache: {:?}", err);
        }

        unsafe {
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.raw));
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.raw));
            self.raw.destroy_pipeline_cache(self.pipeline_cache, None);
            self.raw.destroy_device(None);
        }
    }
}
