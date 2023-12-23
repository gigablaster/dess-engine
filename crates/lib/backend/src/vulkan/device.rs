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
use dess_common::{Handle, HotColdPool, SentinelPoolStrategy};
use gpu_alloc::{Dedicated, Request};
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use gpu_descriptor_ash::AshDescriptorDevice;
use log::{info, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{mem, slice};

use crate::vulkan::frame::MAX_TEMP_MEMORY;
use crate::vulkan::{save_pipeline_cache, BufferDesc, ExecutionContext, ImageViewDesc};
use crate::{BackendError, BackendResult, BufferUsage};

use super::{
    frame::Frame, Buffer, DescriptorAllocator, DropList, GpuAllocator, GpuMemory, Image, Instance,
    PhysicalDevice, ToDrop, UniformStorage,
};
use super::{
    load_or_create_pipeline_cache, DescriptorHandle, DescriptorSetCreateInfo, DescriptorSetInfo,
    DescriptorStorage, FrameContext, Index, Program, RasterPipelineCreateDesc, Staging, Swapchain,
};

pub type ImageHandle = Handle<vk::Image>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type ProgramHandle = Index<Program>;
pub type RasterPipelineHandle = Handle<(vk::Pipeline, vk::PipelineLayout)>;

pub enum FrameResult {
    Rendered,
    NeedRecreate,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BufferSlice {
    pub handle: BufferHandle,
    pub offset: u32,
    pub size: u32,
}

impl BufferSlice {
    pub fn new(buffer: BufferHandle, offset: u32, size: u32) -> Self {
        Self {
            handle: buffer,
            offset,
            size,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.handle.is_valid()
    }

    pub fn part(&self, offset: u32) -> Self {
        Self {
            handle: self.handle,
            offset: self.offset + offset,
            size: self
                .size
                .checked_sub(offset)
                .expect("Buffer part offset must be less than it's size"),
        }
    }
}

pub(crate) type ImageStorage = HotColdPool<vk::Image, Image, SentinelPoolStrategy<vk::Image>>;
pub(crate) type BufferStorage = HotColdPool<vk::Buffer, Buffer, SentinelPoolStrategy<vk::Buffer>>;
pub(crate) type ProgramStorage = Vec<Program>;
pub(crate) type RasterPipelineStorage =
    HotColdPool<(vk::Pipeline, vk::PipelineLayout), RasterPipelineCreateDesc>;

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
    pub(crate) raw: ash::Device,
    frames: [Mutex<Frame>; 2],
    pub(crate) instance: Instance,
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
    pub(crate) pipelines: RwLock<RasterPipelineStorage>,
    pub(crate) pipeline_cache: vk::PipelineCache,
    pub queue_familt_index: u32,
    current_cpu_frame: AtomicUsize,
    pub(crate) temp_buffer: vk::Buffer,
    temp_buffer_memory: Option<GpuMemory>,
    temp_buffer_handle: BufferHandle,
    pub(crate) descriptor_layouts: Mutex<HashMap<DescriptorSetCreateInfo, DescriptorSetInfo>>,
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({})", vk::Handle::as_raw(self.raw.handle()))
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn new(instance: Instance, pdevice: PhysicalDevice) -> BackendResult<Arc<Self>> {
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
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchronization2)
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

        let temp_buffer_type =
            BufferUsage::Index | BufferUsage::Vertex | BufferUsage::Uniform | BufferUsage::Storage;
        let temp_buffer_desc = vk::BufferCreateInfo::builder()
            .size((MAX_TEMP_MEMORY * 2) as u64)
            .usage(temp_buffer_type.into())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[universal_queue_index])
            .build();
        let temp_buffer = unsafe { device.create_buffer(&temp_buffer_desc, None) }?;
        Self::set_object_name_impl(&instance, &device, temp_buffer, "Temp buffer");
        let requirements = unsafe { device.get_buffer_memory_requirements(temp_buffer) };
        let mut temp_buffer_memory = Self::allocate_impl(
            &device,
            &mut memory_allocator,
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | gpu_alloc::UsageFlags::HOST_ACCESS,
            true,
        )?;
        unsafe {
            device.bind_buffer_memory(
                temp_buffer,
                *temp_buffer_memory.memory(),
                temp_buffer_memory.offset(),
            )
        }?;
        let temp_mapping = unsafe {
            temp_buffer_memory.map(
                AshMemoryDevice::wrap(&device),
                0,
                temp_buffer_desc.size as _,
            )
        }?;
        let frames = [
            Mutex::new(Frame::new(&device, 0, temp_mapping, universal_queue_index)?),
            Mutex::new(Frame::new(
                &device,
                MAX_TEMP_MEMORY,
                temp_mapping,
                universal_queue_index,
            )?),
        ];

        let uniform_storage = UniformStorage::new(&instance, &device, &mut memory_allocator)?;
        let mut buffer_storage = BufferStorage::default();
        let temp_buffer_handle = buffer_storage.push(
            temp_buffer,
            Buffer {
                raw: temp_buffer,
                desc: BufferDesc {
                    size: temp_buffer_desc.size as usize,
                    ty: temp_buffer_type,
                },
                memory: None,
            },
        );

        Ok(Arc::new(Self {
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
            buffer_storage: RwLock::new(buffer_storage),
            current_drop_list: Mutex::default(),
            descriptor_storage: RwLock::default(),
            dirty_descriptors: Mutex::default(),
            program_storage: RwLock::default(),
            pipelines: RwLock::default(),
            raw: device,
            queue_familt_index: universal_queue_index,
            current_cpu_frame: AtomicUsize::new(0),
            temp_buffer,
            temp_buffer_memory: Some(temp_buffer_memory),
            temp_buffer_handle,
            descriptor_layouts: Mutex::default(),
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

    pub fn frame<F: FnOnce(&FrameContext) -> BackendResult<()>>(
        &self,
        swapchain: &Swapchain,
        frame_fn: F,
    ) -> BackendResult<FrameResult> {
        let current_cpu_frame = self.current_cpu_frame.load(Ordering::Acquire);
        puffin::profile_function!();
        let mut frame = self.frames[current_cpu_frame].lock();
        unsafe {
            self.raw.wait_for_fences(
                &[frame.present_cb.fence, frame.main_cb.fence],
                true,
                u64::MAX,
            )
        }?;
        {
            let mut memory_allocator = self.memory_allocator.lock();
            let mut descriptor_allocator = self.descriptor_allocator.lock();
            let mut uniforms = self.uniform_storage.lock();
            frame.reset(
                &self.raw,
                &mut memory_allocator,
                &mut descriptor_allocator,
                &mut uniforms,
            )?;
        }
        frame
            .drop_list
            .replace(mem::take(&mut self.current_drop_list.lock()));

        // Upload staging and descriptor sets
        let staging_semaphore = self.staging.lock().upload(self)?;
        self.update_descriptor_sets()?;

        let backbuffer = match swapchain.acquire_next_image()? {
            crate::vulkan::AcquiredSurface::NeedRecreate => return Ok(FrameResult::NeedRecreate),
            crate::vulkan::AcquiredSurface::Image(image) => image,
        };
        let context = FrameContext {
            frame: &frame,
            render_area: swapchain.render_area(),
            target_view: backbuffer
                .image
                .get_or_create_view(&self.raw, ImageViewDesc::color())?,
            target_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            temp_buffer_handle: self.temp_buffer_handle,
            passes: Mutex::default(),
        };
        frame_fn(&context)?;

        {
            puffin::profile_scope!("Record frame");
            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();

            unsafe { self.raw.begin_command_buffer(frame.main_cb.raw, &info) }?;
            let _ = self.scoped_label(frame.main_cb.raw, "Render");

            let barrier = vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .src_queue_family_index(self.queue_familt_index)
                .dst_queue_family_index(self.queue_familt_index)
                .image(backbuffer.image.raw)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            let dependency = vk::DependencyInfo::builder()
                .image_memory_barriers(slice::from_ref(&barrier))
                .build();
            unsafe {
                self.raw
                    .cmd_pipeline_barrier2(frame.main_cb.raw, &dependency)
            };

            let execution_context = ExecutionContext {
                universal_queue: self.queue_familt_index,
                device: self,
                frame: &frame,
                images: &self.image_storage.read(),
                buffers: &self.buffer_storage.read(),
                pipelines: &self.pipelines.read(),
                descriptors: &self.descriptor_storage.read(),
            };
            execution_context.execute(context.passes.into_inner().iter())?;
        }
        unsafe { self.raw.end_command_buffer(frame.main_cb.raw) }.unwrap();

        self.submit(
            &frame.main_cb,
            &[
                staging_semaphore,
                (
                    backbuffer.acquire_semaphore,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                ),
            ],
            &[(
                frame.finished,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )],
        )?;
        {
            puffin::profile_scope!("Present");
            unsafe {
                self.raw.begin_command_buffer(
                    frame.present_cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )?;
                let _ = self.scoped_label(frame.present_cb.raw, "Present");
                let barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .src_queue_family_index(self.queue_familt_index)
                    .dst_queue_family_index(self.queue_familt_index)
                    .image(backbuffer.image.raw)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();
                let dependency = vk::DependencyInfo::builder()
                    .image_memory_barriers(slice::from_ref(&barrier))
                    .build();
                self.raw
                    .cmd_pipeline_barrier2(frame.present_cb.raw, &dependency);
                self.raw.end_command_buffer(frame.present_cb.raw)?;
            };
            self.submit(
                &frame.present_cb,
                &[(
                    frame.finished,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )],
                &[(backbuffer.rendering_finished, vk::PipelineStageFlags2::NONE)],
            )?;
            swapchain.present_image(backbuffer, *self.universal_queue.lock());
        }

        let next_cpu_frame = (current_cpu_frame + 1) % 2;
        assert_eq!(
            self.current_cpu_frame
                .compare_exchange(
                    current_cpu_frame,
                    next_cpu_frame,
                    Ordering::Release,
                    Ordering::Acquire
                )
                .unwrap(),
            current_cpu_frame
        );
        Ok(FrameResult::Rendered)
    }

    pub fn submit(
        &self,
        cb: &CommandBuffer,
        wait: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        triggers: &[(vk::Semaphore, vk::PipelineStageFlags2)],
    ) -> BackendResult<()> {
        puffin::profile_function!();
        let command_buffers = vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cb.raw)
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
            self.raw.reset_fences(&[cb.fence])?;
            self.raw
                .queue_submit2(*queue, slice::from_ref(&info), cb.fence)
        }?;

        Ok(())
    }

    pub(crate) fn destroy_resource<T: Debug + Default + Copy + Eq, U: ToDrop + Debug>(
        &self,
        handle: Handle<T>,
        storage: &RwLock<HotColdPool<T, U, SentinelPoolStrategy<T>>>,
    ) {
        let item: Option<(T, U)> = storage.write().remove(handle);
        if let Some((_, mut item)) = item {
            item.to_drop(&mut self.current_drop_list.lock());
        }
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
        info!("Cleanup...");
        unsafe { self.raw.device_wait_idle() }.unwrap();
        let mut memory_allocator = self.memory_allocator.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        let mut uniform_storage = self.uniform_storage.lock();

        self.current_drop_list.lock().purge(
            &self.raw,
            &mut memory_allocator,
            &mut descriptor_allocator,
            &mut uniform_storage,
        );
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

        self.pipelines
            .write()
            .drain()
            .filter_map(|((pipeline, _), _)| (pipeline != vk::Pipeline::null()).then_some(pipeline))
            .for_each(|pipeline| unsafe { self.raw.destroy_pipeline(pipeline, None) });

        self.program_storage
            .write()
            .drain(..)
            .for_each(|x| x.free(&self.raw));

        self.frames.iter().for_each(|x| {
            x.lock().free(
                &self.raw,
                &mut memory_allocator,
                &mut descriptor_allocator,
                &mut uniform_storage,
            )
        });

        uniform_storage.free(&self.raw, &mut memory_allocator);
        self.staging.lock().free(&self.raw, &mut memory_allocator);

        if let Some(memory) = self.temp_buffer_memory.take() {
            unsafe {
                memory_allocator.dealloc(AshMemoryDevice::wrap(&self.raw), memory);
                self.raw.destroy_buffer(self.temp_buffer, None);
            }
        }
        if let Err(err) = save_pipeline_cache(&self.raw, &self.pdevice, self.pipeline_cache) {
            warn!("Failed to save pipeline cache: {:?}", err);
        }

        for (_, sampler) in self.samplers.drain() {
            unsafe { self.raw.destroy_sampler(sampler, None) };
        }

        for (_, layout) in self.descriptor_layouts.lock().drain() {
            layout.free(&self.raw);
        }

        unsafe {
            descriptor_allocator.cleanup(AshDescriptorDevice::wrap(&self.raw));
            memory_allocator.cleanup(AshMemoryDevice::wrap(&self.raw));
            self.raw.destroy_pipeline_cache(self.pipeline_cache, None);
            self.raw.destroy_device(None);
        }
    }
}
