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

use core::panic;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    fmt::Debug,
    mem::size_of,
    slice::{self, from_raw_parts},
    sync::{Arc, Mutex},
};

use arrayvec::ArrayVec;
use ash::{
    extensions::khr,
    vk::{self, Handle},
};
use log::info;

use crate::vulkan::{staging::Staging, BackendError, BufferAllocator, ImageAllocator};

use super::{
    droplist::DropList, staging::StagingError, BackendResult, Buffer, CommandBuffer, FrameContext,
    FreeGpuResource, ImageMemory, Instance, PhysicalDevice, QueueFamily, RingBuffer,
};

const STAGING_SIZE: u64 = 32 * 1024 * 1024;
const BUFFER_CACHE_SIZE: u64 = 64 * 1024 * 1024;
const IMAGE_CHUNK_SIZE: u64 = 256 * 1024 * 1024;
const IMAGE_CHUNK_THRESHOLD: u64 = 2 * 1024 * 1024;
const UNIFORM_BUFFER_SIZE: u64 = 8 * 1024 * 1024;
const DYN_GEO_BUFFER_SIZE: u64 = 32 * 1024 * 1024;

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

pub struct SubmitWaitDesc {
    pub semaphore: vk::Semaphore,
    pub stage: vk::PipelineStageFlags,
}

pub struct Device {
    instance: Instance,
    pub raw: ash::Device,
    pub pdevice: PhysicalDevice,
    pub graphics_queue: Queue,
    pub transfer_queue: Queue,
    samplers: HashMap<SamplerDesc, vk::Sampler>,
    setup_pool: vk::CommandPool,
    setup_cb: CommandBuffer,
    frames: [Mutex<Arc<FrameContext>>; 2],
    drop_lists: [Mutex<DropList>; 2],
    geo_cache: Mutex<BufferAllocator>,
    image_cache: Mutex<ImageAllocator>,
    uniform_buffer: RingBuffer,
    dynamic_buffer: RingBuffer,
    staging: Mutex<Staging>,
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

        let image_cache = ImageAllocator::new(IMAGE_CHUNK_SIZE, IMAGE_CHUNK_THRESHOLD);
        let geo_cache = BufferAllocator::new(
            &device,
            &pdevice,
            BUFFER_CACHE_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let staging = Staging::new(&device, &pdevice, STAGING_SIZE)?;
        let uniform_buffer = RingBuffer::new(
            &device,
            &pdevice,
            UNIFORM_BUFFER_SIZE,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let dynamic_buffer = RingBuffer::new(
            &device,
            &pdevice,
            DYN_GEO_BUFFER_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        Ok(Arc::new(Self {
            instance,
            pdevice,
            graphics_queue: Self::create_queue(&device, graphics_queue),
            transfer_queue: Self::create_queue(&device, transfer_queue),
            setup_cb,
            samplers: Self::generate_samplers(&device),
            raw: device,
            frames,
            drop_lists,
            setup_pool,
            image_cache: Mutex::new(image_cache),
            geo_cache: Mutex::new(geo_cache),
            staging: Mutex::new(staging),
            uniform_buffer,
            dynamic_buffer,
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

    fn commit_staging(&self) -> BackendResult<()> {
        puffin::profile_scope!("commit staging");
        let mut staging = self.staging.lock().unwrap();
        if staging.is_empty() {
            return Ok(());
        }
        self.setup_cb.wait(&self.raw)?;
        self.setup_cb.reset(&self.raw)?;
        {
            let _recorder = self.setup_cb.record(self)?;
            staging.upload(
                &self.raw,
                self.setup_cb.raw,
                self.transfer_queue.family.index,
                self.graphics_queue.family.index,
            );
        }
        self.submit_transfer(&self.setup_cb, &[], &[])?;
        Ok(())
    }

    pub fn begin_frame(&self) -> BackendResult<Arc<FrameContext>> {
        puffin::profile_scope!("begin frame");
        let mut frame0 = self.frames[0].lock().unwrap();
        {
            if let Some(frame0) = Arc::get_mut(&mut frame0) {
                self.commit_staging()?;
                unsafe {
                    self.raw.wait_for_fences(
                        &[frame0.presentation_cb.fence, self.setup_cb.fence],
                        true,
                        u64::MAX,
                    )
                }?;
                let mut image_cache = self.image_cache.lock().unwrap();
                let mut geo_cache = self.geo_cache.lock().unwrap();
                self.drop_lists[0].lock().unwrap().free(
                    &self.raw,
                    &mut image_cache,
                    &mut geo_cache,
                );
                self.uniform_buffer.commit(&self.raw)?;
                self.dynamic_buffer.commit(&self.raw)?;
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

    pub fn instance(&self) -> &ash::Instance {
        &self.instance.raw
    }

    pub fn wait(&self) {
        unsafe { self.raw.device_wait_idle().unwrap() };
    }

    pub(crate) fn with_drop_list<F: FnOnce(&mut DropList)>(&self, cb: F) {
        let mut list = self.drop_lists[0].lock().unwrap();
        cb(&mut list);
    }

    pub fn get_sampler(&self, desc: SamplerDesc) -> Option<vk::Sampler> {
        self.samplers.get(&desc).copied()
    }

    pub(crate) fn set_object_name<T: Handle>(&self, object: T, name: &str) -> BackendResult<()> {
        if let Some(debug_utils) = self.instance.get_debug_utils() {
            let name = CString::new(name).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(object.as_raw())
                .object_name(&name)
                .build();
            unsafe { debug_utils.set_debug_utils_object_name(self.raw.handle(), &name_info) }?;
        }
        Ok(())
    }

    pub fn create_buffer(&self, size: usize) -> BackendResult<Buffer> {
        let mut geo_cache = self.geo_cache.lock().unwrap();
        geo_cache.allocate(size as _)
    }

    pub fn drop_buffer(&self, buffer: Buffer) {
        self.with_drop_list(|drop_list| {
            drop_list.drop_buffer(buffer);
        });
    }

    pub fn create_buffer_from<T: Sized>(&self, data: &[T]) -> BackendResult<Buffer> {
        let size = data.len() * size_of::<T>();
        let buffer = self.create_buffer(size)?;
        let ptr = data.as_ptr() as *const u8;
        let data = unsafe { from_raw_parts(ptr, size) };
        self.upload_buffer(&buffer, data)?;

        Ok(buffer)
    }

    pub fn upload_buffer(&self, buffer: &Buffer, data: &[u8]) -> BackendResult<()> {
        let mut staging = self.staging.lock().unwrap();
        match staging.upload_buffer(&self.raw, buffer, data) {
            Ok(_) => Ok(()),
            Err(StagingError::NeedUpload) => {
                self.commit_staging()?;
                unsafe {
                    self.raw
                        .wait_for_fences(slice::from_ref(&self.setup_cb.fence), true, u64::MAX)
                }?;
                match staging.upload_buffer(&self.raw, buffer, data) {
                    Ok(()) => Ok(()),
                    Err(StagingError::NeedUpload) => {
                        panic!("Shit happened when trying to stage data right after uploading")
                    }
                    Err(StagingError::VulkanError(vk)) => Err(BackendError::Vulkan(vk)),
                }
            }
            Err(StagingError::VulkanError(vk)) => Err(BackendError::Vulkan(vk)),
        }
    }

    pub(crate) fn allocate_image(&self, image: vk::Image) -> BackendResult<ImageMemory> {
        let mut image_cache = self.image_cache.lock().unwrap();
        let memory = image_cache.allocate(&self.raw, &self.pdevice, image)?;
        unsafe {
            self.raw
                .bind_image_memory(image, memory.memory, memory.offset as _)
        }?;

        Ok(memory)
    }

    pub fn push_uniforms<T: Sized>(&self, data: &[T]) -> Buffer {
        self.uniform_buffer.push(data)
    }

    pub fn push_dynamic_geo<T: Sized>(&self, data: &[T]) -> Buffer {
        self.dynamic_buffer.push(data)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.device_wait_idle().ok() };
        let mut image_cache = self.image_cache.lock().unwrap();
        let mut geo_cache = self.geo_cache.lock().unwrap();
        let staging = self.staging.lock().unwrap();
        self.setup_cb.free(&self.raw);
        unsafe { self.raw.destroy_command_pool(self.setup_pool, None) };
        self.drop_lists.iter().for_each(|list| {
            let mut list = list.lock().unwrap();
            list.free(&self.raw, &mut image_cache, &mut geo_cache);
        });
        self.frames.iter().for_each(|frame| {
            let mut frame = frame.lock().unwrap();
            let frame = Arc::get_mut(&mut frame).unwrap();
            frame.free(&self.raw);
        });
        self.samplers.iter().for_each(|(_, sampler)| unsafe {
            self.raw.destroy_sampler(*sampler, None);
        });
        staging.free(&self.raw);
        image_cache.free(&self.raw);
        geo_cache.free(&self.raw);
        self.uniform_buffer.free(&self.raw);
        self.dynamic_buffer.free(&self.raw);
        unsafe { self.raw.destroy_device(None) };
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Vulkan Device")
    }
}
