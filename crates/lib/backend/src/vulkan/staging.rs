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
    mem::{self},
    ptr::{copy_nonoverlapping, NonNull},
    slice,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use dess_common::BumpAllocator;
use gpu_alloc_ash::AshMemoryDevice;

use crate::{
    vulkan::{Buffer, CommandBuffer, Device, Image},
    BackendError, BackendResult, ImageSubresourceData,
};

use super::{GpuAllocator, GpuMemory, Instance, PhysicalDevice};

const STAGES: usize = 4;
const BUFFER_SIZE: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

pub struct Staging {
    pool: vk::CommandPool,
    tranfser_cbs: ArrayVec<CommandBuffer, STAGES>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    mapping: NonNull<u8>,
    buffer: vk::Buffer,
    memory: Option<GpuMemory>,
    semaphores: ArrayVec<vk::Semaphore, STAGES>,
    render_semaphores: ArrayVec<vk::Semaphore, STAGES>,
    last: Option<usize>,
    current: usize,
    main_queue_family_index: u32,
    transfer_queue_family_index: u32,
    pending_buffer_barriers: Vec<vk::BufferMemoryBarrier>,
    pending_image_barriers: Vec<vk::ImageMemoryBarrier>,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

impl Staging {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        allocator: &mut GpuAllocator,
        main_queue_family_index: u32,
        transfer_queue_family_index: u32,
    ) -> BackendResult<Self> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(transfer_queue_family_index),
                None,
            )
        }?;
        let mut tranfser_cbs = ArrayVec::new();
        let mut semaphores = ArrayVec::new();
        let mut render_semaphores = ArrayVec::new();
        for index in 0..STAGES {
            let cb = CommandBuffer::primary(device, pool)?;
            let semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None) }?;
            let render_semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None) }?;
            Device::set_object_name_impl(instance, device, cb.raw, &format!("Staging CB {index}"));
            Device::set_object_name_impl(
                instance,
                device,
                semaphore,
                &format!("Staging-staging semaphore {index}"),
            );
            Device::set_object_name_impl(
                instance,
                device,
                render_semaphore,
                &format!("Staging-render semaphore {index}"),
            );
            tranfser_cbs.push(cb);
            semaphores.push(semaphore);
            render_semaphores.push(render_semaphore);
        }
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size((BUFFER_SIZE * STAGES) as _),
                None,
            )
        }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut memory = Device::allocate_impl(
            device,
            allocator,
            requirements,
            gpu_alloc::UsageFlags::HOST_ACCESS,
            false,
        )?;
        unsafe { device.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;
        let mapping =
            unsafe { memory.map(AshMemoryDevice::wrap(device), 0, BUFFER_SIZE * STAGES) }?;

        Ok(Self {
            pool,
            tranfser_cbs,
            allocator: BumpAllocator::new(
                BUFFER_SIZE,
                pdevice.properties.limits.buffer_image_granularity.max(512) as _,
            ),
            upload_buffers: HashMap::with_capacity(64),
            upload_images: HashMap::with_capacity(64),
            mapping,
            buffer,
            memory: Some(memory),
            semaphores,
            render_semaphores,
            last: None,
            current: 0,
            main_queue_family_index,
            transfer_queue_family_index,
            pending_buffer_barriers: Vec::default(),
            pending_image_barriers: Vec::default(),
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        device: &Device,
        target: &Buffer,
        offset: usize,
        data: &[T],
    ) -> BackendResult<()> {
        let mut current_offset = 0;
        loop {
            let data_len = mem::size_of_val(data);
            let pushed = self.try_push_buffer(
                target,
                offset + current_offset,
                data_len - current_offset,
                unsafe { (data.as_ptr() as *const u8).add(current_offset) },
            )?;
            current_offset += pushed;
            if current_offset == data_len {
                return Ok(());
            } else {
                self.upload(device, false)?;
            }
        }
    }

    pub fn upload_image(
        &mut self,
        device: &Device,
        target: &Image,
        data: &[ImageSubresourceData],
    ) -> BackendResult<()> {
        for (mip, data) in data.iter().enumerate() {
            while !self.try_push_mip(target, mip as _, data)? {
                self.upload(device, false)?;
            }
        }
        Ok(())
    }

    fn try_push_mip(
        &mut self,
        target: &Image,
        mip: u32,
        data: &ImageSubresourceData,
    ) -> BackendResult<bool> {
        let size = data.data.len();
        if size > BUFFER_SIZE {
            return Err(BackendError::TooBig);
        }
        if let Some(allocated_offset) = self.allocator.allocate(size) {
            let buffer_offset = BUFFER_SIZE * self.current + allocated_offset;
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mapping.as_ptr().add(buffer_offset),
                    size,
                )
            };
            let op = vk::BufferImageCopy::builder()
                .image_extent(vk::Extent3D {
                    width: target.desc.dims[0] >> mip,
                    height: target.desc.dims[1] >> mip,
                    depth: 1,
                })
                .buffer_offset(buffer_offset as _)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            self.upload_images
                .entry(target.raw)
                .or_default()
                .push(ImageUploadRequest(op, range));

            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn try_push_buffer(
        &mut self,
        target: &Buffer,
        offset: usize,
        bytes: usize,
        data: *const u8,
    ) -> BackendResult<usize> {
        let can_send = self.allocator.validate(bytes);
        let allocated = self.allocator.allocate(can_send).unwrap(); // Already checked that allocator can allocate enough space
        let src_offset = BUFFER_SIZE * self.current + allocated;
        unsafe { copy_nonoverlapping(data, self.mapping.as_ptr().add(src_offset), can_send) };
        let op = vk::BufferCopy::builder()
            .src_offset(src_offset as _)
            .dst_offset(offset as _)
            .size(can_send as _)
            .build();
        self.upload_buffers.entry(target.raw).or_default().push(op);

        Ok(can_send)
    }

    pub fn upload(
        &mut self,
        device: &Device,
        will_wait: bool,
    ) -> BackendResult<(vk::Semaphore, vk::PipelineStageFlags)> {
        puffin::profile_function!();
        let cb = &self.tranfser_cbs[self.current];
        let fence = cb.fence;
        let cb = cb.raw;

        unsafe {
            device
                .raw
                .wait_for_fences(slice::from_ref(&fence), true, u64::MAX)?;
            device.raw.reset_fences(slice::from_ref(&fence))?;
            device
                .raw
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();
            device.raw.begin_command_buffer(cb, &info)?;
        };

        {
            let _ = device.scoped_label(cb, "Submit staged data");
            self.barrier_before(&device.raw, cb);
            self.copy_buffers(&device.raw, cb);
            self.copy_images(&device.raw, cb);
            self.barrier_after(&device.raw, cb);
        }

        {
            unsafe { device.raw.end_command_buffer(cb) }?;
        }
        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];
        let mut triggers = ArrayVec::<_, 2>::new();
        triggers.push(semaphore);
        if will_wait {
            triggers.push(render_semaphore);
        }

        if let Some(last) = self.last {
            device.submit_transfer(
                &self.tranfser_cbs[self.current],
                &[(self.semaphores[last], vk::PipelineStageFlags::TRANSFER)],
                &triggers,
            )?;
        } else {
            device.submit_transfer(&self.tranfser_cbs[self.current], &[], &triggers)?;
        }

        self.last = Some(self.current);
        self.current += 1;
        self.current %= STAGES;
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        Ok((render_semaphore, vk::PipelineStageFlags::TRANSFER))
    }

    pub fn execute_pending_barriers(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &self.pending_buffer_barriers,
                &self.pending_image_barriers,
            );
            self.pending_buffer_barriers.clear();
            self.pending_image_barriers.clear();
        };
    }

    fn barrier_before(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::builder()
                    .buffer(*x.0)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .offset(op.dst_offset)
                    .size(op.size)
                    .build();
                buffer_barriers.push(barrier);
            })
        });
        let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
        let mut image_barriers = Vec::with_capacity(size);
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier::builder()
                    // .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(*x.0)
                    .subresource_range(op.1)
                    .build();
                image_barriers.push(barrier);
            })
        });
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &buffer_barriers,
                &image_barriers,
            )
        };
    }

    fn barrier_after(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::builder()
                    .buffer(*x.0)
                    .src_queue_family_index(self.transfer_queue_family_index)
                    .dst_queue_family_index(self.main_queue_family_index)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .offset(op.dst_offset)
                    .size(op.size)
                    .build();
                self.pending_buffer_barriers.push(barrier);
            })
        });
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self.transfer_queue_family_index)
                    .dst_queue_family_index(self.main_queue_family_index)
                    .image(*x.0)
                    .subresource_range(op.1)
                    .build();
                self.pending_image_barriers.push(barrier);
            })
        });

        if self.transfer_queue_family_index != self.main_queue_family_index {
            let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
            let mut buffer_barriers = Vec::with_capacity(size);
            self.upload_buffers.iter().for_each(|x| {
                x.1.iter().for_each(|op| {
                    let barrier = vk::BufferMemoryBarrier::builder()
                        .buffer(*x.0)
                        .src_queue_family_index(self.transfer_queue_family_index)
                        .dst_queue_family_index(self.main_queue_family_index)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .offset(op.dst_offset)
                        .size(op.size)
                        .build();
                    buffer_barriers.push(barrier);
                })
            });
            let size = self.upload_images.iter().map(|x| x.1.len()).sum::<usize>();
            let mut image_barriers = Vec::with_capacity(size);
            self.upload_images.iter().for_each(|x| {
                x.1.iter().for_each(|op| {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(self.transfer_queue_family_index)
                        .dst_queue_family_index(self.main_queue_family_index)
                        .image(*x.0)
                        .subresource_range(op.1)
                        .build();
                    image_barriers.push(barrier);
                })
            });
            unsafe {
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &buffer_barriers,
                    &image_barriers,
                )
            };
        }
    }

    fn copy_buffers(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_buffers.iter().for_each(|x| {
            unsafe { device.cmd_copy_buffer(cb, self.buffer, *x.0, x.1) };
        })
    }

    fn copy_images(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cb,
                    self.buffer,
                    *x.0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        })
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(memory) = self.memory.take() {
            unsafe {
                allocator.dealloc(AshMemoryDevice::wrap(device), memory);
                for cb in self.tranfser_cbs.drain(..) {
                    cb.free(device);
                }
                device.destroy_buffer(self.buffer, None);
                device.destroy_command_pool(self.pool, None);
                for index in 0..STAGES {
                    device.destroy_semaphore(self.semaphores[index], None);
                    device.destroy_semaphore(self.render_semaphores[index], None);
                }
            }
        }
    }
}
