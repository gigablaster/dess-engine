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
    BackendError, BackendResult,
};

use super::{GpuAllocator, GpuMemory, Instance, PhysicalDevice};

const STAGES: usize = 4;
const BUFFER_SIZE: usize = 32 * 1024 * 1024;

pub struct ImageSubresourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
}

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy2, vk::ImageSubresourceRange);

pub struct Staging {
    pool: vk::CommandPool,
    tranfser_cbs: ArrayVec<CommandBuffer, STAGES>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy2>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    mapping: NonNull<u8>,
    buffer: vk::Buffer,
    memory: Option<GpuMemory>,
    semaphores: ArrayVec<vk::Semaphore, STAGES>,
    render_semaphores: ArrayVec<vk::Semaphore, STAGES>,
    last: Option<usize>,
    current: usize,
    queue_family_index: u32,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

impl Staging {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        pdevice: &PhysicalDevice,
        allocator: &mut GpuAllocator,
        queue_family_index: u32,
    ) -> BackendResult<Self> {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family_index),
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
                BUFFER_SIZE * STAGES,
                pdevice
                    .properties
                    .limits
                    .optimal_buffer_copy_offset_alignment as _,
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
            queue_family_index,
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        device: &Device,
        target: &Buffer,
        offset: usize,
        data: &[T],
    ) -> BackendResult<()> {
        if let Some(memory) = &target.memory {
            if memory.props().contains(
                gpu_alloc::MemoryPropertyFlags::DEVICE_LOCAL
                    | gpu_alloc::MemoryPropertyFlags::HOST_VISIBLE,
            ) {
                // Special experimental case for UMA - we just map and memcpy.
                unsafe {
                    let ptr = device.raw.map_memory(
                        *memory.memory(),
                        memory.offset(),
                        memory.size(),
                        vk::MemoryMapFlags::empty(),
                    )? as *mut u8;
                    copy_nonoverlapping(data.as_ptr() as *const u8, ptr, mem::size_of_val(data));
                    if !memory
                        .props()
                        .contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT)
                    {
                        let range = vk::MappedMemoryRange::builder()
                            .memory(*memory.memory())
                            .offset(memory.offset())
                            .size(memory.size())
                            .build();
                        device
                            .raw
                            .flush_mapped_memory_ranges(slice::from_ref(&range))?;
                    }
                    device.raw.unmap_memory(*memory.memory());
                }
                return Ok(());
            }
        }
        let mut current_offset = 0;
        loop {
            let pushed = self.try_push_buffer(
                target,
                offset + current_offset,
                data.len() - current_offset,
                unsafe { data.as_ptr().add(current_offset) },
            )?;
            current_offset += pushed;
            if current_offset == data.len() {
                return Ok(());
            } else {
                self.upload(device)?;
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
                self.upload(device)?;
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
        if data.data.len() > BUFFER_SIZE {
            return Err(BackendError::TooBig);
        }
        if let Some(allocated) = self.allocator.allocate(data.data.len()) {
            let buffer_offset = BUFFER_SIZE * self.current + allocated;
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mapping.as_ptr().add(buffer_offset),
                    data.data.len(),
                )
            };
            let op = vk::BufferImageCopy2::builder()
                .buffer_image_height(target.desc.extent[0] >> mip)
                .buffer_row_length(data.row_pitch as _)
                .image_extent(vk::Extent3D {
                    width: target.desc.extent[0] >> mip,
                    height: target.desc.extent[1] >> mip,
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

    fn try_push_buffer<T: Sized>(
        &mut self,
        target: &Buffer,
        offset: usize,
        count: usize,
        data: *const T,
    ) -> BackendResult<usize> {
        let byte_size = mem::size_of::<T>() * count;
        let can_send = self.allocator.validate(byte_size);
        let allocated = self.allocator.allocate(can_send).unwrap(); // Already checked that allocator can allocate enough space
        let src_offset = BUFFER_SIZE * self.current + allocated;
        unsafe {
            copy_nonoverlapping(
                data as *const u8,
                self.mapping.as_ptr().add(src_offset),
                can_send,
            )
        };
        let op = vk::BufferCopy2::builder()
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
    ) -> BackendResult<(vk::Semaphore, vk::PipelineStageFlags2)> {
        puffin::profile_function!();
        let cb = &self.tranfser_cbs[self.current];

        unsafe {
            device
                .raw
                .wait_for_fences(slice::from_ref(&cb.fence), true, u64::MAX)?;
            device.raw.reset_fences(slice::from_ref(&cb.fence))?;
            device
                .raw
                .reset_command_buffer(cb.raw, vk::CommandBufferResetFlags::empty())?;
            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();
            device.raw.begin_command_buffer(cb.raw, &info)?;
        };

        {
            let _ = device.scoped_label(cb.raw, "Submit staged data");
            self.barrier_before(&device.raw, cb.raw);
            self.copy_buffers(&device.raw, cb.raw);
            self.copy_images(&device.raw, cb.raw);
            self.barrier_after(&device.raw, cb.raw);
        }

        {
            unsafe { device.raw.end_command_buffer(cb.raw) }?;
        }
        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];

        if let Some(last) = self.last {
            device.submit(
                cb,
                &[(self.semaphores[last], vk::PipelineStageFlags2::TRANSFER)],
                &[
                    (semaphore, vk::PipelineStageFlags2::TRANSFER),
                    (
                        render_semaphore,
                        vk::PipelineStageFlags2::VERTEX_INPUT
                            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    ),
                ],
            )?;
        } else {
            device.submit(
                cb,
                &[],
                &[
                    (semaphore, vk::PipelineStageFlags2::TRANSFER),
                    (
                        render_semaphore,
                        vk::PipelineStageFlags2::VERTEX_INPUT
                            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    ),
                ],
            )?;
        }

        self.last = Some(self.current);
        self.current += 1;
        self.current %= STAGES;
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        Ok((render_semaphore, vk::PipelineStageFlags2::TRANSFER))
    }

    fn barrier_before(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier2::builder()
                    .buffer(*x.0)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
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
                let barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(*x.0)
                    .subresource_range(op.1)
                    .build();
                image_barriers.push(barrier);
            })
        });
        let dependencies = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .buffer_memory_barriers(&buffer_barriers)
            .image_memory_barriers(&image_barriers)
            .build();

        unsafe { device.cmd_pipeline_barrier2(cb, &dependencies) };
    }

    fn barrier_after(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier2::builder()
                    .buffer(*x.0)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
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
                let barrier = vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(*x.0)
                    .subresource_range(op.1)
                    .build();
                image_barriers.push(barrier);
            })
        });
        let dependencies = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .buffer_memory_barriers(&buffer_barriers)
            .image_memory_barriers(&image_barriers)
            .build();

        unsafe { device.cmd_pipeline_barrier2(cb, &dependencies) };
    }

    fn copy_buffers(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_buffers.iter().for_each(|x| {
            let info = vk::CopyBufferInfo2::builder()
                .src_buffer(self.buffer)
                .dst_buffer(*x.0)
                .regions(x.1)
                .build();
            unsafe { device.cmd_copy_buffer2(cb, &info) };
        })
    }

    fn copy_images(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            let info = vk::CopyBufferToImageInfo2::builder()
                .src_buffer(self.buffer)
                .dst_image(*x.0)
                .regions(&regions)
                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .build();
            unsafe { device.cmd_copy_buffer_to_image2(cb, &info) }
        })
    }

    pub fn free(&mut self, device: &ash::Device, allocator: &mut GpuAllocator) {
        if let Some(memory) = self.memory.take() {
            unsafe {
                allocator.dealloc(AshMemoryDevice::wrap(device), memory);
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
