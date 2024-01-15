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

use std::{
    collections::HashMap,
    mem::{self},
    ptr::{copy_nonoverlapping, NonNull},
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use dess_backend::{
    AsVulkan, Buffer, BufferCreateDesc, CommandBuffer, CommandBufferRecorder, Device, Image,
    ImageSubresourceData,
};
use dess_common::BumpAllocator;
use parking_lot::Mutex;

use crate::Error;

#[derive(Debug, Clone, Copy)]
struct ImageUploadRequest(vk::BufferImageCopy, vk::ImageSubresourceRange);

#[derive(Debug)]
pub struct Staging {
    device: Arc<Device>,
    command_buffers: Vec<CommandBuffer>,
    allocator: BumpAllocator,
    upload_buffers: HashMap<vk::Buffer, Vec<vk::BufferCopy>>,
    upload_images: HashMap<vk::Image, Vec<ImageUploadRequest>>,
    buffer: Buffer,
    mapping: NonNull<u8>,
    semaphores: Vec<vk::Semaphore>,
    render_semaphores: Vec<vk::Semaphore>,
    last: Option<usize>,
    current: usize,
    page_size: usize,
    pages: usize,
    pending_buffer_barriers: Mutex<Vec<vk::BufferMemoryBarrier>>,
    pending_image_barriers: Mutex<Vec<vk::ImageMemoryBarrier>>,
}

unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

pub struct StagingDesc {
    pub page_size: usize,
    pub pages: usize,
}

impl Default for StagingDesc {
    fn default() -> Self {
        Self::new(32 * 1024 * 1024, 4)
    }
}

impl StagingDesc {
    pub fn new(page_size: usize, pages: usize) -> Self {
        Self { page_size, pages }
    }
}

fn create_semaphore(device: &Device, index: usize) -> Result<vk::Semaphore, Error> {
    let semaphore = unsafe {
        device
            .get()
            .create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
    }?;
    device.set_object_name(semaphore, format!("Staging semaphore {index}"));
    Ok(semaphore)
}

impl Staging {
    pub fn new(device: &Arc<Device>, desc: StagingDesc) -> Result<Self, Error> {
        let transfer_cbs = Vec::from_iter(
            (0..desc.pages)
                .map(|n| CommandBuffer::transfer(device, Some(format!("Staging {n}"))).unwrap()),
        );
        let semaphores =
            Vec::from_iter((0..desc.pages).map(|n| create_semaphore(device, n).unwrap()));
        let render_semaphores =
            Vec::from_iter((0..desc.pages).map(|n| create_semaphore(device, n).unwrap()));

        let mut buffer = Buffer::new(
            device,
            BufferCreateDesc::upload(desc.page_size * desc.pages)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .name("Staging buffer")
                .dedicated(true),
        )?;
        Ok(Self {
            device: device.clone(),
            command_buffers: transfer_cbs,
            allocator: BumpAllocator::new(desc.page_size),
            upload_buffers: HashMap::with_capacity(64),
            upload_images: HashMap::with_capacity(64),
            mapping: buffer.map()?,
            buffer,
            semaphores,
            render_semaphores,
            last: None,
            current: 0,
            pending_buffer_barriers: Mutex::default(),
            pending_image_barriers: Mutex::default(),
            page_size: desc.page_size,
            pages: desc.pages,
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        target: &Buffer,
        offset: usize,
        data: &[T],
    ) -> Result<(), Error> {
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
                self.upload_impl(false)?;
            }
        }
    }

    pub fn upload_image(
        &mut self,
        target: &Image,
        data: &[ImageSubresourceData],
    ) -> Result<(), Error> {
        for (mip, data) in data.iter().enumerate() {
            while !self.try_push_mip(target, mip as _, data)? {
                self.upload_impl(false)?;
            }
        }
        Ok(())
    }

    fn try_push_mip(
        &mut self,
        target: &Image,
        mip: u32,
        data: &ImageSubresourceData,
    ) -> Result<bool, Error> {
        let size = data.data.len();
        if size > self.page_size {
            return Err(Error::ImageTooBig);
        }
        let aligment = self
            .device
            .physical_device()
            .properties()
            .limits
            .buffer_image_granularity;
        if let Some(allocated_offset) = self.allocator.allocate(size, aligment as _) {
            let buffer_offset = self.page_size * self.current + allocated_offset;
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mapping.as_ptr().add(buffer_offset),
                    size,
                )
            };
            let op = vk::BufferImageCopy::builder()
                .image_extent(vk::Extent3D {
                    width: target.desc().dims[0] >> mip,
                    height: target.desc().dims[1] >> mip,
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
                .entry(target.as_vk())
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
    ) -> Result<usize, Error> {
        let aligment = self
            .device
            .physical_device()
            .properties()
            .limits
            .optimal_buffer_copy_offset_alignment;
        let can_send = self.allocator.validate(bytes, aligment as _);
        let allocated = self.allocator.allocate(bytes, aligment as _).unwrap(); // Already checked that allocator can allocate enough space
        let src_offset = self.page_size * self.current + allocated;
        unsafe { copy_nonoverlapping(data, self.mapping.as_ptr().add(src_offset), can_send) };
        let op = vk::BufferCopy::builder()
            .src_offset(src_offset as _)
            .dst_offset(offset as _)
            .size(can_send as _)
            .build();
        self.upload_buffers
            .entry(target.as_vk())
            .or_default()
            .push(op);

        Ok(can_send)
    }

    pub fn upload(&mut self) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        self.upload_impl(true)
    }

    fn upload_impl(
        &mut self,
        client_will_wait: bool,
    ) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        puffin::profile_function!();
        let cb = &self.command_buffers[self.current];

        cb.wait()?;
        cb.reset()?;
        {
            let recorder = cb.record();

            self.barrier_before(&recorder);
            self.copy_buffers(&recorder);
            self.copy_images(&recorder);
            self.barrier_after(&recorder);
        }

        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];
        let mut triggers = ArrayVec::<_, 2>::new();
        triggers.push(semaphore);
        if client_will_wait {
            triggers.push(render_semaphore);
        }

        if let Some(last) = self.last {
            self.device.submit_transfer(
                &self.command_buffers[self.current],
                &[(self.semaphores[last], vk::PipelineStageFlags::TRANSFER)],
                &triggers,
            )?;
        } else {
            self.device
                .submit_transfer(&self.command_buffers[self.current], &[], &triggers)?;
        }

        self.last = Some(self.current);
        self.current += 1;
        self.current %= self.pages;
        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        Ok((render_semaphore, vk::PipelineStageFlags::TRANSFER))
    }

    pub fn execute_pending_barriers(&self, recorder: &CommandBufferRecorder) {
        let mut pending_buffer_barriers = self.pending_buffer_barriers.lock();
        let mut pending_image_barriers = self.pending_image_barriers.lock();

        recorder.barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::BY_REGION,
            &pending_buffer_barriers,
            &pending_image_barriers,
        );
        pending_buffer_barriers.clear();
        pending_image_barriers.clear();
    }

    fn barrier_before(&self, recorder: &CommandBufferRecorder) {
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
        recorder.barrier(
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::BY_REGION,
            &buffer_barriers,
            &image_barriers,
        );
    }

    fn barrier_after(&self, recorder: &CommandBufferRecorder) {
        let mut pending_buffer_barriers = self.pending_buffer_barriers.lock();
        let mut pending_image_barriers = self.pending_image_barriers.lock();

        self.upload_buffers.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::BufferMemoryBarrier::builder()
                    .buffer(*x.0)
                    .src_queue_family_index(self.device.transfer_queue_index())
                    .dst_queue_family_index(self.device.main_queue_index())
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .offset(op.dst_offset)
                    .size(op.size)
                    .build();
                pending_buffer_barriers.push(barrier);
            })
        });
        self.upload_images.iter().for_each(|x| {
            x.1.iter().for_each(|op| {
                let barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self.device.transfer_queue_index())
                    .dst_queue_family_index(self.device.main_queue_index())
                    .image(*x.0)
                    .subresource_range(op.1)
                    .build();
                pending_image_barriers.push(barrier);
            })
        });

        if self.device.transfer_queue_index() != self.device.main_queue_index() {
            let size = self.upload_buffers.iter().map(|x| x.1.len()).sum::<usize>();
            let mut buffer_barriers = Vec::with_capacity(size);
            self.upload_buffers.iter().for_each(|x| {
                x.1.iter().for_each(|op| {
                    let barrier = vk::BufferMemoryBarrier::builder()
                        .buffer(*x.0)
                        .src_queue_family_index(self.device.transfer_queue_index())
                        .dst_queue_family_index(self.device.main_queue_index())
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
                        .src_queue_family_index(self.device.transfer_queue_index())
                        .dst_queue_family_index(self.device.main_queue_index())
                        .image(*x.0)
                        .subresource_range(op.1)
                        .build();
                    image_barriers.push(barrier);
                })
            });
            recorder.barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &buffer_barriers,
                &image_barriers,
            );
        }
    }

    fn copy_buffers(&self, recorder: &CommandBufferRecorder) {
        self.upload_buffers
            .iter()
            .for_each(|x| recorder.copy_buffer(self.buffer.as_vk(), *x.0, x.1))
    }

    fn copy_images(&self, recorder: &CommandBufferRecorder) {
        self.upload_images.iter().for_each(|x| {
            let regions = x.1.iter().map(|x| x.0).collect::<Vec<_>>();
            recorder.copy_buffer_to_image(self.buffer.as_vk(), *x.0, &regions);
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.upload_impl(false).unwrap();
        unsafe { self.device.get().device_wait_idle().unwrap() };
        for index in 0..self.pages {
            unsafe {
                self.device
                    .get()
                    .destroy_semaphore(self.semaphores[index], None);
                self.device
                    .get()
                    .destroy_semaphore(self.render_semaphores[index], None);
            }
        }
    }
}
