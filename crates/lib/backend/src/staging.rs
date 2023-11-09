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
    fmt::Debug,
    mem::size_of_val,
    ptr::{copy_nonoverlapping, NonNull},
    slice,
    sync::Arc,
};

use ash::vk;
use dess_common::memory::BumpAllocator;

use parking_lot::Mutex;
use vk_sync::{cmd::pipeline_barrier, AccessType, BufferBarrier, ImageBarrier};

use crate::{
    vulkan::{
        Buffer, BufferDesc, CommandBuffer, CommandBufferRecorder, CommandPool, Device, Image,
        Semaphore, SubImage, SubmitWait,
    },
    BackendError, GpuResource,
};

const STAGES: usize = 4;

pub struct ImageSubresourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
}

#[derive(Debug, Clone)]
struct ImageUploadRequest {
    pub op: vk::BufferImageCopy,
    pub subresource: vk::ImageSubresourceRange,
    pub dst: Arc<Image>,
}

struct StagingInner {
    device: Arc<Device>,
    pool: CommandPool,
    tranfser_cbs: Vec<CommandBuffer>,
    size: u64,
    allocator: BumpAllocator,
    upload_buffers: HashMap<Arc<Buffer>, Vec<vk::BufferCopy>>,
    upload_images: Vec<ImageUploadRequest>,
    mappings: Vec<NonNull<u8>>,
    buffers: Vec<Buffer>,
    semaphores: Vec<Semaphore>,
    render_semaphores: Vec<Semaphore>,
    last: Option<usize>,
    current: usize,
    index: u64,
}

unsafe impl Send for StagingInner {}
unsafe impl Sync for StagingInner {}

impl StagingInner {
    pub fn new(device: &Arc<Device>, size: usize) -> Result<Self, BackendError> {
        let mut pool = CommandPool::new(
            device.raw(),
            device.queue_family_index(),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            vk::CommandBufferLevel::PRIMARY,
        )?;
        let mut buffers = (0..STAGES)
            .map(|x| {
                let buffer = Buffer::new(
                    device,
                    BufferDesc::host_only(size, vk::BufferUsageFlags::TRANSFER_SRC).dedicated(true),
                )
                .unwrap();
                buffer.name(&format!("staging {}", x));
                buffer
            })
            .collect::<Vec<_>>();
        let semaphores = (0..STAGES)
            .map(|index| {
                let semaphore = Semaphore::new(device.raw()).unwrap();
                device.set_object_name(semaphore.raw, &format!("Staging sempahore {}", index));
                semaphore
            })
            .collect::<Vec<_>>();
        let render_semaphores = (0..STAGES)
            .map(|index| {
                let semaphore = Semaphore::new(device.raw()).unwrap();
                device.set_object_name(
                    semaphore.raw,
                    &format!("Staging->Render sempahore {}", index),
                );
                semaphore
            })
            .collect::<Vec<_>>();

        let tranfser_cbs = (0..STAGES)
            .map(|index| {
                let cb = pool.get_or_create(device.raw()).unwrap();
                device.set_object_name(cb.raw(), &format!("Stage CB {}", index));
                cb
            })
            .collect::<Vec<_>>();
        tranfser_cbs.iter().for_each(|cb| {
            device.set_object_name(cb.raw(), "staging - transfer cb");
        });
        let mappings = buffers
            .iter_mut()
            .map(|buffer| buffer.map().unwrap())
            .collect::<Vec<_>>();
        Ok(Self {
            device: device.clone(),
            pool,
            buffers,
            semaphores,
            render_semaphores,
            last: None,
            current: 0,
            tranfser_cbs,
            size: size as _,
            allocator: BumpAllocator::new(
                size as _,
                64.max(
                    device
                        .physical_device()
                        .properties()
                        .limits
                        .buffer_image_granularity as u32,
                ),
            ),
            upload_buffers: HashMap::with_capacity(32),
            upload_images: Vec::with_capacity(32),
            mappings,
            index: 0,
        })
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        buffer: &Arc<Buffer>,
        offset: u64,
        data: &[T],
    ) -> Result<(), BackendError> {
        let size = size_of_val(data);
        assert!(size as u64 <= self.size);
        if !self.try_push_buffer(buffer, offset, size, data.as_ptr() as *const u8) {
            self.upload()?;
            if !self.try_push_buffer(buffer, offset, size, data.as_ptr() as *const u8) {
                panic!(
                    "Despite just pushing entire staging buffer we still can't push data in it."
                );
            }
        }

        Ok(())
    }

    pub fn upload_image(
        &mut self,
        image: &Arc<Image>,
        data: &[&ImageSubresourceData],
    ) -> Result<(), BackendError> {
        for (mip, data) in data.iter().enumerate() {
            if !self.try_push_image_mip(image, mip as _, data) {
                self.upload()?;
                if !self.try_push_image_mip(image, mip as _, data) {
                    panic!(
                        "Despite just pushing entire staging buffer we still can't push data in it."
                    );
                }
            }
        }

        Ok(())
    }

    fn try_push_image_mip(
        &mut self,
        image: &Arc<Image>,
        mip: u32,
        data: &ImageSubresourceData,
    ) -> bool {
        if let Some(staging_offset) = self.allocator.allocate(data.data.len() as _) {
            unsafe {
                copy_nonoverlapping(
                    data.data.as_ptr(),
                    self.mappings[self.current]
                        .as_ptr()
                        .add(staging_offset as _),
                    data.data.len(),
                )
            }
            let op = vk::BufferImageCopy::builder()
                .buffer_image_height(image.desc().extent[0])
                .buffer_row_length(data.row_pitch as _)
                .image_extent(vk::Extent3D {
                    width: image.desc().extent[0],
                    height: image.desc().extent[1],
                    depth: 1,
                })
                .buffer_offset(staging_offset as _)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_subresource(image.subresource_layer(0, mip, vk::ImageAspectFlags::COLOR))
                .build();

            let request = ImageUploadRequest {
                op,
                dst: image.clone(),
                subresource: image
                    .subresource_range(SubImage::LayerAndMip(0, mip), vk::ImageAspectFlags::COLOR),
            };
            self.upload_images.push(request);
            return true;
        }

        false
    }

    fn try_push_buffer(
        &mut self,
        buffer: &Arc<Buffer>,
        offset: u64,
        size: usize,
        data: *const u8,
    ) -> bool {
        if let Some(staging_offset) = self.allocator.allocate(size as _) {
            unsafe {
                copy_nonoverlapping(
                    data,
                    self.mappings[self.current]
                        .as_ptr()
                        .add(staging_offset as _),
                    size,
                )
            }
            let op = vk::BufferCopy {
                src_offset: staging_offset as u64,
                dst_offset: offset,
                size: size as u64,
            };
            self.upload_buffers
                .entry(buffer.clone())
                .or_default()
                .push(op);
            true
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.upload_buffers.is_empty() && self.upload_images.is_empty()
    }

    pub fn upload(&mut self) -> Result<Option<SubmitWait>, BackendError> {
        if self.upload_images.is_empty() && self.upload_buffers.is_empty() {
            return Ok(None);
        }
        self.tranfser_cbs[self.current].wait(self.device.raw())?;
        self.tranfser_cbs[self.current].reset(self.device.raw())?;

        self.tranfser_cbs[self.current].record(self.device.raw(), |recorder| {
            self.device.cmd_begin_label(
                self.tranfser_cbs[self.current].raw(),
                &format!("Sending staging queue #{}", self.index),
            );

            self.barrier_pre(&recorder);
            self.copy_buffers(&recorder);
            self.copy_images(&recorder);
            self.barrier_after(&recorder);

            self.device
                .cmd_end_label(self.tranfser_cbs[self.current].raw());
        })?;
        let semaphore = self.semaphores[self.current];
        let render_semaphore = self.render_semaphores[self.current];

        if let Some(last) = self.last {
            self.device.submit_transfer(
                &self.tranfser_cbs[self.current],
                &[SubmitWait::Transfer(self.semaphores[last])],
                &[semaphore, render_semaphore],
            )?;
        } else {
            self.device.submit_transfer(
                &self.tranfser_cbs[self.current],
                &[],
                &[semaphore, render_semaphore],
            )?;
        }

        self.allocator.reset();
        self.upload_buffers.clear();
        self.upload_images.clear();

        self.last = Some(self.current);
        self.current += 1;
        self.current %= STAGES;

        Ok(Some(SubmitWait::Transfer(render_semaphore)))
    }

    fn barrier_pre(&self, recorder: &CommandBufferRecorder) {
        let image_barriers = self
            .upload_images
            .iter()
            .map(|request| ImageBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::TransferWrite],
                src_queue_family_index: self.device.queue_family_index(),
                dst_queue_family_index: self.device.queue_family_index(),
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                image: request.dst.raw(),
                range: request.subresource,
            })
            .collect::<Vec<_>>();

        let size = self.upload_buffers.values().map(|x| x.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        for (target, requests) in &self.upload_buffers {
            requests.iter().for_each(|buffer| {
                buffer_barriers.push(BufferBarrier {
                    previous_accesses: target.access_type(),
                    next_accesses: &[AccessType::TransferWrite],
                    src_queue_family_index: self.device.queue_family_index(),
                    dst_queue_family_index: self.device.queue_family_index(),
                    buffer: target.raw(),
                    offset: buffer.dst_offset as usize,
                    size: buffer.size as usize,
                });
            })
        }

        pipeline_barrier(
            recorder.device,
            *recorder.cb,
            None,
            &buffer_barriers,
            &image_barriers,
        );
    }

    fn barrier_after(&self, recorder: &CommandBufferRecorder) {
        let image_barriers = self
            .upload_images
            .iter()
            .map(|request| ImageBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer],
                src_queue_family_index: self.device.queue_family_index(),
                dst_queue_family_index: self.device.queue_family_index(),
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: false,
                image: request.dst.raw(),
                range: request.subresource,
            })
            .collect::<Vec<_>>();

        let size = self.upload_buffers.values().map(|x| x.len()).sum::<usize>();
        let mut buffer_barriers = Vec::with_capacity(size);
        for (target, requests) in &self.upload_buffers {
            requests.iter().for_each(|buffer| {
                buffer_barriers.push(BufferBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: target.access_type(),
                    src_queue_family_index: self.device.queue_family_index(),
                    dst_queue_family_index: self.device.queue_family_index(),
                    buffer: target.raw(),
                    offset: buffer.dst_offset as usize,
                    size: buffer.size as usize,
                });
            })
        }

        pipeline_barrier(
            recorder.device,
            *recorder.cb,
            None,
            &buffer_barriers,
            &image_barriers,
        );
    }

    fn copy_buffers(&self, recorder: &CommandBufferRecorder) {
        self.upload_buffers.iter().for_each(|(buffer, requests)| {
            unsafe {
                recorder.device.cmd_copy_buffer(
                    *recorder.cb,
                    self.buffers[self.current].raw(),
                    buffer.raw(),
                    requests,
                )
            };
        });
    }

    fn copy_images(&self, recorder: &CommandBufferRecorder) {
        self.upload_images.iter().for_each(|requests| unsafe {
            recorder.device.cmd_copy_buffer_to_image(
                *recorder.cb,
                self.buffers[self.current].raw(),
                requests.dst.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&requests.op),
            )
        });
    }
}

impl Drop for StagingInner {
    fn drop(&mut self) {
        self.tranfser_cbs
            .drain(..)
            .for_each(|cb| self.pool.retire(cb));
        self.pool.free(self.device.raw());
        self.semaphores
            .iter()
            .for_each(|semaphore| semaphore.free(self.device.raw()));
        self.render_semaphores
            .iter()
            .for_each(|semaphore| semaphore.free(self.device.raw()));
    }
}

pub struct Staging {
    inner: Mutex<StagingInner>,
}

impl Staging {
    pub fn new(device: &Arc<Device>, size: usize) -> Result<Self, BackendError> {
        Ok(Self {
            inner: Mutex::new(StagingInner::new(device, size)?),
        })
    }

    pub fn upload_buffer<T: Sized>(
        &self,
        buffer: &Arc<Buffer>,
        offset: u64,
        data: &[T],
    ) -> Result<(), BackendError> {
        self.inner.lock().upload_buffer(buffer, offset, data)
    }

    pub fn upload_image(
        &self,
        image: &Arc<Image>,
        data: &[&ImageSubresourceData],
    ) -> Result<(), BackendError> {
        self.inner.lock().upload_image(image, data)
    }

    pub fn upload(&self) -> Result<Option<SubmitWait>, BackendError> {
        self.inner.lock().upload()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}
