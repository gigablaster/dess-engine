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
    mem::size_of,
    ptr::{copy_nonoverlapping, NonNull},
    slice,
    sync::Arc,
};

use ash::vk;
use dess_common::memory::BumpAllocator;
use dess_render_backend::{
    Buffer, BufferDesc, BufferView, CommandBuffer, CommandBufferRecorder, Device, Image, SubImage,
};
use vk_sync::{cmd::pipeline_barrier, AccessType, BufferBarrier, ImageBarrier};

use crate::RenderResult;

pub struct ImageSubresourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
}

#[derive(Debug, Copy, Clone)]
struct BufferUploadRequest {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
    pub dst: vk::Buffer,
}

#[derive(Debug, Copy, Clone)]
struct ImageUploadRequest {
    pub op: vk::BufferImageCopy,
    pub subresource: vk::ImageSubresourceRange,
    pub dst: vk::Image,
}

pub struct Staging {
    device: Arc<Device>,
    buffer: Buffer,
    tranfser_cb: CommandBuffer,
    size: u64,
    allocator: BumpAllocator,
    upload_buffers: Vec<BufferUploadRequest>,
    upload_images: Vec<ImageUploadRequest>,
    mapping: Option<NonNull<u8>>,
    index: u64,
}

impl Staging {
    pub fn new(device: &Arc<Device>, size: usize) -> RenderResult<Self> {
        let buffer = Buffer::transfer(
            device,
            BufferDesc::host_only(size, vk::BufferUsageFlags::TRANSFER_SRC).dedicated(true),
            Some("staging"),
        )?;
        let tranfser_cb = CommandBuffer::new(&device.raw, device.transfer_queue.family.index)?;
        Ok(Self {
            device: device.clone(),
            buffer,
            tranfser_cb,
            size: size as _,
            allocator: BumpAllocator::new(
                size as _,
                device.pdevice.properties.limits.buffer_image_granularity,
            ),
            upload_buffers: Vec::with_capacity(128),
            upload_images: Vec::with_capacity(32),
            mapping: None,
            index: 0,
        })
    }

    fn map_buffer(&mut self) -> RenderResult<NonNull<u8>> {
        Ok(self.buffer.map()?)
    }

    fn unmap_buffer(&mut self) {
        self.buffer.unmap();
    }

    pub fn upload_buffer<T: Sized>(
        &mut self,
        buffer: &impl BufferView,
        data: &[T],
    ) -> RenderResult<u64> {
        let size = data.len() * size_of::<T>();
        assert!(size as u64 <= self.size);
        assert_eq!(buffer.size(), size as u64);
        self.tranfser_cb.wait(&self.device.raw)?;
        if self.mapping.is_none() {
            self.mapping = Some(self.map_buffer()?);
        }
        let mapping = self.mapping.unwrap();
        if !self.try_push_buffer(buffer, data.as_ptr() as *const u8, mapping.as_ptr(), size) {
            self.upload()?;
            if !self.try_push_buffer(buffer, data.as_ptr() as *const u8, mapping.as_ptr(), size) {
                panic!(
                    "Despite just pushing entire staging buffer we still can't push data in it."
                );
            }
        }

        Ok(self.index)
    }

    pub fn upload_image(
        &mut self,
        image: &Image,
        data: &[&ImageSubresourceData],
    ) -> RenderResult<u64> {
        for (mip, data) in data.iter().enumerate() {
            self.tranfser_cb.wait(&self.device.raw)?;
            if self.mapping.is_none() {
                self.mapping = Some(self.map_buffer()?);
            }

            let mapping = self.mapping.unwrap();
            if !self.try_push_image_mip(image, mip as _, data, mapping.as_ptr()) {
                self.upload()?;
                if !self.try_push_image_mip(image, mip as _, data, mapping.as_ptr()) {
                    panic!(
                        "Despite just pushing entire staging buffer we still can't push data in it."
                    );
                }
            }
        }

        Ok(self.index)
    }

    fn try_push_image_mip(
        &mut self,
        image: &Image,
        mip: u32,
        data: &ImageSubresourceData,
        mapping: *mut u8,
    ) -> bool {
        if let Some(offset) = self.allocator.allocate(data.data.len() as _) {
            unsafe { copy_nonoverlapping(data.data.as_ptr(), mapping, data.data.len()) }
            let op = vk::BufferImageCopy::builder()
                .buffer_image_height(image.desc.extent[0])
                .buffer_row_length(data.row_pitch as _)
                .image_extent(vk::Extent3D {
                    width: image.desc.extent[0],
                    height: image.desc.extent[1],
                    depth: 1,
                })
                .buffer_offset(offset)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_subresource(image.subresource_layer(0, mip, vk::ImageAspectFlags::COLOR))
                .build();

            let request = ImageUploadRequest {
                op,
                dst: image.raw,
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
        buffer: &impl BufferView,
        data: *const u8,
        mapping: *mut u8,
        size: usize,
    ) -> bool {
        if let Some(offset) = self.allocator.allocate(size as _) {
            unsafe { copy_nonoverlapping(data, mapping.add(offset as _), size) }
            let request = BufferUploadRequest {
                dst: buffer.buffer(),
                src_offset: offset,
                dst_offset: buffer.offset(),
                size: buffer.size(),
            };
            self.upload_buffers.push(request);
            true
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.upload_buffers.is_empty() && self.upload_images.is_empty()
    }

    pub fn current_queue(&self) -> u64 {
        self.index
    }

    pub fn wait(&self) -> RenderResult<()> {
        self.tranfser_cb.wait(&self.device.raw)?;

        Ok(())
    }

    pub fn upload(&mut self) -> RenderResult<()> {
        if self.upload_images.is_empty() && self.upload_buffers.is_empty() {
            return Ok(());
        }
        self.tranfser_cb.wait(&self.device.raw)?;
        self.tranfser_cb.reset(&self.device.raw)?;
        if self.mapping.is_some() {
            self.unmap_buffer();
            self.mapping = None;
        }

        self.tranfser_cb.record(&self.device.raw, |recorder| {
            self.barrier_pre(
                &recorder,
                &self.upload_buffers,
                &self.upload_images,
                self.device.graphics_queue.family.index,
                self.device.transfer_queue.family.index,
            );
            self.copy_buffers(&recorder, &self.upload_buffers);
            self.copy_images(&recorder, &self.upload_images);
            self.barrier_after(
                &recorder,
                &self.upload_buffers,
                &self.upload_images,
                self.device.transfer_queue.family.index,
                self.device.graphics_queue.family.index,
            );
        })?;

        self.allocator.reset();
        self.upload_buffers.clear();

        self.index += 1;

        Ok(())
    }

    fn barrier_pre(
        &self,
        recorder: &CommandBufferRecorder,
        buffers: &[BufferUploadRequest],
        images: &[ImageUploadRequest],
        from: u32,
        to: u32,
    ) {
        let buffer_barriers = buffers
            .iter()
            .map(|request| BufferBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::TransferWrite],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                buffer: request.dst,
                offset: request.dst_offset as _,
                size: request.size as _,
            })
            .collect::<Vec<_>>();

        let image_barriers = images
            .iter()
            .map(|request| ImageBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::TransferWrite],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                image: request.dst,
                range: request.subresource,
            })
            .collect::<Vec<_>>();

        pipeline_barrier(
            recorder.device,
            *recorder.cb,
            None,
            &buffer_barriers,
            &image_barriers,
        );
    }

    fn barrier_after(
        &self,
        recorder: &CommandBufferRecorder,
        buffers: &[BufferUploadRequest],
        images: &[ImageUploadRequest],
        from: u32,
        to: u32,
    ) {
        let buffer_barriers = buffers
            .iter()
            .map(|request| BufferBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[
                    AccessType::VertexBuffer,
                    AccessType::IndexBuffer,
                    AccessType::AnyShaderReadUniformBuffer,
                ],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                buffer: request.dst,
                offset: request.dst_offset as _,
                size: request.size as _,
            })
            .collect::<Vec<_>>();

        let image_barriers = images
            .iter()
            .map(|request| ImageBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                image: request.dst,
                range: request.subresource,
            })
            .collect::<Vec<_>>();

        pipeline_barrier(
            recorder.device,
            *recorder.cb,
            None,
            &buffer_barriers,
            &image_barriers,
        );
    }

    fn copy_buffers(&self, recorder: &CommandBufferRecorder, requests: &[BufferUploadRequest]) {
        requests.iter().for_each(|request| {
            let region = vk::BufferCopy {
                src_offset: request.src_offset,
                dst_offset: request.dst_offset,
                size: request.size,
            };
            unsafe {
                recorder.device.cmd_copy_buffer(
                    *recorder.cb,
                    self.buffer.raw,
                    request.dst,
                    slice::from_ref(&region),
                )
            };
        });
    }

    fn copy_images(&self, recorder: &CommandBufferRecorder, requests: &[ImageUploadRequest]) {
        requests.iter().for_each(|requests| unsafe {
            recorder.device.cmd_copy_buffer_to_image(
                *recorder.cb,
                self.buffer.raw,
                requests.dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&requests.op),
            )
        });
    }
}
