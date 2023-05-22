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

use std::{ptr::copy_nonoverlapping, slice, sync::Arc};

use ash::vk;
use dess_common::memory::BumpAllocator;
use dess_render_backend::{
    Buffer, BufferDesc, BufferView, CommandBuffer, CommandBufferRecorder, Device,
};
use vk_sync::{cmd::pipeline_barrier, AccessType, BufferBarrier};

use crate::RenderResult;

#[derive(Debug, Copy, Clone)]
struct BufferUploadRequest {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
    pub dst: vk::Buffer,
}

#[derive(Debug, Copy, Clone)]
struct ImageUploadRequest {
    pub src_offset: u64,
    pub dst_offset: vk::Offset3D,
    pub dst_subresource: vk::ImageSubresourceLayers,
    pub dst: vk::Image,
    pub dst_layout: vk::ImageLayout,
}

pub struct Staging {
    device: Arc<Device>,
    buffer: Buffer,
    tranfser_cb: CommandBuffer,
    size: u64,
    allocator: BumpAllocator,
    upload_buffers: Vec<BufferUploadRequest>,
    upload_images: Vec<BufferUploadRequest>,
    mapping: Option<*mut u8>,
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

    fn map_buffer(&mut self) -> RenderResult<*mut u8> {
        Ok(self.buffer.map()?)
    }

    fn unmap_buffer(&mut self) {
        self.buffer.unmap();
    }

    pub fn upload_buffer(
        &mut self,
        _device: &ash::Device,
        buffer: &impl BufferView,
        data: &[u8],
    ) -> RenderResult<u64> {
        assert!(data.len() as u64 <= self.size);
        assert_eq!(buffer.size(), data.len() as u64);
        self.tranfser_cb.wait(&self.device.raw)?;
        if self.mapping.is_none() {
            self.mapping = Some(self.map_buffer()?);
        }
        let mapping = self.mapping.unwrap();
        if !self.try_push_buffer(buffer, data, mapping) {
            self.upload()?;
            if !self.try_push_buffer(buffer, data, mapping) {
                panic!(
                    "Despite just pushing entire staging buffer we still can't push data in it."
                );
            }
        }

        Ok(self.index)
    }

    fn try_push_buffer(&mut self, buffer: &impl BufferView, data: &[u8], mapping: *mut u8) -> bool {
        if let Some(offset) = self.allocator.allocate(data.len() as _) {
            unsafe { copy_nonoverlapping(data.as_ptr(), mapping.add(offset as _), data.len()) }
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
        self.tranfser_cb.wait(&self.device.raw)?;
        self.tranfser_cb.reset(&self.device.raw)?;
        if self.mapping.is_some() {
            self.unmap_buffer();
            self.mapping = None;
        }

        self.tranfser_cb.record(&self.device.raw, |recorder| {
            // Record buffer uploads
            self.move_requests_to_queue(
                &recorder,
                &self.upload_buffers,
                self.device.graphics_queue.family.index,
                self.device.transfer_queue.family.index,
            );
            self.copy_buffers(&recorder, &self.upload_buffers);
            self.move_requests_to_queue(
                &recorder,
                &self.upload_buffers,
                self.device.transfer_queue.family.index,
                self.device.graphics_queue.family.index,
            );
        })?;

        self.allocator.reset();
        self.upload_buffers.clear();

        self.index += 1;

        Ok(())
    }

    fn move_requests_to_queue(
        &self,
        recorder: &CommandBufferRecorder,
        requests: &[BufferUploadRequest],
        from: u32,
        to: u32,
    ) {
        let barriers = requests
            .iter()
            .map(|request| BufferBarrier {
                previous_accesses: &[
                    AccessType::TransferWrite,
                    AccessType::VertexBuffer,
                    AccessType::IndexBuffer,
                ],
                next_accesses: &[
                    AccessType::TransferWrite,
                    AccessType::VertexBuffer,
                    AccessType::IndexBuffer,
                ],
                src_queue_family_index: from,
                dst_queue_family_index: to,
                buffer: request.dst,
                offset: request.dst_offset as _,
                size: request.size as _,
            })
            .collect::<Vec<_>>();
        pipeline_barrier(recorder.device, *recorder.cb, None, &barriers, &[]);
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
}
