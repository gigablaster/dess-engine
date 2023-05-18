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

use std::{ptr::copy_nonoverlapping, slice};

use ash::vk;
use log::debug;
use vk_sync::{cmd::pipeline_barrier, AccessType, BufferBarrier};

use super::{
    memory::{allocate_vram, DynamicAllocator},
    BackendResult, Buffer, FreeGpuResource, PhysicalDevice,
};

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
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    granularity: u64,
    allocator: DynamicAllocator, // We supposed to use bump allocator but it will work in same way.
    upload_buffers: Vec<BufferUploadRequest>,
    upload_images: Vec<BufferUploadRequest>,
    mapping: Option<*mut u8>,
}

#[derive(Debug)]
pub enum StagingError {
    NeedUpload,
    VulkanError(vk::Result),
}

impl From<vk::Result> for StagingError {
    fn from(value: vk::Result) -> Self {
        StagingError::VulkanError(value)
    }
}

impl Staging {
    pub fn new(device: &ash::Device, pdevice: &PhysicalDevice, size: u64) -> BackendResult<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(size)
            .flags(vk::BufferCreateFlags::empty())
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;
        let requirement = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (_, memory) = allocate_vram(
            device,
            pdevice,
            requirement.size,
            requirement.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            None,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self {
            buffer,
            memory,
            size,
            granularity: pdevice.properties.limits.buffer_image_granularity,
            allocator: DynamicAllocator::new(
                size,
                pdevice.properties.limits.buffer_image_granularity,
            ),
            upload_buffers: Vec::with_capacity(128),
            upload_images: Vec::with_capacity(32),
            mapping: None,
        })
    }

    fn map_buffer(&self, device: &ash::Device) -> Result<*mut u8, StagingError> {
        Ok(
            unsafe { device.map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty()) }?
                as *mut u8,
        )
    }

    fn unmap_buffer(&self, device: &ash::Device) {
        unsafe { device.unmap_memory(self.memory) };
    }

    pub fn upload_buffer(
        &mut self,
        device: &ash::Device,
        buffer: &Buffer,
        data: &[u8],
    ) -> Result<(), StagingError> {
        assert!(data.len() as u64 <= self.size);
        assert_eq!(buffer.size, data.len() as u64);
        if self.mapping.is_none() {
            self.mapping = Some(self.map_buffer(device)?);
        }
        let mapping = self.mapping.unwrap();
        if let Ok(block) = self.allocator.alloc(data.len() as _) {
            unsafe {
                copy_nonoverlapping(data.as_ptr(), mapping.add(block.offset as _), data.len())
            }
            let request = BufferUploadRequest {
                dst: buffer.buffer,
                src_offset: block.offset,
                dst_offset: buffer.offset,
                size: buffer.size,
            };
            self.upload_buffers.push(request);
            debug!("Query buffer upload {:?}", request);
        } else {
            debug!("No more space in staging - request upload");
            return Err(StagingError::NeedUpload);
        }

        Ok(())
    }

    pub fn upload(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        transfer_queue_index: u32,
        graphics_queue_index: u32,
    ) {
        if self.upload_buffers.is_empty() && self.upload_images.is_empty() {
            return;
        }
        if self.mapping.is_some() {
            self.unmap_buffer(device);
            self.mapping = None;
        }

        // Move main buffer to transfer queue and keep it there
        let barrier = BufferBarrier {
            previous_accesses: &[AccessType::HostWrite, AccessType::TransferRead],
            next_accesses: &[AccessType::HostWrite, AccessType::TransferRead],
            src_queue_family_index: 0,
            dst_queue_family_index: transfer_queue_index,
            buffer: self.buffer,
            offset: 0,
            size: self.size as _,
        };
        pipeline_barrier(device, cb, None, &[barrier], &[]);

        // Record buffer uploads
        self.move_requests_to_queue(
            &self.upload_buffers,
            device,
            cb,
            graphics_queue_index,
            transfer_queue_index,
        );
        self.copy_buffers(&self.upload_buffers, device, cb);
        self.move_requests_to_queue(
            &self.upload_buffers,
            device,
            cb,
            transfer_queue_index,
            graphics_queue_index,
        );

        self.allocator = DynamicAllocator::new(self.size, self.granularity);
        self.upload_buffers.clear();
    }

    fn move_requests_to_queue(
        &self,
        requests: &[BufferUploadRequest],
        device: &ash::Device,
        cb: vk::CommandBuffer,
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
        pipeline_barrier(device, cb, None, &barriers, &[]);
    }

    fn copy_buffers(
        &self,
        requests: &[BufferUploadRequest],
        device: &ash::Device,
        cb: vk::CommandBuffer,
    ) {
        requests.iter().for_each(|request| {
            let region = vk::BufferCopy {
                src_offset: request.src_offset,
                dst_offset: request.dst_offset,
                size: request.size,
            };
            debug!("Upload request {:?}", request);
            unsafe {
                device.cmd_copy_buffer(cb, self.buffer, request.dst, slice::from_ref(&region))
            };
        });
    }
}

impl FreeGpuResource for Staging {
    fn free(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}
