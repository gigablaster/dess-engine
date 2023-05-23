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

use std::{fmt::Debug, mem::size_of, sync::Arc};

use ash::vk;
use dess_common::memory::DynamicAllocator;
use dess_render_backend::{Buffer, BufferDesc, BufferView, Device, PipelineVertex};

use crate::RenderResult;

pub type Index = u16;

#[derive(Debug, Clone)]
pub struct CachedBuffer {
    pub buffer: Arc<Buffer>,
    pub offset: u32,
    pub size: u32,
}

impl BufferView for CachedBuffer {
    fn buffer(&self) -> vk::Buffer {
        self.buffer.raw
    }

    fn offset(&self) -> u64 {
        self.offset as _
    }

    fn size(&self) -> u64 {
        self.size as _
    }
}

#[derive(Debug, Clone)]
pub struct StaticGeometry {
    pub vertices: CachedBuffer,
    pub indices: CachedBuffer,
    pub name: Option<String>,
}

#[derive(Debug)]
pub struct GeometryCache {
    pub buffer: Arc<Buffer>,
    allocator: DynamicAllocator,
}

impl GeometryCache {
    pub fn new(device: &Arc<Device>, size: usize) -> RenderResult<Self> {
        let buffer = Arc::new(Buffer::graphics(
            device,
            BufferDesc::gpu_only(
                size,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .dedicated(true),
            Some("static geometry cache"),
        )?);
        let allocator = DynamicAllocator::new(
            size as _,
            64.max(device.pdevice.properties.limits.non_coherent_atom_size),
        );

        Ok(Self { buffer, allocator })
    }

    pub fn allocate(&mut self, size: usize) -> RenderResult<CachedBuffer> {
        if let Some(offset) = self.allocator.alloc(size as _) {
            Ok(CachedBuffer {
                buffer: self.buffer.clone(),
                offset: offset as _,
                size: size as _,
            })
        } else {
            Err(crate::RenderError::NotEnoughCacheMemory)
        }
    }

    pub fn deallocate(&mut self, buffer: CachedBuffer) {
        self.allocator.free(buffer.offset as _);
    }

    pub fn create<T: PipelineVertex>(
        &mut self,
        vertex_count: usize,
        index_count: usize,
        name: Option<&str>,
    ) -> RenderResult<StaticGeometry> {
        if let Ok(vertex_buffer) = self.allocate(vertex_count * size_of::<T>()) {
            if let Ok(index_buffer) = self.allocate(index_count * size_of::<Index>()) {
                return Ok(StaticGeometry {
                    vertices: vertex_buffer,
                    indices: index_buffer,
                    name: name.map(|name| name.to_string()),
                });
            }
            self.deallocate(vertex_buffer);
        }

        Err(crate::RenderError::NotEnoughCacheMemory)
    }
}
