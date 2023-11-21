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

use std::{collections::HashSet, mem, slice};

use arrayvec::ArrayVec;
use ash::vk;
use dess_common::{Handle, Pool};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use log::debug;

use crate::{BackendError, BackendResult};

use super::{
    Buffer, BufferHandle, BufferStorage, DescriptorSet, DescriptorSetInfo, Device, Image,
    ImageHandle, ImageStorage, ImageViewDesc, UniformStorage,
};

#[derive(Debug, Clone, Copy)]
pub struct BindingPoint<T> {
    pub binding: u32,
    pub data: Option<T>,
}

const MAX_UNIFORMS: usize = 8;
const MAX_DYNAMIC_UNIFORMS: usize = 2;
const MAX_IMAGES: usize = 8;
const MAX_DESCRIPTORS_PER_UPDATE: usize = 512;

#[derive(Debug)]
pub struct DescriptorData {
    pub(crate) descriptor: Option<DescriptorSet>,
    pub(crate) static_uniforms: ArrayVec<BindingPoint<(u32, u32)>, MAX_UNIFORMS>,
    pub(crate) dynamic_uniforms: ArrayVec<BindingPoint<vk::Buffer>, MAX_DYNAMIC_UNIFORMS>,
    pub(crate) images: ArrayVec<BindingPoint<(vk::ImageView, vk::ImageLayout)>, MAX_IMAGES>,
    pub(crate) count: DescriptorTotalCount,
    pub(crate) layout: vk::DescriptorSetLayout,
}

pub type DescriptorHandle = Handle<vk::DescriptorSet, Box<DescriptorData>>;
pub(crate) type DescriptorStorage = Pool<vk::DescriptorSet, Box<DescriptorData>>;

pub trait PushAndGetRef<T> {
    fn add(&mut self, data: T) -> &T;
}

impl<T, const CAP: usize> PushAndGetRef<T> for ArrayVec<T, CAP> {
    fn add(&mut self, data: T) -> &T {
        let index = self.len();
        self.push(data);
        &self[index]
    }
}

impl DescriptorData {
    pub fn is_valid(&self) -> bool {
        self.static_uniforms
            .iter()
            .all(|buffer| buffer.data.is_some())
            && self.images.iter().all(|image| image.data.is_some())
            && self
                .dynamic_uniforms
                .iter()
                .all(|buffer| buffer.data.is_some())
            && self.descriptor.is_some()
    }
}

pub struct UpdateDescriptorContext<'a> {
    device: &'a Device,
    uniforms: &'a mut UniformStorage,
    storage: &'a mut DescriptorStorage,
    dirty: &'a mut HashSet<DescriptorHandle>,
    images: &'a ImageStorage,
    buffers: &'a BufferStorage,
    retired_descriptors: Vec<DescriptorData>,
    retired_uniforms: Vec<u32>,
}

impl<'a> UpdateDescriptorContext<'a> {
    pub fn create(&mut self, set: &DescriptorSetInfo) -> Result<DescriptorHandle, BackendError> {
        let static_uniforms = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER {
                    Some(BindingPoint::<(u32, u32)> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<ArrayVec<_, MAX_UNIFORMS>>();
        let images = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    Some(BindingPoint::<(vk::ImageView, vk::ImageLayout)> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<ArrayVec<_, MAX_IMAGES>>();
        let dynamic_uniforms = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(BindingPoint::<vk::Buffer> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<ArrayVec<_, MAX_DYNAMIC_UNIFORMS>>();
        let count = DescriptorTotalCount {
            combined_image_sampler: images.len() as _,
            uniform_buffer: static_uniforms.len() as _,
            ..Default::default()
        };
        let handle = self.storage.push(
            vk::DescriptorSet::null(),
            Box::new(DescriptorData {
                descriptor: None,
                static_uniforms,
                dynamic_uniforms,
                images,
                count,
                layout: set.layout,
            }),
        );

        self.dirty.insert(handle);
        Ok(handle)
    }

    pub fn resolve(&self, handle: DescriptorHandle) -> Option<vk::DescriptorSet> {
        self.storage
            .get_hot(handle)
            .copied()
            .filter(|value| *value != vk::DescriptorSet::null())
    }

    pub fn remove(&mut self, handle: DescriptorHandle) {
        if let Some((_, value)) = self.storage.remove(handle) {
            value
                .static_uniforms
                .iter()
                .filter_map(|x| x.data)
                .for_each(|x| self.retired_uniforms.push(x.0));
            self.retired_descriptors.push(*value);
        }
    }

    pub fn bind_image(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        if let Some(image) = self.images.get_cold(image) {
            self.bind_image_direct(handle, binding, image, layout)
        } else {
            debug!("Attemt to bind invalid image handle {}", image);
            Ok(())
        }
    }

    pub fn bind_dynamic_uniform(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        buffer: BufferHandle,
    ) -> Result<(), BackendError> {
        if let Some(buffer) = self.buffers.get_cold(buffer) {
            self.bind_dynamic_uniform_direct(handle, binding, buffer)
        } else {
            debug!("Attemt to bind invalid buffer handle {}", buffer);
            Ok(())
        }
    }

    pub fn bind_image_direct(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        image: &Image,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        if let Some(desc) = self.storage.get_cold_mut(handle) {
            let image_bind = desc
                .images
                .iter_mut()
                .find(|point| point.binding == binding as u32);
            if let Some(point) = image_bind {
                point.data = Some((
                    image.get_or_create_view(&self.device.raw, ImageViewDesc::default())?,
                    layout,
                ));
                self.dirty.insert(handle);
            }
        }

        Ok(())
    }

    pub fn bind_uniform<T: Sized>(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        data: &T,
    ) -> Result<(), BackendError> {
        if let Some(desc) = self.storage.get_cold_mut(handle) {
            let desc = desc
                .static_uniforms
                .iter_mut()
                .find(|point| point.binding == binding as u32);
            if let Some(point) = desc {
                if let Some(old) = point
                    .data
                    .replace((self.uniforms.push(data)? as u32, mem::size_of::<T>() as u32))
                {
                    self.retired_uniforms.push(old.0);
                }
                self.dirty.insert(handle);
            }
        }
        Ok(())
    }

    pub fn bind_dynamic_uniform_direct(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        buffer: &Buffer,
    ) -> Result<(), BackendError> {
        if let Some(desc) = self.storage.get_cold_mut(handle) {
            let desc = desc
                .dynamic_uniforms
                .iter_mut()
                .find(|point| point.binding == binding as u32);
            if let Some(point) = desc {
                point.data = Some(buffer.raw);
                self.dirty.insert(handle);
            }
        }
        Ok(())
    }
}

impl<'a> Drop for UpdateDescriptorContext<'a> {
    fn drop(&mut self) {
        let mut drop_list = self.device.current_drop_list.lock();

        self.retired_descriptors
            .drain(..)
            .filter_map(|x| x.descriptor)
            .for_each(|x| drop_list.free_descriptor_set(x));
        self.retired_uniforms
            .drain(..)
            .for_each(|x| drop_list.free_uniform(x as _));
    }
}

impl Device {
    pub(crate) fn update_descriptor_sets(&self) -> Result<(), BackendError> {
        puffin::profile_function!();
        let mut drop_list = self.current_drop_list.lock();
        let mut allocator = self.descriptor_allocator.lock();
        let mut dirty = self.dirty_descriptors.lock();
        let mut storage = self.descriptor_storage.write();
        for handle in dirty.iter() {
            if let Some(desc) = storage.get_cold_mut(*handle) {
                if let Some(old) = desc.descriptor.take() {
                    drop_list.free_descriptor_set(old);
                }
                desc.descriptor = Some(
                    unsafe {
                        allocator.allocate(
                            AshDescriptorDevice::wrap(&self.raw),
                            &desc.layout,
                            DescriptorSetLayoutCreateFlags::empty(),
                            &desc.count,
                            1,
                        )
                    }?
                    .remove(0),
                );
            }
        }
        let mut writes = ArrayVec::<_, MAX_DESCRIPTORS_PER_UPDATE>::new();
        let mut buffers = ArrayVec::<_, MAX_DESCRIPTORS_PER_UPDATE>::new();
        let mut images = ArrayVec::<_, MAX_DESCRIPTORS_PER_UPDATE>::new();
        let mut uniforms = self.uniform_storage.lock();
        dirty.iter().for_each(|x| {
            if writes.is_full() || buffers.is_full() || images.is_full() {
                unsafe { self.raw.update_descriptor_sets(&writes, &[]) };
                writes.clear();
                buffers.clear();
                images.clear();
            }
            Self::prepare_descriptor(
                *x,
                &mut uniforms,
                &mut storage,
                &mut writes,
                &mut images,
                &mut buffers,
            );
        });
        if !writes.is_empty() {
            unsafe { self.raw.update_descriptor_sets(&writes, &[]) };
        }

        // Make dirty descriptors valid if they have eveything
        dirty.iter().for_each(|handle| {
            if let Some(desc) = storage.get_cold_mut(*handle) {
                if desc.is_valid() {
                    if let Some(descriptor) = &desc.descriptor {
                        let raw = *descriptor.raw();
                        storage.replace_hot(*handle, raw);
                    }
                }
            }
        });
        // Discard old data
        dirty.retain(|x| !Self::is_valid_descriptor_impl(&storage, *x));

        Ok(())
    }

    fn prepare_descriptor(
        handle: DescriptorHandle,
        uniforms: &mut UniformStorage,
        storage: &mut DescriptorStorage,
        writes: &mut ArrayVec<vk::WriteDescriptorSet, MAX_DESCRIPTORS_PER_UPDATE>,
        images: &mut ArrayVec<vk::DescriptorImageInfo, MAX_DESCRIPTORS_PER_UPDATE>,
        buffers: &mut ArrayVec<vk::DescriptorBufferInfo, MAX_DESCRIPTORS_PER_UPDATE>,
    ) {
        if let Some(desc) = storage.get_cold(handle) {
            if !desc.is_valid() {
                return;
            }
            if let Some(descriptor) = &desc.descriptor {
                desc.images
                    .iter()
                    .map(|binding| {
                        let image = binding.data.as_ref().unwrap();
                        let image = vk::DescriptorImageInfo::builder()
                            .image_view(image.0)
                            .image_layout(image.1)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .image_info(slice::from_ref(images.add(image)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.static_uniforms
                    .iter()
                    .map(|binding| {
                        let buffer = binding.data.unwrap();
                        let data = uniforms.raw;
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(data)
                            .offset(buffer.0 as _)
                            .range(buffer.1 as _)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .buffer_info(slice::from_ref(buffers.add(buffer)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.dynamic_uniforms
                    .iter()
                    .map(|binding| {
                        let data = &binding.data.unwrap();
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(*data)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .buffer_info(slice::from_ref(buffers.add(buffer)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
            }
        }
    }

    pub fn with_descriptors<F: FnOnce(UpdateDescriptorContext) -> BackendResult<()>>(
        &self,
        cb: F,
    ) -> BackendResult<()> {
        let context = UpdateDescriptorContext {
            device: self,
            uniforms: &mut self.uniform_storage.lock(),
            storage: &mut self.descriptor_storage.write(),
            dirty: &mut self.dirty_descriptors.lock(),
            images: &self.image_storage.read(),
            buffers: &self.buffer_storage.read(),
            retired_descriptors: Vec::with_capacity(MAX_DESCRIPTORS_PER_UPDATE),
            retired_uniforms: Vec::with_capacity(MAX_DESCRIPTORS_PER_UPDATE),
        };
        cb(context)
    }

    fn is_valid_descriptor_impl(
        container: &Pool<vk::DescriptorSet, Box<DescriptorData>>,
        handle: DescriptorHandle,
    ) -> bool {
        if let Some(desc) = container.get_cold(handle) {
            desc.is_valid()
        } else {
            false
        }
    }
}