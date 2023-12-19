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
    collections::{HashMap, HashSet},
    mem, slice,
};

use ash::vk;
use dess_common::{Handle, HotColdPool, SentinelPoolStrategy, TempList};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use smol_str::SmolStr;

use crate::{BackendError, BackendResult};

use super::{
    BufferHandle, BufferSlice, BufferStorage, DescriptorSet, DescriptorSetInfo, Device,
    ImageHandle, ImageStorage, ImageViewDesc, ProgramHandle, ProgramStorage, UniformStorage,
};

#[derive(Debug, Clone, Copy)]
pub struct BindingPoint<T> {
    pub binding: u32,
    pub data: Option<T>,
}

const BASIC_DESCIPTOR_UPDATE_COUNT: usize = 512;

#[derive(Debug)]
pub struct DescriptorData {
    pub(crate) descriptor: Option<DescriptorSet>,
    pub(crate) static_uniforms: Vec<BindingPoint<(u32, u32)>>,
    pub(crate) dynamic_uniforms: Vec<BindingPoint<vk::Buffer>>,
    pub(crate) storage_buffers: Vec<BindingPoint<(BufferHandle, vk::Buffer, u32, u32)>>,
    pub(crate) storage_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    pub(crate) sampled_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    pub(crate) count: DescriptorTotalCount,
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) names: HashMap<SmolStr, usize>,
}

pub type DescriptorHandle = Handle<vk::DescriptorSet>;
pub(crate) type DescriptorStorage =
    HotColdPool<vk::DescriptorSet, Box<DescriptorData>, SentinelPoolStrategy<vk::DescriptorSet>>;

pub trait PushAndGetRef<T> {
    fn push_get_ref(&mut self, data: T) -> &T;
}

impl<T> PushAndGetRef<T> for Vec<T> {
    fn push_get_ref(&mut self, data: T) -> &T {
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
            && self.sampled_images.iter().all(|image| image.data.is_some())
            && self
                .storage_buffers
                .iter()
                .all(|buffer| buffer.data.is_some())
            && self.storage_images.iter().all(|image| image.data.is_some())
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
    programs: &'a ProgramStorage,
    images: &'a ImageStorage,
    buffers: &'a BufferStorage,
    retired_descriptors: Vec<DescriptorData>,
    retired_uniforms: Vec<u32>,
}

/// Allows client to manipulate descriptor set
impl<'a> UpdateDescriptorContext<'a> {
    pub fn from_program(
        &mut self,
        program: ProgramHandle,
        index: usize,
    ) -> Result<DescriptorHandle, BackendError> {
        let set = &self
            .programs
            .get(program.index())
            .ok_or(BackendError::InvalidHandle)?
            .sets[index];
        self.from_info(set)
    }
    /// Created descriptor set from descriptor set info
    ///
    /// Descriptor set info can be extracted from Program as an example.
    pub fn from_info(&mut self, set: &DescriptorSetInfo) -> Result<DescriptorHandle, BackendError> {
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
            .collect::<Vec<_>>();
        let sampled_images = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::SAMPLED_IMAGE {
                    Some(
                        BindingPoint::<(ImageHandle, vk::ImageView, vk::ImageLayout)> {
                            binding: *index,
                            data: None,
                        },
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_uniforms = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(BindingPoint::<vk::Buffer> {
                        binding: *index,
                        data: Some(self.device.temp_buffer),
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let storage_buffers = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::STORAGE_BUFFER {
                    Some(BindingPoint::<(BufferHandle, vk::Buffer, u32, u32)> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let storage_images = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::STORAGE_IMAGE {
                    Some(
                        BindingPoint::<(ImageHandle, vk::ImageView, vk::ImageLayout)> {
                            binding: *index,
                            data: None,
                        },
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let count = DescriptorTotalCount {
            sampled_image: sampled_images.len() as _,
            uniform_buffer: static_uniforms.len() as _,
            storage_buffer: storage_buffers.len() as _,
            storage_image: storage_images.len() as _,
            ..Default::default()
        };
        let names = set
            .names
            .iter()
            .map(|(name, slot)| (name.into(), *slot as usize))
            .collect::<HashMap<SmolStr, _>>();
        let handle = self.storage.push(
            vk::DescriptorSet::null(),
            Box::new(DescriptorData {
                descriptor: None,
                static_uniforms,
                dynamic_uniforms,
                storage_buffers,
                storage_images,
                sampled_images,
                count,
                layout: set.layout,
                names,
            }),
        );

        self.dirty.insert(handle);
        Ok(handle)
    }

    /// Mark descriptor set as invalid.
    ///
    /// It will be destroyed when current frame finished rendering..
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

    pub fn bind_storage_image(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        let data = self
            .images
            .get_cold(image)
            .ok_or(BackendError::InvalidHandle)?;
        debug_assert!(data.desc.usage.contains(vk::ImageUsageFlags::STORAGE));
        let desc = self
            .storage
            .get_cold_mut(handle)
            .ok_or(BackendError::InvalidHandle)?;
        let image_bind = desc
            .storage_images
            .iter_mut()
            .find(|point| point.binding == binding as u32);
        if let Some(point) = image_bind {
            point.data = Some((
                image,
                data.get_or_create_view(&self.device.raw, ImageViewDesc::default())?,
                layout,
            ));
            self.dirty.insert(handle);
        }

        Ok(())
    }

    pub fn bind_storage_image_by_name(
        &mut self,
        handle: DescriptorHandle,
        name: &str,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        let binding = self
            .storage
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .names
            .get(name)
            .copied()
            .ok_or(BackendError::NotFound)?;
        self.bind_storage_image(handle, binding, image, layout)
    }

    /// Bind image to descriptor set
    pub fn bind_image(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        let data = self
            .images
            .get_cold(image)
            .ok_or(BackendError::InvalidHandle)?;
        debug_assert!(data.desc.usage.contains(vk::ImageUsageFlags::SAMPLED));
        let desc = self
            .storage
            .get_cold_mut(handle)
            .ok_or(BackendError::InvalidHandle)?;
        let image_bind = desc
            .sampled_images
            .iter_mut()
            .find(|point| point.binding == binding as u32);
        if let Some(point) = image_bind {
            point.data = Some((
                image,
                data.get_or_create_view(&self.device.raw, ImageViewDesc::default())?,
                layout,
            ));
            self.dirty.insert(handle);
        }

        Ok(())
    }

    /// Bind image to descriptor set by name
    pub fn bind_image_by_name(
        &mut self,
        handle: DescriptorHandle,
        name: &str,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        let binding = self
            .storage
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .names
            .get(name)
            .copied()
            .ok_or(BackendError::NotFound)?;
        self.bind_image(handle, binding, image, layout)
    }

    /// Bind uniform to descriptor set by name
    pub fn bind_uniform_by_name<T: Sized>(
        &mut self,
        handle: DescriptorHandle,
        name: &str,
        data: &T,
    ) -> Result<(), BackendError> {
        let binding = self
            .storage
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .names
            .get(name)
            .copied()
            .ok_or(BackendError::NotFound)?;
        self.bind_uniform(handle, binding, data)
    }

    pub fn bind_uniform<T: Sized>(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        data: &T,
    ) -> Result<(), BackendError> {
        let desc = self
            .storage
            .get_cold_mut(handle)
            .ok_or(BackendError::InvalidHandle)?;
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
        Ok(())
    }

    pub fn bind_uniform_raw_by_name(
        &mut self,
        handle: DescriptorHandle,
        name: &str,
        data: &[u8],
    ) -> Result<(), BackendError> {
        let binding = self
            .storage
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .names
            .get(name)
            .copied()
            .ok_or(BackendError::NotFound)?;
        self.bind_uniform_raw(handle, binding, data)
    }

    pub fn bind_uniform_raw(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        data: &[u8],
    ) -> Result<(), BackendError> {
        let desc = self
            .storage
            .get_cold_mut(handle)
            .ok_or(BackendError::InvalidHandle)?;
        let desc = desc
            .static_uniforms
            .iter_mut()
            .find(|point| point.binding == binding as u32);
        if let Some(point) = desc {
            if let Some(old) = point.data.replace((
                unsafe { self.uniforms.push_raw(data.as_ptr(), data.len()) }? as u32,
                data.len() as u32,
            )) {
                self.retired_uniforms.push(old.0);
            }
            self.dirty.insert(handle);
        }
        Ok(())
    }

    pub fn bind_storage_buffer(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        buffer: BufferSlice,
    ) -> BackendResult<()> {
        let desc = self
            .storage
            .get_cold_mut(handle)
            .ok_or(BackendError::InvalidHandle)?;
        let raw = self
            .buffers
            .get_cold(buffer.handle)
            .ok_or(BackendError::InvalidHandle)?;
        debug_assert!(raw
            .desc
            .usage
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        let buffer_bind = desc
            .storage_buffers
            .iter_mut()
            .find(|point| point.binding == binding as u32);
        if let Some(point) = buffer_bind {
            point.data = Some((buffer.handle, raw.raw, buffer.offset, buffer.size));
            self.dirty.insert(handle);
        }

        Ok(())
    }

    pub fn bind_storage_buffer_by_name(
        &mut self,
        handle: DescriptorHandle,
        name: &str,
        buffer: BufferSlice,
    ) -> BackendResult<()> {
        let binding = self
            .storage
            .get_cold(handle)
            .ok_or(BackendError::InvalidHandle)?
            .names
            .get(name)
            .copied()
            .ok_or(BackendError::NotFound)?;
        self.bind_storage_buffer(handle, binding, buffer)
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
        let mut writes = Vec::with_capacity(BASIC_DESCIPTOR_UPDATE_COUNT);
        let mut buffers = TempList::new();
        let mut images = TempList::new();
        let mut uniforms = self.uniform_storage.lock();
        dirty.iter().for_each(|x| {
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
                        storage.replace(*handle, raw);
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
        writes: &mut Vec<vk::WriteDescriptorSet>,
        images: &mut TempList<vk::DescriptorImageInfo>,
        buffers: &mut TempList<vk::DescriptorBufferInfo>,
    ) {
        if let Some(desc) = storage.get_cold(handle) {
            if !desc.is_valid() {
                return;
            }
            if let Some(descriptor) = &desc.descriptor {
                desc.sampled_images
                    .iter()
                    .map(|binding| {
                        let image = binding.data.as_ref().unwrap();
                        let image = vk::DescriptorImageInfo::builder()
                            .image_view(image.1)
                            .image_layout(image.2)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .image_info(slice::from_ref(images.add(image)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.storage_images
                    .iter()
                    .map(|binding| {
                        let image = binding.data.as_ref().unwrap();
                        let image = vk::DescriptorImageInfo::builder()
                            .image_view(image.1)
                            .image_layout(image.2)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .image_info(slice::from_ref(images.add(image)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
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
                desc.storage_buffers
                    .iter()
                    .map(|binding| {
                        let data = &binding.data.unwrap();
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(data.1)
                            .offset(data.2 as u64)
                            .range(data.3 as u64)
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

    pub fn with_descriptors<F>(&self, cb: F) -> BackendResult<()>
    where
        F: FnOnce(&mut UpdateDescriptorContext) -> BackendResult<()>,
    {
        let mut context = UpdateDescriptorContext {
            device: self,
            uniforms: &mut self.uniform_storage.lock(),
            storage: &mut self.descriptor_storage.write(),
            dirty: &mut self.dirty_descriptors.lock(),
            images: &self.image_storage.read(),
            buffers: &self.buffer_storage.read(),
            programs: &self.program_storage.read(),
            retired_descriptors: Vec::with_capacity(BASIC_DESCIPTOR_UPDATE_COUNT),
            retired_uniforms: Vec::with_capacity(BASIC_DESCIPTOR_UPDATE_COUNT),
        };
        cb(&mut context)
    }

    fn is_valid_descriptor_impl(
        container: &HotColdPool<
            vk::DescriptorSet,
            Box<DescriptorData>,
            SentinelPoolStrategy<vk::DescriptorSet>,
        >,
        handle: DescriptorHandle,
    ) -> bool {
        if let Some(desc) = container.get_cold(handle) {
            desc.is_valid()
        } else {
            false
        }
    }
}
